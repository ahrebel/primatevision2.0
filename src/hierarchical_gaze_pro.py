#!/usr/bin/env python
import os, re, argparse, copy
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from typing import Dict, List
from sklearn.model_selection import KFold
from tqdm import tqdm

def create_grid(screen_width, screen_height, n_cols, n_rows):
   grid = []
   cell_w = screen_width / n_cols
   cell_h = screen_height / n_rows
   region_id = 1
   for r in range(n_rows):
       for c in range(n_cols):
           x1 = c * cell_w
           y1 = r * cell_h
           x2 = (c + 1) * cell_w
           y2 = (r + 1) * cell_h
           grid.append((c, r, x1, y1, x2, y2, region_id))
           region_id += 1
   return grid

def get_coarse_grid():
   return {
       "A": {"xmin": 0, "xmax": 512, "ymin": 0, "ymax": 384},
       "B": {"xmin": 512, "xmax": 1024, "ymin": 0, "ymax": 384},
       "C": {"xmin": 0, "xmax": 512, "ymin": 384, "ymax": 768},
       "D": {"xmin": 512, "xmax": 1024, "ymin": 384, "ymax": 768},
   }

def get_fine_grid(coarse_label: str, coarse_grid: Dict[str, Dict[str, float]]):
   b = coarse_grid[coarse_label]
   xm = (b["xmin"] + b["xmax"]) / 2
   ym = (b["ymin"] + b["ymax"]) / 2
   return {
       coarse_label + "A": {"xmin": b["xmin"], "xmax": xm, "ymin": b["ymin"], "ymax": ym},
       coarse_label + "B": {"xmin": xm, "xmax": b["xmax"], "ymin": b["ymin"], "ymax": ym},
       coarse_label + "C": {"xmin": b["xmin"], "xmax": xm, "ymin": ym, "ymax": b["ymax"]},
       coarse_label + "D": {"xmin": xm, "xmax": b["xmax"], "ymin": ym, "ymax": b["ymax"]},
   }

def compute_head_pose_approx(nose_x, nose_y, clx, cly, crx, cry):
   dx_l = clx - nose_x
   dy_l = cly - nose_y
   dx_r = crx - nose_x
   dy_r = cry - nose_y
   yaw = np.arctan2(dy_r, dx_r) - np.arctan2(dy_l, dx_l)
   pitch = (dy_l + dy_r) / 2.0
   slope_l = dy_l / (dx_l + 1e-7)
   slope_r = dy_r / (dx_r + 1e-7)
   roll = slope_r - slope_l
   return pitch, yaw, roll

REQUIRED_COLUMNS = [
   "nose_x", "nose_y",
   "corner_left_x", "corner_left_y", "corner_right_x", "corner_right_y",
   "left_pupil_x", "left_pupil_y", "right_pupil_x", "right_pupil_y",
   "screen_x", "screen_y"
]

class CoarseGazeDataset(Dataset):
   def __init__(self, csv_path: str, balanced_ce: bool=False, normalize: bool=False):
       df = pd.read_csv(csv_path)
       df.drop_duplicates(inplace=True)
       for col in REQUIRED_COLUMNS:
           if col not in df.columns:
               raise ValueError(f"Missing '{col}' in {csv_path}")
       df["screen_x"] = df["screen_x"].clip(lower=0, upper=1024)
       df["screen_y"] = df["screen_y"].clip(lower=0, upper=768)
       self.head_data = df[["nose_x", "nose_y", "corner_left_x", "corner_left_y", "corner_right_x", "corner_right_y"]].values.astype(np.float32)
       self.pupil_data = df[["left_pupil_x", "left_pupil_y", "right_pupil_x", "right_pupil_y"]].values.astype(np.float32)
       self.screen_xy = df[["screen_x", "screen_y"]].values.astype(np.float32)
       pose_list = []
       for i in range(len(df)):
           nx, ny, clx, cly, crx, cry = self.head_data[i]
           pitch, yaw, roll = compute_head_pose_approx(nx, ny, clx, cly, crx, cry)
           pose_list.append([pitch, yaw, roll])
       self.pose_data = np.array(pose_list, dtype=np.float32)
       self.input_data = np.concatenate([self.head_data, self.pupil_data, self.pose_data], axis=1)
       self.normalize = normalize
       if normalize:
           self.mean_ = self.input_data.mean(axis=0)
           self.std_ = self.input_data.std(axis=0) + 1e-7
           self.input_data = (self.input_data - self.mean_) / self.std_
       self.grid = get_coarse_grid()
       self.grid_keys = sorted(self.grid.keys())
       self.labels = []
       self.rel_coords = []
       for pt in self.screen_xy:
           assigned = False
           for i, key in enumerate(self.grid_keys):
               b = self.grid[key]
               if b["xmin"] <= pt[0] < b["xmax"] and b["ymin"] <= pt[1] < b["ymax"]:
                   self.labels.append(i)
                   rx = (pt[0] - b["xmin"]) / (b["xmax"] - b["xmin"] + 1e-7)
                   ry = (pt[1] - b["ymin"]) / (b["ymax"] - b["ymin"] + 1e-7)
                   self.rel_coords.append([rx, ry])
                   assigned = True
                   break
           if not assigned:
               raise ValueError(f"Point {pt} not in any coarse region!?")
       self.labels = np.array(self.labels, dtype=np.int64)
       self.rel_coords = np.array(self.rel_coords, dtype=np.float32)
       self.balanced_ce = balanced_ce
       self.class_weights = None
       if balanced_ce:
           from collections import Counter
           freq = Counter(self.labels.tolist())
           total = len(self.labels)
           weights = [total / (freq.get(i, 1) + 1e-7) for i in range(len(self.grid_keys))]
           self.class_weights = torch.tensor(weights, dtype=torch.float32)
           print(f"[Coarse] Balanced CE: counts={dict(freq)}, weights={weights}")
   def __len__(self):
       return len(self.labels)
   def __getitem__(self, idx):
       x = torch.tensor(self.input_data[idx], dtype=torch.float32)
       label = torch.tensor(self.labels[idx], dtype=torch.long)
       rel = torch.tensor(self.rel_coords[idx], dtype=torch.float32)
       return x, label, rel

class FineGazeDataset(Dataset):
   def __init__(self, csv_path: str, coarse_label: str, balanced_ce=False, normalize: bool=False):
       df = pd.read_csv(csv_path)
       df.drop_duplicates(inplace=True)
       for col in REQUIRED_COLUMNS:
           if col not in df.columns:
               raise ValueError(f"Missing '{col}' in {csv_path}")
       df["screen_x"] = df["screen_x"].clip(lower=0, upper=1024)
       df["screen_y"] = df["screen_y"].clip(lower=0, upper=768)
       head_data = df[["nose_x", "nose_y", "corner_left_x", "corner_left_y", "corner_right_x", "corner_right_y"]].values.astype(np.float32)
       pupil_data = df[["left_pupil_x", "left_pupil_y", "right_pupil_x", "right_pupil_y"]].values.astype(np.float32)
       screen_xy = df[["screen_x", "screen_y"]].values.astype(np.float32)
       pose_list = []
       for i in range(len(df)):
           nx, ny, clx, cly, crx, cry = head_data[i]
           pitch, yaw, roll = compute_head_pose_approx(nx, ny, clx, cly, crx, cry)
           pose_list.append([pitch, yaw, roll])
       pose_data = np.array(pose_list, dtype=np.float32)
       combined_data = np.concatenate([head_data, pupil_data, pose_data], axis=1)
       cgrid = get_coarse_grid()
       if coarse_label not in cgrid:
           raise ValueError(f"coarse_label {coarse_label} not recognized.")
       box = cgrid[coarse_label]
       valid_idx = [i for i, (sx, sy) in enumerate(screen_xy) if box["xmin"] <= sx < box["xmax"] and box["ymin"] <= sy < box["ymax"]]
       if not valid_idx:
           raise ValueError(f"No samples in quadrant {coarse_label} found in {csv_path}.")
       self.screen_xy = screen_xy[valid_idx]
       self.input_data = combined_data[valid_idx]
       self.normalize = normalize
       if normalize:
           self.mean_ = self.input_data.mean(axis=0)
           self.std_ = self.input_data.std(axis=0) + 1e-7
           self.input_data = (self.input_data - self.mean_) / self.std_
       self.fine_grid = get_fine_grid(coarse_label, cgrid)
       self.fine_keys = sorted(self.fine_grid.keys())
       self.labels = []
       self.rel_coords = []
       for pt in self.screen_xy:
           assigned = False
           for i, key in enumerate(self.fine_keys):
               sb = self.fine_grid[key]
               if sb["xmin"] <= pt[0] < sb["xmax"] and sb["ymin"] <= pt[1] < sb["ymax"]:
                   self.labels.append(i)
                   rx = (pt[0] - sb["xmin"]) / (sb["xmax"] - sb["xmin"] + 1e-7)
                   ry = (pt[1] - sb["ymin"]) / (sb["ymax"] - sb["ymin"] + 1e-7)
                   self.rel_coords.append([rx, ry])
                   assigned = True
                   break
           if not assigned:
               raise ValueError(f"Point {pt} not in any sub-grid for quadrant {coarse_label}!")
       self.labels = np.array(self.labels, dtype=np.int64)
       self.rel_coords = np.array(self.rel_coords, dtype=np.float32)
       self.balanced_ce = balanced_ce
       self.class_weights = None
       if balanced_ce:
           from collections import Counter
           freq = Counter(self.labels.tolist())
           total = len(self.labels)
           weights = [total / (freq.get(i, 1) + 1e-7) for i in range(len(self.fine_keys))]
           self.class_weights = torch.tensor(weights, dtype=torch.float32)
           print(f"[Fine {coarse_label}] Balanced CE: counts={dict(freq)}, weights={weights}")
   def __len__(self):
       return len(self.labels)
   def __getitem__(self, idx):
       x = torch.tensor(self.input_data[idx], dtype=torch.float32)
       label = torch.tensor(self.labels[idx], dtype=torch.long)
       rel = torch.tensor(self.rel_coords[idx], dtype=torch.float32)
       return x, label, rel

class ResidualBlock(nn.Module):
   def __init__(self, dim, dropout_p=0.1):
       super().__init__()
       self.fc1 = nn.Linear(dim, dim)
       self.bn1 = nn.BatchNorm1d(dim)
       self.fc2 = nn.Linear(dim, dim)
       self.bn2 = nn.BatchNorm1d(dim)
       self.relu = nn.ReLU()
       self.drop = nn.Dropout(dropout_p)
   def forward(self, x):
       identity = x
       out = self.fc1(x)
       out = self.bn1(out)
       out = self.relu(out)
       out = self.drop(out)
       out = self.fc2(out)
       out = self.bn2(out)
       out += identity
       out = self.relu(out)
       return out

class SharedEncoder(nn.Module):
   def __init__(self, in_dim=13, embed_dim=256, dropout_p=0.1):
       super().__init__()
       self.fc_in = nn.Linear(in_dim, embed_dim)
       self.bn_in = nn.BatchNorm1d(embed_dim)
       self.res1 = ResidualBlock(embed_dim, dropout_p)
       self.res2 = ResidualBlock(embed_dim, dropout_p)
       self.fc_out = nn.Linear(embed_dim, embed_dim)
       self.drop = nn.Dropout(dropout_p)
       self.relu = nn.ReLU()
   def forward(self, x):
       x = self.fc_in(x)
       x = self.bn_in(x)
       x = self.relu(x)
       x = self.drop(x)
       x = self.res1(x)
       x = self.res2(x)
       x = self.fc_out(x)
       return x

class CoarseGazeNet(nn.Module):
   def __init__(self, encoder_dim=256, hidden_dim=128, num_classes=4):
       super().__init__()
       self.shared_encoder = SharedEncoder(in_dim=13, embed_dim=encoder_dim)
       self.bn_head = nn.BatchNorm1d(encoder_dim)
       self.fc_comb = nn.Linear(encoder_dim, hidden_dim)
       self.bn_comb = nn.BatchNorm1d(hidden_dim)
       self.class_out = nn.Linear(hidden_dim, num_classes)
       self.reg_out = nn.Linear(hidden_dim, 2)
       self.relu = nn.ReLU()
       self.drop = nn.Dropout(0.1)
   def forward(self, head, pupil):
       batch_size = head.size(0)
       x = torch.zeros(batch_size, 13, device=head.device)
       left_eye_x = pupil[:, 0, 0]
       left_eye_y = pupil[:, 0, 1]
       right_eye_x = pupil[:, 1, 0]
       right_eye_y = pupil[:, 1, 1]
       eye_center_x = (left_eye_x + right_eye_x) / 2
       eye_center_y = (left_eye_y + right_eye_y) / 2
       nose_x = head[:, 0, 0]
       nose_y = head[:, 0, 1]
       gaze_x = nose_x - eye_center_x
       gaze_y = nose_y - eye_center_y
       gaze_length = torch.sqrt(gaze_x**2 + gaze_y**2)
       gaze_x = gaze_x / (gaze_length + 1e-7)
       gaze_y = gaze_y / (gaze_length + 1e-7)
       x[:, 0] = gaze_x
       x[:, 1] = gaze_y
       x[:, 2] = (left_eye_x - eye_center_x)
       x[:, 3] = (left_eye_y - eye_center_y)
       x[:, 4] = (right_eye_x - eye_center_x)
       x[:, 5] = (right_eye_y - eye_center_y)
       x[:, 6] = (head[:, 1, 0] - eye_center_x)
       x[:, 7] = (head[:, 1, 1] - eye_center_y)
       x[:, 8] = (head[:, 2, 0] - eye_center_x)
       x[:, 9] = (head[:, 2, 1] - eye_center_y)
       eye_dist = torch.sqrt((right_eye_x - left_eye_x)**2 + (right_eye_y - left_eye_y)**2)
       x[:, 2:10] = x[:, 2:10] / (eye_dist.unsqueeze(1) + 1e-7)
       x[:, 10] = torch.atan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x)
       x[:, 11] = torch.atan2(gaze_y, gaze_x)
       x[:, 12] = torch.atan2(head[:, 2, 1] - head[:, 1, 1],
                             head[:, 2, 0] - head[:, 1, 0])
       f = self.shared_encoder(x)
       f = self.fc_comb(f)
       f = self.bn_comb(f)
       f = self.relu(f)
       f = self.drop(f)
       logits = self.class_out(f)
       coords = torch.clamp(self.reg_out(f), 0, 1)
       return logits, coords

class FineGazeNet(nn.Module):
   def __init__(self, embed_dim=256, num_fine=4, dropout_p=0.1):
       super().__init__()
       self.shared_encoder = SharedEncoder(in_dim=13, embed_dim=embed_dim, dropout_p=dropout_p)
       self.bn_head = nn.BatchNorm1d(embed_dim)
       self.fc_comb = nn.Linear(embed_dim, 128)
       self.bn_comb = nn.BatchNorm1d(128)
       self.relu = nn.ReLU()
       self.drop = nn.Dropout(dropout_p)
       self.class_out = nn.Linear(128, num_fine)
       self.reg_out = nn.Linear(128, 2)
       self.sigmoid = nn.Sigmoid()
   def forward(self, x):
       embed = self.shared_encoder(x)
       h = self.bn_head(embed)
       h = self.relu(h)
       h = self.drop(h)
       h = self.fc_comb(h)
       h = self.bn_comb(h)
       h = self.relu(h)
       h = self.drop(h)
       logits = self.class_out(h)
       rel_xy = torch.clamp(self.reg_out(h), 0, 1)
       return logits, rel_xy

def create_balanced_sampler(labels: np.ndarray):
   from collections import Counter
   freq = Counter(labels.tolist())
   total = len(labels)
   weight_arr = []
   for i in range(total):
       label = labels[i]
       w = total / (freq[label] + 1e-7)
       weight_arr.append(w)
   sampler = WeightedRandomSampler(weight_arr, num_samples=total, replacement=True)
   return sampler

def train_one_epoch(model, dataloader, optimizer, ce_loss_fn, mse_loss_fn, alpha, device, scheduler=None):
   model.train()
   total_loss = 0.0
   total_samples = 0
   for x, label, rel in dataloader:
       x = x.to(device)
       label = label.to(device)
       rel = rel.to(device)
       optimizer.zero_grad()
       if isinstance(model, CoarseGazeNet):
           head = x[:, :6].view(-1, 3, 2)
           pupil = x[:, 6:10].view(-1, 2, 2)
           logits, pred_rel = model(head, pupil)
       else:
           logits, pred_rel = model(x)
       loss_ce = ce_loss_fn(logits, label)
       loss_mse = mse_loss_fn(pred_rel, rel)
       loss = alpha * loss_ce + (1 - alpha) * loss_mse
       loss.backward()
       optimizer.step()
       if scheduler and isinstance(scheduler, OneCycleLR):
           scheduler.step()
       total_loss += loss.item() * x.size(0)
       total_samples += x.size(0)
   return total_loss / total_samples

def validate_one_epoch(model, dataloader, ce_loss_fn, mse_loss_fn, alpha, device):
   model.eval()
   total_loss = 0.0
   total_samples = 0
   correct = 0
   with torch.no_grad():
       for x, label, rel in dataloader:
           x = x.to(device)
           label = label.to(device)
           rel = rel.to(device)
           if isinstance(model, CoarseGazeNet):
               head = x[:, :6].view(-1, 3, 2)
               pupil = x[:, 6:10].view(-1, 2, 2)
               logits, pred_rel = model(head, pupil)
           else:
               logits, pred_rel = model(x)
           loss_ce = ce_loss_fn(logits, label)
           loss_mse = mse_loss_fn(pred_rel, rel)
           loss = alpha * loss_ce + (1 - alpha) * loss_mse
           total_loss += loss.item() * x.size(0)
           total_samples += x.size(0)
           preds = torch.argmax(logits, dim=1)
           correct += (preds == label).sum().item()
   val_loss = total_loss / total_samples
   val_acc = (correct / total_samples) * 100.0
   return val_loss, val_acc

def train_model(dataset: Dataset, model: nn.Module, output_path: str, device="cpu", batch_size=32, epochs=100, patience=10, alpha=0.8, balanced_ce=False, lr=1e-3, use_onecycle=False, kfold=0):
   device = torch.device(device)
   model.to(device)
   ce_loss_fn = (nn.CrossEntropyLoss(weight=dataset.class_weights.to(device)) if getattr(dataset, "class_weights", None) is not None else nn.CrossEntropyLoss())
   mse_loss_fn = nn.MSELoss()
   if kfold > 1:
       print(f"=== K-Fold Cross Validation (k={kfold}) ===")
       all_val_accs = []
       kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
       idxs = np.arange(len(dataset))
       for fold, (train_idx, val_idx) in enumerate(kf.split(idxs), 1):
           print(f"--- Fold {fold} ---")
           train_ds = Subset(dataset, train_idx)
           val_ds = Subset(dataset, val_idx)
           fold_model = copy.deepcopy(model).to(device)
           optimizer = optim.AdamW(fold_model.parameters(), lr=lr)
           if use_onecycle:
               steps_per_epoch = len(DataLoader(train_ds, batch_size=batch_size, shuffle=True))
               scheduler = OneCycleLR(optimizer, max_lr=lr*10, total_steps=steps_per_epoch*epochs, pct_start=0.3, anneal_strategy="cos")
           else:
               scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
           best_val_loss = float("inf")
           best_state = None
           no_improve = 0
           train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
           val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
           for epoch in range(1, epochs+1):
               train_loss = train_one_epoch(fold_model, train_loader, optimizer, ce_loss_fn, mse_loss_fn, alpha, device, scheduler)
               val_loss, val_acc = validate_one_epoch(fold_model, val_loader, ce_loss_fn, mse_loss_fn, alpha, device)
               if not use_onecycle:
                   scheduler.step(val_loss)
               print(f"[Fold {fold} Epoch {epoch}/{epochs}] TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, ValAcc={val_acc:.1f}%")
               if val_loss < best_val_loss:
                   best_val_loss = val_loss
                   best_state = fold_model.state_dict()
                   no_improve = 0
               else:
                   no_improve += 1
                   if no_improve >= patience:
                       print("Early stopping triggered!")
                       break
           if best_state is not None:
               fold_model.load_state_dict(best_state)
           _, final_acc = validate_one_epoch(fold_model, val_loader, ce_loss_fn, mse_loss_fn, alpha, device)
           all_val_accs.append(final_acc)
           print(f"[Fold {fold}] Final Val Acc: {final_acc:.2f}%")
       avg_acc = np.mean(all_val_accs)
       print(f"=== Average cross-val accuracy over {kfold} folds: {avg_acc:.2f}% ===")
       save_dict = {"model_state": fold_model.state_dict()}
       if getattr(dataset, "normalize", False):
           save_dict["mean_"] = dataset.mean_.tolist()
           save_dict["std_"] = dataset.std_.tolist()
       torch.save(save_dict, output_path)
       print(f"KFold-based model saved to {output_path}")
   else:
       n_total = len(dataset)
       n_train = int(0.8 * n_total)
       n_val = n_total - n_train
       train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
       sampler = None
       if balanced_ce:
           train_labels = [dataset[i][1].item() for i in train_ds.indices]
           sampler = create_balanced_sampler(np.array(train_labels))
       train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler if sampler else None, shuffle=(sampler is None))
       val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
       optimizer = optim.AdamW(model.parameters(), lr=lr)
       if use_onecycle:
           steps_per_epoch = len(train_loader)
           scheduler = OneCycleLR(optimizer, max_lr=lr*10, total_steps=steps_per_epoch*epochs, pct_start=0.3, anneal_strategy="cos")
       else:
           scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
       best_val_loss = float("inf")
       best_state = None
       no_improve = 0
       for epoch in range(1, epochs+1):
           train_loss = train_one_epoch(model, train_loader, optimizer, ce_loss_fn, mse_loss_fn, alpha, device, scheduler)
           val_loss, val_acc = validate_one_epoch(model, val_loader, ce_loss_fn, mse_loss_fn, alpha, device)
           if not use_onecycle:
               scheduler.step(val_loss)
           print(f"[Epoch {epoch}/{epochs}] TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, ValAcc={val_acc:.1f}%")
           if val_loss < best_val_loss:
               best_val_loss = val_loss
               best_state = model.state_dict()
               no_improve = 0
           else:
               no_improve += 1
               if no_improve >= patience:
                   print("Early stopping triggered!")
                   break
       if best_state is None:
           best_state = model.state_dict()
       save_dict = {"model_state": best_state}
       if getattr(dataset, "normalize", False):
           save_dict["mean_"] = dataset.mean_.tolist()
           save_dict["std_"] = dataset.std_.tolist()
       os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
       torch.save(save_dict, output_path)
       print(f"Model saved to {output_path}")

def load_coarse_model(model_path, device):
   state = torch.load(model_path, map_location=device)
   model = CoarseGazeNet(encoder_dim=256, hidden_dim=128)
   if 'model_state' in state:
       model.load_state_dict(state['model_state'])
   else:
       model.load_state_dict(state)
   if 'mean_' in state:
       model.mean_ = state['mean_']
   if 'std_' in state:
       model.std_ = state['std_']
   model.to(device)
   model.eval()
   return model

def load_fine_model(model_path, device="cpu"):
   data = torch.load(model_path, map_location=device)
   if "model_state" not in data:
       raise KeyError(f"'model_state' not found in {list(data.keys())}")
   model = FineGazeNet(embed_dim=256, num_fine=4, dropout_p=0.1).to(device)
   model.load_state_dict(data["model_state"])
   model.eval()
   stats = {}
   if "mean_" in data and "std_" in data:
       stats["mean_"] = np.array(data["mean_"], dtype=np.float32)
       stats["std_"] = np.array(data["std_"], dtype=np.float32)
   return model, stats

def extract_elapsed_from_filename(filename):
   match = re.search(r"screenshot_(\d+\.\d{3})", filename)
   return float(match.group(1)) if match else None

def generate_heatmap(df, out_path, screen_w=1024, screen_h=768, sigma=20):
   heat = np.zeros((screen_h, screen_w), dtype=np.float32)
   for _, row in df.iterrows():
       x = int(row["screen_x"])
       y = int(row["screen_y"])
       if 0 <= x < screen_w and 0 <= y < screen_h:
           heat[y, x] += 1
   heat = gaussian_filter(heat, sigma=sigma)
   if heat.max() > 0:
       heat /= heat.max()
   plt.figure(figsize=(12,9))
   plt.imshow(heat, cmap="hot", origin="upper")
   plt.colorbar(label="Normalized density")
   plt.title("Gaze Heatmap")
   plt.axis("off")
   plt.savefig(out_path, dpi=300, bbox_inches="tight")
   plt.close()

def analyze_sections(df, out_path, n_cols=5, n_rows=5, screen_w=1024, screen_h=768):
   counts = df["region"].value_counts().sort_index()
   import pandas as pd
   base = pd.Series(0, index=range(1, n_cols*n_rows+1))
   counts = counts.add(base, fill_value=0)
   perc = counts / counts.sum() * 100 if counts.sum() > 0 else counts*0
   fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,7))
   counts.plot(kind="bar", ax=ax1)
   ax1.set_title("Gaze Count by Region")
   ax1.set_xlabel("Region ID")
   ax1.set_ylabel("Count")
   perc.plot(kind="bar", ax=ax2)
   ax2.set_title("Gaze Percentage by Region")
   ax2.set_xlabel("Region ID")
   ax2.set_ylabel("Percentage (%)")
   plt.tight_layout()
   plt.savefig(out_path, dpi=300)
   plt.close()

def visualize_prediction(img_path, x, y, quadrant, confidence, output_path, debug=False,
                          nose_x=None, nose_y=None,
                          left_pupil_x=None, left_pupil_y=None,
                          right_pupil_x=None, right_pupil_y=None):
   if debug:
       print(f"Loading image from: {img_path}")
   img = cv2.imread(img_path)
   if img is None:
       if debug:
           print(f"Failed to load image: {img_path}")
       return
   h, w = img.shape[:2]
   cv2.line(img, (w//2, 0), (w//2, h), (255,255,255), 1)
   cv2.line(img, (0, h//2), (w, h//2), (255,255,255), 1)
   size = 10
   pt1 = (int(x), int(y - size))
   pt2 = (int(x - size), int(y + size))
   pt3 = (int(x + size), int(y + size))
   triangle = np.array([pt1, pt2, pt3], np.int32)
   triangle = triangle.reshape((-1,1,2))
   cv2.fillPoly(img, [triangle], (0,0,255))
   if debug:
       print(f"Saving prediction visualization to: {output_path}")
   cv2.imwrite(output_path, img)

def compute_gaze_target(nose_x, nose_y, left_pupil_x, left_pupil_y,
                          right_pupil_x, right_pupil_y, screen_width, screen_height):
   eye_center_x = (left_pupil_x + right_pupil_x) / 2
   eye_center_y = (left_pupil_y + right_pupil_y) / 2
   gaze_x = nose_x - eye_center_x
   gaze_y = nose_y - eye_center_y
   gaze_length = np.sqrt(gaze_x**2 + gaze_y**2)
   if gaze_length > 0:
       gaze_x /= gaze_length
       gaze_y /= gaze_length
   scale_x = screen_width / 2
   scale_y = screen_height / 2
   screen_x = eye_center_x + (gaze_x * scale_x)
   screen_y = eye_center_y + (gaze_y * scale_y)
   screen_x = screen_x * (screen_width / 640)
   screen_y = screen_y * (screen_height / 480)
   screen_x = np.clip(screen_x, 0, screen_width)
   screen_y = np.clip(screen_y, 0, screen_height)
   return screen_x, screen_y

def analyze_hierarchical(landmarks_csv, coarse_model_path, fine_model_dir=None, output_csv=None, output_heatmap=None, output_sections=None, screenshot_folder=None, screenshot_output_folder=None, screenshot_tolerance=0.5, screen_width=1024, screen_height=768, n_cols=5, n_rows=5, device="cpu", detailed=False, coarse_only=False, debug=True):
   print("Loading coarse model from", coarse_model_path)
   coarse_model = load_coarse_model(coarse_model_path, device)
   df = pd.read_csv(landmarks_csv)
   if debug:
       print("Available columns:", df.columns.tolist())
   screenshot_map = {}
   if screenshot_folder and screenshot_output_folder:
       print(f"Screenshot input folder: {screenshot_folder}")
       print(f"Screenshot output folder: {screenshot_output_folder}")
       os.makedirs(screenshot_output_folder, exist_ok=True)
       screenshots = sorted([f for f in os.listdir(screenshot_folder) if f.startswith('screenshot_')])
       if debug:
           print(f"Found {len(screenshots)} screenshots")
           print("Example files:", screenshots[:5])
       for fname in screenshots:
           try:
               time_str = fname.replace('screenshot_', '').replace('.png', '')
               seconds, ms = time_str.split('.')
               timestamp = float(f"{int(seconds)}.{ms}")
               screenshot_map[timestamp] = fname
               if debug:
                   print(f"Mapped {fname} -> {timestamp}s")
           except Exception as e:
               if debug:
                   print(f"Could not parse timestamp from filename: {fname} - {str(e)}")
               continue
   results = []
   coarse_grid = get_coarse_grid()
   with torch.no_grad():
       for idx, row in tqdm(df.iterrows(), total=len(df)):
           try:
               geo_x, geo_y = compute_gaze_target(row['nose_x'], row['nose_y'],
                                                  row['left_pupil_x'], row['left_pupil_y'],
                                                  row['right_pupil_x'], row['right_pupil_y'],
                                                  screen_width, screen_height)
               head_tensor = torch.tensor([[[row['nose_x'], row['nose_y']],
                                              [row['corner_left_x'], row['corner_left_y']],
                                              [row['corner_right_x'], row['corner_right_y']]]], dtype=torch.float32)
               pupil_tensor = torch.tensor([[[row['left_pupil_x'], row['left_pupil_y']],
                                               [row['right_pupil_x'], row['right_pupil_y']]]], dtype=torch.float32)
               logits, reg_output = coarse_model(head_tensor, pupil_tensor)
               conf, pred = torch.max(torch.softmax(logits, dim=1), dim=1)
               c_conf = conf.item()
               c_label = "ABCD"[pred.item()]
               b = coarse_grid[c_label]
               x_model = b["xmin"] + reg_output[0, 0].item() * (b["xmax"] - b["xmin"])
               y_model = b["ymin"] + reg_output[0, 1].item() * (b["ymax"] - b["ymin"])
               x_model = np.clip(x_model, b["xmin"], b["xmax"])
               y_model = np.clip(y_model, b["ymin"], b["ymax"])
               if not (b["xmin"] <= x_model < b["xmax"] and b["ymin"] <= y_model < b["ymax"]):
                   x_model, y_model = geo_x, geo_y
               x, y = x_model, y_model
               quadrant = c_label
               if detailed and idx % 100 == 0:
                   print(f"\nFrame {idx}:")
                   print(f"Input nose: ({row['nose_x']:.1f}, {row['nose_y']:.1f})")
                   print(f"Input pupils: L({row['left_pupil_x']:.1f}, {row['left_pupil_y']:.1f}), R({row['right_pupil_x']:.1f}, {row['right_pupil_y']:.1f})")
                   print(f"Geometric prediction: ({geo_x:.1f}, {geo_y:.1f})")
                   print(f"Model prediction: ({x:.1f}, {y:.1f}) in quadrant {quadrant} (confidence: {c_conf:.3f})")
               if screenshot_folder and screenshot_output_folder:
                   try:
                       elapsed = float(row['elapsed'])
                   except:
                       elapsed = idx / 30.0
                   closest_time = None
                   min_diff = screenshot_tolerance
                   for time in screenshot_map.keys():
                       diff = abs(time - elapsed)
                       if diff < min_diff:
                           min_diff = diff
                           closest_time = time
                   if closest_time is not None:
                       img_path = os.path.join(screenshot_folder, screenshot_map[closest_time])
                       out_path = os.path.join(screenshot_output_folder, f"pred_{screenshot_map[closest_time]}")
                       if debug and idx % 100 == 0:
                           print(f"\nProcessing screenshot for elapsed={elapsed:.3f}s:")
                           print(f"Matched to screenshot at {closest_time:.3f}s (diff={min_diff:.3f}s)")
                           print(f"Input: {img_path}")
                           print(f"Output: {out_path}")
                       visualize_prediction(img_path, x, y, quadrant, c_conf, out_path, debug=(idx % 100 == 0))
                   elif debug and idx % 100 == 0:
                       print(f"\nNo matching screenshot found for elapsed={elapsed:.3f}s")
                       print(f"Available timestamps: {sorted(screenshot_map.keys())}")
               results.append({
                   'frame': idx,
                   'elapsed': row.get('elapsed', idx/30.0),
                   'screen_x': float(x),
                   'screen_y': float(y),
                   'coarse_region': c_label,
                   'confidence': c_conf,
                   'region': "ABCD".find(c_label) + 1,
                   'method': 'coarse_only' if coarse_only else 'fine'
               })
           except Exception as ex:
               if detailed:
                   print(f"Error processing frame {idx}: {str(ex)}")
                   if debug:
                       import traceback
                       traceback.print_exc()
               results.append({
                   'frame': idx,
                   'elapsed': row.get('elapsed', idx/30.0),
                   'screen_x': np.nan,
                   'screen_y': np.nan,
                   'coarse_region': 'UNKNOWN',
                   'confidence': 0.0,
                   'region': 0,
                   'method': 'error'
               })
   out_df = pd.DataFrame(results)
   if output_csv:
       out_df.to_csv(output_csv, index=False)
       print(f"Saved predictions to {output_csv}")
   return out_df

def evaluate_pipeline(data_csv, coarse_model_path, fine_model_dir, device="cpu", print_confusion=True):
   print("Evaluation not fully implemented. Modify as needed.")
   pass

def main():
   parser = argparse.ArgumentParser(description="Enhanced Hierarchical Gaze Estimation Pipeline")
   subp = parser.add_subparsers(dest="mode")
   p_coarse = subp.add_parser("coarse_train", help="Train the coarse model")
   p_coarse.add_argument("--data", required=True)
   p_coarse.add_argument("--output", required=True)
   p_coarse.add_argument("--device", default="cpu")
   p_coarse.add_argument("--batch_size", type=int, default=64)
   p_coarse.add_argument("--epochs", type=int, default=100)
   p_coarse.add_argument("--patience", type=int, default=10)
   p_coarse.add_argument("--alpha", type=float, default=0.8)
   p_coarse.add_argument("--lr", type=float, default=1e-3)
   p_coarse.add_argument("--balanced_ce", action="store_true")
   p_coarse.add_argument("--onecycle", action="store_true", help="Use OneCycleLR scheduler")
   p_coarse.add_argument("--normalize", action="store_true", help="Enable input normalization")
   p_coarse.add_argument("--kfold", type=int, default=0, help="Number of folds for KFold cross-validation (0=off)")
   p_fine = subp.add_parser("fine_train", help="Train the fine model (for one quadrant or all)")
   p_fine.add_argument("--data", required=True)
   p_fine.add_argument("--coarse_label", required=True, help="One of A, B, C, D or 'all'")
   p_fine.add_argument("--output", required=True)
   p_fine.add_argument("--device", default="cpu")
   p_fine.add_argument("--batch_size", type=int, default=64)
   p_fine.add_argument("--epochs", type=int, default=100)
   p_fine.add_argument("--patience", type=int, default=10)
   p_fine.add_argument("--alpha", type=float, default=0.8)
   p_fine.add_argument("--lr", type=float, default=1e-3)
   p_fine.add_argument("--balanced_ce", action="store_true")
   p_fine.add_argument("--onecycle", action="store_true", help="Use OneCycleLR scheduler")
   p_fine.add_argument("--normalize", action="store_true", help="Enable input normalization")
   p_fine.add_argument("--kfold", type=int, default=0, help="Number of folds for KFold cross-validation (0=off)")
   p_ana = subp.add_parser("analyze", help="Run inference on landmark data")
   p_ana.add_argument("--landmarks", required=True)
   p_ana.add_argument("--coarse_model", required=True)
   p_ana.add_argument("--fine_model_dir", default="", help="Optional fine model directory")
   p_ana.add_argument("--output_csv", default=None)
   p_ana.add_argument("--output_heatmap", default=None)
   p_ana.add_argument("--output_sections", default=None)
   p_ana.add_argument("--screenshot_folder", default=None)
   p_ana.add_argument("--screenshot_output_folder", default=None)
   p_ana.add_argument("--screenshot_tolerance", type=float, default=0.5)
   p_ana.add_argument("--screen_width", type=int, default=1024)
   p_ana.add_argument("--screen_height", type=int, default=768)
   p_ana.add_argument("--n_cols", type=int, default=5)
   p_ana.add_argument("--n_rows", type=int, default=5)
   p_ana.add_argument("--device", default="cpu")
   p_ana.add_argument("--detailed", action="store_true")
   p_ana.add_argument("--normalize", action="store_true", help="Apply normalization using training stats")
   p_eval = subp.add_parser("evaluate", help="Evaluate on a labeled CSV")
   p_eval.add_argument("--data_csv", required=True)
   p_eval.add_argument("--coarse_model", required=True)
   p_eval.add_argument("--fine_model_dir", required=True)
   p_eval.add_argument("--device", default="cpu")
   p_eval.add_argument("--confusion", action="store_true")
   args = parser.parse_args()
   if args.mode == "coarse_train":
       ds = CoarseGazeDataset(args.data, balanced_ce=args.balanced_ce, normalize=args.normalize)
       model = CoarseGazeNet(encoder_dim=256, hidden_dim=128)
       train_model(ds, model, args.output, device=args.device, batch_size=args.batch_size,
                   epochs=args.epochs, patience=args.patience, alpha=args.alpha,
                   balanced_ce=args.balanced_ce, lr=args.lr, use_onecycle=args.onecycle,
                   kfold=args.kfold)
   elif args.mode == "fine_train":
       if args.coarse_label.lower() == "all":
           outdir = args.output
           os.makedirs(outdir, exist_ok=True)
           for cl in ["A", "B", "C", "D"]:
               print(f"Training fine model for quadrant {cl}...")
               ds = FineGazeDataset(args.data, coarse_label=cl, balanced_ce=args.balanced_ce, normalize=args.normalize)
               model = FineGazeNet(embed_dim=256, num_fine=4, dropout_p=0.1)
               out_path = os.path.join(outdir, f"fine_model_{cl}.pt")
               train_model(ds, model, out_path, device=args.device, batch_size=args.batch_size,
                           epochs=args.epochs, patience=args.patience, alpha=args.alpha,
                           balanced_ce=args.balanced_ce, lr=args.lr, use_onecycle=args.onecycle,
                           kfold=args.kfold)
       else:
           ds = FineGazeDataset(args.data, coarse_label=args.coarse_label.upper(), balanced_ce=args.balanced_ce, normalize=args.normalize)
           model = FineGazeNet(embed_dim=256, num_fine=4, dropout_p=0.1)
           train_model(ds, model, args.output, device=args.device, batch_size=args.batch_size,
                       epochs=args.epochs, patience=args.patience, alpha=args.alpha,
                       balanced_ce=args.balanced_ce, lr=args.lr, use_onecycle=args.onecycle,
                       kfold=args.kfold)
   elif args.mode == "analyze":
       analyze_hierarchical(landmarks_csv=args.landmarks,
                            coarse_model_path=args.coarse_model,
                            fine_model_dir=args.fine_model_dir,
                            output_csv=args.output_csv,
                            output_heatmap=args.output_heatmap,
                            output_sections=args.output_sections,
                            screenshot_folder=args.screenshot_folder,
                            screenshot_output_folder=args.screenshot_output_folder,
                            screenshot_tolerance=args.screenshot_tolerance,
                            screen_width=args.screen_width,
                            screen_height=args.screen_height,
                            n_cols=args.n_cols,
                            n_rows=args.n_rows,
                            device=args.device,
                            detailed=args.detailed,
                            coarse_only=True,
                            debug=True)
   elif args.mode == "evaluate":
       evaluate_pipeline(data_csv=args.data_csv, coarse_model_path=args.coarse_model,
                         fine_model_dir=args.fine_model_dir, device=args.device,
                         print_confusion=args.confusion)
   else:
       parser.print_help()

if __name__ == "__main__":
   main()
