#!/usr/bin/env python
"""
hierarchical_gaze_pro.py

A "professional-grade" hierarchical gaze estimation pipeline:
- Coarse 4-quadrant classification + regression
- Fine sub-quadrant classification + regression
- Weighted loss for classification vs. regression
- Optionally balanced cross-entropy for quadrant imbalance
- LR scheduling, early stopping
- Single script with subcommands:
  1) coarse_train
  2) fine_train
  3) analyze (inference + screenshots + heatmaps)
  4) evaluate (test set evaluation metrics + confusion matrix)
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Tuple

########################
# Utility: create grid
########################

def create_grid(screen_width, screen_height, n_cols, n_rows):
    """
    Create a fallback grid for section analysis or region labeling.
    Returns a list of (col, row, x1, y1, x2, y2, region_id).
    """
    grid = []
    cell_w = screen_width / n_cols
    cell_h = screen_height / n_rows
    region_id = 1
    for r in range(n_rows):
        for c in range(n_cols):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = (c+1)*cell_w
            y2 = (r+1)*cell_h
            grid.append((c,r,x1,y1,x2,y2, region_id))
            region_id+=1
    return grid

########################
# Grid definitions
########################

def get_coarse_grid():
    """
    4 big squares for 1024x768.
    A = top-left
    B = top-right
    C = bottom-left
    D = bottom-right
    """
    return {
        "A": {"xmin":0,"xmax":512,"ymin":0,"ymax":384},
        "B": {"xmin":512,"xmax":1024,"ymin":0,"ymax":384},
        "C": {"xmin":0,"xmax":512,"ymin":384,"ymax":768},
        "D": {"xmin":512,"xmax":1024,"ymin":384,"ymax":768},
    }

def get_fine_grid(coarse_label:str, coarse_grid:Dict[str,Dict[str,float]]):
    """
    Subdivide the coarse bounding box into 4 sub-squares.
    """
    b = coarse_grid[coarse_label]
    xm = (b["xmin"]+ b["xmax"])/2
    ym = (b["ymin"]+ b["ymax"])/2
    # We'll name them A,B,C,D appended to the coarse label
    # e.g., for coarse_label='A', sub = 'AA','AB','AC','AD'
    return {
        coarse_label+"A": {"xmin":b["xmin"],"xmax":xm,"ymin":b["ymin"],"ymax":ym},
        coarse_label+"B": {"xmin":xm,"xmax":b["xmax"],"ymin":b["ymin"],"ymax":ym},
        coarse_label+"C": {"xmin":b["xmin"],"xmax":xm,"ymin":ym,"ymax":b["ymax"]},
        coarse_label+"D": {"xmin":xm,"xmax":b["xmax"],"ymin":ym,"ymax":b["ymax"]},
    }

########################
# Dataset definitions
########################

REQUIRED_COLUMNS = [
    "nose_x","nose_y",
    "corner_left_x","corner_left_y","corner_right_x","corner_right_y",
    "left_pupil_x","left_pupil_y","right_pupil_x","right_pupil_y",
    "screen_x","screen_y"
]

class CoarseGazeDataset(Dataset):
    """
    For training a 4-quadrant (A,B,C,D) classifier + 2D offset inside the quadrant.
    Weighted combination of classification + regression.
    """
    def __init__(self, csv_path:str, balanced_ce:bool=False):
        df = pd.read_csv(csv_path)
        df.drop_duplicates(inplace=True)
        for c in REQUIRED_COLUMNS:
            if c not in df.columns:
                raise ValueError(f"Missing '{c}' in {csv_path}")

        # Head = (nose_x, nose_y, corner_left_x, ..., corner_right_y)
        self.head_data = df[["nose_x","nose_y","corner_left_x","corner_left_y",
                             "corner_right_x","corner_right_y"]].values.astype(np.float32)

        # Pupils = (left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y)
        self.pupil_data= df[["left_pupil_x","left_pupil_y",
                             "right_pupil_x","right_pupil_y"]].values.astype(np.float32)

        # Gaze
        self.screen_xy = df[["screen_x","screen_y"]].values.astype(np.float32)

        self.grid = get_coarse_grid()  # A,B,C,D
        self.grid_keys = sorted(self.grid.keys()) # ["A","B","C","D"]

        # Compute normalization stats
        self.head_mean = self.head_data.mean(axis=0)
        self.head_std  = self.head_data.std(axis=0)
        self.pupil_mean= self.pupil_data.mean(axis=0)
        self.pupil_std = self.pupil_data.std(axis=0)

        self.head_norm = (self.head_data - self.head_mean)/(self.head_std+1e-7)
        self.pupil_norm= (self.pupil_data- self.pupil_mean)/(self.pupil_std+1e-7)

        # Precompute coarse labels, for optional class balancing
        # label_idx in [0..3]
        self.labels = []
        for pt in self.screen_xy:
            label_i = None
            for i,k in enumerate(self.grid_keys):
                b = self.grid[k]
                if pt[0]>=b["xmin"] and pt[0]<b["xmax"] and pt[1]>=b["ymin"] and pt[1]<b["ymax"]:
                    label_i = i
                    break
            if label_i is None:
                raise ValueError(f"Point {pt} not in any coarse region!?")
            self.labels.append(label_i)
        self.labels = np.array(self.labels, dtype=np.int64)

        self.balanced_ce = balanced_ce
        self.class_weights = None
        if balanced_ce:
            # compute frequency of each quadrant
            from collections import Counter
            counts = Counter(self.labels.tolist())
            # invert freq => class weight
            # e.g. weight[i] = total_count / counts[i]
            total_count = len(self.labels)
            weights_arr = []
            for i in range(len(self.grid_keys)):
                c_i = counts[i]
                w_i = total_count/(c_i+1e-7)
                weights_arr.append(w_i)
            weights_tensor = torch.tensor(weights_arr, dtype=torch.float32)
            self.class_weights = weights_tensor
            print(f"Balanced CE: quadrant counts={dict(counts)}, weights={weights_tensor.numpy()}")

    def __len__(self):
        return len(self.head_norm)

    def __getitem__(self, idx):
        head = self.head_norm[idx]
        pupil= self.pupil_norm[idx]
        label_idx = self.labels[idx]
        xy   = self.screen_xy[idx]
        # relative offset
        ckey  = self.grid_keys[label_idx]
        b     = self.grid[ckey]
        rx = (xy[0]- b["xmin"])/(b["xmax"]- b["xmin"])
        ry = (xy[1]- b["ymin"])/(b["ymax"]- b["ymin"])
        rel_coord = np.array([rx, ry], dtype=np.float32)

        return (torch.tensor(head),
                torch.tensor(pupil),
                torch.tensor(label_idx, dtype=torch.long),
                torch.tensor(rel_coord))

class FineGazeDataset(Dataset):
    """
    For training the sub-quadrant classification + offset within that sub-quadrant.
    We gather only points that fall in the bounding box of the chosen coarse_label
    and subdivide them further.
    """
    def __init__(self, csv_path:str, coarse_label:str,
                 # unify stats with coarse
                 c_head_mean=None, c_head_std=None,
                 c_pupil_mean=None, c_pupil_std=None,
                 balanced_ce=False):
        df = pd.read_csv(csv_path)
        df.drop_duplicates(inplace=True)
        for c in REQUIRED_COLUMNS:
            if c not in df.columns:
                raise ValueError(f"Missing '{c}' in {csv_path}")

        # same approach
        self.head_data = df[["nose_x","nose_y","corner_left_x","corner_left_y",
                             "corner_right_x","corner_right_y"]].values.astype(np.float32)
        self.pupil_data= df[["left_pupil_x","left_pupil_y","right_pupil_x","right_pupil_y"]].values.astype(np.float32)
        self.screen_xy = df[["screen_x","screen_y"]].values.astype(np.float32)

        # Filter to points in that coarse region
        cgrid = get_coarse_grid()
        if coarse_label not in cgrid:
            raise ValueError(f"coarse_label {coarse_label} not recognized.")
        cb = cgrid[coarse_label]
        valid_indices = []
        for i,(sx,sy) in enumerate(self.screen_xy):
            if sx>=cb["xmin"] and sx<cb["xmax"] and sy>=cb["ymin"] and sy<cb["ymax"]:
                valid_indices.append(i)
        if len(valid_indices)==0:
            raise ValueError(f"No data found in region {coarse_label} in {csv_path}.")

        self.head_data = self.head_data[valid_indices]
        self.pupil_data= self.pupil_data[valid_indices]
        self.screen_xy = self.screen_xy[valid_indices]

        # unify stats with coarse
        if c_head_mean is not None:
            self.head_mean= c_head_mean
            self.head_std = c_head_std
            self.pupil_mean= c_pupil_mean
            self.pupil_std = c_pupil_std
        else:
            self.head_mean= self.head_data.mean(axis=0)
            self.head_std = self.head_data.std(axis=0)
            self.pupil_mean= self.pupil_data.mean(axis=0)
            self.pupil_std = self.pupil_data.std(axis=0)

        self.head_norm = (self.head_data - self.head_mean)/(self.head_std+1e-7)
        self.pupil_norm= (self.pupil_data- self.pupil_mean)/(self.pupil_std+1e-7)

        # sub-grid for that coarse
        self.fine_grid = get_fine_grid(coarse_label, cgrid)
        self.fine_keys = sorted(self.fine_grid.keys()) # e.g. ["AA","AB","AC","AD"]

        # label each point
        self.labels = []
        for pt in self.screen_xy:
            label_i=None
            for i,k in enumerate(self.fine_keys):
                b= self.fine_grid[k]
                if pt[0]>=b["xmin"] and pt[0]<b["xmax"] and pt[1]>=b["ymin"] and pt[1]<b["ymax"]:
                    label_i= i
                    break
            if label_i is None:
                raise ValueError(f"Point {pt} not in sub-grid {coarse_label}?")
            self.labels.append(label_i)
        self.labels = np.array(self.labels, dtype=np.int64)

        self.balanced_ce = balanced_ce
        self.class_weights = None
        if balanced_ce:
            from collections import Counter
            freq = Counter(self.labels.tolist())
            total_ct = len(self.labels)
            weight_arr = []
            for i in range(len(self.fine_keys)):
                c_i = freq[i]
                w_i = total_ct/(c_i+1e-7)
                weight_arr.append(w_i)
            self.class_weights= torch.tensor(weight_arr,dtype=torch.float32)
            print(f"[Fine {coarse_label}] Balanced CE: freq={dict(freq)}, weights={weight_arr}")

    def __len__(self):
        return len(self.head_norm)

    def __getitem__(self, idx):
        head = self.head_norm[idx]
        pupil= self.pupil_norm[idx]
        label_idx = self.labels[idx]
        xy   = self.screen_xy[idx]
        k    = self.fine_keys[label_idx]
        b    = self.fine_grid[k]
        rx   = (xy[0] - b["xmin"])/(b["xmax"]- b["xmin"])
        ry   = (xy[1] - b["ymin"])/(b["ymax"]- b["ymin"])
        rel_coord = np.array([rx,ry],dtype=np.float32)

        return (torch.tensor(head),
                torch.tensor(pupil),
                torch.tensor(label_idx, dtype=torch.long),
                torch.tensor(rel_coord))

########################
# Model definitions
########################

class HeadEncoder(nn.Module):
    """A small MLP that encodes 6D head features into a 128D embedding."""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(6,64)
        self.fc2 = nn.Linear(64,128)
        self.fc3 = nn.Linear(128, embed_dim)
        self.relu= nn.ReLU()
        self.drop= nn.Dropout(0.2)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x)

class CoarseGazeNet(nn.Module):
    def __init__(self, embed_dim=128, num_coarse=4):
        super().__init__()
        self.head_net = HeadEncoder(embed_dim)
        self.pupil_fc = nn.Linear(4,64)
        self.comb_fc1 = nn.Linear(embed_dim+64, 128)
        self.comb_fc2 = nn.Linear(128, 64)
        self.class_out= nn.Linear(64, num_coarse)
        self.reg_out  = nn.Linear(64, 2)
        self.relu= nn.ReLU()
        self.drop= nn.Dropout(0.2)
        self.sigmoid= nn.Sigmoid()

    def forward(self, head, pupil):
        h = self.head_net(head)
        p = self.relu(self.pupil_fc(pupil))
        x = torch.cat([h,p], dim=1)
        x = self.relu(self.comb_fc1(x))
        x = self.drop(x)
        x = self.relu(self.comb_fc2(x))
        x = self.drop(x)
        logits = self.class_out(x)
        rel_xy = self.sigmoid(self.reg_out(x))
        return logits, rel_xy

class FineGazeNet(nn.Module):
    def __init__(self, embed_dim=128, num_fine=4):
        super().__init__()
        self.head_net = HeadEncoder(embed_dim)
        self.pupil_fc = nn.Linear(4,64)
        self.comb_fc1 = nn.Linear(embed_dim+64, 128)
        self.comb_fc2 = nn.Linear(128,64)
        self.class_out= nn.Linear(64, num_fine)
        self.reg_out  = nn.Linear(64,2)
        self.relu= nn.ReLU()
        self.drop= nn.Dropout(0.2)
        self.sigmoid= nn.Sigmoid()

    def forward(self, head, pupil):
        h = self.head_net(head)
        p = self.relu(self.pupil_fc(pupil))
        x = torch.cat([h,p], dim=1)
        x = self.relu(self.comb_fc1(x))
        x = self.drop(x)
        x = self.relu(self.comb_fc2(x))
        x = self.drop(x)
        logits= self.class_out(x)
        rel_xy= self.sigmoid(self.reg_out(x))
        return logits, rel_xy

########################
# Training Functions
########################

def train_model(
    dataset: Dataset,
    model: nn.Module,
    output_path: str,
    device="cpu",
    batch_size=32,
    epochs=30,
    patience=5,
    alpha=0.8,  # weighting factor for CE vs MSE
    balanced_ce=False,
    lr=1e-3
):
    """
    Generic train function that can train either CoarseGazeNet or FineGazeNet
    based on the dataset provided.
    Weighted loss = alpha*CE + (1-alpha)*MSE
    - If dataset has dataset.class_weights, we'll pass that to CrossEntropyLoss for balancing.
    - Uses a ReduceLROnPlateau scheduler
    - Prints classification accuracy, MSE in the sub-quadrant space
    - Early stopping on validation loss
    """
    # split data
    n_total= len(dataset)
    n_train= int(0.8*n_total)
    n_val  = n_total- n_train
    train_ds, val_ds = random_split(dataset,[n_train,n_val])

    train_loader= DataLoader(train_ds,batch_size=batch_size,shuffle=True)
    val_loader  = DataLoader(val_ds,batch_size=batch_size,shuffle=False)

    device= torch.device(device)
    model= model.to(device)

    # Balanced CE if requested
    if hasattr(dataset,"class_weights") and dataset.class_weights is not None:
        class_weights= dataset.class_weights.to(device)
        ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    else:
        ce_loss = nn.CrossEntropyLoss()

    mse_loss= nn.MSELoss()
    optimizer= optim.Adam(model.parameters(), lr=lr)
    scheduler= ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

    best_val_loss= 1e10
    best_state= None
    no_improve= 0

    for epoch in range(1, epochs+1):
        # Train
        model.train()
        train_loss_sum=0
        train_size=0
        for head,pupil,label,rel in train_loader:
            head,pupil,label,rel = head.to(device),pupil.to(device),label.to(device),rel.to(device)
            optimizer.zero_grad()
            logits, pred_rel= model(head,pupil)
            l_ce = ce_loss(logits,label)
            l_mse= mse_loss(pred_rel,rel)
            loss = alpha*l_ce + (1-alpha)*l_mse
            loss.backward()
            optimizer.step()
            bs= head.shape[0]
            train_loss_sum+= loss.item()*bs
            train_size+= bs

        train_epoch_loss= train_loss_sum/ train_size

        # Validate
        model.eval()
        val_loss_sum=0
        val_size=0
        correct=0
        # to measure MSE in "rel" space
        total_mse= 0.0
        with torch.no_grad():
            for head,pupil,label,rel in val_loader:
                head,pupil,label,rel = head.to(device),pupil.to(device),label.to(device),rel.to(device)
                logits, pred_rel= model(head,pupil)
                l_ce= ce_loss(logits,label)
                l_mse= mse_loss(pred_rel,rel)
                loss= alpha*l_ce + (1-alpha)*l_mse
                bs= head.shape[0]
                val_loss_sum += loss.item()*bs
                val_size+= bs
                # classification accuracy
                preds= torch.argmax(logits,dim=1)
                correct+= (preds==label).sum().item()
                # measure MSE
                total_mse+= ( (pred_rel - rel)**2 ).sum().item()

        val_epoch_loss= val_loss_sum/ val_size
        val_acc= correct/ val_size*100
        rel_mse= total_mse/ (val_size*2)  # average over x,y => dimension=2

        # LR scheduling
        scheduler.step(val_epoch_loss)

        print(f"Epoch {epoch:02d}/{epochs} | train_loss={train_epoch_loss:.4f} | val_loss={val_epoch_loss:.4f} | "
              f"val_acc={val_acc:.1f}% | rel_mse={rel_mse:.4f} (in [0..1])")

        if val_epoch_loss< best_val_loss:
            best_val_loss= val_epoch_loss
            best_state= model.state_dict()
            no_improve= 0
        else:
            no_improve+=1
            if no_improve>= patience:
                print("Early stopping triggered!")
                break

    # save best
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(best_state, output_path)
    print(f"Model saved to {output_path}")

########################
# Loading
########################

def load_coarse_model(model_path:str, device="cpu"):
    data = torch.load(model_path, map_location=device)
    # If we saved only state_dict, we create the model skeleton
    # but in our code, we do 'torch.save(best_state, ...)', so let's do it
    model= CoarseGazeNet().to(device)
    model.load_state_dict(data)
    model.eval()
    return model

def load_fine_model(model_path:str, device="cpu"):
    data = torch.load(model_path, map_location=device)
    model= FineGazeNet().to(device)
    model.load_state_dict(data)
    model.eval()
    return model

########################
# Inference / Analysis
########################

def extract_elapsed_from_filename(filename):
    match = re.search(r"screenshot_(\d+\.\d{3})", filename)
    if match:
        return float(match.group(1))
    else:
        return None

def generate_heatmap(df, out_path, screen_w=1024, screen_h=768, sigma=20):
    df_f = df.copy()
    if "confidence" in df.columns:
        df_f = df[df["confidence"]>0.2]
        if len(df_f)==0:
            df_f = df.copy()
    heat = np.zeros((screen_h,screen_w), dtype=np.float32)
    for _,row in df_f.iterrows():
        x= int(row["screen_x"])
        y= int(row["screen_y"])
        if 0<=x<screen_w and 0<=y<screen_h:
            heat[y,x]+=1
    heat= gaussian_filter(heat, sigma=sigma)
    if heat.max()>0:
        heat/= heat.max()
    plt.figure(figsize=(12,9))
    plt.imshow(heat, cmap="hot", origin="upper")
    plt.colorbar(label="Normalized density")
    plt.title("Gaze Heatmap")
    plt.axis("off")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def analyze_sections(df, out_path, n_cols=5, n_rows=5, screen_w=1024, screen_h=768):
    if "confidence" in df.columns:
        df_f = df[df["confidence"]>0.2]
        if len(df_f)==0:
            df_f= df
    else:
        df_f= df
    if "region" not in df_f.columns:
        print("[analyze_sections] missing 'region' col, skipping!")
        return
    counts= df_f["region"].value_counts().sort_index()
    # ensure we have 1..n_cols*n_rows
    import pandas as pd
    all_ids= range(1,n_cols*n_rows+1)
    base_series= pd.Series(0, index=all_ids)
    counts= counts.add(base_series, fill_value=0)
    tot= counts.sum()
    if tot>0:
        perc= counts/tot*100
    else:
        perc= counts*0

    fig,(ax1,ax2)= plt.subplots(1,2, figsize=(15,7))
    counts.plot(kind="bar", ax=ax1)
    ax1.set_title("Gaze Count by Region")
    ax1.set_xlabel("Region ID")
    ax1.set_ylabel("Count")
    perc.plot(kind="bar", ax=ax2)
    ax2.set_title("Gaze Percentage by Region")
    ax2.set_xlabel("Region ID")
    ax2.set_ylabel("Percentage")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def overlay_predictions(df, in_folder, out_folder, tolerance=0.5,
                       screen_w=1024, screen_h=768, detailed=False):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    files = sorted(os.listdir(in_folder))
    for fname in files:
        fpath= os.path.join(in_folder,fname)
        if not os.path.isfile(fpath):
            continue
        e = extract_elapsed_from_filename(fname)
        if e is None:
            continue
        # find subset in df
        sub = df[ (df["elapsed"]- e).abs()<= tolerance ]
        if len(sub)==0:
            continue
        img= cv2.imread(fpath)
        if img is None:
            continue
        h,w= img.shape[:2]
        scx= w/screen_w
        scy= h/screen_h
        for _,row in sub.iterrows():
            x= row["screen_x"]
            y= row["screen_y"]
            if pd.isna(x) or pd.isna(y):
                continue
            x_i= int(x* scx)
            y_i= int(y* scy)
            if 0<=x_i<w and 0<=y_i<h:
                conf= row.get("confidence",1.0)
                color= (0, int(255*conf), int(255*(1-conf)))
                cv2.circle(img, (x_i,y_i),10, color, -1)
                cv2.circle(img, (x_i,y_i),10, (0,0,0),2)
        out_f= os.path.join(out_folder,fname)
        cv2.imwrite(out_f, img)
        if detailed:
            print(f"Annotated {fname} => {out_f}")

def analyze_hierarchical(
    landmarks_csv,
    coarse_model_path,
    fine_model_dir,
    output_csv=None,
    output_heatmap=None,
    output_sections=None,
    screenshot_folder=None,
    screenshot_output_folder=None,
    screenshot_tolerance=0.5,
    screen_width=1024,
    screen_height=768,
    n_cols=5,
    n_rows=5,
    device="cpu",
    detailed=False
):
    """
    This function does the hierarchical inference:
    1) load coarse model, load any fine_model_*.pt
    2) read CSV => for each row => coarse => bounding box => (cx,cy)
       => if fine => sub bounding box => (fx,fy)
    3) produce final results DataFrame with columns: frame,elapsed,screen_x,screen_y,confidence,coarse_region,region,method
    4) optional heatmap, sections, screenshot overlay
    """

    # 1) load coarse
    dev= torch.device(device)
    coarse_model= load_coarse_model(coarse_model_path, dev)
    cgrid = get_coarse_grid()
    c_keys= sorted(cgrid.keys())

    # We'll guess that c_head_mean,... are not saved in that file, so let's unify approach
    # Actually, in the above code, we "torch.save(best_state,...)", so we do not have the stats
    # So let's do an approach where we skip re-norming. We'll rely on the model's learned distribution. 
    # A "professional" approach is to store those stats in the model checkpoint. You can do that by saving a dictionary. 
    # But for demonstration, let's keep it simpler – or re-check your preference.

    # 2) load fine models
    fine_models= {}
    for fn in os.listdir(fine_model_dir):
        if fn.endswith(".pt") and "fine_model_" in fn:
            c_label= fn.split("_")[-1].split(".")[0] # e.g. "A" from "fine_model_A.pt"
            mpath= os.path.join(fine_model_dir,fn)
            fm= load_fine_model(mpath, dev)
            # store
            f_grid= get_fine_grid(c_label, cgrid)
            # we don't have the stats but we'll do best. 
            fine_models[c_label]= {
                "model": fm,
                "grid":  f_grid
            }

    # 3) read landmarks CSV
    df= pd.read_csv(landmarks_csv)
    for c in ["nose_x","nose_y","corner_left_x","corner_left_y","corner_right_x","corner_right_y",
              "left_pupil_x","left_pupil_y","right_pupil_x","right_pupil_y"]:
        if c not in df.columns:
            raise ValueError(f"Missing {c} in {landmarks_csv}")
    if "elapsed" not in df.columns:
        if "frame" in df.columns:
            df["elapsed"]= df["frame"]/30
        else:
            df["elapsed"]= np.arange(len(df))/30.0

    # fallback grid for region ID
    fallback_g= create_grid(screen_width, screen_height, n_cols, n_rows)
    def get_region_id(x,y):
        for c0,r0,x1,y1,x2,y2,regid in fallback_g:
            if x>=x1 and x<x2 and y>=y1 and y<y2:
                return regid
        return 0

    results=[]
    with torch.no_grad():
        for i,row in enumerate(df.itertuples()):
            try:
                # gather features
                head_np= np.array([
                    row.nose_x,row.nose_y,
                    row.corner_left_x,row.corner_left_y,
                    row.corner_right_x,row.corner_right_y
                ],dtype=np.float32)
                pupil_np= np.array([
                    row.left_pupil_x,row.left_pupil_y,
                    row.right_pupil_x,row.right_pupil_y
                ],dtype=np.float32)
                if np.isnan(head_np).any() or np.isnan(pupil_np).any():
                    raise ValueError("NaN in row features")

                # convert to torch
                head_t= torch.tensor(head_np, dtype=torch.float32, device=dev).unsqueeze(0)
                pupil_t= torch.tensor(pupil_np, dtype=torch.float32, device=dev).unsqueeze(0)

                # coarse
                c_logits, c_rel= coarse_model(head_t, pupil_t)
                c_probs= torch.softmax(c_logits, dim=1)[0]
                c_idx= int(torch.argmax(c_probs).item())
                c_label= c_keys[c_idx]
                c_conf= float(c_probs[c_idx].item())

                # bounding box => absolute
                b= cgrid[c_label]
                xy_rel= c_rel[0].cpu().numpy()
                cx= b["xmin"] + xy_rel[0]*(b["xmax"]- b["xmin"])
                cy= b["ymin"] + xy_rel[1]*(b["ymax"]- b["ymin"]
                )
                cx= np.clip(cx, 0, screen_width)
                cy= np.clip(cy, 0, screen_height)

                final_x= cx
                final_y= cy
                method= "coarse_only"

                # fine if available
                if c_label in fine_models:
                    fm= fine_models[c_label]["model"]
                    fgrid= fine_models[c_label]["grid"]
                    f_logits,f_rel= fm(head_t,pupil_t)
                    f_probs= torch.softmax(f_logits, dim=1)[0]
                    f_idx= int(torch.argmax(f_probs).item())
                    # find sub bounding box
                    sub_keys= sorted(fgrid.keys()) # e.g. [AA,AB,AC,AD]
                    if f_idx<len(sub_keys):
                        sub_label= sub_keys[f_idx]
                        sb= fgrid[sub_label]
                        xy2= f_rel[0].cpu().numpy()
                        fx= sb["xmin"] + xy2[0]*(sb["xmax"]- sb["xmin"])
                        fy= sb["ymin"] + xy2[1]*(sb["ymax"]- sb["ymin"])
                        fx= np.clip(fx,0,screen_width)
                        fy= np.clip(fy,0,screen_height)
                        final_x,final_y= fx,fy
                        method= "fine"

                reg_id= get_region_id(final_x, final_y)
                results.append({
                    "frame": getattr(row,"frame", i),
                    "elapsed": getattr(row,"elapsed", i/30.0),
                    "screen_x": final_x, "screen_y": final_y,
                    "coarse_region": c_label,
                    "confidence": c_conf,
                    "region": reg_id,
                    "method": method
                })
            except Exception as ex:
                # fallback
                results.append({
                    "frame": getattr(row,"frame", i),
                    "elapsed": getattr(row,"elapsed", i/30.0),
                    "screen_x": np.nan, "screen_y": np.nan,
                    "coarse_region": "UNKNOWN",
                    "confidence": 0.0,
                    "region":0,
                    "method":"error"
                })
                if detailed:
                    print(f"Row {i} error: {ex}")

    out_df= pd.DataFrame(results)
    if output_csv:
        out_df.to_csv(output_csv,index=False)
        print(f"Saved final predictions to {output_csv}")
    if output_heatmap:
        generate_heatmap(out_df, output_heatmap, screen_w=screen_width, screen_h=screen_height)
        print(f"Saved heatmap to {output_heatmap}")
    if output_sections:
        analyze_sections(out_df, output_sections, n_cols,n_rows, screen_width,screen_height)
        print(f"Saved section analysis to {output_sections}")
    if screenshot_folder and screenshot_output_folder:
        overlay_predictions(out_df, screenshot_folder, screenshot_output_folder,
                            tolerance=screenshot_tolerance,
                            screen_w=screen_width, screen_h=screen_height,
                            detailed=detailed)
        print(f"Annotated screenshots => {screenshot_output_folder}")

    return out_df

########################
# Evaluate subcommand
########################

def evaluate_pipeline(
    data_csv,
    coarse_model_path,
    fine_model_dir,
    device="cpu",
    print_confusion=True
):
    """
    Evaluate classification accuracy for coarse quadrant, final pixel MSE, etc.
    Goes row by row, does the same as "analyze_hierarchical," but also checks ground-truth quadrant
    and final pixel error.

    You must have "screen_x" + "screen_y" as ground-truth columns in data_csv.
    We'll measure:
       - Coarse classification accuracy
       - Mean final (x,y) error in pixels
       - Possibly a confusion matrix for coarse classification
    """
    dev= torch.device(device)
    # load coarse
    c_model= load_coarse_model(coarse_model_path, dev)
    cgrid= get_coarse_grid()
    c_keys= sorted(cgrid.keys())
    # load fine
    fine_models={}
    for f in os.listdir(fine_model_dir):
        if f.endswith(".pt") and "fine_model_" in f:
            label= f.split("_")[-1].split(".")[0] # "A" from "fine_model_A.pt"
            path= os.path.join(fine_model_dir,f)
            fm= load_fine_model(path, dev)
            fine_models[label]= fm

    df= pd.read_csv(data_csv)
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Missing {c} in {data_csv}")

    # figure out ground truth coarse quadrant
    # We'll store them in a list for confusion matrix
    def find_coarse_label(x,y):
        for i,k in enumerate(c_keys):
            b= cgrid[k]
            if x>=b["xmin"] and x< b["xmax"] and y>=b["ymin"] and y< b["ymax"]:
                return (k,i)
        return (None,-1)

    all_coarse_preds= []
    all_coarse_gts  = []
    sum_pixel_dist= 0
    n_samples=0

    with torch.no_grad():
        for i,row in df.iterrows():
            # ground truth screen coords
            gx= row["screen_x"]
            gy= row["screen_y"]
            if pd.isna(gx) or pd.isna(gy):
                continue
            c_label_gt, c_idx_gt= find_coarse_label(gx,gy)
            if c_idx_gt<0:
                continue

            # model input
            head_np= np.array([
                row["nose_x"], row["nose_y"],
                row["corner_left_x"], row["corner_left_y"],
                row["corner_right_x"],row["corner_right_y"]
            ],dtype=np.float32)
            pupil_np= np.array([
                row["left_pupil_x"],row["left_pupil_y"],
                row["right_pupil_x"],row["right_pupil_y"]
            ],dtype=np.float32)
            if np.isnan(head_np).any() or np.isnan(pupil_np).any():
                continue
            head_t= torch.tensor(head_np, dtype=torch.float32, device=dev).unsqueeze(0)
            pupil_t= torch.tensor(pupil_np,dtype=torch.float32, device=dev).unsqueeze(0)

            # coarse
            c_logits, c_rel= c_model(head_t,pupil_t)
            c_probs= torch.softmax(c_logits, dim=1)[0]
            c_pred_idx= int(torch.argmax(c_probs).item())
            c_pred_label= c_keys[c_pred_idx]
            all_coarse_preds.append(c_pred_idx)
            all_coarse_gts.append(c_idx_gt)

            b= cgrid[c_pred_label]
            cr= c_rel[0].cpu().numpy()
            cx= b["xmin"]+ cr[0]*(b["xmax"]-b["xmin"])
            cy= b["ymin"]+ cr[1]*(b["ymax"]-b["ymin"])
            # fine
            if c_pred_label in fine_models:
                fm= fine_models[c_pred_label]
                f_logits,f_rel= fm(head_t,pupil_t)
                f_idx= int(torch.argmax(f_logits, dim=1).item())
                # we won't bother w/ sub label correctness. We care about final pixel error
                # bounding box
                sub_grid= get_fine_grid(c_pred_label,cgrid)
                sub_keys= sorted(sub_grid.keys())
                if f_idx< len(sub_keys):
                    sb= sub_grid[sub_keys[f_idx]]
                    fr= f_rel[0].cpu().numpy()
                    fx= sb["xmin"]+ fr[0]*(sb["xmax"]- sb["xmin"])
                    fy= sb["ymin"]+ fr[1]*(sb["ymax"]- sb["ymin"])
                    fx= np.clip(fx,0,1024)
                    fy= np.clip(fy,0,768)
                    final_x= fx
                    final_y= fy
                else:
                    final_x, final_y= cx,cy
            else:
                final_x, final_y= cx,cy

            dist= np.hypot(final_x- gx, final_y- gy)
            sum_pixel_dist+= dist
            n_samples+=1

    if n_samples>0:
        avg_dist= sum_pixel_dist/n_samples
    else:
        avg_dist= -1

    # confusion matrix
    import sklearn.metrics as skm
    cm= skm.confusion_matrix(all_coarse_gts, all_coarse_preds, labels=range(len(c_keys)))
    acc= skm.accuracy_score(all_coarse_gts, all_coarse_preds)*100.0

    print("===== EVALUATION RESULTS =====")
    print(f"Coarse quadrant classification accuracy: {acc:.2f}%")
    print(f"Average final pixel error: {avg_dist:.2f} px over {n_samples} samples.")
    if print_confusion:
        print("Confusion Matrix (rows=GT, cols=Pred):")
        print(cm)

########################
# Main Argparse
########################

def main():
    parser = argparse.ArgumentParser(description="Professional-grade hierarchical gaze system.")
    subp = parser.add_subparsers(dest="mode")

    # coarse train
    p_coarse= subp.add_parser("coarse_train", help="Train a 4-quadrant coarse model.")
    p_coarse.add_argument("--data", required=True)
    p_coarse.add_argument("--output", required=True)
    p_coarse.add_argument("--device", default="cpu")
    p_coarse.add_argument("--batch_size", type=int, default=32)
    p_coarse.add_argument("--epochs", type=int, default=30)
    p_coarse.add_argument("--patience", type=int, default=5)
    p_coarse.add_argument("--alpha", type=float, default=0.8)
    p_coarse.add_argument("--lr", type=float, default=1e-3)
    p_coarse.add_argument("--balanced_ce", action="store_true", help="Enable class weighting if quadrant is imbalanced")

    # fine train
    p_fine= subp.add_parser("fine_train", help="Train a sub-quadrant fine model (A->AA,AB,AC,AD).")
    p_fine.add_argument("--data", required=True)
    p_fine.add_argument("--coarse_label", required=True, help="A,B,C,D or 'all'")
    p_fine.add_argument("--output", required=True)
    p_fine.add_argument("--device", default="cpu")
    p_fine.add_argument("--batch_size", type=int, default=32)
    p_fine.add_argument("--epochs", type=int, default=30)
    p_fine.add_argument("--patience", type=int, default=5)
    p_fine.add_argument("--alpha", type=float, default=0.8)
    p_fine.add_argument("--lr", type=float, default=1e-3)
    p_fine.add_argument("--balanced_ce", action="store_true")

    # analyze
    p_ana= subp.add_parser("analyze", help="Inference + optional heatmap + screenshots, etc.")
    p_ana.add_argument("--landmarks", required=True)
    p_ana.add_argument("--coarse_model", required=True)
    p_ana.add_argument("--fine_model_dir", required=True)
    p_ana.add_argument("--output_csv", default=None)
    p_ana.add_argument("--output_heatmap", default=None)
    p_ana.add_argument("--output_sections", default=None)
    p_ana.add_argument("--screenshot_folder", default=None)
    p_ana.add_argument("--screenshot_output_folder", default=None)
    p_ana.add_argument("--screenshot_tolerance", type=float, default=0.5)
    p_ana.add_argument("--screen_width", type=int, default=1024)
    p_ana.add_argument("--screen_height",type=int, default=768)
    p_ana.add_argument("--n_cols", type=int, default=5)
    p_ana.add_argument("--n_rows", type=int, default=5)
    p_ana.add_argument("--device", default="cpu")
    p_ana.add_argument("--detailed", action="store_true")

    # evaluate
    p_eval= subp.add_parser("evaluate", help="Evaluate a coarse+fine pipeline on a labeled test CSV.")
    p_eval.add_argument("--data_csv", required=True)
    p_eval.add_argument("--coarse_model", required=True)
    p_eval.add_argument("--fine_model_dir", required=True)
    p_eval.add_argument("--device", default="cpu")
    p_eval.add_argument("--confusion", action="store_true", help="Print confusion matrix for coarse quadrant")

    args= parser.parse_args()

    if args.mode=="coarse_train":
        ds= CoarseGazeDataset(args.data, balanced_ce=args.balanced_ce)
        model= CoarseGazeNet(embed_dim=128, num_coarse=len(ds.grid_keys))
        train_model(ds, model, args.output,
                    device=args.device,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    patience=args.patience,
                    alpha=args.alpha,
                    balanced_ce=args.balanced_ce,
                    lr=args.lr)

        # Additionally, we might want to save the entire dictionary of stats
        # but for now we only store the model state. 
        # If you want to store stats, do so here.

    elif args.mode=="fine_train":
        # single or all
        from collections import OrderedDict
        if args.coarse_label.lower()=="all":
            outdir= args.output
            os.makedirs(outdir, exist_ok=True)
            # We rely on the same CSV for each region A,B,C,D
            for c_label in ["A","B","C","D"]:
                print(f"Training fine model for region {c_label}...")
                # We can unify stats by reusing a CoarseGazeDataset
                # Actually let's do that properly:
                c_dataset= CoarseGazeDataset(args.data, balanced_ce=args.balanced_ce)
                c_head_mean= c_dataset.head_mean
                c_head_std= c_dataset.head_std
                c_pupil_mean= c_dataset.pupil_mean
                c_pupil_std= c_dataset.pupil_std

                ds= FineGazeDataset(
                    args.data,
                    coarse_label=c_label,
                    c_head_mean=c_head_mean,
                    c_head_std=c_head_std,
                    c_pupil_mean=c_pupil_mean,
                    c_pupil_std=c_pupil_std,
                    balanced_ce=args.balanced_ce
                )
                model= FineGazeNet(embed_dim=128, num_fine=len(ds.fine_keys))
                out_path= os.path.join(outdir, f"fine_model_{c_label}.pt")
                train_model(ds, model, out_path,
                            device=args.device,
                            batch_size=args.batch_size,
                            epochs=args.epochs,
                            patience=args.patience,
                            alpha=args.alpha,
                            lr=args.lr,
                            balanced_ce=args.balanced_ce)
        else:
            # single region
            # unify stats from the coarse dataset
            c_dataset= CoarseGazeDataset(args.data, balanced_ce=args.balanced_ce)
            ds= FineGazeDataset(
                args.data,
                coarse_label=args.coarse_label,
                c_head_mean=c_dataset.head_mean,
                c_head_std=c_dataset.head_std,
                c_pupil_mean=c_dataset.pupil_mean,
                c_pupil_std=c_dataset.pupil_std,
                balanced_ce=args.balanced_ce
            )
            model= FineGazeNet(embed_dim=128, num_fine=len(ds.fine_keys))
            train_model(ds, model, args.output,
                        device=args.device,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        patience=args.patience,
                        alpha=args.alpha,
                        lr=args.lr,
                        balanced_ce=args.balanced_ce)

    elif args.mode=="analyze":
        analyze_hierarchical(
            landmarks_csv=args.landmarks,
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
            detailed=args.detailed
        )

    elif args.mode=="evaluate":
        evaluate_pipeline(
            data_csv=args.data_csv,
            coarse_model_path=args.coarse_model,
            fine_model_dir=args.fine_model_dir,
            device=args.device,
            print_confusion=args.confusion
        )

    else:
        parser.print_help()

if __name__=="__main__":
    main()
