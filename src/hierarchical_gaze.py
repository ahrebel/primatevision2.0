import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

#########################
# Grid Definitions
#########################

def get_coarse_grid():
    return {
        "A": {"xmin": 0, "xmax": 1024/2, "ymin": 0, "ymax": 768/2},
        "B": {"xmin": 1024/2, "xmax": 1024, "ymin": 0, "ymax": 768/2},
        "C": {"xmin": 0, "xmax": 1024/2, "ymin": 768/2, "ymax": 768},
        "D": {"xmin": 1024/2, "xmax": 1024, "ymin": 768/2, "ymax": 768},
    }

def get_fine_grid(coarse_label, coarse_grid):
    bounds = coarse_grid[coarse_label]
    xmin, xmax, ymin, ymax = bounds["xmin"], bounds["xmax"], bounds["ymin"], bounds["ymax"]
    xm = (xmin + xmax) / 2
    ym = (ymin + ymax) / 2
    return {
        coarse_label + "A": {"xmin": xmin, "xmax": xm, "ymin": ymin, "ymax": ym},
        coarse_label + "B": {"xmin": xm, "xmax": xmax, "ymin": ymin, "ymax": ym},
        coarse_label + "C": {"xmin": xmin, "xmax": xm, "ymin": ym, "ymax": ymax},
        coarse_label + "D": {"xmin": xm, "xmax": xmax, "ymin": ym, "ymax": ymax},
    }

#########################
# Datasets
#########################

class CoarseGazeDataset(Dataset):
    def __init__(self, csv_path, grid=None):
        df = pd.read_csv(csv_path)
        df.drop_duplicates(inplace=True)
        required = [
            'corner_left_x', 'corner_left_y',
            'corner_right_x', 'corner_right_y',
            'nose_x', 'nose_y',
            'left_pupil_x', 'left_pupil_y',
            'right_pupil_x', 'right_pupil_y',
            'screen_x', 'screen_y'
        ]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in CSV.")
        
        self.head_data = df[['corner_left_x','corner_left_y',
                             'corner_right_x','corner_right_y',
                             'nose_x','nose_y']].values.astype(np.float32)
        self.pupil_data = df[['left_pupil_x','left_pupil_y',
                              'right_pupil_x','right_pupil_y']].values.astype(np.float32)
        self.screen = df[['screen_x','screen_y']].values.astype(np.float32)
        
        self.head_mean = self.head_data.mean(axis=0)
        self.head_std = self.head_data.std(axis=0)
        self.pupil_mean = self.pupil_data.mean(axis=0)
        self.pupil_std = self.pupil_data.std(axis=0)
        
        self.head_normalized = (self.head_data - self.head_mean) / (self.head_std + 1e-7)
        self.pupil_normalized = (self.pupil_data - self.pupil_mean) / (self.pupil_std + 1e-7)
        
        self.grid = grid if grid is not None else get_coarse_grid()
        self.grid_keys = sorted(self.grid.keys())
    
    def __len__(self):
        return len(self.head_data)
    
    def __getitem__(self, idx):
        head = self.head_normalized[idx]
        pupil = self.pupil_normalized[idx]
        screen_pt = self.screen[idx]
        label_idx = None
        rel_coord = None
        for i, key in enumerate(self.grid_keys):
            bounds = self.grid[key]
            if (screen_pt[0] >= bounds["xmin"] and screen_pt[0] < bounds["xmax"] and
                screen_pt[1] >= bounds["ymin"] and screen_pt[1] < bounds["ymax"]):
                label_idx = i
                rel_x = (screen_pt[0] - bounds["xmin"]) / (bounds["xmax"] - bounds["xmin"])
                rel_y = (screen_pt[1] - bounds["ymin"]) / (bounds["ymax"] - bounds["ymin"])
                rel_coord = np.array([rel_x, rel_y], dtype=np.float32)
                break
        if label_idx is None:
            raise ValueError(f"Screen point {screen_pt} does not fall into any grid region.")
        return (torch.tensor(head),
                torch.tensor(pupil),
                torch.tensor(label_idx, dtype=torch.long),
                torch.tensor(rel_coord))

class FineGazeDataset(Dataset):
    def __init__(self, csv_path, coarse_label, coarse_grid=None, fine_grid=None, 
                 head_mean=None, head_std=None, pupil_mean=None, pupil_std=None):
        df = pd.read_csv(csv_path)
        df.drop_duplicates(inplace=True)
        required = [
            'corner_left_x', 'corner_left_y',
            'corner_right_x', 'corner_right_y',
            'nose_x', 'nose_y',
            'left_pupil_x', 'left_pupil_y',
            'right_pupil_x', 'right_pupil_y',
            'screen_x', 'screen_y'
        ]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in CSV.")
        
        self.head_data = df[['corner_left_x','corner_left_y',
                             'corner_right_x','corner_right_y',
                             'nose_x','nose_y']].values.astype(np.float32)
        self.pupil_data = df[['left_pupil_x','left_pupil_y',
                              'right_pupil_x','right_pupil_y']].values.astype(np.float32)
        self.screen = df[['screen_x','screen_y']].values.astype(np.float32)
        
        if coarse_grid is None:
            coarse_grid = get_coarse_grid()
        bounds = coarse_grid[coarse_label]
        indices = []
        for i, screen_pt in enumerate(self.screen):
            if (screen_pt[0] >= bounds["xmin"] and screen_pt[0] < bounds["xmax"] and
                screen_pt[1] >= bounds["ymin"] and screen_pt[1] < bounds["ymax"]):
                indices.append(i)
        if len(indices) == 0:
            raise ValueError(f"No samples found in coarse region {coarse_label}.")
        self.head_data = self.head_data[indices]
        self.pupil_data = self.pupil_data[indices]
        self.screen = self.screen[indices]
        
        self.head_mean = head_mean if head_mean is not None else self.head_data.mean(axis=0)
        self.head_std = head_std if head_std is not None else self.head_data.std(axis=0)
        self.pupil_mean = pupil_mean if pupil_mean is not None else self.pupil_data.mean(axis=0)
        self.pupil_std = pupil_std if pupil_std is not None else self.pupil_data.std(axis=0)
        
        self.head_normalized = (self.head_data - self.head_mean) / (self.head_std + 1e-7)
        self.pupil_normalized = (self.pupil_data - self.pupil_mean) / (self.pupil_std + 1e-7)
        
        if fine_grid is None:
            self.fine_grid = get_fine_grid(coarse_label, coarse_grid)
        else:
            self.fine_grid = fine_grid
        self.fine_keys = sorted(self.fine_grid.keys())
    
    def __len__(self):
        return len(self.head_data)
    
    def __getitem__(self, idx):
        head = self.head_normalized[idx]
        pupil = self.pupil_normalized[idx]
        screen_pt = self.screen[idx]
        label_idx = None
        rel_coord = None
        for i, key in enumerate(self.fine_keys):
            bounds = self.fine_grid[key]
            if (screen_pt[0] >= bounds["xmin"] and screen_pt[0] < bounds["xmax"] and
                screen_pt[1] >= bounds["ymin"] and screen_pt[1] < bounds["ymax"]):
                label_idx = i
                rel_x = (screen_pt[0] - bounds["xmin"]) / (bounds["xmax"] - bounds["xmin"])
                rel_y = (screen_pt[1] - bounds["ymin"]) / (bounds["ymax"] - bounds["ymin"])
                rel_coord = np.array([rel_x, rel_y], dtype=np.float32)
                break
        if label_idx is None:
            raise ValueError("Screen point does not fall into any fine grid region")
        return (torch.tensor(head),
                torch.tensor(pupil),
                torch.tensor(label_idx, dtype=torch.long),
                torch.tensor(rel_coord))

#########################
# Model Definitions
#########################

class CoarseHeadPoseNet(nn.Module):
    def __init__(self, embed_dim=128):
        super(CoarseHeadPoseNet, self).__init__()
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class CoarseGazeNet(nn.Module):
    def __init__(self, embed_dim=128, num_coarse=4):
        super(CoarseGazeNet, self).__init__()
        self.head_net = CoarseHeadPoseNet(embed_dim)
        self.pupil_fc = nn.Linear(4, 64)
        self.combined_fc1 = nn.Linear(embed_dim + 64, 128)
        self.combined_fc2 = nn.Linear(128, 64)
        self.class_out = nn.Linear(64, num_coarse)
        self.reg_out = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, head, pupil):
        head_embed = self.head_net(head)
        pupil_feat = self.relu(self.pupil_fc(pupil))
        combined = torch.cat([head_embed, pupil_feat], dim=1)
        x = self.relu(self.combined_fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.combined_fc2(x))
        x = self.dropout(x)
        logits = self.class_out(x)
        rel_coord = self.sigmoid(self.reg_out(x))
        return logits, rel_coord

class FineGazeNet(nn.Module):
    def __init__(self, embed_dim=128, num_fine=4):
        super(FineGazeNet, self).__init__()
        self.head_net = CoarseHeadPoseNet(embed_dim)
        self.pupil_fc = nn.Linear(4, 64)
        self.combined_fc1 = nn.Linear(embed_dim + 64, 128)
        self.combined_fc2 = nn.Linear(128, 64)
        self.class_out = nn.Linear(64, num_fine)
        self.reg_out = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, head, pupil):
        head_embed = self.head_net(head)
        pupil_feat = self.relu(self.pupil_fc(pupil))
        combined = torch.cat([head_embed, pupil_feat], dim=1)
        x = self.relu(self.combined_fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.combined_fc2(x))
        x = self.dropout(x)
        logits = self.class_out(x)
        rel_coord = self.sigmoid(self.reg_out(x))
        return logits, rel_coord

#########################
# Training Routines
#########################

def train_coarse(csv_path, output_model_path, device='cpu', batch_size=32, epochs=50, patience=10):
    device = torch.device(device)
    dataset = CoarseGazeDataset(csv_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    model = CoarseGazeNet(embed_dim=128, num_coarse=len(dataset.grid_keys)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    
    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for head, pupil, label, rel_coord in train_loader:
            head, pupil, label, rel_coord = head.to(device), pupil.to(device), label.to(device), rel_coord.to(device)
            optimizer.zero_grad()
            logits, pred_rel = model(head, pupil)
            loss = ce_loss(logits, label) + mse_loss(pred_rel, rel_coord)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * head.size(0)
        train_loss = total_loss / len(train_ds)
        
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for head, pupil, label, rel_coord in test_loader:
                head, pupil, label, rel_coord = head.to(device), pupil.to(device), label.to(device), rel_coord.to(device)
                logits, pred_rel = model(head, pupil)
                loss = ce_loss(logits, label) + mse_loss(pred_rel, rel_coord)
                total_loss += loss.item() * head.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == label).sum().item()
                total += head.size(0)
        test_loss = total_loss / len(test_ds)
        test_acc = correct / total
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
        
        # Check if this is the best model so far
        if test_loss < best_loss:
            best_loss = test_loss
            best_state = model.state_dict()
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1  # Increment patience counter
            
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs)")
            break
    
    save_dict = {
        'state_dict': best_state,
        'grid': dataset.grid,
        'grid_keys': dataset.grid_keys,
        'head_mean': dataset.head_mean,
        'head_std': dataset.head_std,
        'pupil_mean': dataset.pupil_mean,
        'pupil_std': dataset.pupil_std,
    }
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(save_dict, output_model_path)
    print(f"Coarse model saved to {output_model_path}")

def train_fine(csv_path, coarse_label, output_model_path, device='cpu', batch_size=32, epochs=50, patience=10):
    device = torch.device(device)
    dataset = FineGazeDataset(csv_path, coarse_label)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    model = FineGazeNet(embed_dim=128, num_fine=len(dataset.fine_keys)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    
    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for head, pupil, label, rel_coord in train_loader:
            head, pupil, label, rel_coord = head.to(device), pupil.to(device), label.to(device), rel_coord.to(device)
            optimizer.zero_grad()
            logits, pred_rel = model(head, pupil)
            loss = ce_loss(logits, label) + mse_loss(pred_rel, rel_coord)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * head.size(0)
        train_loss = total_loss / len(train_ds)
        
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for head, pupil, label, rel_coord in test_loader:
                head, pupil, label, rel_coord = head.to(device), pupil.to(device), label.to(device), rel_coord.to(device)
                logits, pred_rel = model(head, pupil)
                loss = ce_loss(logits, label) + mse_loss(pred_rel, rel_coord)
                total_loss += loss.item() * head.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == label).sum().item()
                total += head.size(0)
        test_loss = total_loss / len(test_ds)
        test_acc = correct / total
        print(f"[Fine {coarse_label}] Epoch {epoch}: Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
        
        # Check if this is the best model so far
        if test_loss < best_loss:
            best_loss = test_loss
            best_state = model.state_dict()
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1  # Increment patience counter
            
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs)")
            break
    
    save_dict = {
        'state_dict': best_state,
        'fine_grid': dataset.fine_grid,
        'fine_keys': dataset.fine_keys,
        'head_mean': dataset.head_mean,
        'head_std': dataset.head_std,
        'pupil_mean': dataset.pupil_mean,
        'pupil_std': dataset.pupil_std,
    }
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(save_dict, output_model_path)
    print(f"Fine model for coarse region {coarse_label} saved to {output_model_path}")

#########################
# Inference Routines
#########################

def load_coarse_model(model_path, device):
    data = torch.load(model_path, map_location=device)
    grid = data['grid']
    grid_keys = data['grid_keys']
    head_mean = data['head_mean']
    head_std = data['head_std']
    pupil_mean = data['pupil_mean']
    pupil_std = data['pupil_std']
    num_coarse = len(grid_keys)
    model = CoarseGazeNet(embed_dim=128, num_coarse=num_coarse).to(device)
    model.load_state_dict(data['state_dict'])
    model.eval()
    return model, grid, grid_keys, head_mean, head_std, pupil_mean, pupil_std

def load_fine_model(model_path, device):
    data = torch.load(model_path, map_location=device)
    fine_grid = data['fine_grid']
    fine_keys = data['fine_keys']
    head_mean = data['head_mean']
    head_std = data['head_std']
    pupil_mean = data['pupil_mean']
    pupil_std = data['pupil_std']
    num_fine = len(fine_keys)
    model = FineGazeNet(embed_dim=128, num_fine=num_fine).to(device)
    model.load_state_dict(data['state_dict'])
    model.eval()
    return model, fine_grid, fine_keys, head_mean, head_std, pupil_mean, pupil_std

def preprocess_sample(sample, head_mean, head_std, pupil_mean, pupil_std):
    head = np.array([sample['corner_left_x'], sample['corner_left_y'],
                     sample['corner_right_x'], sample['corner_right_y'],
                     sample['nose_x'], sample['nose_y']], dtype=np.float32)
    pupil = np.array([sample['left_pupil_x'], sample['left_pupil_y'],
                      sample['right_pupil_x'], sample['right_pupil_y']], dtype=np.float32)
    head_norm = (head - head_mean) / (head_std + 1e-7)
    pupil_norm = (pupil - pupil_mean) / (pupil_std + 1e-7)
    return torch.tensor(head_norm).unsqueeze(0), torch.tensor(pupil_norm).unsqueeze(0)

def infer(sample, coarse_model, coarse_grid, coarse_keys, head_mean, head_std, pupil_mean, pupil_std, fine_model_dir, device, detailed=False):
    head, pupil = preprocess_sample(sample, head_mean, head_std, pupil_mean, pupil_std)
    with torch.no_grad():
        logits, coarse_rel = coarse_model(head.to(device), pupil.to(device))
        if isinstance(logits, tuple):
            logits = logits[0]
        coarse_probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        coarse_pred_idx = np.argmax(coarse_probs)
        coarse_label = coarse_keys[coarse_pred_idx]
        bounds = coarse_grid[coarse_label]
        coarse_abs_x = bounds["xmin"] + coarse_rel[0,0].item() * (bounds["xmax"] - bounds["xmin"])
        coarse_abs_y = bounds["ymin"] + coarse_rel[0,1].item() * (bounds["ymax"] - bounds["ymin"])
        
        if detailed:
            print(f"Inference debug: coarse_probs: {coarse_probs}")
            print(f"Coarse predicted label: {coarse_label} with bounds {bounds}")
        
        fine_model_path = os.path.join(fine_model_dir, f"fine_model_{coarse_label}.pt")
        if os.path.exists(fine_model_path):
            fine_model, fine_grid, fine_keys, f_head_mean, f_head_std, f_pupil_mean, f_pupil_std = load_fine_model(fine_model_path, device)
            head_f, pupil_f = preprocess_sample(sample, f_head_mean, f_head_std, f_pupil_mean, f_pupil_std)
            with torch.no_grad():
                fine_logits, fine_rel = fine_model(head_f.to(device), pupil_f.to(device))
                if isinstance(fine_logits, tuple):
                    fine_logits = fine_logits[0]
                fine_probs = torch.softmax(fine_logits, dim=1).cpu().numpy().flatten()
                fine_pred_idx = np.argmax(fine_probs)
            fine_label = fine_keys[fine_pred_idx]
            fine_bounds = fine_grid[fine_label]
            fine_abs_x = fine_bounds["xmin"] + fine_rel[0,0].item() * (fine_bounds["xmax"] - fine_bounds["xmin"])
            fine_abs_y = fine_bounds["ymin"] + fine_rel[0,1].item() * (fine_bounds["ymax"] - fine_bounds["ymin"])
            final_x = fine_abs_x
            final_y = fine_abs_y
            method = "fine"
            if detailed:
                print(f"Fine predicted label: {fine_label} with bounds {fine_bounds}")
                print(f"Fine probabilities: {fine_probs}")
        else:
            final_x = coarse_abs_x
            final_y = coarse_abs_y
            method = "coarse"
    
    return {
        "coarse_label": coarse_label,
        "coarse_abs": (coarse_abs_x, coarse_abs_y),
        "final_abs": (final_x, final_y),
        "method": method
    }

#########################
# Main Routine with Subcommands
#########################

def main():
    parser = argparse.ArgumentParser(description="Hierarchical Gaze Estimation")
    subparsers = parser.add_subparsers(dest="mode", help="Mode of operation")

    # Coarse training subcommand
    parser_coarse = subparsers.add_parser("coarse_train", help="Train the coarse model")
    parser_coarse.add_argument("--data", required=True, help="Path to CSV file")
    parser_coarse.add_argument("--output", required=True, help="Path to save coarse model (.pt)")
    parser_coarse.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser_coarse.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser_coarse.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser_coarse.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")

    # Fine training subcommand: set coarse_label to "all" to train models for all regions.
    parser_fine = subparsers.add_parser("fine_train", help="Train fine models for a specific coarse square or all (A, B, C, D)")
    parser_fine.add_argument("--data", required=True, help="Path to CSV file")
    parser_fine.add_argument("--coarse_label", required=True, help="Coarse square label (A, B, C, D or 'all' for all regions)")
    parser_fine.add_argument("--output", required=True, help="Path to save fine model (.pt) or directory when training all")
    parser_fine.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser_fine.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser_fine.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser_fine.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")

    # Inference subcommand
    parser_inf = subparsers.add_parser("inference", help="Run inference on a CSV of samples")
    parser_inf.add_argument("--data_csv", required=True, help="CSV file with gaze samples")
    parser_inf.add_argument("--coarse_model", required=True, help="Path to coarse model (.pt)")
    parser_inf.add_argument("--fine_model_dir", required=True, help="Directory with fine models named as fine_model_<coarse_label>.pt")
    parser_inf.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser_inf.add_argument("--detailed", action="store_true", help="Enable detailed debug prints")

    args = parser.parse_args()

    if args.mode == "coarse_train":
        train_coarse(args.data, args.output, device=args.device, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience)
    elif args.mode == "fine_train":
        if args.coarse_label.lower() == "all":
            if not os.path.isdir(args.output):
                os.makedirs(args.output, exist_ok=True)
            for label in sorted(get_coarse_grid().keys()):
                output_path = os.path.join(args.output, f"fine_model_{label}.pt")
                print(f"Training fine model for coarse region {label}...")
                train_fine(args.data, label, output_path, device=args.device, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience)
        else:
            train_fine(args.data, args.coarse_label, args.output, device=args.device, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience)
    elif args.mode == "inference":
        device = args.device
        coarse_model, coarse_grid, coarse_keys, head_mean, head_std, pupil_mean, pupil_std = load_coarse_model(args.coarse_model, device)
        df = pd.read_csv(args.data_csv)
        for i, row in df.iterrows():
            sample = row.to_dict()
            res = infer(sample, coarse_model, coarse_grid, coarse_keys, head_mean, head_std, pupil_mean, pupil_std, args.fine_model_dir, device, detailed=args.detailed)
            print(f"Sample {i}: Coarse region {res['coarse_label']} predicted at {res['coarse_abs']}, final prediction {res['final_abs']} using {res['method']} method")
            if i >= 9:
                break
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
