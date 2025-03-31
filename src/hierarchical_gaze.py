import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

class GazeDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df.drop_duplicates(inplace=True)

        needed = [
            'corner_left_x','corner_left_y',
            'corner_right_x','corner_right_y',
            'nose_x','nose_y',
            'left_pupil_x','left_pupil_y',
            'right_pupil_x','right_pupil_y',
            'screen_x','screen_y'
        ]
        for c in needed:
            if c not in df.columns:
                raise ValueError(f"Column '{c}' missing from {csv_path}")

        self.head_data = df[[
            'corner_left_x','corner_left_y',
            'corner_right_x','corner_right_y',
            'nose_x','nose_y'
        ]].values.astype(np.float32)

        self.pupil_data = df[[
            'left_pupil_x','left_pupil_y',
            'right_pupil_x','right_pupil_y'
        ]].values.astype(np.float32)

        self.screen = df[['screen_x','screen_y']].values.astype(np.float32)

        self.head_mean = self.head_data.mean(axis=0)
        self.head_std = self.head_data.std(axis=0)
        self.pupil_mean = self.pupil_data.mean(axis=0)
        self.pupil_std = self.pupil_data.std(axis=0)
        self.screen_mean = self.screen.mean(axis=0)
        self.screen_std = self.screen.std(axis=0)

        self.head_normalized = (self.head_data - self.head_mean) / (self.head_std + 1e-7)
        self.pupil_normalized = (self.pupil_data - self.pupil_mean) / (self.pupil_std + 1e-7)
        self.screen_normalized = (self.screen - self.screen_mean) / (self.screen_std + 1e-7)

        print("\nData Ranges:")
        print(f"Screen X: {self.screen[:,0].min():.1f} to {self.screen[:,0].max():.1f}")
        print(f"Screen Y: {self.screen[:,1].min():.1f} to {self.screen[:,1].max():.1f}")

    def __len__(self):
        return len(self.head_data)

    def __getitem__(self, idx):
        return (
            self.head_normalized[idx],
            self.pupil_normalized[idx],
            self.screen_normalized[idx]
        )

    def denormalize_screen(self, normalized_coords):
        return normalized_coords * self.screen_std + self.screen_mean

class HeadPoseNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        return self.fc(x)

class PupilNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 2)
        )

    def forward(self, head_embed, pupil):
        x = torch.cat([head_embed, pupil], dim=1)
        return self.fc(x)

class TwoStageNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.head_net = HeadPoseNet(embed_dim=embed_dim)
        self.pupil_net = PupilNet(embed_dim=embed_dim)

    def forward(self, head_input, pupil_input):
        h = self.head_net(head_input)
        out = self.pupil_net(h, pupil_input)
        return out

def train_one_lr(
    dataset, lr=1e-3, batch_size=32, max_epochs=50000,
    patience=1000, device='cpu'
):
    test_size = int(len(dataset)*0.2)
    train_size = len(dataset) - test_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = TwoStageNet(embed_dim=128).to(device)
    criterion = nn.MSELoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=max_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        div_factor=10,
        final_div_factor=100,
        anneal_strategy='cos'
    )

    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(1, max_epochs+1):
        model.train()
        train_losses = []
        for batch in train_loader:
            head_data, pupil_data, targets = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            outputs = model(head_data, pupil_data)
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())
        
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch in test_loader:
                head_data, pupil_data, targets = [b.to(device) for b in batch]
                outputs = model(head_data, pupil_data)
                loss = criterion(outputs, targets)
                test_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)
        
        print(f"Epoch {epoch:4d} | LR={optimizer.param_groups[0]['lr']:.6f} | "
              f"TrainLoss={train_loss:.4f} | TestLoss={test_loss:.4f}")
        
        if test_loss < best_loss:
            best_loss = test_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
    return best_loss, best_state

def final_train_on_full(
    dataset, model_state, lr, batch_size=32, epochs=5, device='mps'
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = TwoStageNet(embed_dim=128).to(device)
    model.load_state_dict(model_state)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for head, pupil, scr in loader:
            head, pupil, scr = head.to(device), pupil.to(device), scr.to(device)
            optimizer.zero_grad()
            pred = model(head, pupil)
            loss = criterion(pred, scr)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * head.size(0)
            total_count += head.size(0)
        print(f"Refine epoch {ep}/{epochs} | Loss={(total_loss/total_count):.4f}")

    return model.state_dict()

def analyze_data_distribution(dataset, grid_size=13):
    print("\nDetailed Data Analysis:")

    all_data = [dataset[i] for i in range(len(dataset))]
    head = np.stack([d[0] for d in all_data])
    pupil = np.stack([d[1] for d in all_data])
    screen = np.stack([d[2] for d in all_data])
    
    # Use denormalized screen coordinates for analysis
    denorm_screen = dataset.denormalize_screen(screen)
    
    # Create grid with specified size (default 13x13)
    # Use exact screen dimensions rather than min/max of data
    x_min, x_max = 0, 1008  # Standard screen width
    y_min, y_max = 0, 749   # Standard screen height
    
    x_bins = np.linspace(x_min, x_max, grid_size + 1)
    y_bins = np.linspace(y_min, y_max, grid_size + 1)
    
    hist, _, _ = np.histogram2d(denorm_screen[:,0], denorm_screen[:,1], bins=(x_bins, y_bins))
    
    print("\nScreen Coverage (samples per region):")
    print(f"  Grid size: {grid_size}x{grid_size}")
    print("  Min samples in any region:", int(hist.min()))
    print("  Max samples in any region:", int(hist.max()))
    print("  Mean samples per region:", int(hist.mean()))
    print("  Empty regions:", np.sum(hist == 0))
    
    # Avoid division by zero
    if hist.min() > 0:
        print(f"  Ratio max/min: {hist.max()/hist.min():.1f}x")
    else:
        print("  Ratio max/min: ∞ (some regions have zero samples)")
    
    print("\nHead Pose Ranges:")
    print(f"  Left corner X: {head[:,0].min():.1f} to {head[:,0].max():.1f}")
    print(f"  Left corner Y: {head[:,1].min():.1f} to {head[:,1].max():.1f}")
    print(f"  Right corner X: {head[:,2].min():.1f} to {head[:,2].max():.1f}")
    print(f"  Right corner Y: {head[:,3].min():.1f} to {head[:,3].max():.1f}")
    print(f"  Nose X: {head[:,4].min():.1f} to {head[:,4].max():.1f}")
    print(f"  Nose Y: {head[:,5].min():.1f} to {head[:,5].max():.1f}")
    
    print("\nPupil Position Ranges:")
    print(f"  Left pupil X: {pupil[:,0].min():.1f} to {pupil[:,0].max():.1f}")
    print(f"  Left pupil Y: {pupil[:,1].min():.1f} to {pupil[:,1].max():.1f}")
    print(f"  Right pupil X: {pupil[:,2].min():.1f} to {pupil[:,2].max():.1f}")
    print(f"  Right pupil Y: {pupil[:,3].min():.1f} to {pupil[:,3].max():.1f}")
    
    print("\nScreen Coverage Heatmap (0=low, 9=high samples):")
    if hist.max() > 0:  # Avoid division by zero
        hist_normalized = hist / hist.max()
    else:
        hist_normalized = hist
        
    for i in range(grid_size):
        row = ""
        for j in range(grid_size):
            intensity = int(hist_normalized[j, grid_size-1-i] * 9)
            row += str(intensity) + " "
        print(row)

def evaluate_model_with_error_bars(model, dataset, device='cpu', n_bootstrap=100, confidence=0.95):
    """
    Evaluate model performance with error bars using bootstrap resampling.
    
    Parameters:
    -----------
    model : nn.Module
        The trained model to evaluate
    dataset : Dataset
        The dataset to evaluate on
    device : str
        Device to use for computation
    n_bootstrap : int
        Number of bootstrap samples to generate
    confidence : float
        Confidence level for the error bars (0-1)
    
    Returns:
    --------
    dict
        Dictionary containing error metrics with confidence intervals
    """
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    import torch
    
    model.eval()
    
    # Create a dataloader for the full dataset
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Collect all predictions and targets
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for head, pupil, target in loader:
            head = head.to(device)
            pupil = pupil.to(device)
            target = target.to(device)
            
            pred = model(head, pupil)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Denormalize predictions and targets
    denorm_preds = dataset.denormalize_screen(all_preds)
    denorm_targets = dataset.denormalize_screen(all_targets)
    
    # Calculate pixel errors
    pixel_errors = np.sqrt(np.sum((denorm_preds - denorm_targets)**2, axis=1))
    
    # Calculate basic statistics
    mean_error = np.mean(pixel_errors)
    median_error = np.median(pixel_errors)
    percentile_95 = np.percentile(pixel_errors, 95)
    
    # Bootstrap to get confidence intervals
    bootstrap_means = []
    bootstrap_medians = []
    bootstrap_95percentiles = []
    
    n_samples = len(pixel_errors)
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_sample = pixel_errors[indices]
        
        bootstrap_means.append(np.mean(bootstrap_sample))
        bootstrap_medians.append(np.median(bootstrap_sample))
        bootstrap_95percentiles.append(np.percentile(bootstrap_sample, 95))
    
    # Calculate confidence intervals
    alpha = (1 - confidence) / 2
    ci_low_idx = int(n_bootstrap * alpha)
    ci_high_idx = int(n_bootstrap * (1 - alpha))
    
    bootstrap_means.sort()
    bootstrap_medians.sort()
    bootstrap_95percentiles.sort()
    
    mean_ci = (bootstrap_means[ci_low_idx], bootstrap_means[ci_high_idx])
    median_ci = (bootstrap_medians[ci_low_idx], bootstrap_medians[ci_high_idx])
    percentile_95_ci = (bootstrap_95percentiles[ci_low_idx], bootstrap_95percentiles[ci_high_idx])
    
    # Calculate per-axis errors
    x_errors = np.abs(denorm_preds[:, 0] - denorm_targets[:, 0])
    y_errors = np.abs(denorm_preds[:, 1] - denorm_targets[:, 1])
    
    mean_x_error = np.mean(x_errors)
    mean_y_error = np.mean(y_errors)
    
    # Bootstrap for per-axis errors
    bootstrap_x_means = []
    bootstrap_y_means = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_x = x_errors[indices]
        bootstrap_y = y_errors[indices]
        
        bootstrap_x_means.append(np.mean(bootstrap_x))
        bootstrap_y_means.append(np.mean(bootstrap_y))
    
    bootstrap_x_means.sort()
    bootstrap_y_means.sort()
    
    x_error_ci = (bootstrap_x_means[ci_low_idx], bootstrap_x_means[ci_high_idx])
    y_error_ci = (bootstrap_y_means[ci_low_idx], bootstrap_y_means[ci_high_idx])
    
    # Return all metrics
    return {
        'mean_error': mean_error,
        'mean_error_ci': mean_ci,
        'median_error': median_error,
        'median_error_ci': median_ci,
        'percentile_95': percentile_95,
        'percentile_95_ci': percentile_95_ci,
        'mean_x_error': mean_x_error,
        'mean_x_error_ci': x_error_ci,
        'mean_y_error': mean_y_error,
        'mean_y_error_ci': y_error_ci,
        'raw_errors': pixel_errors
    }

def train_two_stage_net(
    csv_path,
    output_path,
    lr_candidates=[1e-3],
    max_epochs=50000,
    patience=1000,
    batch_size=32,
    device='cpu',
    grid_size=13
):
    dataset = GazeDataset(csv_path)
    analyze_data_distribution(dataset, grid_size=grid_size)
    print(f"Using device: {device}")

    print("\nTraining data statistics:")
    print(f"X range: {dataset.screen[:,0].min():.2f} to {dataset.screen[:,0].max():.2f}")
    print(f"Y range: {dataset.screen[:,1].min():.2f} to {dataset.screen[:,1].max():.2f}")
    print(f"X mean: {dataset.screen[:,0].mean():.2f}, std: {dataset.screen[:,0].std():.2f}")
    print(f"Y mean: {dataset.screen[:,1].mean():.2f}, std: {dataset.screen[:,1].std():.2f}")
    print(f"Number of samples: {len(dataset)}")

    best_overall_loss = float('inf')
    best_lr = None
    best_state = None

    overall_start = time.time()
    for lr in lr_candidates:
        print(f"\n=== Training with LR={lr} ===")
        candidate_start = time.time()
        test_loss, model_state = train_one_lr(
            dataset,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            device=device
        )
        candidate_elapsed = time.time() - candidate_start
        print(f"LR={lr} training time: {candidate_elapsed:.2f} seconds, Test Loss: {test_loss:.4f}")
        
        if test_loss < best_overall_loss:
            best_overall_loss = test_loss
            best_lr = lr
            best_state = model_state
    
    overall_elapsed = time.time() - overall_start
    print(f"\nTotal training time: {overall_elapsed:.2f} seconds")
    print(f"Best LR: {best_lr}, Best Test Loss: {best_overall_loss:.4f}")
    
    # Final training on full dataset with best LR
    print("\nFinal training on full dataset...")
    final_state = final_train_on_full(dataset, best_state, best_lr, device=device)
    
    # Evaluate model with error bars
    print("\nEvaluating model performance with error bars...")
    model = TwoStageNet(embed_dim=128).to(device)
    model.load_state_dict(final_state)
    
    error_metrics = evaluate_model_with_error_bars(model, dataset, device=device)
    
    print("\nModel Performance Metrics (in pixels):")
    print(f"Mean Error: {error_metrics['mean_error']:.2f} pixels")
    print(f"95% CI: ({error_metrics['mean_error_ci'][0]:.2f}, {error_metrics['mean_error_ci'][1]:.2f})")
    print(f"Median Error: {error_metrics['median_error']:.2f} pixels")
    print(f"95% CI: ({error_metrics['median_error_ci'][0]:.2f}, {error_metrics['median_error_ci'][1]:.2f})")
    print(f"95th Percentile Error: {error_metrics['percentile_95']:.2f} pixels")
    print(f"95% CI: ({error_metrics['percentile_95_ci'][0]:.2f}, {error_metrics['percentile_95_ci'][1]:.2f})")
    print(f"Mean X Error: {error_metrics['mean_x_error']:.2f} pixels")
    print(f"95% CI: ({error_metrics['mean_x_error_ci'][0]:.2f}, {error_metrics['mean_x_error_ci'][1]:.2f})")
    print(f"Mean Y Error: {error_metrics['mean_y_error']:.2f} pixels")
    print(f"95% CI: ({error_metrics['mean_y_error_ci'][0]:.2f}, {error_metrics['mean_y_error_ci'][1]:.2f})")
    
    # Create error distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(error_metrics['raw_errors'], bins=50, alpha=0.7, color='blue')
    plt.axvline(error_metrics['mean_error'], color='red', linestyle='--', label=f'Mean: {error_metrics["mean_error"]:.2f}px')
    plt.axvline(error_metrics['median_error'], color='green', linestyle='--', label=f'Median: {error_metrics["median_error"]:.2f}px')
    plt.axvline(error_metrics['percentile_95'], color='orange', linestyle='--', label=f'95th Percentile: {error_metrics["percentile_95"]:.2f}px')
    plt.xlabel('Error (pixels)')
    plt.ylabel('Frequency')
    plt.title('Gaze Prediction Error Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=150)
    print("Error distribution plot saved to error_distribution.png")
    
    # Save model with normalization parameters
    save_dict = {
        'state_dict': final_state,
        'head_mean': dataset.head_mean,
        'head_std': dataset.head_std,
        'pupil_mean': dataset.pupil_mean,
        'pupil_std': dataset.pupil_std,
        'screen_mean': dataset.screen_mean,
        'screen_std': dataset.screen_std,
        'error_metrics': error_metrics
    }
    
    torch.save(save_dict, output_path)
    print(f"Model saved to {output_path}")
    
    return final_state, error_metrics

def main():
    parser = argparse.ArgumentParser(
        description="Two-stage net with automatic LR search, early stopping, and timing."
    )
    parser.add_argument("--data", required=True, help="Calibration CSV path.")
    parser.add_argument("--output", required=True, help="Where to save the final .pt model.")
    parser.add_argument("--lr_candidates", nargs="+", type=float, default=[1e-3],
                        help="List of candidate LRs to try (space-separated).")
    parser.add_argument("--max_epochs", type=int, default=50000,
                        help="Upper bound on epochs if improvement continues.")
    parser.add_argument("--patience", type=int, default=1000,
                        help="Early stopping patience in epochs.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--device", type=str, default='cpu',
                        help="Device to use for training.")
    parser.add_argument("--grid_size", type=int, default=13,
                        help="Grid size for data analysis.")
    args = parser.parse_args()

    train_two_stage_net(
        csv_path=args.data,
        output_path=args.output,
        lr_candidates=args.lr_candidates,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
        grid_size=args.grid_size
    )

if __name__ == "__main__":
    main()
