#!/usr/bin/env python
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
from datetime import datetime
from section_mapping import create_grid, get_region_for_point
from glob import glob
import re

def parse_filename_time(filepath):
    base = os.path.basename(filepath)
    name, _ = os.path.splitext(base)
    try:
        parts = name.split('_')
        if len(parts) < 3:
            return None
        date_str = parts[0]
        time_str = parts[1]
        ms_str = parts[2].zfill(6)
        dt = datetime.strptime(f"{date_str}_{time_str}_{ms_str}", "%Y%m%d_%H%M%S_%f")
        return dt.timestamp()
    except Exception:
        return None

def overlay_predictions_on_screenshots(df, screenshot_folder, output_folder, tolerance=0.5, detailed=False):
    """Overlay gaze predictions on screenshots with red stars"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all screenshot files with multiple extensions
    screenshot_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        screenshot_files.extend(glob(os.path.join(screenshot_folder, ext)))
    
    screenshot_files = sorted(screenshot_files)
    
    if not screenshot_files:
        print(f"No screenshots found in {screenshot_folder}")
        return
    
    if detailed:
        print(f"Found {len(screenshot_files)} screenshots")
    
    # Check if time column exists
    has_time = "time" in df.columns
    if not has_time:
        print("No time column in data, using sequential matching instead")
        # Create a sequential time column
        df = df.copy()
        df["time"] = np.linspace(0, 100, len(df))
        has_time = True
    
    # Analyze time format in the dataframe
    df_time_range = df["time"].max() - df["time"].min()
    df_time_format = "seconds"
    
    if df_time_range > 1000000:  # Likely milliseconds since epoch
        df_time_format = "epoch_ms"
        # Convert to seconds for easier comparison
        df = df.copy()
        df["time"] = df["time"] / 1000.0
    elif df_time_range > 10000:  # Likely seconds since epoch
        df_time_format = "epoch_s"
    
    if detailed:
        print(f"Detected time format in data: {df_time_format}")
        print(f"Time range in data: {df['time'].min():.2f} to {df['time'].max():.2f}")
    
    # Extract timestamps from filenames
    screenshot_times = []
    
    # Try to parse timestamps using our custom function
    for file in screenshot_files:
        timestamp = parse_filename_time(file)
        screenshot_times.append(timestamp)
    
    # If all timestamps are None, use sequential matching
    if all(t is None for t in screenshot_times):
        if detailed:
            print("Could not extract timestamps from filenames, using sequential matching")
        screenshot_times = np.linspace(df["time"].min(), df["time"].max(), len(screenshot_files))
    else:
        # Filter out None values
        valid_times = [t for t in screenshot_times if t is not None]
        if valid_times:
            # Check if we need to align the time ranges
            ss_min = min(valid_times)
            ss_max = max(valid_times)
            ss_range = ss_max - ss_min
            
            df_min = df["time"].min()
            df_max = df["time"].max()
            df_range = df_max - df_min
            
            # If ranges are very different, try to align them
            if abs(ss_range / df_range - 1) > 0.5 or abs(ss_min - df_min) > df_range * 0.1:
                if detailed:
                    print(f"Time ranges differ: screenshots ({ss_min:.2f}-{ss_max:.2f}) vs data ({df_min:.2f}-{df_max:.2f})")
                    print("Aligning time ranges...")
                
                # Normalize both to 0-1 range, then scale to data range
                for i, t in enumerate(screenshot_times):
                    if t is not None:
                        normalized = (t - ss_min) / ss_range
                        screenshot_times[i] = df_min + normalized * df_range
                
                if detailed:
                    print(f"Aligned screenshot times to range: {min([t for t in screenshot_times if t is not None]):.2f}-{max([t for t in screenshot_times if t is not None]):.2f}")
    
    # Process each screenshot
    processed_count = 0
    
    for i, (file, time) in enumerate(zip(screenshot_files, screenshot_times)):
        if time is None:
            if detailed:
                print(f"No timestamp for {file}, skipping")
            continue
        
        # Find gaze points within time tolerance
        mask = abs(df["time"] - time) <= tolerance
        if not mask.any():
            if detailed:
                print(f"No gaze points found for screenshot at time {time:.2f} (tolerance: {tolerance}s)")
            
            # Try with increased tolerance if no matches
            increased_tolerance = tolerance * 5
            mask = abs(df["time"] - time) <= increased_tolerance
            if not mask.any():
                if detailed:
                    print(f"Still no matches with increased tolerance ({increased_tolerance}s)")
                continue
            elif detailed:
                print(f"Found matches with increased tolerance ({increased_tolerance}s)")
        
        # Get the closest gaze point
        closest_idx = (abs(df["time"] - time)).idxmin()
        time_diff = abs(df.loc[closest_idx, "time"] - time)
        
        if detailed and i < 5:  # Show details for first few files
            print(f"File: {os.path.basename(file)}, Time: {time:.2f}, Closest data time: {df.loc[closest_idx, 'time']:.2f}, Diff: {time_diff:.2f}s")
        
        x = df.loc[closest_idx, "screen_x"]
        y = df.loc[closest_idx, "screen_y"]
        
        # Load the screenshot
        img = cv2.imread(file)
        if img is None:
            print(f"Could not load {file}, skipping")
            continue
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Scale coordinates to match screenshot dimensions
        # This is critical for accuracy - scale the predicted coordinates to match the screenshot size
        x_scale = width / 1080  # Assuming model was trained on 1080 width
        y_scale = height / 728  # Assuming model was trained on 728 height
        
        scaled_x = int(x * x_scale)
        scaled_y = int(y * y_scale)
        
        # Make sure coordinates are within image bounds
        scaled_x = max(0, min(scaled_x, width-1))
        scaled_y = max(0, min(scaled_y, height-1))
        
        if detailed and i < 5:
            print(f"  Original coords: ({x:.1f}, {y:.1f}), Scaled: ({scaled_x}, {scaled_y}), Image size: {width}x{height}")
        
        # Draw a red star at the gaze position
        cv2.circle(img, (scaled_x, scaled_y), 10, (0, 0, 255), -1)
        
        # Then draw a star pattern on top
        star_size = 15
        for angle in range(0, 360, 45):
            rad = np.radians(angle)
            end_x = int(scaled_x + star_size * np.cos(rad))
            end_y = int(scaled_y + star_size * np.sin(rad))
            # Keep line endpoints within image bounds
            end_x = max(0, min(end_x, width-1))
            end_y = max(0, min(end_y, height-1))
            cv2.line(img, (scaled_x, scaled_y), (end_x, end_y), (255, 255, 255), 2)
        
        # Add confidence score if available
        if "confidence" in df.columns:
            confidence = df.loc[closest_idx, "confidence"]
            cv2.putText(img, f"Conf: {confidence:.2f}", 
                       (scaled_x + 20, scaled_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add time information
        cv2.putText(img, f"Time: {time:.2f}s", 
                   (20, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save the output image
        output_file = os.path.join(output_folder, os.path.basename(file))
        cv2.imwrite(output_file, img)
        processed_count += 1
    
    if detailed:
        print(f"Processed {processed_count} screenshots with gaze overlay")
    else:
        print(f"Saved {processed_count} screenshots with gaze overlay to {output_folder}")

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

def analyze_data_distribution(df, grid_size=13):
    x_min, x_max = df['screen_x'].min(), df['screen_x'].max()
    y_min, y_max = df['screen_y'].min(), df['screen_y'].max()
    x_bins = np.linspace(x_min, x_max, grid_size + 1)
    y_bins = np.linspace(y_min, y_max, grid_size + 1)
    hist, _, _ = np.histogram2d(df['screen_x'], df['screen_y'], bins=(x_bins, y_bins))
    print("\nScreen Coverage (samples per region):")
    print(f"  Min samples: {int(hist.min())}")
    print(f"  Max samples: {int(hist.max())}")
    print(f"  Mean samples: {int(hist.mean())}")
    print(f"  Median samples: {int(np.median(hist))}")
    empty_regions = np.sum(hist == 0)
    if empty_regions > 0:
        print(f"  Empty regions: {empty_regions}")
    if hist.min() > 0:
        print(f"  Ratio max/min: {hist.max()/hist.min():.1f}x")
    return hist

def visualize_distribution(hist, title="Screen Coverage"):
    hist_normalized = hist / hist.max()
    print(f"\n{title} Heatmap (0=low, 9=high samples):")
    for i in range(hist.shape[0]):
        row = ""
        for j in range(hist.shape[1]):
            intensity = int(hist_normalized[i, j] * 9)
            row += str(intensity) + " "
        print(row)
    plt.figure(figsize=(10, 8))
    plt.imshow(hist, cmap='hot', origin='upper')
    plt.colorbar(label='Sample count')
    plt.title(title)
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.close()

def analyze_gaze_2stage(
    landmarks_csv,
    model_path,
    screen_width,
    screen_height,
    n_cols=10,
    n_rows=10,
    output_heatmap=None,
    output_sections=None,
    device='cpu',
    force_range=True,
    save_raw_plot=False,
    save_distributions=False,
    save_debug_csv=False,
    detailed=False,
    outlier_removal=True,
    temporal_smoothing=True,
    confidence_threshold=0.8,
    screenshot_folder=None,
    screenshot_output_folder=None,
    screenshot_tolerance=0.5
):
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        saved_dict = torch.load(model_path, map_location=device)
    if detailed:
        print("\nSaved model contents:")
        for key in saved_dict:
            print(f"  {key}")
    model = TwoStageNet(embed_dim=128).to(device)
    if 'state_dict' in saved_dict:
        model.load_state_dict(saved_dict['state_dict'])
        screen_mean = saved_dict['screen_mean']
        screen_std = saved_dict['screen_std']
        head_mean = saved_dict['head_mean']
        head_std = saved_dict['head_std']
        pupil_mean = saved_dict['pupil_mean']
        pupil_std = saved_dict['pupil_std']
    else:
        try:
            model.load_state_dict(saved_dict)
            screen_mean = [504.0, 374.5]
            screen_std = [291.0, 216.5]
            head_mean = [0]*6
            head_std = [1]*6
            pupil_mean = [0]*4
            pupil_std = [1]*4
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    model.eval()
    if torch.is_tensor(screen_mean):
        screen_mean = screen_mean.numpy()
    if torch.is_tensor(screen_std):
        screen_std = screen_std.numpy()
    if torch.is_tensor(head_mean):
        head_mean = head_mean.numpy()
    if torch.is_tensor(head_std):
        head_std = head_std.numpy()
    if torch.is_tensor(pupil_mean):
        pupil_mean = pupil_mean.numpy()
    if torch.is_tensor(pupil_std):
        pupil_std = pupil_std.numpy()
    if detailed:
        print("\nModel loaded successfully")
        print(f"Screen normalization - mean: {screen_mean}, std: {screen_std}")
        print(f"Head normalization - mean shape: {np.array(head_mean).shape}, std shape: {np.array(head_std).shape}")
        print(f"Pupil normalization - mean shape: {np.array(pupil_mean).shape}, std shape: {np.array(pupil_std).shape}")
    df = pd.read_csv(landmarks_csv)
    needed = [
        'corner_left_x','corner_left_y',
        'corner_right_x','corner_right_y',
        'nose_x','nose_y',
        'left_pupil_x','left_pupil_y',
        'right_pupil_x','right_pupil_y'
    ]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' missing in {landmarks_csv}")
    df['ipd'] = np.sqrt((df['right_pupil_x'] - df['left_pupil_x'])**2 + (df['right_pupil_y'] - df['left_pupil_y'])**2)
    df['eye_aspect_ratio'] = np.abs((df['corner_right_y'] - df['corner_left_y']) / (df['corner_right_x'] - df['corner_left_x'] + 1e-5))
    valid_frames = ((df['ipd'] > 20) & (df['ipd'] < 300) & (df['eye_aspect_ratio'] < 1.0))
    if valid_frames.sum() < len(df):
        if detailed:
            print(f"\nFiltered out {len(df) - valid_frames.sum()} frames with invalid geometry")
        df = df[valid_frames].reset_index(drop=True)
    head_data = df[['corner_left_x','corner_left_y','corner_right_x','corner_right_y','nose_x','nose_y']].values.astype(np.float32)
    pupil_data = df[['left_pupil_x','left_pupil_y','right_pupil_x','right_pupil_y']].values.astype(np.float32)
    if detailed:
        print(f"Using device: {device}")
        print("\nInput data statistics:")
        print(f"Head data range: {head_data.min():.1f} to {head_data.max():.1f}")
        print(f"Pupil data range: {pupil_data.min():.1f} to {pupil_data.max():.1f}")
    head_norm = (head_data - head_mean) / (np.array(head_std) + 1e-7)
    pupil_norm = (pupil_data - pupil_mean) / (np.array(pupil_std) + 1e-7)
    if detailed:
        print("\nNormalized input statistics:")
        print(f"Head normalized range: {head_norm.min():.1f} to {head_norm.max():.1f}")
        print(f"Pupil normalized range: {pupil_norm.min():.1f} to {pupil_norm.max():.1f}")
    head_t = torch.from_numpy(head_norm.astype(np.float32)).to(device)
    pupil_t = torch.from_numpy(pupil_norm.astype(np.float32)).to(device)
    with torch.no_grad():
        head_embeds = model.head_net(head_t).cpu().numpy()
        embedding_norms = np.linalg.norm(head_embeds, axis=1)
        confidence_scores = (embedding_norms - embedding_norms.min()) / (embedding_norms.max() - embedding_norms.min() + 1e-7)
        normalized_preds = model(head_t, pupil_t).cpu().numpy()
    if detailed:
        print("\nNormalized prediction statistics:")
        print(f"X range: {normalized_preds[:,0].min():.1f} to {normalized_preds[:,0].max():.1f}")
        print(f"Y range: {normalized_preds[:,1].min():.1f} to {normalized_preds[:,1].max():.1f}")
        print(f"Confidence scores: {confidence_scores.min():.3f} to {confidence_scores.max():.3f}")
    preds = normalized_preds * np.array(screen_std) + np.array(screen_mean)
    df["screen_x"] = preds[:,0]
    df["screen_y"] = preds[:,1]
    df["confidence"] = confidence_scores
    if force_range:
        x_min, x_max = df["screen_x"].min(), df["screen_x"].max()
        y_min, y_max = df["screen_y"].min(), df["screen_y"].max()
        if (x_max - x_min) < screen_width * 0.8 or (y_max - y_min) < screen_height * 0.8:
            if detailed:
                print("\nPredictions too constrained, forcing full range mapping")
            valid_idx = df["screen_x"].notna()
            norm_x = (df.loc[valid_idx, "screen_x"] - df.loc[valid_idx, "screen_x"].min()) / (df.loc[valid_idx, "screen_x"].max() - df.loc[valid_idx, "screen_x"].min() + 1e-8)
            norm_y = (df.loc[valid_idx, "screen_y"] - df.loc[valid_idx, "screen_y"].min()) / (df.loc[valid_idx, "screen_y"].max() - df.loc[valid_idx, "screen_y"].min() + 1e-8)
            df.loc[valid_idx, "screen_x"] = norm_x * screen_width * 0.9 + screen_width * 0.05
            df.loc[valid_idx, "screen_y"] = norm_y * screen_height * 0.9 + screen_height * 0.05
            if detailed:
                print("Applied full-range mapping:")
                print(f"X range: {df['screen_x'].min():.1f} to {df['screen_x'].max():.1f}")
                print(f"Y range: {df['screen_y'].min():.1f} to {df['screen_y'].max():.1f}")
    df["screen_x"] = df["screen_x"].clip(0, screen_width)
    df["screen_y"] = df["screen_y"].clip(0, screen_height)
    if save_raw_plot:
        plt.figure(figsize=(12, 12 * screen_height/screen_width))
        if "time" in df.columns:
            sizes = df["confidence"] * 10 + 1
            plt.scatter(df["screen_x"], df["screen_y"], alpha=0.6, s=sizes, c=df["time"], cmap='viridis', marker='.')
            plt.colorbar(label="Time (s)")
        else:
            plt.scatter(df["screen_x"], df["screen_y"], alpha=0.6, s=3, c='blue', marker='.')
        plt.title("Gaze Points (Screen Coordinates)")
        plt.xlabel("Screen Width (pixels)")
        plt.ylabel("Screen Height (pixels)")
        plt.grid(True, linestyle=':', alpha=0.3)
        plt.axvline(x=screen_width/2, color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=screen_height/2, color='r', linestyle='--', alpha=0.3)
        plt.xlim(0, screen_width)
        plt.ylim(screen_height, 0)
        plt.gca().set_aspect('equal')
        plt.savefig('raw_gaze_points.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\nRaw gaze points plot saved to raw_gaze_points.png")
    grid = create_grid(screen_width, screen_height, n_cols, n_rows)
    def assign_region(x, y):
        if pd.isna(x) or pd.isna(y):
            return None
        return get_region_for_point(x, y, grid)
    df["region"] = df.apply(lambda r: assign_region(r["screen_x"], r["screen_y"]), axis=1)
    if "time" in df.columns:
        times = df["time"].dropna().values
        if len(times) > 1:
            dt = np.mean(np.diff(times))
            if detailed:
                print(f"\nTime step: {dt:.3f} seconds")
        else:
            dt = 0.0
    else:
        dt = 0.0
        if detailed:
            print("\nNo time column found; dt set to 0")
    region_counts = df["region"].dropna().value_counts().sort_index()
    region_times = {}
    for rid in range(n_rows * n_cols):
        region_times[rid] = region_counts.get(rid, 0) * dt
    if output_sections:
        out_df = pd.DataFrame(list(region_times.items()), columns=["region","time_spent"])
        out_df.to_csv(output_sections, index=False)
        print(f"Section durations saved to {output_sections}")
    if output_heatmap:
        heatmap_data = np.zeros((n_rows, n_cols), dtype=np.float32)
        for rid, t_spent in region_times.items():
            row_idx = rid // n_cols
            col_idx = rid % n_cols
            if 0 <= row_idx < n_rows and 0 <= col_idx < n_cols:
                heatmap_data[row_idx, col_idx] = t_spent
        smoothed_heatmap = gaussian_filter(heatmap_data, sigma=1.0)
        plt.figure(figsize=(12,8))
        plt.imshow(smoothed_heatmap, cmap="hot", origin="upper", interpolation="gaussian")
        plt.colorbar(label="Time Spent (s)")
        plt.title("Gaze Duration Heatmap")
        plt.xlabel("Screen Width")
        plt.ylabel("Screen Height")
        for i in range(n_cols + 1):
            plt.axvline(x=i - 0.5, color='white', linestyle=':')
        for i in range(n_rows + 1):
            plt.axhline(y=i - 0.5, color='white', linestyle=':')
        center_x = n_cols / 2 - 0.5
        center_y = n_rows / 2 - 0.5
        plt.plot(center_x, center_y, 'w+', markersize=10)
        plt.text(0, 0, "Top-Left", color='white', fontsize=8, horizontalalignment='left', verticalalignment='top')
        plt.text(n_cols-1, 0, "Top-Right", color='white', fontsize=8, horizontalalignment='right', verticalalignment='top')
        plt.text(0, n_rows-1, "Bottom-Left", color='white', fontsize=8, horizontalalignment='left', verticalalignment='bottom')
        plt.text(n_cols-1, n_rows-1, "Bottom-Right", color='white', fontsize=8, horizontalalignment='right', verticalalignment='bottom')
        plt.savefig(output_heatmap, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved to {output_heatmap}")
    if detailed:
        print("\nGaze Distribution Statistics:")
        print(f"Total regions visited: {len(region_counts)}/{n_rows*n_cols}")
        print(f"Most viewed region: {region_counts.index[0]} ({region_counts.iloc[0]} samples)")
        print(f"Least viewed region: {region_counts.index[-1]} ({region_counts.iloc[-1]} samples)")
        print(f"Mean samples per region: {region_counts.mean():.1f}")
        print(f"Median samples per region: {region_counts.median():.1f}")
        x_range = df["screen_x"].max() - df["screen_x"].min()
        y_range = df["screen_y"].max() - df["screen_y"].min()
        print(f"\nScreen Coverage:")
        print(f"X range: {df['screen_x'].min():.1f} to {df['screen_x'].max():.1f} ({x_range:.1f} pixels)")
        print(f"Y range: {df['screen_y'].min():.1f} to {df['screen_y'].max():.1f} ({y_range:.1f} pixels)")
    if save_debug_csv:
        df.to_csv("debug_predictions.csv", index=False)
        print("\nDebug CSV saved to debug_predictions.csv")
    if screenshot_folder and screenshot_output_folder:
        print("\nOverlaying predictions on screenshots...")
        overlay_predictions_on_screenshots(df, screenshot_folder, screenshot_output_folder, tolerance=screenshot_tolerance, detailed=detailed)
    
    if temporal_smoothing:
        window_size = 5
        if detailed:
            print(f"\nApplying temporal smoothing with window size {window_size}")
        
        df["screen_x"] = df["screen_x"].rolling(window=window_size, center=True).mean().fillna(df["screen_x"])
        df["screen_y"] = df["screen_y"].rolling(window=window_size, center=True).mean().fillna(df["screen_y"])
    
    if outlier_removal:
        if detailed:
            print("\nRemoving outliers")
            print(f"Initial points: {len(df)}")
        
        if "confidence" in df.columns:
            high_conf = df["confidence"] >= confidence_threshold
            if detailed:
                print(f"Points with confidence >= {confidence_threshold}: {high_conf.sum()}")
            
            if high_conf.sum() > len(df) * 0.5:
                df = df[high_conf].reset_index(drop=True)
                if detailed:
                    print(f"After confidence filtering: {len(df)}")
        
        x_median = df["screen_x"].median()
        y_median = df["screen_y"].median()
        x_std = df["screen_x"].std()
        y_std = df["screen_y"].std()
        
        df["distance"] = np.sqrt(
            ((df["screen_x"] - x_median) / x_std) ** 2 + 
            ((df["screen_y"] - y_median) / y_std) ** 2
        )
        
        inliers = df["distance"] <= 3.0
        if inliers.sum() > len(df) * 0.8:
            df = df[inliers].reset_index(drop=True)
            if detailed:
                print(f"After spatial outlier removal: {len(df)}")
        
        df = df.drop(columns=["distance"])

def main():
    parser = argparse.ArgumentParser(description="Analyze Gaze with Two-Stage Net and optional screenshot overlay")
    parser.add_argument("--landmarks_csv", required=True, help="CSV file with facial landmarks")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--screen_width", type=int, required=True, help="Screen width in pixels")
    parser.add_argument("--screen_height", type=int, required=True, help="Screen height in pixels")
    parser.add_argument("--n_cols", type=int, default=10, help="Number of columns in grid")
    parser.add_argument("--n_rows", type=int, default=10, help="Number of rows in grid")
    parser.add_argument("--output_heatmap", help="Path to save heatmap image")
    parser.add_argument("--output_sections", help="Path to save section durations CSV")
    parser.add_argument("--force_range", action="store_true", help="Force predictions to cover full screen range")
    parser.add_argument("--save_raw_plot", action="store_true", help="Save raw gaze points plot")
    parser.add_argument("--save_distributions", action="store_true", help="Save coordinate distribution plots")
    parser.add_argument("--save_debug_csv", action="store_true", help="Save debug CSV with predictions")
    parser.add_argument("--detailed", action="store_true", help="Print detailed information")
    parser.add_argument("--no_outlier_removal", action="store_true", help="Disable outlier removal")
    parser.add_argument("--no_temporal_smoothing", action="store_true", help="Disable temporal smoothing")
    parser.add_argument("--confidence_threshold", type=float, default=0.8, help="Confidence threshold (0-1)")
    parser.add_argument("--screenshot_folder", help="Folder containing screenshot images")
    parser.add_argument("--screenshot_output_folder", help="Folder to save screenshots with overlay")
    parser.add_argument("--screenshot_tolerance", type=float, default=0.5, help="Time tolerance for matching screenshots (seconds)")
    args = parser.parse_args()
    analyze_gaze_2stage(
        args.landmarks_csv,
        args.model,
        args.screen_width,
        args.screen_height,
        args.n_cols,
        args.n_rows,
        args.output_heatmap,
        args.output_sections,
        device='cpu',
        force_range=args.force_range,
        save_raw_plot=args.save_raw_plot,
        save_distributions=args.save_distributions,
        save_debug_csv=args.save_debug_csv,
        detailed=args.detailed,
        outlier_removal=not args.no_outlier_removal,
        temporal_smoothing=not args.no_temporal_smoothing,
        confidence_threshold=args.confidence_threshold,
        screenshot_folder=args.screenshot_folder,
        screenshot_output_folder=args.screenshot_output_folder,
        screenshot_tolerance=args.screenshot_tolerance
    )

if __name__ == "__main__":
    main()