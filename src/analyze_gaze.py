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
from hierarchical_gaze import load_coarse_model, load_fine_model, preprocess_sample, infer

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
        
        x = df.loc[closest_idx, "final_x"]
        y = df.loc[closest_idx, "final_y"]
        quadrant = df.loc[closest_idx, "quadrant"]
        
        # Load the screenshot
        img = cv2.imread(file)
        if img is None:
            print(f"Could not load {file}, skipping")
            continue
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Scale coordinates to match screenshot dimensions
        # This is critical for accuracy - scale the predicted coordinates to match the screenshot size
        x_scale = width / 1024  # Assuming model was trained on 1024 width
        y_scale = height / 768  # Assuming model was trained on 768 height
        
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

def analyze_gaze_hierarchical(
    landmarks_csv,
    coarse_model_path,
    fine_model_dir,
    output_csv=None,
    device='cpu',
    detailed=False,
    screenshot_folder=None,
    screenshot_output_folder=None,
    screenshot_tolerance=0.5
):
    print(f"Loading coarse model from {coarse_model_path}")
    coarse_model, coarse_grid, coarse_keys, head_mean, head_std, pupil_mean, pupil_std = load_coarse_model(coarse_model_path, device)
    
    print(f"Loading landmarks from {landmarks_csv}")
    df = pd.read_csv(landmarks_csv)
    
    results = []
    
    print(f"Processing {len(df)} samples...")
    for i, row in df.iterrows():
        sample = row.to_dict()
        res = infer(sample, coarse_model, coarse_grid, coarse_keys, head_mean, head_std, pupil_mean, pupil_std, fine_model_dir, device)
        
        results.append({
            'time': sample.get('time', i),
            'coarse_label': res['coarse_label'],
            'quadrant': res['coarse_label'] if res['method'] == 'coarse' else res['method'],
            'final_x': res['final_abs'][0],
            'final_y': res['final_abs'][1],
            'method': res['method']
        })
        
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i} samples...")
    
    results_df = pd.DataFrame(results)
    
    if output_csv:
        results_df.to_csv(output_csv, index=False)
        print(f"Saved results to {output_csv}")
    
    if screenshot_folder and screenshot_output_folder:
        print(f"Overlaying predictions on screenshots...")
        overlay_predictions_on_screenshots(
            results_df, 
            screenshot_folder, 
            screenshot_output_folder,
            tolerance=screenshot_tolerance,
            detailed=detailed
        )
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="Analyze gaze using hierarchical model")
    
    # Required arguments
    parser.add_argument("--landmarks_csv", required=True, help="CSV file with facial landmarks")
    parser.add_argument("--coarse_model", dest="model", required=True, help="Path to coarse model")
    parser.add_argument("--fine_model_dir", required=True, help="Directory with fine models")
    parser.add_argument("--screen_width", type=int, default=1024, help="Screen width in pixels")
    parser.add_argument("--screen_height", type=int, default=768, help="Screen height in pixels")
    
    # Optional arguments
    parser.add_argument("--n_cols", type=int, default=4, help="Number of columns for grid analysis")
    parser.add_argument("--n_rows", type=int, default=4, help="Number of rows for grid analysis")
    parser.add_argument("--output_csv", help="Path to save output CSV")
    parser.add_argument("--output_heatmap", help="Path to save heatmap image")
    parser.add_argument("--output_sections", help="Path to save sections analysis CSV")
    parser.add_argument("--force_range", action="store_true", help="Force predictions to stay within screen bounds")
    parser.add_argument("--save_raw_plot", action="store_true", help="Save raw gaze plot")
    parser.add_argument("--save_distributions", action="store_true", help="Save X and Y distributions")
    parser.add_argument("--save_debug_csv", action="store_true", help="Save debug CSV with all data")
    parser.add_argument("--detailed", action="store_true", help="Print detailed information")
    parser.add_argument("--no_outlier_removal", action="store_true", help="Disable outlier removal")
    parser.add_argument("--no_temporal_smoothing", action="store_true", help="Disable temporal smoothing")
    parser.add_argument("--confidence_threshold", type=float, default=0.0, help="Confidence threshold for predictions")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--screenshot_folder", help="Folder with screenshots")
    parser.add_argument("--screenshot_output_folder", help="Folder to save annotated screenshots")
    parser.add_argument("--screenshot_tolerance", type=float, default=0.5, help="Time tolerance for matching screenshots")
    
    args = parser.parse_args()
    
    analyze_gaze_hierarchical(
        args.landmarks_csv,
        args.model,
        args.fine_model_dir,
        args.output_csv,
        args.device,
        args.detailed,
        args.screenshot_folder,
        args.screenshot_output_folder,
        args.screenshot_tolerance
    )

if __name__ == "__main__":
    main()