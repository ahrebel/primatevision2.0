#!/usr/bin/env python
import argparse
import os
import re
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

# Import grid functions from section_mapping
from section_mapping import create_grid, get_region_for_point

# Import functions from hierarchical_gaze
from hierarchical_gaze import load_coarse_model, load_fine_model, infer

def extract_elapsed_from_filename(filename):
    """
    Extracts the elapsed time (in seconds) from the screenshot filename.
    Assumes the filename format is "screenshot_XXX.XXX.png" where XXX.XXX 
    represents the elapsed seconds (with millisecond precision).
    """
    match = re.search(r'screenshot_(\d+\.\d{3})', filename)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Filename {filename} does not contain a valid elapsed time.")

def analyze_gaze(
    landmarks_csv, 
    coarse_model_path, 
    fine_model_dir, 
    output_csv=None,
    output_heatmap=None, 
    output_sections=None, 
    screen_width=1024, 
    screen_height=768,
    n_cols=5, 
    n_rows=5, 
    device="cpu",
    detailed=False,
    screenshot_folder=None,
    screenshot_output_folder=None,
    screenshot_tolerance=0.5  # in seconds
):
    """
    Analyze gaze data using hierarchical gaze models.

    This version expects the landmarks CSV to have been produced by process_video,
    meaning it should include a numeric 'elapsed' column (seconds since video start)
    and optionally a 'datetime' column. Screenshots should be named as
      screenshot_XXX.XXX.png
    where XXX.XXX is the elapsed time in seconds (with millisecond precision).

    Args:
        landmarks_csv: Path to CSV with facial landmarks (must contain an 'elapsed' column)
        coarse_model_path: Path to coarse gaze model
        fine_model_dir: Directory containing fine gaze models
        output_csv: Path to save processed results CSV
        output_heatmap: Path to save heatmap visualization image
        output_sections: Path to save section analysis image
        screen_width: Width of screen in pixels (analysis target)
        screen_height: Height of screen in pixels (analysis target)
        n_cols: Number of columns for section analysis
        n_rows: Number of rows for section analysis
        device: Device to run the models on ('cpu' or 'cuda')
        detailed: Whether to print detailed logging info
        screenshot_folder: Folder containing screenshots
        screenshot_output_folder: Folder to save annotated screenshots
        screenshot_tolerance: Time tolerance (in seconds) for matching screenshots
    Returns:
        DataFrame with processed gaze data
    """
    # Set device
    device = torch.device(device)
    
    # Load coarse model and set to evaluation mode
    if detailed:
        print(f"Loading coarse model from {coarse_model_path}")
    try:
        coarse_model, coarse_grid, coarse_keys, head_mean, head_std, pupil_mean, pupil_std = load_coarse_model(coarse_model_path, device)
        coarse_model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load coarse model: {e}")
    
    # Load fine models and set them to evaluation mode
    fine_models = {}
    for model_file in os.listdir(fine_model_dir):
        model_path = os.path.join(fine_model_dir, model_file)
        if os.path.isfile(model_path) and model_file.endswith('.pt'):
            try:
                fine_model = torch.load(model_path, map_location=device)
                fine_model.eval()
                # Extract the region identifier from the filename (e.g., 'A' from 'fine_model_A.pt')
                region_id = model_file.split('_')[-1].split('.')[0]
                fine_models[region_id] = fine_model
                if detailed:
                    print(f"Loaded fine model for region {region_id} from {model_path}")
            except Exception as e:
                print(f"Failed to load fine model {model_file}: {e}")
    
    # Load landmarks data
    if detailed:
        print(f"Loading landmarks from {landmarks_csv}")
    df = pd.read_csv(landmarks_csv)
    
    # Ensure the 'elapsed' column exists (in seconds)
    if 'elapsed' not in df.columns:
        if 'time' in df.columns:
            df['elapsed'] = df['time']
        elif 'frame' in df.columns:
            df['elapsed'] = df['frame'] / 30.0  # assume 30 fps if not provided
        else:
            df['elapsed'] = np.arange(len(df)) / 30.0
    
    # Create a simple grid for region mapping
    fallback_grid = []
    cell_width = screen_width / n_cols
    cell_height = screen_height / n_rows
    for row in range(n_rows):
        for col in range(n_cols):
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = (col + 1) * cell_width
            y2 = (row + 1) * cell_height
            region_id = row * n_cols + col + 1  # 1-based region ID
            fallback_grid.append((col, row, x1, y1, x2, y2, region_id))
    
    def get_region(x, y):
        for cell in fallback_grid:
            _, _, x1, y1, x2, y2, region_id = cell
            if x1 <= x < x2 and y1 <= y < y2:
                return region_id
        return 0
    
    results = []
    for i, row in enumerate(df.itertuples()):
        if i % 100 == 0 and detailed:
            print(f"Processing row {i}/{len(df)}")
        try:
            head_features = np.array([
                row.nose_x, row.nose_y,
                row.corner_left_x, row.corner_left_y,
                row.corner_right_x, row.corner_right_y
            ])
            pupil_features = np.array([
                row.left_pupil_x, row.left_pupil_y,
                row.right_pupil_x, row.right_pupil_y
            ])
            
            # Normalize features
            head_features = (head_features - head_mean) / head_std
            pupil_features = (pupil_features - pupil_mean) / pupil_std
            
            head_tensor = torch.tensor(head_features, dtype=torch.float32).to(device)
            pupil_tensor = torch.tensor(pupil_features, dtype=torch.float32).to(device)
            
            # Step 1: Use coarse model to predict the quadrant
            with torch.no_grad():
                coarse_output = coarse_model(head_tensor.unsqueeze(0), pupil_tensor.unsqueeze(0))
                if isinstance(coarse_output, tuple):
                    coarse_output = coarse_output[0]
                coarse_probs = torch.softmax(coarse_output, dim=1).cpu().numpy().flatten()
                coarse_region_idx = np.argmax(coarse_probs)
                coarse_region = coarse_keys[coarse_region_idx]  # Get the region label (A, B, C, D, etc.)
                coarse_confidence = coarse_probs[coarse_region_idx]
            
            # Step 2: Use the corresponding fine model for that quadrant
            if coarse_region in fine_models:
                fine_model = fine_models[coarse_region]
                with torch.no_grad():
                    fine_output = fine_model(head_tensor.unsqueeze(0), pupil_tensor.unsqueeze(0))
                    if isinstance(fine_output, tuple):
                        fine_output = fine_output[0]
                    # Get the precise coordinates within the quadrant
                    fine_coords = fine_output.cpu().numpy().flatten()
                    # Map these coordinates to screen coordinates
                    # Assuming fine_coords are normalized (0-1) within the quadrant
                    # We need to map them to the specific quadrant's area on screen
                    
                    # Find the quadrant boundaries from coarse_grid
                    for grid_cell in coarse_grid:
                        if grid_cell[0] == coarse_region:  # Match the region label
                            x1, y1, x2, y2 = grid_cell[1:5]  # Extract quadrant boundaries
                            break
                    else:
                        # Fallback if quadrant not found
                        x1, y1 = 0, 0
                        x2, y2 = screen_width, screen_height
                    
                    # Map normalized coordinates to screen coordinates within the quadrant
                    screen_x = x1 + fine_coords[0] * (x2 - x1)
                    screen_y = y1 + fine_coords[1] * (y2 - y1)
                    
                    # Ensure coordinates are within screen bounds
                    screen_x = max(0, min(screen_width, screen_x))
                    screen_y = max(0, min(screen_height, screen_y))
            else:
                # Fallback to coarse prediction if no fine model exists for this region
                # Get the center of the predicted coarse region
                for grid_cell in coarse_grid:
                    if grid_cell[0] == coarse_region:
                        x1, y1, x2, y2 = grid_cell[1:5]
                        screen_x = (x1 + x2) / 2
                        screen_y = (y1 + y2) / 2
                        break
                else:
                    # If region not found in grid, use a default
                    screen_x = screen_width / 2
                    screen_y = screen_height / 2
            
            # Store the results
            result = {
                'frame': getattr(row, 'frame', i),
                'elapsed': getattr(row, 'elapsed', float(i) / 30.0),
                'screen_x': screen_x,
                'screen_y': screen_y,
                'confidence': coarse_confidence,
                'coarse_region': coarse_region,
                'region': get_region(screen_x, screen_y)
            }
            results.append(result)
        except Exception as e:
            if detailed:
                print(f"Error processing row {i}: {str(e)}")
            # Add a placeholder result for failed predictions
            results.append({
                'frame': getattr(row, 'frame', i),
                'elapsed': getattr(row, 'elapsed', float(i) / 30.0),
                'screen_x': np.nan,
                'screen_y': np.nan,
                'confidence': 0.0,
                'coarse_region': 'Unknown',
                'region': 0
            })
    
    results_df = pd.DataFrame(results)
    
    if output_csv:
        if detailed:
            print(f"Saving results to {output_csv}")
        results_df.to_csv(output_csv, index=False)
    
    if output_heatmap:
        if detailed:
            print(f"Generating heatmap at {output_heatmap}")
        generate_heatmap(results_df, output_heatmap, screen_width, screen_height)
    
    if output_sections:
        if detailed:
            print(f"Generating section analysis at {output_sections}")
        analyze_sections(results_df, output_sections, n_cols, n_rows, screen_width, screen_height)
    
    # Process screenshots using elapsed times
    if screenshot_folder and screenshot_output_folder:
        if detailed:
            print(f"Processing screenshots from {screenshot_folder}")
        overlay_predictions_on_screenshots(
            results_df, 
            screenshot_folder, 
            screenshot_output_folder, 
            tolerance=screenshot_tolerance,
            detailed=detailed,
            target_screen_width=screen_width,
            target_screen_height=screen_height
        )
    
    return results_df

def generate_heatmap(df, output_path, screen_width, screen_height, sigma=20):
    df_filtered = df[df['confidence'] > 0.2].copy()
    if len(df_filtered) == 0:
        print("Warning: No high-confidence predictions for heatmap")
        df_filtered = df.copy()
    
    heatmap = np.zeros((screen_height, screen_width))
    for _, row in df_filtered.iterrows():
        x, y = int(row['screen_x']), int(row['screen_y'])
        if 0 <= x < screen_width and 0 <= y < screen_height:
            heatmap[y, x] += 1
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    plt.figure(figsize=(12, 9))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Normalized Density')
    plt.title('Gaze Heatmap')
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_sections(df, output_path, n_cols, n_rows, screen_width, screen_height):
    df_filtered = df[df['confidence'] > 0.2].copy()
    if len(df_filtered) == 0:
        print("Warning: No high-confidence predictions for section analysis")
        df_filtered = df.copy()
    
    grid = create_grid(screen_width, screen_height, n_cols, n_rows)
    
    region_counts = df_filtered['region'].value_counts().sort_index()
    all_regions = pd.Series(0, index=range(1, n_cols*n_rows+1))
    region_counts = region_counts.add(all_regions, fill_value=0)
    
    total = region_counts.sum()
    if total > 0:
        region_percentages = (region_counts / total) * 100
    else:
        region_percentages = region_counts * 0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    region_counts.plot(kind='bar', ax=ax1)
    ax1.set_title('Gaze Count by Region')
    ax1.set_xlabel('Region')
    ax1.set_ylabel('Count')
    region_percentages.plot(kind='bar', ax=ax2)
    ax2.set_title('Gaze Percentage by Region')
    ax2.set_xlabel('Region')
    ax2.set_ylabel('Percentage (%)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def overlay_predictions_on_screenshots(
    results_df, 
    screenshot_folder, 
    screenshot_output_folder, 
    tolerance=0.5,  # in seconds
    detailed=False,
    target_screen_width=1024,
    target_screen_height=768
):
    if not os.path.exists(screenshot_output_folder):
        os.makedirs(screenshot_output_folder)
    
    screenshots = sorted(os.listdir(screenshot_folder))
    for i, screenshot_file in enumerate(screenshots):
        if detailed:
            print(f"Processing screenshot {i+1}/{len(screenshots)}: {screenshot_file}")
        screenshot_path = os.path.join(screenshot_folder, screenshot_file)
        try:
            screenshot_elapsed = extract_elapsed_from_filename(screenshot_file)
        except ValueError as e:
            print(e)
            continue
        
        time_diff = np.abs(results_df['elapsed'] - screenshot_elapsed)
        matching_indices = time_diff[time_diff <= tolerance].index
        
        if len(matching_indices) == 0:
            if detailed:
                print(f"No matching gaze data for screenshot {screenshot_file} (elapsed {screenshot_elapsed})")
            continue
        
        img = cv2.imread(screenshot_path)
        if img is None:
            print(f"Could not load screenshot {screenshot_file}")
            continue
        
        img_height, img_width = img.shape[:2]
        # Use provided target screen dimensions for scaling
        scale_x = img_width / target_screen_width
        scale_y = img_height / target_screen_height
        
        for idx in matching_indices:
            row = results_df.loc[idx]
            if np.isnan(row['screen_x']) or np.isnan(row['screen_y']):
                continue
            x = int(row['screen_x'] * scale_x)
            y = int(row['screen_y'] * scale_y)
            if 0 <= x < img_width and 0 <= y < img_height:
                confidence = row['confidence']
                color = (0, int(255 * confidence), int(255 * (1 - confidence)))
                cv2.circle(img, (x, y), 10, color, -1)
                cv2.circle(img, (x, y), 10, (0, 0, 0), 2)
        
        output_file = os.path.join(screenshot_output_folder, os.path.basename(screenshot_file))
        cv2.imwrite(output_file, img)
        if detailed:
            print(f"Annotated screenshot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze gaze data and overlay predictions on screenshots.")
    parser.add_argument("--landmarks", required=True, help="Path to the landmarks CSV file")
    parser.add_argument("--coarse_model", required=True, help="Path to the coarse model file")
    parser.add_argument("--fine_model_dir", required=True, help="Directory containing fine-tuned models")
    parser.add_argument("--output_csv", help="Path to save the results CSV")
    parser.add_argument("--screen_width", type=int, default=1024, help="Screen width in pixels")
    parser.add_argument("--screen_height", type=int, default=768, help="Screen height in pixels")
    parser.add_argument("--device", default="cpu", help="Device to run the models on (cpu or cuda)")
    parser.add_argument("--detailed", action="store_true", help="Enable detailed logging")
    parser.add_argument("--screenshot_folder", help="Folder containing screenshots")
    parser.add_argument("--screenshot_output", help="Folder to save annotated screenshots")
    parser.add_argument("--screenshot_tolerance", type=float, default=0.5, help="Time tolerance (in seconds) for matching screenshots")
    
    args = parser.parse_args()
    
    analyze_gaze(
        args.landmarks,
        args.coarse_model,
        args.fine_model_dir,
        output_csv=args.output_csv,
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        device=args.device,
        detailed=args.detailed,
        screenshot_folder=args.screenshot_folder,
        screenshot_output_folder=args.screenshot_output,
        screenshot_tolerance=args.screenshot_tolerance
    )

if __name__ == "__main__":
    main()
