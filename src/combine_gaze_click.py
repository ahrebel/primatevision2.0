import argparse
import os
import pandas as pd
import numpy as np

def parse_time_column(df, time_col='time'):
    if df[time_col].dtype.kind in ('i','f'):
        df['time_sec'] = df[time_col].astype(float)
    else:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        if df[time_col].isnull().any():
            raise ValueError(f"Could not parse some {time_col} as datetimes.")
        earliest = df[time_col].min()
        df['time_sec'] = (df[time_col] - earliest).dt.total_seconds()
    return df

def load_click_data(click_path):
    df = pd.read_csv(click_path)

    if 'time' not in df.columns:
        if 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp': 'time'})
            print(f"Renamed 'timestamp' column to 'time' in {click_path}")
        elif 'elapsed_seconds' in df.columns:
            df = df.rename(columns={'elapsed_seconds': 'time'})
            print(f"Renamed 'elapsed_seconds' column to 'time' in {click_path}")
        else:
            raise ValueError(f"No time column found in {click_path}. Need 'time', 'timestamp', or 'elapsed_seconds'.")

    rename_map = {}
    if 'x' in df.columns:         rename_map['x'] = 'screen_x'
    if 'y' in df.columns:         rename_map['y'] = 'screen_y'
    if 'clickX' in df.columns:    rename_map['clickX'] = 'screen_x'
    if 'clickY' in df.columns:    rename_map['clickY'] = 'screen_y'

    if rename_map:
        df = df.rename(columns=rename_map)

    required_cols = ['time','screen_x','screen_y']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' missing in {click_path}.")

    df = parse_time_column(df, time_col='time')
    return df

def combine_gaze_click(gaze_csv, click_file, output_csv, max_time_diff=0.2, gaze_offset=0.1):
    gaze_df = pd.read_csv(gaze_csv)
    
    if 'time' not in gaze_df.columns:
        if 'datetime' in gaze_df.columns:
            gaze_df = gaze_df.rename(columns={'datetime': 'time'})
            print(f"Renamed 'datetime' column to 'time' in {gaze_csv}")
        elif 'elapsed' in gaze_df.columns:
            gaze_df = gaze_df.rename(columns={'elapsed': 'time'})
            print(f"Renamed 'elapsed' column to 'time' in {gaze_csv}")
        else:
            raise ValueError(f"No time column found in {gaze_csv}. Need 'time', 'datetime', or 'elapsed'.")
    
    needed = [
        'time',
        'corner_left_x','corner_left_y',
        'corner_right_x','corner_right_y',
        'left_pupil_x','left_pupil_y',
        'right_pupil_x','right_pupil_y',
        'nose_x','nose_y',
        'dist_nose_left_pupil','dist_nose_right_pupil',
        'roll_angle_approx'
    ]
    for col in needed:
        if col not in gaze_df.columns:
            raise ValueError(f"Column '{col}' missing from {gaze_csv}. Check your process_video step.")

    gaze_df = parse_time_column(gaze_df, time_col='time')
    click_df = load_click_data(click_file)

    combined = []
    for _, c_row in click_df.iterrows():
        c_time = c_row['time_sec']
        candidates = gaze_df[
            (gaze_df['time_sec'] >= c_time + gaze_offset) &
            (gaze_df['time_sec'] <= c_time + max_time_diff)
        ]
        if candidates.empty:
            continue
        idx = candidates.index[0] 
        best = gaze_df.loc[idx]
        row_out = {
            'time': best['time'],
            'corner_left_x': best['corner_left_x'],
            'corner_left_y': best['corner_left_y'],
            'corner_right_x': best['corner_right_x'],
            'corner_right_y': best['corner_right_y'],
            'left_pupil_x': best['left_pupil_x'],
            'left_pupil_y': best['left_pupil_y'],
            'right_pupil_x': best['right_pupil_x'],
            'right_pupil_y': best['right_pupil_y'],
            'nose_x': best['nose_x'],
            'nose_y': best['nose_y'],
            'dist_nose_left_pupil': best['dist_nose_left_pupil'],
            'dist_nose_right_pupil': best['dist_nose_right_pupil'],
            'roll_angle_approx': best['roll_angle_approx'],
            'screen_x': c_row['screen_x'],
            'screen_y': c_row['screen_y']
        }
        combined.append(row_out)

    if not combined:
        print("No matches found. Check time columns or max_time_diff.")
        return

    out_df = pd.DataFrame(combined)

    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        try:
            existing = pd.read_csv(output_csv)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
        merged = pd.concat([existing, out_df], ignore_index=True)
        merged.drop_duplicates(inplace=True)
        merged.to_csv(output_csv, index=False)
        print(f"Appended to '{output_csv}' (duplicates removed).")
    else:
        out_df.drop_duplicates(inplace=True)
        out_df.to_csv(output_csv, index=False)
        print(f"Created new calibration CSV '{output_csv}'.")

def main():
    parser = argparse.ArgumentParser(
        description="Merge gaze CSV + click data with a time tolerance. Default max_time_diff=0.2s, gaze_offset=0.1s"
    )
    parser.add_argument("--gaze_csv", required=True)
    parser.add_argument("--click_file", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--max_time_diff", type=float, default=0.2)
    parser.add_argument("--gaze_offset", type=float, default=0.1,
                      help="Time offset in seconds to look ahead for gaze data after click")
    args = parser.parse_args()

    combine_gaze_click(args.gaze_csv, args.click_file, args.output_csv, 
                      args.max_time_diff, args.gaze_offset)

if __name__ == "__main__":
    main()
