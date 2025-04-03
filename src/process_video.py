import cv2
import os
import argparse
import tempfile
import numpy as np
import pandas as pd
import deeplabcut
import time
from datetime import datetime, timedelta

def smooth_landmarks_inplace(df, window_size=3):
    columns = [
        "left_pupil_x", "left_pupil_y",
        "right_pupil_x", "right_pupil_y",
        "corner_left_x", "corner_left_y",
        "corner_right_x", "corner_right_y",
        "nose_x", "nose_y",
        "dist_nose_left_pupil", "dist_nose_right_pupil",
        "roll_angle_approx"
    ]
    for col in columns:
        if col in df.columns:
            df[col] = df[col].rolling(window=window_size, center=True, min_periods=1).median()
    return df

def compute_distances_and_roll(
    left_pupil_x, left_pupil_y,
    right_pupil_x, right_pupil_y,
    corner_left_x, corner_left_y,
    corner_right_x, corner_right_y,
    nose_x, nose_y
):
    dist_nose_left_pupil = np.nan
    dist_nose_right_pupil = np.nan
    roll_angle_approx = np.nan

    if not any(np.isnan([nose_x, nose_y, left_pupil_x, left_pupil_y])):
        dist_nose_left_pupil = np.hypot(nose_x - left_pupil_x, nose_y - left_pupil_y)
    if not any(np.isnan([nose_x, nose_y, right_pupil_x, right_pupil_y])):
        dist_nose_right_pupil = np.hypot(nose_x - right_pupil_x, nose_y - right_pupil_y)

    if not any(np.isnan([corner_left_x, corner_left_y, corner_right_x, corner_right_y])):
        dx = corner_right_x - corner_left_x
        dy = corner_right_y - corner_left_y
        roll_angle_approx = np.degrees(np.arctan2(dy, dx))

    return dist_nose_left_pupil, dist_nose_right_pupil, roll_angle_approx

def extract_video_start_from_filename(video_filename):
    base = os.path.basename(video_filename)
    parts = base.split('_')
    if len(parts) < 4:
        raise ValueError("Video filename does not contain a valid start datetime.")
    date_str = parts[1]
    time_str = parts[2]
    ms_part = parts[3].split('.')[0] 
    ms_str = ms_part.zfill(6)
    timestamp_str = f"{date_str}_{time_str}_{ms_str}"
    return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f")

def process_video(
    video_path,
    config_path,
    output_csv_path,
    skip_frames=False,
    resize_factor=1.0,
    smooth_window=0,
    labeled_frame_output=None,
    video_start_datetime=None
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    new_width = int(width * resize_factor)
    new_height = int(height * resize_factor)

    sample_frame = None
    sample_frame_idx = None
    frame_map = []

    read_start_time = time.time()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video_path = os.path.join(temp_dir, "combined_temp.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (new_width, new_height))

        global_frame_idx = 0
        processed_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if skip_frames:
                _ = cap.read() 
                global_frame_idx += 1

            if resize_factor != 1.0:
                frame = cv2.resize(frame, (new_width, new_height))

            out_writer.write(frame)

            if sample_frame is None:
                sample_frame = frame.copy()
                sample_frame_idx = processed_count

            frame_map.append((processed_count, global_frame_idx, global_frame_idx / fps))
            processed_count += 1
            global_frame_idx += 1

        out_writer.release()
        cap.release()
        read_elapsed = time.time() - read_start_time
        print(f"Finished writing {processed_count} frames to {temp_video_path} in {read_elapsed:.2f}s")

        print("Running DLC analyze_videos on combined_temp.mp4...")
        dlc_start = time.time()
        deeplabcut.analyze_videos(
            config_path, [temp_video_path],
            save_as_csv=True,
            destfolder=temp_dir,
            videotype=".mp4"
        )
        dlc_elapsed = time.time() - dlc_start
        print(f"DLC analysis done in {dlc_elapsed:.2f}s")

        csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
        if not csv_files:
            raise ValueError("No DLC CSV output found. Check your DLC config/training.")
        dlc_result_csv = os.path.join(temp_dir, csv_files[0])
        dlc_df = pd.read_csv(dlc_result_csv, header=[1, 2])

        if video_start_datetime is None:
            video_start = extract_video_start_from_filename(video_path)
            print(f"Extracted video start datetime from filename: {video_start}")
        else:
            video_start = datetime.strptime(video_start_datetime, '%Y-%m-%d %H:%M:%S')
        frame_rate = fps

        final_data = []
        for i in range(len(dlc_df)):
            proc_idx, orig_idx, orig_time = frame_map[i]
            elapsed = orig_idx / frame_rate 
            timestamp = video_start + timedelta(seconds=elapsed)

            left_pupil_x  = dlc_df[("left_pupil",  "x")].iloc[i]
            left_pupil_y  = dlc_df[("left_pupil",  "y")].iloc[i]
            right_pupil_x = dlc_df[("right_pupil", "x")].iloc[i]
            right_pupil_y = dlc_df[("right_pupil", "y")].iloc[i]
            corner_left_x  = dlc_df[("corner_left",  "x")].iloc[i]
            corner_left_y  = dlc_df[("corner_left",  "y")].iloc[i]
            corner_right_x = dlc_df[("corner_right", "x")].iloc[i]
            corner_right_y = dlc_df[("corner_right", "y")].iloc[i]
            nose_x         = dlc_df[("nose", "x")].iloc[i]
            nose_y         = dlc_df[("nose", "y")].iloc[i]

            dnlp, dnrp, roll_ang = compute_distances_and_roll(
                left_pupil_x, left_pupil_y,
                right_pupil_x, right_pupil_y,
                corner_left_x, corner_left_y,
                corner_right_x, corner_right_y,
                nose_x, nose_y
            )

            final_data.append({
                "frame": orig_idx,
                "datetime": timestamp,
                "elapsed": elapsed,
                "left_pupil_x": left_pupil_x,
                "left_pupil_y": left_pupil_y,
                "right_pupil_x": right_pupil_x,
                "right_pupil_y": right_pupil_y,
                "corner_left_x": corner_left_x,
                "corner_left_y": corner_left_y,
                "corner_right_x": corner_right_x,
                "corner_right_y": corner_right_y,
                "nose_x": nose_x,
                "nose_y": nose_y,
                "dist_nose_left_pupil": dnlp,
                "dist_nose_right_pupil": dnrp,
                "roll_angle_approx": roll_ang
            })

        df_out = pd.DataFrame(final_data)
        if smooth_window > 0:
            df_out = smooth_landmarks_inplace(df_out, window_size=smooth_window)

        if not os.path.exists(output_csv_path):
            print(f"Creating new file: {output_csv_path}")
        df_out.to_csv(output_csv_path, index=False)
        print(f"Final CSV saved to {output_csv_path}")

        if labeled_frame_output and sample_frame is not None:
            print(f"Drawing landmarks on sample frame index: {sample_frame_idx}")
            row = df_out[df_out["frame"] == sample_frame_idx]
            if not row.empty:
                row = row.iloc[0]
                landmarks = {
                    "left_pupil":   (row["left_pupil_x"],   row["left_pupil_y"]),
                    "right_pupil":  (row["right_pupil_x"],  row["right_pupil_y"]),
                    "corner_left":  (row["corner_left_x"],  row["corner_left_y"]),
                    "corner_right": (row["corner_right_x"], row["corner_right_y"]),
                    "nose":         (row["nose_x"],         row["nose_y"])
                }
                for name, (lx, ly) in landmarks.items():
                    if pd.notnull(lx) and pd.notnull(ly):
                        cv2.circle(sample_frame, (int(lx), int(ly)), 3, (0,255,0), -1)
                cv2.imwrite(labeled_frame_output, sample_frame)
                print(f"Labeled frame saved to {labeled_frame_output}")
            else:
                print("Could not find matching row for sample frame. No labeled image saved.")

    return df_out

def main():
    parser = argparse.ArgumentParser(
        description="Process a video for DLC in a single pass, capturing all frames and computing derived features."
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--config", required=True, help="DLC config.yaml path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--skip_frames", action="store_true", help="Skip next frame if set")
    parser.add_argument("--resize_factor", type=float, default=1.0, help="Scale frames")
    parser.add_argument("--smooth_window", type=int, default=0, help="Rolling median smoothing window")
    parser.add_argument("--labeled_frame_output", default=None, help="Optional labeled frame output path")
    parser.add_argument("--video_start_datetime", default=None, help="Optional video start date-time in 'YYYY-MM-DD HH:MM:SS' format")
    args = parser.parse_args()

    process_video(
        args.video,
        args.config,
        args.output,
        skip_frames=args.skip_frames,
        resize_factor=args.resize_factor,
        smooth_window=args.smooth_window,
        labeled_frame_output=args.labeled_frame_output,
        video_start_datetime=args.video_start_datetime
    )

if __name__ == "__main__":
    main()
