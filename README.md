# PrimateVision

**PrimateVision** is an advanced **eye-tracking system** designed for Rhesus macaques (and humans) that combines **DeepLabCut (DLC)** for multi-landmark detection with a **Two-Stage Neural Network** to map raw eye coordinates onto screen coordinates. It supports offline processing of pre-recorded videos, producing detailed **heatmaps** and CSV summaries of gaze/fixation data.

![Animated Eyes](https://github.com/user-attachments/assets/0f245b14-ec20-4a11-868a-ae207a7dfa1d)

---

## Overview

PrimateVision simplifies the **eye-tracking** workflow by:

- **Detecting eye and nose landmarks** (e.g., pupil corners, nose tip) via DeepLabCut.  
- **Mapping** these landmarks to screen coordinates using a hierarchical square-based **Two-Stage Neural Network** model that separates head pose encoding from pupil-based gaze prediction.
- **Analyzing** gaze/fixation patterns for touchscreen-based tasks with advanced visualization tools.

---

## Key Features

- **Offline Video Processing**  
  - Single pass to extract landmarks (pupils, corners, nose) and derived features like distances and angles.

- **Hierarchical Square-Based Training**
  - Initial coarse training (A, B, C, D squares).
  - Optional fine-tuning for finer sub-squares (AA, AB, AC, AD).

- **Two-Stage Neural Network Architecture**  
  - **Head Pose Encoder**: Extracts head orientation features.
  - **Pupil Decoder**: Combines head pose with pupil data for gaze prediction.

- **DeepLabCut Integration**
  - Seamless landmark detection.

- **Advanced Visualization**  
  - Customizable gaze heatmaps.
  - CSV summaries of fixation data.

- **Cross-Platform**  
  - macOS and Windows compatibility, CPU-friendly.

---

## Prerequisites & Installation

### Requirements
- Python 3.8+
- PyTorch, OpenCV, pandas, numpy, matplotlib
- DeepLabCut

### Setup

```bash
pip install -r requirements.txt
pip install deeplabcut[gui]
```

---

## Data Preparation

### Video Requirements
- Format: MP4, ≥ 720p resolution
- Consistent lighting, stable conditions

### DLC Landmarks
- `left_pupil`, `right_pupil`, `corner_left`, `corner_right`, `nose`

### Calibration Data
- Combined CSV file with:
  - Time stamps
  - Landmark positions
  - Screen coordinates (`screen_x`, `screen_y`)
  - Square labels (e.g., `A`, `B`, `AA`, `AB`)

---

## Workflow & Usage

### 1. Landmark Detection (DeepLabCut)

```bash
python process_video.py \
  --video path/to/video.mp4 \
  --config path/to/dlc_config.yaml \
  --output landmarks_output.csv
```

### 2. Calibration Merge

```bash
python combine_gaze_click.py \
  --gaze_csv landmarks_output.csv \
  --click_file calibration_touches.csv \
  --output_csv training_data.csv \
  --max_time_diff 0.2
```

---

# Neural Network Training


Run coarse training followed immediately by fine-tuning for all available squares in one command:

```bash
python hierarchical_gaze.py \
  --data training_data.csv \
  --output trained_model.pt \
  --fine_output fine_tuned_models \
  --fine_tune_all_squares \
  --device cpu \
  --batch_size 32 \
  --coarse_lr 0.001 \
  --fine_lr 0.0005 \
  --coarse_max_epochs 50000 \
  --fine_max_epochs 10000 \
  --coarse_patience 1000 \
  --fine_patience 500
```

---

## Gaze Analysis & Visualization

```bash
python analyze_gaze.py \
  --landmarks landmarks_output.csv \
  --model trained_model.pt \
  --screen_width 1024 \
  --screen_height 768 \
  --grid_size 13 \
  --output_heatmap gaze_heatmap.png \
  --output_csv gaze_summary.csv
```

---

## PrimateVision GUI (Optional)

- PyQt5-based interface for easy interaction with all components.

```bash
pip install pyqt5
python primatevision_app.py
```

---

## Troubleshooting

- **Landmark issues:** Retrain DLC model with additional frames.
- **Diagonal gaze patterns:** Use extensive calibration data or enable `--force_range`.

---

## Contributing

- Contributions are welcomed through GitHub issues and pull requests.

---

## Future Improvements

- Real-time gaze prediction
- Enhanced GUI functionality
- Adaptive and incremental learning capabilities

---

## License

GPL-3.0

---

## Security Disclaimer

Use within secure lab environments. No security guarantees for public deployments.

**Happy Eye-Tracking!**

---

## Acknowledgements

- DeepLabCut: https://github.com/DeepLabCut/DeepLabCut
- PyTorch: https://pytorch.org/
- OpenCV: https://opencv.org/