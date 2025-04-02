# PrimateVision

**PrimateVision** is a robust **eye-tracking system** designed primarily for Rhesus macaques (and compatible with humans). It integrates **DeepLabCut (DLC)** for landmark detection (e.g., pupil and nose points) with a **two-stage neural network** that maps these landmarks to on-screen gaze coordinates. It supports **offline** analysis of recorded videos, offering **heatmaps**, CSV summaries, and optional **GUI** functionality.

![Animated Eyes](https://github.com/user-attachments/assets/0f245b14-ec20-4a11-868a-ae207a7dfa1d)

---

## Overview

PrimateVision streamlines eye-tracking for touchscreen-based tasks by:

- **Detecting facial landmarks** (nose tip, pupil corners, etc.) via DeepLabCut.
- **Mapping** the detected coordinates to screen coordinates using a **two-stage** (coarse and fine) neural network.
- **Generating** visual analytics, including gaze heatmaps and fixation distributions.

---

## Key Features

- **Offline Landmark Extraction**  
  - Single-pass detection of pupil and nose landmarks from an input video.

- **Hierarchical Modeling**  
  - Coarse model to predict which quadrant (A, B, C, D) the gaze falls into.  
  - Fine models (A1, A2, etc.) for sub-squares within each quadrant.

- **Two-Stage Neural Network**  
  - **Head Pose Encoder**: Learns robust representation of head position/orientation.  
  - **Pupil Decoder**: Uses head pose + pupil positions to estimate precise gaze coordinates.

- **DeepLabCut Integration**  
  - Smoothly ties into DLC pipelines for landmark detection.

- **Visualization Tools**  
  - Automatic **gaze heatmaps**.  
  - CSV summaries of gaze coordinates for further analysis.

- **Cross-Platform & Lightweight**  
  - Mac and Windows friendly, CPU-compatible (no CUDA required unless desired).

---

## Prerequisites

1. **Python 3.8** or newer  
2. **Conda** (recommended) or **pip + virtualenv** for environment management  
3. **PyTorch** for neural network training  
4. **OpenCV**, **numpy**, **pandas**, **matplotlib**, etc.  
5. **DeepLabCut** (and associated TF dependencies) if you plan to retrain or reconfigure DLC  
6. (Optional) **PyQt5** if you plan to use the PrimateVision GUI

---

## Installation

Below are general guidelines for installing PrimateVision and its dependencies. Adjust versions as needed based on your system.

1. **Clone or Download**  
   ```bash
   git clone https://github.com/ahrebel/primatevision2.0.git
   cd primatevision2.0
   ```

2. **Create & Activate Environment**  
   **Using Conda** (recommended):
   ```bash
   conda create -n primatevision python=3.8 -c conda-forge
   conda activate primatevision
   ```
   Or, **using pip + virtualenv**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Requirements**  
   ```bash
   pip install -r requirements.txt
   ```
   Make sure `PyTorch`, `OpenCV`, `matplotlib`, and other libraries are included either in `requirements.txt` or installed via:
   ```bash
   pip install torch torchvision opencv-python pandas scikit-learn matplotlib
   ```

4. **Install DeepLabCut**  
   ```bash
   pip install deeplabcut pyyaml tensorflow tensorpack tf-slim 'deeplabcut[gui]'
   ```
   > Some systems (e.g., Apple Silicon) may need:  
   > `pip install --upgrade tensorflow_macos==2.12.0`

5. **(Optional) PyQt5 for GUI**  
   ```bash
   pip install pyqt5
   ```

---

## Data Preparation

1. **Video Requirements**  
   - MP4 format, at least 720p resolution.  
   - Consistent lighting/conditions for accurate DLC detection.

2. **DeepLabCut Landmarks**  
   - The system expects the following points:  
     - `left_pupil`, `right_pupil` (pupil center or corners)  
     - `corner_left`, `corner_right` (eye corners)  
     - `nose` (nose tip)  

3. **Calibration Data**  
   - A CSV containing known screen touches or calibrations.  
   - Example columns: `elapsed_time`, `screen_x`, `screen_y`, plus any labels as needed.

---

## Workflow & Usage

### 1. **Process Video with DeepLabCut**  
Use `process_video.py` to run DLC on a video and output a CSV of tracked landmarks:

```bash
python process_video.py \
  --video path/to/video.mp4 \
  --config path/to/dlc_config.yaml \
  --output landmarks_output.csv
```

- **Arguments**:
  - `--video`: The input video file.
  - `--config`: DeepLabCut config YAML.
  - `--output`: The resulting CSV of landmarks.

### 2. **Combine Gaze & Calibration Data**  
If you have a separate file with **known calibration touches** (e.g., where the subject physically touched the screen), you can merge it with your DLC output using `combine_gaze_click.py`:

```bash
python combine_gaze_click.py \
  --gaze_csv landmarks_output.csv \
  --click_file calibration_touches.csv \
  --output_csv training_data.csv \
  --max_time_diff 0.2
```
- Merges each gaze frame with the closest screen touch event within `0.2` seconds, populating `screen_x`, `screen_y`.

---

## Neural Network Training

### Overview

PrimateVision uses a **hierarchical** approach:
1. **Coarse Model**: Predict quadrant (A, B, C, D).  
2. **Fine Models**: For each quadrant, a sub-model refines the coordinates.

### 1. **Train the Coarse Model**  
```bash
python hierarchical_gaze.py coarse_train \
  --data training_data.csv \
  --output models/coarse_model.pt \
  --batch_size 32 \
  --epochs 5000 \
  --patience 1000 \
  --device cpu
```

### 2. **Train the Fine Models**  
You can either train each quadrant individually:
```bash
python hierarchical_gaze.py fine_train \
  --data training_data.csv \
  --coarse_label A \
  --output models/fine_model_A.pt \
  --epochs 3000 \
  --device cpu
```
...or train **all** quadrants at once:
```bash
python hierarchical_gaze.py fine_train \
  --data training_data.csv \
  --coarse_label all \
  --output models/fine_tuned \
  --epochs 3000 \
  --device cpu
```
This will create multiple `.pt` files inside `models/fine_tuned/` (e.g., `fine_model_A.pt`, `fine_model_B.pt`, etc.).

---

## Gaze Analysis & Visualization

Once you have a trained coarse model (and optionally fine models), run `analyze_gaze.py` to:

1. **Infer** gaze coordinates (coarse + fine).  
2. **Generate** a heatmap or region-based CSV summary.

Example:

```bash
python analyze_gaze.py \
  --landmarks landmarks_output.csv \
  --coarse_model models/coarse_model.pt \
  --fine_model_dir models/fine_tuned \
  --output_csv results/gaze_predictions.csv \
  --output_heatmap results/gaze_heatmap.png \
  --screen_width 1024 \
  --screen_height 768 \
  --device cpu \
  --detailed
```

- **Arguments**:
  - `--landmarks`: The CSV with DLC landmarks (e.g., `landmarks_output.csv`).
  - `--coarse_model`: Path to your coarse model file (`.pt`).
  - `--fine_model_dir`: Directory with `fine_model_*.pt` files.
  - `--output_csv`: Where to store the final gaze predictions.
  - `--output_heatmap`: (Optional) Output image path for a heatmap.
  - `--screen_width`, `--screen_height`: Screen dimensions.
  - `--detailed`: Enables extra logging.

---

## (Optional) PrimateVision GUI

A PyQt5-based GUI (`primatevision_app.py`) can guide you through:

- Video selection & DLC configuration
- Model training & calibration merging
- On-screen visualization

```bash
pip install pyqt5
python primatevision_app.py
```

---

## Troubleshooting

- **“IsADirectoryError” when saving a `.pt` file**  
  You may have created a **directory** named `fine_model_A.pt` instead of a file. Delete or rename that folder, and ensure that only its **parent directory** is created with `os.makedirs()`.

- **NaN or unstable training**  
  - Check that your calibration data has correct `screen_x`, `screen_y`.
  - Ensure no extreme outliers in landmarks.
  - Use more training samples or better coverage of positions.

- **DLC KeyError or Missing Column**  
  - Confirm your DLC config includes the correct body parts: `nose, left_pupil, right_pupil, corner_left, corner_right`.
  - Re-run `process_video.py` after adjusting DLC if needed.

- **Inconsistent Gaze**  
  - Provide more calibration points across different screen regions.
  - Increase training epochs or reduce `--patience` for each model.

---

## Contributing

We welcome contributions via GitHub issues and pull requests. Whether reporting a bug or proposing a new feature, please submit a clear description and relevant examples.

---

## License

This project is licensed under the terms of the **GPL-3.0** license.

---

## Acknowledgements

- **DeepLabCut** – https://github.com/DeepLabCut/DeepLabCut  
- **PyTorch** – https://pytorch.org/  
- **OpenCV** – https://opencv.org/  

**Thank you for using PrimateVision!** If you have any questions or feedback, please open an issue on our [GitHub page](https://github.com/ahrebel/primatevision2.0).
