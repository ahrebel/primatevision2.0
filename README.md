# PrimateVision

**PrimateVision** is an advanced **eye-tracking system** designed for Rhesus macaques (and humans) that combines **DeepLabCut (DLC)** for multi-landmark detection with a **Two-Stage Neural Network** to map raw eye coordinates onto screen coordinates. It supports offline processing of pre-recorded videos, producing detailed **heatmaps** and CSV summaries of gaze/fixation data.

![Animated Eyes](https://github.com/user-attachments/assets/0f245b14-ec20-4a11-868a-ae207a7dfa1d)

---

## Table of Contents

1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Prerequisites & Installation](#prerequisites--installation)  
4. [Data Preparation](#data-preparation)  
5. [DeepLabCut Model Training](#deeplabcut-model-training)  
6. [Workflow & Usage](#workflow--usage)  
7. [Calibration & Processing](#calibration--processing)  
8. [Neural Network Architecture](#neural-network-architecture)
9. [Gaze Analysis & Visualization](#gaze-analysis--visualization)  
10. [PrimateVision GUI](#primatevision-gui)  
11. [Troubleshooting](#troubleshooting)  
12. [Contributing](#contributing)  
13. [Future Improvements](#future-improvements)  
14. [License](#license)  
15. [Security Disclaimer](#security-disclaimer)

---

## 1. Overview

PrimateVision simplifies the **eye-tracking** workflow by:

- **Detecting eye and nose landmarks** (e.g., pupil corners, nose tip) via DeepLabCut.  
- **Mapping** these landmarks to screen coordinates using a **Two-Stage Neural Network** model that separates head pose encoding from pupil-based gaze prediction.
- **Analyzing** gaze/fixation patterns for touchscreen-based tasks with advanced visualization tools.

You can integrate your own DLC-trained model (with corners, pupils, and nose labeled), calibrate the neural network with gaze → screen data, and run a single offline pass to process all experimental videos. The system supports CPU-only usage on macOS and Windows, with optional GPU acceleration for DeepLabCut.

---

## 2. Key Features

- **Offline Video Processing**  
  - Single pass on each video to extract eye + nose landmarks and produce a final CSV of gaze coordinates.
  - Automatic computation of derived features like distances between landmarks and roll angle.

- **DeepLabCut Integration**  
  - Label, train, and evaluate your DLC model, including the **`nose`** body part for better head orientation coverage.
  - Seamless processing pipeline from raw video to landmark detection.

- **Two-Stage Neural Network Architecture**  
  - **Head Pose Encoder**: Extracts head orientation features from corner and nose positions.  
  - **Pupil Decoder**: Combines head pose embedding with pupil positions to predict screen coordinates.  
  - Handles complex head movements and gaze patterns better than traditional methods.

- **Anti-Diagonal Correction**  
  - Automatically detects and corrects strong diagonal correlations in predictions.  
  - Creates more natural, distributed gaze patterns that better reflect actual viewing behavior.

- **Advanced Visualization**  
  - Generate heatmaps showing time spent in each screen region with customizable grid resolution.  
  - Produce CSV summaries of fixation durations for further analysis.  
  - (Optional) Plot raw gaze points and coordinate distributions for deeper insight.

- **Flexible Output Options**  
  - Control which visualizations and data files are generated via command-line flags.  
  - Save intermediate results for debugging or further analysis.

- **Cross-Platform**  
  - Works on macOS and Windows (CPU-only by default; GPU optional for DLC).

- **No Y-Inversion Required**  
  - If your physical screen and your data both increase in y-value downward, no inversion is needed.

---

## 3. Prerequisites & Installation

### Prerequisites

- **Python 3.8** (or later)
- **Conda or pip** for environment management
- **DeepLabCut** dependencies (TensorFlow, etc.) if training or re-configuring DLC
- **PyTorch** for the neural network model
- **OpenCV** for video and image I/O
- **PyQt5** (if you want to use the optional GUI)

### Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/ahrebel/primatevision.git
   cd primatevision
   ```

2. **Set Up Your Environment**  
   **Conda (recommended)**:
   ```bash
   conda create -n primatevision -c conda-forge python=3.8 \
       pytables hdf5 lzo opencv numpy pandas matplotlib \
       scipy tqdm statsmodels pytorch torchvision
   conda activate primatevision
   ```
   **pip + virtualenv** (alternative):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install Required Packages**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Install DeepLabCut**  
   ```bash
   pip install deeplabcut pyyaml tensorflow tensorpack tf-slim 'deeplabcut[gui]'
   ```
   > *If you see `ModuleNotFoundError: No module named 'keras.legacy_tf_layers'`, try:*
   > ```bash
   > pip install --upgrade tensorflow_macos==2.12.0
   > ```

5. **Install PyTorch** (if not included in requirements.txt)
   ```bash
   pip install torch torchvision
   ```

6. **(Optional) Install PyQt5**  
   ```bash
   pip install pyqt5
   ```
   This is required only if you want to use the PrimateVision GUI.

---

## 4. Data Preparation

### Video Requirements

- **Format**: MP4 or AVI (MP4 recommended)
- **Resolution**: Sufficient to clearly see eye features (720p or higher recommended)
- **Framerate**: Consistent throughout the video (30fps typical)
- **Lighting**: Consistent, avoiding strong shadows or reflections on eyes
- **Head Position**: Varied positions during calibration to capture different angles

### Landmark Requirements

Your DeepLabCut model should be trained to detect these key landmarks:
- **left_pupil**: Center of the left eye pupil
- **right_pupil**: Center of the right eye pupil
- **corner_left**: Left corner of the eye (inner or outer depending on your setup)
- **corner_right**: Right corner of the eye (inner or outer depending on your setup)
- **nose**: Tip of the nose (crucial for head orientation)

### Calibration Data

For training the neural network, you'll need:
1. **Landmark Data**: CSV file with detected landmarks from a calibration video
2. **Screen Coordinates**: CSV file with known screen positions (from touches/clicks)
3. **Time Synchronization**: Both files must have a time column for alignment

---

## 5. DeepLabCut Model Training

1. **Launch the DLC GUI**  
   ```bash
   python -m deeplabcut
   ```

2. **Create a New Project**  
   - Provide a name (e.g., "PrimateGaze")
   - Add your calibration video(s)
   - Define keypoints: `left_pupil`, `right_pupil`, `corner_left`, `corner_right`, `nose`

3. **Extract & Label Frames**  
   - Extract frames that cover varied head poses and lighting conditions
   - Label all five landmarks in each frame
   - Ensure the nose tip is clearly labeled in each frame

4. **Train the Model**  
   ```bash
   python src/train_model.py
   ```
   Or use the DLC GUI's training interface.

   For CPU-only usage, consider a lightweight backbone like `mobilenet_v2_1.0`.

5. **Evaluate & Refine**  
   - Use DLC's evaluation tools to check accuracy
   - Add more labeled frames in areas where detection is poor
   - Re-train as needed

---

## 6. Workflow & Usage

PrimateVision's pipeline consists of these main steps:

1. **Landmark Detection**: Process videos with DeepLabCut to extract eye and nose landmarks  
2. **Calibration**: Combine landmark data with known screen positions  
3. **Model Training**: Train the Two-Stage Neural Network to map landmarks to screen coordinates  
4. **(Optional) Validate Model**: Check predictions on known data to ensure accuracy  
5. **Experimental Analysis**: Process new/experimental videos with the trained model  
6. **Visualization**: Generate heatmaps and other visualizations of gaze patterns  

### Typical Workflow

```
(1) DLC → (2) Landmarks CSV → (3) Calibration Merge → (4) Train 2-Stage Net
                                                          ↓
                             (5) New/Experimental Video → (Landmarks) → (6) Gaze Analysis → Heatmaps, Summaries
```

---

## 7. Calibration & Processing

### 7.1 Process Video for Landmark Detection

```bash
python src/process_video.py \
  --video /path/to/video.mp4 \
  --config /path/to/dlc_config.yaml \
  --output landmarks_output.csv \
  --smooth_window 3 \
  --labeled_frame_output sample_frame.jpg
```

This command:
1. Processes the video using your trained DeepLabCut model  
2. Extracts landmarks for each frame (pupils, corners, nose)  
3. Computes derived features (distances, angles)  
4. Applies temporal smoothing if requested  
5. Saves a sample frame with landmarks visualized  

**Key Parameters:**
- `--video`: Path to the input video file
- `--config`: Path to your DeepLabCut `config.yaml`
- `--output`: Where to save the CSV with detected landmarks
- `--smooth_window`: Apply rolling median smoothing
- `--labeled_frame_output`: Save a sample frame with landmarks drawn
- `--skip_frames`: (Optional) Process every other frame for speed
- `--resize_factor`: (Optional) Scale factor for frame size (e.g., 0.5)

### 7.2 Combine with Calibration Data

```bash
python src/combine_gaze_click.py \
  --gaze_csv landmarks_output.csv \
  --click_file calibration_touches.csv \
  --output_csv training_data.csv \
  --max_time_diff 0.2
```

This command:
1. Loads the landmarks CSV from the processed calibration video
2. Loads the calibration data with known screen positions
3. Matches landmarks to screen positions based on timestamps
4. Creates a combined dataset for training the neural network

**Key Parameters:**
- `--gaze_csv`: CSV file with landmark data
- `--click_file`: CSV file with calibration touches/clicks
- `--output_csv`: Where to save the combined training data
- `--max_time_diff`: Max time difference (in seconds) for matching

---

## 8. Neural Network Architecture

PrimateVision uses a Two-Stage Neural Network architecture that separates head pose encoding from gaze prediction:

### Head Pose Encoder

The first stage processes the corner and nose positions to create a head pose embedding.

### Pupil Decoder

The second stage combines the head pose embedding with pupil positions to predict screen coordinates:

---

## 9. Gaze Analysis & Visualization

### 9.1 Train the Two-Stage Neural Network

```bash
python src/train_two_stage_net.py \
  --data training_data.csv \
  --output data/trained_model/two_stage_net.pt \
  --lr_candidates 0.001 0.0005 0.0001 \
  --max_epochs 50000 \
  --patience 1000 \
  --batch_size 32 \
  --device cpu
```

1. Loads the combined calibration data  
2. Tries different learning rates to find the optimal one  
3. Uses early stopping to prevent overfitting  
4. Performs a final refinement on the full dataset with the best learning rate  
5. Saves the trained model

---

### 9.2 Process New Experimental Videos

**Once your Two-Stage Net is trained, you can process additional (new) videos to predict gaze.**  
First, extract landmarks from the new video:

```bash
python src/process_video.py \
  --video /path/to/new_experimental_video.mp4 \
  --config /path/to/dlc_config.yaml \
  --output new_landmarks_output.csv
```

This gives you a CSV of landmarks for each frame in the new video.

---

### 9.3 Analyze Gaze with the Trained Model

Finally, run the two-stage network inference on your new landmarks CSV to generate gaze predictions, heatmaps, and summary CSVs:

```bash
python src/analyze_gaze_2stage.py \
  --landmarks_csv new_landmarks_output.csv \
  --model data/trained_model/two_stage_net.pt \
  --screen_width 1080 \
  --screen_height 728 \
  --n_cols 9 \
  --n_rows 9 \
  --output_heatmap gaze_heatmap.png \
  --output_sections section_durations.csv
```

**Parameters:**
- `--landmarks_csv`: CSV file with landmark data from the new experimental video
- `--model`: Path to the trained Two-Stage Net model
- `--screen_width/--screen_height`: Dimensions of your screen in pixels
- `--n_cols/--n_rows`: Grid resolution for the heatmap
- `--output_heatmap`: Where to save the heatmap visualization
- `--output_sections`: CSV summary of time spent in each grid region

**Optional Flags**:
- `--save_raw_plot`: Save a scatter plot of raw gaze points
- `--save_debug_csv`: Output a debug CSV with predicted `(screen_x, screen_y)` per frame
- `--force_range`: Apply anti-diagonal correction if needed

---

## 10. PrimateVision GUI

For a **user-friendly** approach, we provide a **PyQt5 desktop GUI** that combines all major steps:

1. **DLC GUI**: Launch official DLC for labeling/training.  
2. **Process Video**: Single-call DLC approach for each video.  
3. **Calibration Merge**: Combine gaze CSV + touchscreen data → calibration CSV.  
4. **Train Neural Network**: Train the Two-Stage Neural Network model.  
5. **Analyze Gaze**: Load the final landmarks + model, produce a heatmap + CSV of fixation durations.

**How to Use the GUI**:

1. **Install PyQt5** (if not done already):  
   ```bash
   pip install pyqt5
   ```
2. **Run**:  
   ```bash
   python primatevision_app.py
   ```
3. **Tabs**:
   - **DLC GUI**: Opens `deeplabcut` in a separate window for labeling frames.  
   - **Process Video**: Takes a video + DLC config, outputs a CSV (plus optional labeled frame).  
   - **Calibration Merge**: Combines the gaze CSV with touchscreen clicks → calibration CSV.  
   - **Train Neural Network**: Trains the Two-Stage Neural Network model.  
   - **Analyze Gaze**: Loads the final landmarks + model, produces a heatmap + CSV of fixation durations.

---

## 11. Troubleshooting

### Landmark Detection Issues

1. **Nose Landmark Not Found**  
   - **Problem**: The nose landmark is missing in DLC output.  
   - **Solution**: Ensure you labeled the nose in your DLC training data and that it's included in your `config.yaml`.  

2. **Poor Landmark Detection**
   - **Problem**: Landmarks are inaccurate or jittery.  
   - **Solution**:  
     - Increase the number of labeled frames in DLC training  
     - Use `--smooth_window` to apply temporal smoothing  
     - Check lighting conditions in your video  

### Neural Network Issues

1. **Diagonal Gaze Pattern**
   - **Problem**: Gaze predictions form a diagonal line.  
   - **Solution**: Use `--force_range` for anti-diagonal correction; ensure calibration covers full screen.  

2. **Limited Screen Coverage**
   - **Problem**: Predictions only cover a small portion of the screen.  
   - **Solution**:  
     - Ensure calibration data spans the entire screen  
     - Increase `n_cols` and `n_rows` if you want finer resolution  

### Visualization Issues

1. **Empty Heatmap**
   - **Problem**: Heatmap shows no or minimal activity.  
   - **Solution**:  
     - Check predictions are within screen bounds  
     - Use `--save_raw_plot` to see actual gaze points  

2. **Heatmap Inverted**
   - **Problem**: Heatmap appears upside-down.  
   - **Solution**: If your coordinate system is reversed, consider flipping Y in your data.  

### General Issues

1. **Memory Errors**
   - **Solution**:  
     - Use `--skip_frames` or `--resize_factor`  
     - Split longer videos into segments  

2. **GUI Crashes**
   - **Solution**:  
     - Verify PyQt5 installation  
     - Check Python version compatibility (3.8+ recommended)  

---

## 12. Contributing

We welcome contributions and suggestions! Please see:

- [CONTRIBUTING.md](CONTRIBUTING.md) for pull requests, code style, and how to report issues.  
- [Issues](https://github.com/ahrebel/primatevision/issues) to submit bug reports or feature requests.

---

## 13. Future Improvements

- **Neural Network Enhancements**
  - Implement attention mechanisms for better feature extraction
  - Add transformer-based architectures for sequence modeling
  - Support real-time prediction with webcam input
  - Explore multi-task learning (gaze + blink detection)

- **Adaptive Calibration**
  - Implement incremental learning for continuous model improvement
  - Develop auto-calibration techniques using visual saliency

- **Advanced Visualization**
  - Add 3D gaze visualization options
  - Implement scanpath analysis tools
  - Create interactive dashboards for data exploration    **Coming soon!**

- **Performance Optimization**
  - ONNX model export for faster inference
  - Batch processing for multiple videos
  - GPU acceleration for neural network inference

- **Integration & Packaging**
  - Publish as a Python package for easy installation

- **GUI Enhancements**
  - Real-time processing and visualization
  - Progress bars and detailed logging
  - Integrated video player with gaze overlay

---

## 14. License

This project is licensed under the [GPL-3.0 License](LICENSE).  
Refer to the license file for usage and distribution details.

---

## 15. Security Disclaimer

PrimateVision is designed for local laboratory usage without exposure to public networks. We do not provide any security guarantees for networked or internet-facing deployments. Users are responsible for ensuring that sensitive data (e.g., videos, personal information) is protected and stored in compliance with their institution’s policies. We disclaim any liability for security breaches or data exposure resulting from use of this software. If you have security concerns, you are welcome to reach out via an issue, but we cannot guarantee a fix or provide any timely patches for security-related matters.

---

**Happy Tracking!** If you have questions, open an [issue](https://github.com/ahrebel/primatevision/issues) or create a pull request. We appreciate feedback and contributions to make PrimateVision even better.
