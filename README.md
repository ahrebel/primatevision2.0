# MAKO: Macaque Attention and Keypoint Observation

**MAKO** is an open-source, webcam-based eye-tracking system for Rhesus macaques and humans. It uses **DeepLabCut (DLC)** for multi-landmark detection and a **two-stage neural network** to estimate gaze location on a screen from facial keypoints. MAKO is designed for offline video analysis and produces interpretable heatmaps and structured CSV summaries of gaze behavior, offering an alternative to head-mounted tracking systems.

---

## Overview

MAKO streamlines the gaze estimation workflow with:

- **DeepLabCut-based Landmark Detection**  
  Detects key facial points including pupils, eye corners, and nose tip.

- **Hierarchical Gaze Estimation**  
  A two-stage model:
  - **Coarse model:** Classifies gaze into one of four quadrants (A, B, C, D)
  - **Optional fine model:** Further refines location within the selected quadrant (e.g., AA, AB, AC, AD)  
    Robust fallback logic ensures stable predictions when sub-regions are unavailable or uncertain.

- **Integrated Visualization**  
  Generates heatmaps and annotated screenshots with quadrant overlays and gaze target locations.

---

## Key Features

- **Offline Video Analysis**  
  Compatible with pre-recorded videos under standard lab lighting conditions.

- **Hierarchical Spatial Grid**  
  Modular square-based prediction enables interpretable coarse and fine resolution.

- **Two-Stage Neural Architecture**  
  Combines head pose and pupil geometry for refined predictions.

- **DeepLabCut Integration**  
  Works with custom or pretrained DLC models for flexible landmark tracking.

- **Cross-Platform**  
  Compatible with macOS and Windows; optimized for CPU usage.

- **Visual Output Tools**  
  Exportable heatmaps, quadrant maps, and screenshot overlays for data exploration.

- **No Head-Mounted Hardware Required**  

---

## Installation

### Prerequisites
- Python >= 3.8
- Conda (recommended) or virtualenv
- DeepLabCut dependencies (TensorFlow, tf-slim, etc.)
- PyTorch
- OpenCV

### Steps

1. **Clone the Repository**
```bash
git clone https://github.com/ahrebel/MAKO.git
cd MAKO
```

2. **Environment Setup**
**Conda (recommended):**
```bash
conda create -n mako -c conda-forge python=3.8 \
    pytables hdf5 lzo opencv numpy pandas matplotlib \
    scipy tqdm statsmodels pytorch torchvision
conda activate mako
```

**pip (alternative):**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install Required Packages**
```bash
pip install -r requirements.txt
```

4. **Install DeepLabCut**
```bash
pip install deeplabcut pyyaml tensorflow tensorpack tf-slim 'deeplabcut[gui]'
```
> *If you see `keras.legacy_tf_layers` error:*
```bash
pip install --upgrade tensorflow_macos==2.12.0
```

5. **Install PyTorch** (if not already installed)
```bash
pip install torch torchvision
```

---

## Data Preparation

### Video Requirements
- Format: MP4
- Resolution: ≥ 720p
- Lighting: Consistent and well-lit scenes

### Landmarks
Your DLC model should predict:
- `left_pupil`
- `right_pupil`
- `corner_left`
- `corner_right`
- `nose`

### Calibration CSV
A merged dataset containing:
- Timestamps
- Landmark coordinates
- Screen click coordinates (`screen_x`, `screen_y`)
- (Optional) region label (A/B/AA/etc.)

---

## Workflow & Commands

### 1. Landmark Detection (DeepLabCut)
```bash
python process_video.py \
  --video path/to/video.mp4 \
  --config path/to/dlc_config.yaml \
  --output landmarks_output.csv
```
Runs a pre-trained DeepLabCut model on a video and saves landmark predictions to a CSV file.

### 2. Merge Calibration Touches
```bash
python combine_gaze_click.py \
  --gaze_csv landmarks_output.csv \
  --click_file calibration_touches.csv \
  --output_csv training_data.csv \
  --max_time_diff 0.2
```
Synchronizes gaze data with calibration screen taps to prepare training data. The `max_time_diff` argument controls temporal alignment tolerance.

### 3. Train Coarse Model
```bash
python src/hierarchical_gaze_pro.py coarse_train \
  --data training_data.csv \
  --output data/trained_model/coarse_trained_model.pt \
  --batch_size 64 \
  --epochs 1000 \
  --patience 30 \
  --alpha 0.4 \
  --lr 1e-3 \
  --balanced_ce \
  --onecycle \
  --normalize \
  --kfold 5 \
  --device cpu
```
Trains a coarse classifier and regression network. Supports K-Fold validation, normalization, and class balancing. Model checkpoint saved at the specified `--output` path.

### 4. Train Fine Models (Optional)
```bash
python src/hierarchical_gaze_pro.py fine_train \
  --data training_data.csv \
  --coarse_label all \
  --output data/trained_model/fine_tuned_models \
  --batch_size 64 \
  --epochs 1000 \
  --patience 30 \
  --alpha 0.4 \
  --lr 1e-3 \
  --balanced_ce \
  --onecycle \
  --normalize \
  --kfold 5 \
  --device cpu
```
Trains quadrant-specific fine models (AA, AB, etc.) to refine predictions from the coarse stage. Each model is saved in the `--output` directory with filenames like `fine_model_A.pt`, etc.

### 5. Run Gaze Inference and Generate Outputs
```bash
python src/hierarchical_gaze_pro.py analyze \
  --landmarks landmarks_output.csv \
  --coarse_model data/trained_model/coarse_trained_model.pt \
  --screen_width 1024 \
  --screen_height 768 \
  --output_csv results.csv \
  --device cpu \
  --screenshot_folder videos/input/Screenshots \
  --screenshot_output_folder labeled_screenshots \
  --detailed \
  --normalize
```
Predicts gaze location frame-by-frame and creates:
- Annotated screenshots (`--screenshot_output_folder`)
- A results CSV file (`--output_csv`)
- Gaze heatmaps and optional quadrant breakdowns

### 6. Evaluate Model Performance (Experimental)
```bash
python src/hierarchical_gaze_pro.py evaluate \
  --data_csv labeled_data.csv \
  --coarse_model data/trained_model/coarse_trained_model.pt \
  --fine_model_dir data/trained_model/fine_tuned_models \
  --device cpu \
  --confusion
```
Runs performance evaluation, returning metrics like confusion matrix accuracy. Still under active development.

---

## Performance Benchmarks *(in progress)*

For 60 Minutes (1 Hour) of Training Videos at 30fps:
| Metric               | Value              |
|----------------------|--------------------|
| Mean Gaze Error      | *477.54 px*        |
| Accuracy (Coarse)    | *25%*              |
| Accuracy (Fine)      | *0%* (not used)    |
| CPU Inference Time   | *0.34 ms/frame*    |

For 120 Minutes (2 Hours) of Training Videos at 30fps:      (COMING SOON!)
| Metric               | Value              |
|----------------------|--------------------|
| Mean Gaze Error      | *TBD px*           |
| Accuracy (Coarse)    | *TBD %*            |
| Accuracy (Fine)      | *TBD %*            |
| CPU Inference Time   | *TBD ms/frame*     |

For 240 Minutes (4 Hours) of Training Videos at 30fps:      (COMING SOON!)
| Metric               | Value              |
|----------------------|--------------------|
| Mean Gaze Error      | *TBD px*           |
| Accuracy (Coarse)    | *TBD %*            |
| Accuracy (Fine)      | *TBD %*            |
| CPU Inference Time   | *TBD ms/frame*     |

For 360 Minutes (6 Hours) of Training Videos at 30fps:      (COMING SOON!)
| Metric               | Value              |
|----------------------|--------------------|
| Mean Gaze Error      | *TBD px*           |
| Accuracy (Coarse)    | *TBD %*            |
| Accuracy (Fine)      | *TBD %*            |
| CPU Inference Time   | *TBD ms/frame*     |

> Benchmarks will be updated based on internal evaluations. You can contribute your own test results via pull request. Please note that your results may vary based on training data quality, FPS, and other factors. 

---

## Security Disclaimer

MAKO is intended for **controlled research environments**. It does not include sandboxing, authentication, or user access control features. Avoid using MAKO in unsecured networks or public-facing systems.

- Input videos and results are processed/stored locally.
- Users are responsible for complying with their institutions' ethical and data handling policies.

---

## Contributing

Contributions are welcome via issues and pull requests. See the [CONTRIBUTING.md](CONTRIBUTING.md) guide for suggestions.

---

## Citation (In Progress)

If you use MAKO in published work, please cite:

> Rebello, A. (2025). *MAKO: Vision-based gaze tracking for non-human primates.* [Manuscript in preparation]

---

## License

MAKO is released under the [GPL-3.0 License](https://www.gnu.org/licenses/gpl-3.0.en.html).

---

## Acknowledgements

- **DeepLabCut:** [github.com/DeepLabCut/DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)
- **PyTorch:** [pytorch.org](https://pytorch.org/)
- **OpenCV:** [opencv.org](https://opencv.org/)

---

<pre>
███╗   ███╗  █████╗  ██╗  ██╗  ██████╗ 
████╗ ████║ ██╔══██╗ ██║ ██╔╝ ██╔═══██╗
██╔████╔██║ ███████║ █████╔╝  ██║   ██║
██║╚██╔╝██║ ██╔══██║ ██╔═██╗  ██║   ██║
██║ ╚═╝ ██║ ██║  ██ ║██║  ██╗ ╚██████╔╝
╚═╝     ╚═╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝  ╚═════╝
</pre>

