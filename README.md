# MAKO

**MAKO** is an advanced **eye-tracking system** designed for Rhesus macaques (and humans) that combines **DeepLabCut (DLC)** for multi-landmark detection with a **Two-Stage Neural Network** to map raw eye coordinates onto screen coordinates. It supports offline processing of pre-recorded videos, producing detailed **heatmaps** and CSV summaries of gaze/fixation data.

![Animated Eyes](https://github.com/user-attachments/assets/0f245b14-ec20-4a11-868a-ae207a7dfa1d)

---

## Overview

MAKO simplifies the **eye-tracking** workflow by:

- **Detecting Eye and Nose Landmarks:**  
  Leverages DeepLabCut to detect multiple key landmarks (e.g., pupil corners, nose tip) that serve as inputs for gaze estimation.

- **Hierarchical Gaze Mapping:**  
  Uses a two-stage neural network with a hierarchical square-based approach:
  - **Coarse Prediction:** Maps landmarks into one of four primary quadrants (A, B, C, D).  
  - **Fine Refinement (Optional):** If available, a corresponding fine model further refines the prediction within sub-squares (AA, AB, AC, AD).  
    Enhanced error handling is implemented so that if the refined prediction is out-of-bound, the system reverts to the coarse prediction or a geometric fallback.

- **Advanced Analysis and Visualization:**  
  Generates detailed heatmaps and CSV summaries of gaze/fixation data, and visualizes predictions by displaying the predicted viewing spot (green circle) and the quadrant border (blue rectangle) on screenshots.

---

## Key Features

- **Offline Video Processing**  
  - Single-pass extraction of landmarks (pupils, corners, nose) and derived features such as distances and angles.

- **Hierarchical Square-Based Training and Prediction**  
  - **Coarse Training:** Initial mapping into four primary squares (A, B, C, D).  
  - **Fine-Tuning (Optional):** Additional sub-square (AA, AB, AC, AD) refinement to improve accuracy.  
  - **Enhanced Prediction Pipeline:** Incorporates refined calculations, fallbacks to geometric predictions, and robust logging to track intermediate outputs.

- **Two-Stage Neural Network Architecture**  
  - **Head Pose Encoder:** Extracts head orientation features.  
  - **Pupil Decoder:** Combines head pose and pupil data to predict accurate screen coordinates.

- **Seamless DeepLabCut Integration**  
  - Accurate and reliable landmark detection using state-of-the-art DLC.

- **Advanced Visualization Tools**  
  - Customizable gaze heatmaps, detailed CSV summaries, and updated screenshot visualizations.

- **Cross-Platform Compatibility**  
  - Compatible with macOS and Windows; optimized for CPU usage.

---

## Prerequisites & Installation

### Prerequisites

- **Python 3.8** (or later)
- **Conda or pip** for environment management
- **DeepLabCut** dependencies (TensorFlow, etc.) if training or re-configuring DLC
- **PyTorch** for the neural network models
- **OpenCV** for video and image I/O

### Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/ahrebel/MAKO.git
   cd MAKO
   ```

2. **Set Up Your Environment**  
   **Conda (recommended):**
   ```bash
   conda create -n mako -c conda-forge python=3.8 \
       pytables hdf5 lzo opencv numpy pandas matplotlib \
       scipy tqdm statsmodels pytorch torchvision
   conda activate mako
   ```
   **pip + virtualenv (alternative):**
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

---

## Data Preparation

### Video Requirements

- **Format:** MP4, ≥ 720p resolution  
- **Conditions:** Consistent lighting and stable conditions

### DLC Landmarks

The DeepLabCut model should detect:
- `left_pupil`
- `right_pupil`
- `corner_left`
- `corner_right`
- `nose`

### Calibration Data

Prepare a combined CSV file containing:
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

## Neural Network Training & Hierarchical Gaze Pipeline

### Training Commands

#### Coarse Training Command

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

#### Fine-Tuning Command

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
  --device CPU
```

### Enhanced Prediction Pipeline Details

- **Enhanced Prediction Calculations:**  
  The model initially produces a coarse prediction for the quadrant. If a fine model for that quadrant exists, a refined 13-dimensional input vector (including computed head pose features) is passed through the fine model. The system then uses this refined output if it falls within the designated sub-region; otherwise, it falls back to the coarse prediction or a geometric fallback.

- **Simplified Screenshot Visualization:**  
  The visualization now draws only the predicted viewing spot (a green circle) and the quadrant border (a blue rectangle) on each screenshot.

- **Robust Debugging and Fallbacks:**  
  Additional logging and error handling ensure that if a fine model’s prediction is out-of-bound, the system automatically falls back to the coarse prediction, with geometric calculations as a backup when necessary.

---

## Gaze Analysis & Visualization

After training, generate heatmaps and CSV summaries of gaze data using the following command:

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

---

## Evaluate Function

The **evaluate** function is currently a placeholder within the codebase. It is designed to assess the performance of the trained gaze prediction models on labeled data. Future enhancements to this function may include:

- Generating detailed confusion matrices for the coarse and fine classifications.
- Computing performance metrics such as precision, recall, and F1 scores.
- Visualizing misclassified samples or regions where predictions are less accurate.
- Integrating additional statistical analyses to refine model performance.

To run the evaluation (once fully implemented), you would use a command similar to:

```bash
python src/hierarchical_gaze_pro.py evaluate \
  --data_csv labeled_data.csv \
  --coarse_model data/trained_model/coarse_trained_model.pt \
  --fine_model_dir data/trained_model/fine_tuned_models \
  --device cpu \
  --confusion
```

*Note: The evaluate function is a work-in-progress. Adjustments may be needed based on your evaluation criteria and dataset.*

---

## Command Line Arguments Reference

For a full list of CLI arguments and options for each mode, please refer to the [CLI Reference](docs/cli_reference.md).

---

## Troubleshooting

- **Landmark Detection Issues:**  
  If landmarks are inconsistent, consider retraining your DeepLabCut model with additional labeled frames.

- **Calibration Data Challenges:**  
  Erratic predictions can often be resolved by using a larger calibration dataset or adjusting calibration parameters (e.g., `--max_time_diff`).

- **Training Dynamics:**  
  Monitor both the coarse and fine training loss curves. Adjust learning rates, batch sizes, and epoch limits if predictions tend to converge toward the mean or if fine refinement is not effective.

---

## Contributing

Contributions are welcomed through GitHub issues and pull requests. Whether you’re improving documentation, fixing bugs, or adding new features, your help is greatly appreciated.

---

## Future Improvements

- **Real-Time Gaze Prediction:**  
  Implementation of online, real-time processing capabilities.
- **Enhanced Analysis Tools:**  
  Additional interactive visualizations and statistical analyses.
- **Adaptive and Incremental Learning:**  
  Investigating methods for incremental updates to improve model performance over time.

---

## License

MAKO is licensed under the [GPL-3.0 License](https://www.gnu.org/licenses/gpl-3.0.en.html).

---

## Security Disclaimer

Use MAKO within secure lab environments. No security guarantees are provided for public or commercial deployments.

---

## Citation *(In Progress)*

If you use MAKO in your research, please cite the following:

*Rebello, A. (2025). MAKO: Vision-based gaze tracking for non-human primates. [Manuscript in preparation]*


---

## Acknowledgements

- **DeepLabCut:** [DeepLabCut GitHub Repository](https://github.com/DeepLabCut/DeepLabCut)
- **PyTorch:** [PyTorch Official Site](https://pytorch.org/)
- **OpenCV:** [OpenCV Official Site](https://opencv.org/)

---

<pre>
███╗   ███╗  █████╗  ██╗  ██╗  ██████╗ 
████╗ ████║ ██╔══██╗ ██║ ██╔╝ ██╔═══██╗
██╔████╔██║ ███████║ █████╔╝  ██║   ██║
██║╚██╔╝██║ ██╔══██║ ██╔═██╗  ██║   ██║
██║ ╚═╝ ██║ ██║  ██ ║██║  ██╗ ╚██████╔╝
╚═╝     ╚═╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝  ╚═════╝
</pre>
```
