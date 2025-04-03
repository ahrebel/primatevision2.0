# <pre>
███╗   ███╗  █████╗  ██╗  ██╗  ██████╗ 
████╗ ████║ ██╔══██╗ ██║ ██╔╝ ██╔═══██╗
██╔████╔██║ ███████║ █████╔╝  ██║   ██║
██║╚██╔╝██║ ██╔══██║ ██╔═██╗  ██║   ██║
██║ ╚═╝ ██║ ██║  ██ ║██║  ██╗ ╚██████╔╝
╚═╝     ╚═╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝  ╚═════╝
</pre>

**Mako** is an advanced **eye-tracking system** designed for Rhesus macaques (and humans) that combines **DeepLabCut (DLC)** for multi-landmark detection with a **Two-Stage Neural Network** to map raw eye coordinates onto screen coordinates. It supports offline processing of pre-recorded videos, producing detailed **heatmaps** and CSV summaries of gaze/fixation data.

![Animated Eyes](https://github.com/user-attachments/assets/0f245b14-ec20-4a11-868a-ae207a7dfa1d)

---

## Overview

Mako simplifies the **eye-tracking** workflow by:

- **Detecting Eye and Nose Landmarks:**  
  Leverages DeepLabCut to detect multiple key landmarks (e.g., pupil corners, nose tip) that serve as inputs for gaze estimation.

- **Hierarchical Gaze Mapping:**  
  Uses a two-stage neural network with a hierarchical square-based approach:
  - **Coarse Prediction:** Maps landmarks into one of four primary quadrants (A, B, C, D).  
  - **Fine Refinement (Optional):** If available, a corresponding fine model further refines the prediction within sub-squares (AA, AB, AC, AD).  
  Enhanced error handling is implemented so that if the refined prediction is out-of-bound, the system reverts to the coarse prediction or a geometric fallback.

- **Advanced Analysis and Visualization:**  
  Generates detailed heatmaps and CSV summaries of gaze/fixation data, and visualizes predictions by displaying the predicted viewing spot and quadrant borders on screenshots.

---

## Key Features

- **Offline Video Processing**  
  - Single pass extraction of landmarks (pupils, corners, nose) and derived features such as distances and angles.

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
  - Customizable gaze heatmaps, detailed CSV summaries, and updated screenshot visualizations that now clearly show only the predicted viewing spot and quadrant borders.

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
- **PyQt5** (optional, if you want to use the GUI)

### Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/ahrebel/mako.git
   cd mako
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

6. **(Optional) Install PyQt5**  
   ```bash
   pip install pyqt5
   ```
   This is required only if you want to use the mako GUI.

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

**Coarse Training:**  
Run the following command to train the coarse model with K-Fold cross-validation and additional options:
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

**Fine-Tuning:**  
Train fine models for all quadrants (A, B, C, D) with the following command:
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
  The visualization now draws only the predicted viewing spot (green circle) and the quadrant border (blue rectangle) on each screenshot.

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

## Make GUI (Optional)

- **PyQt5-based Interface:**  
  A user-friendly GUI is available for interacting with all components of Mako.
```bash
pip install pyqt5
python mako_app.py
```

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
- **Enhanced GUI Functionality:**  
  Expanding the PyQt5 interface with additional interactive tools.
- **Adaptive and Incremental Learning:**  
  Investigating methods for incremental updates to improve model performance over time.

---

## License

Mako is licensed under the [GPL-3.0 License](https://www.gnu.org/licenses/gpl-3.0.en.html).

---

## Security Disclaimer

Use Mako within secure lab environments. No security guarantees are provided for public or commercial deployments.

---

## Acknowledgements

- **DeepLabCut:** [DeepLabCut GitHub Repository](https://github.com/DeepLabCut/DeepLabCut)
- **PyTorch:** [PyTorch Official Site](https://pytorch.org/)
- **OpenCV:** [OpenCV Official Site](https://opencv.org/)

---

**Happy Eye-Tracking!**

---

This updated README now reflects the changes made to the hierarchical gaze code, includes your specific training and analysis commands, and adds a section outlining the evaluate function. Feel free to further customize or expand this document as your project evolves.

Below is the updated README with all of the previous content—including the changes to the hierarchical gaze code—plus new sections that describe the usage of each command argument (and whether it is required or optional). Any mention of the GUI has been removed.

---

# Mako

**Mako** is an advanced **eye-tracking system** designed for Rhesus macaques (and humans) that combines **DeepLabCut (DLC)** for multi-landmark detection with a **Two-Stage Neural Network** to map raw eye coordinates onto screen coordinates. It supports offline processing of pre-recorded videos, producing detailed **heatmaps** and CSV summaries of gaze/fixation data.

![Animated Eyes](https://github.com/user-attachments/assets/0f245b14-ec20-4a11-868a-ae207a7dfa1d)

---

## Overview

Mako simplifies the **eye-tracking** workflow by:

- **Detecting Eye and Nose Landmarks:**  
  Leverages DeepLabCut to detect multiple key landmarks (e.g., pupil corners, nose tip) that serve as inputs for gaze estimation.

- **Hierarchical Gaze Mapping:**  
  Uses a two-stage neural network with a hierarchical square-based approach:
  - **Coarse Prediction:** Maps landmarks into one of four primary quadrants (A, B, C, D).  
  - **Fine Refinement (Optional):** If available, a corresponding fine model further refines the prediction within sub-squares (AA, AB, AC, AD).  
  Enhanced error handling is implemented so that if the refined prediction is out-of-bound, the system reverts to the coarse prediction or a geometric fallback.

- **Advanced Analysis and Visualization:**  
  Generates detailed heatmaps and CSV summaries of gaze/fixation data, and visualizes predictions by displaying the predicted viewing spot and quadrant borders on screenshots.

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
  - Customizable gaze heatmaps, detailed CSV summaries, and updated screenshot visualizations that now clearly show only the predicted viewing spot (green circle) and the quadrant border (blue rectangle).

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
   git clone https://github.com/ahrebel/Mako.git
   cd Mako
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

## Command Line Arguments

Each mode in the `hierarchical_gaze_pro.py` script accepts a variety of arguments. Below is a summary of the available arguments, organized by mode.

### Coarse Training Mode (`coarse_train`)

- `--data` (Required):  
  Path to the training CSV file containing calibration data.
- `--output` (Required):  
  Destination path for the saved coarse trained model.
- `--batch_size` (Optional, default: 64):  
  Batch size used during training.
- `--epochs` (Optional, default: user-specified, e.g., 1000):  
  Maximum number of training epochs.
- `--patience` (Optional, default: user-specified, e.g., 30):  
  Number of epochs with no improvement before early stopping.
- `--alpha` (Optional, default: user-specified, e.g., 0.4):  
  Weight for the cross-entropy loss (the remainder is used for the MSE loss).
- `--lr` (Optional, default: 1e-3):  
  Learning rate for the optimizer.
- `--balanced_ce` (Optional flag):  
  Enables balanced cross-entropy loss if specified.
- `--onecycle` (Optional flag):  
  Uses the OneCycleLR learning rate scheduler if specified.
- `--normalize` (Optional flag):  
  Applies input normalization to the training data.
- `--kfold` (Optional, default: 0 or user-specified, e.g., 5):  
  Number of folds for K-Fold cross-validation.
- `--device` (Optional, default: "cpu"):  
  Device on which to run the training (e.g., "cpu" or "cuda").

### Fine Training Mode (`fine_train`)

- `--data` (Required):  
  Path to the training CSV file.
- `--coarse_label` (Required):  
  Specifies the quadrant to train for (e.g., A, B, C, D) or `all` to train for all quadrants.
- `--output` (Required):  
  Output folder where the fine-tuned models will be saved.
- `--batch_size` (Optional, default: 64):  
  Batch size used during training.
- `--epochs` (Optional, default: user-specified, e.g., 1000):  
  Maximum number of training epochs.
- `--patience` (Optional, default: user-specified, e.g., 30):  
  Number of epochs with no improvement before early stopping.
- `--alpha` (Optional, default: user-specified, e.g., 0.4):  
  Weight for the cross-entropy loss versus the MSE loss.
- `--lr` (Optional, default: 1e-3):  
  Learning rate for the optimizer.
- `--balanced_ce` (Optional flag):  
  Enables balanced cross-entropy loss if specified.
- `--onecycle` (Optional flag):  
  Uses the OneCycleLR learning rate scheduler if specified.
- `--normalize` (Optional flag):  
  Applies input normalization to the training data.
- `--kfold` (Optional, default: 0 or user-specified, e.g., 5):  
  Number of folds for K-Fold cross-validation.
- `--device` (Optional, default: "cpu"):  
  Device on which to run the training.

### Analysis Mode (`analyze`)

- `--landmarks` (Required):  
  CSV file containing the landmark outputs from DeepLabCut.
- `--coarse_model` (Required):  
  Path to the trained coarse model file.
- `--screen_width` (Optional, default: 1024):  
  Width of the target screen in pixels.
- `--screen_height` (Optional, default: 768):  
  Height of the target screen in pixels.
- `--output_csv` (Optional):  
  Destination CSV file to save the gaze predictions.
- `--device` (Optional, default: "cpu"):  
  Device on which to run the analysis.
- `--screenshot_folder` (Optional):  
  Folder containing screenshot images for overlay visualization.
- `--screenshot_output_folder` (Optional):  
  Folder where annotated screenshots will be saved.
- `--detailed` (Optional flag):  
  Enables detailed logging output during analysis.
- `--normalize` (Optional flag):  
  Applies normalization to the input data using training statistics.

### Evaluate Mode (`evaluate`)

- `--data_csv` (Required):  
  CSV file containing the labeled evaluation data.
- `--coarse_model` (Required):  
  Path to the trained coarse model file.
- `--fine_model_dir` (Required):  
  Directory containing the fine-tuned model files.
- `--device` (Optional, default: "cpu"):  
  Device on which to run the evaluation.
- `--confusion` (Optional flag):  
  If specified, the evaluation will output a confusion matrix and additional performance metrics.

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

Mako is licensed under the [GPL-3.0 License](https://www.gnu.org/licenses/gpl-3.0.en.html).

---

## Security Disclaimer

Use Mako within secure lab environments. No security guarantees are provided for public or commercial deployments.

---

## Acknowledgements

- **DeepLabCut:** [DeepLabCut GitHub Repository](https://github.com/DeepLabCut/DeepLabCut)
- **PyTorch:** [PyTorch Official Site](https://pytorch.org/)
- **OpenCV:** [OpenCV Official Site](https://opencv.org/)

---

**Happy Eye-Tracking!**
