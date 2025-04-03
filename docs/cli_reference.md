## Coarse Training Mode (`coarse_train`)

- **`--data` (Required):**  
  Path to the training CSV file containing calibration data.
- **`--output` (Required):**  
  Destination path for the saved coarse trained model.
- **`--batch_size` (Optional, default: 64):**  
  Batch size used during training.
- **`--epochs` (Optional, e.g., 1000):**  
  Maximum number of training epochs.
- **`--patience` (Optional, e.g., 30):**  
  Number of epochs with no improvement before early stopping.
- **`--alpha` (Optional, e.g., 0.4):**  
  Weight for the cross-entropy loss (the remainder is used for the MSE loss).
- **`--lr` (Optional, default: 1e-3):**  
  Learning rate for the optimizer.
- **`--balanced_ce` (Optional flag):**  
  Enables balanced cross-entropy loss if specified.
- **`--onecycle` (Optional flag):**  
  Uses the OneCycleLR learning rate scheduler if specified.
- **`--normalize` (Optional flag):**  
  Applies input normalization to the training data.
- **`--kfold` (Optional, default: 5):**  
  Number of folds for K-Fold cross-validation.
- **`--device` (Optional, default: "cpu"):**  
  Device on which to run the training (e.g., "cpu" or "cuda").

---

## Fine Training Mode (`fine_train`)

- **`--data` (Required):**  
  Path to the training CSV file.
- **`--coarse_label` (Required):**  
  Specifies the quadrant to train for (e.g., A, B, C, D) or `all` to train for all quadrants.
- **`--output` (Required):**  
  Output folder where the fine-tuned models will be saved.
- **`--batch_size` (Optional, default: 64):**  
  Batch size used during training.
- **`--epochs` (Optional, e.g., 1000):**  
  Maximum number of training epochs.
- **`--patience` (Optional, e.g., 30):**  
  Number of epochs with no improvement before early stopping.
- **`--alpha` (Optional, e.g., 0.4):**  
  Weight for the cross-entropy loss versus the MSE loss.
- **`--lr` (Optional, default: 1e-3):**  
  Learning rate for the optimizer.
- **`--balanced_ce` (Optional flag):**  
  Enables balanced cross-entropy loss if specified.
- **`--onecycle` (Optional flag):**  
  Uses the OneCycleLR learning rate scheduler if specified.
- **`--normalize` (Optional flag):**  
  Applies input normalization to the training data.
- **`--kfold` (Optional, default: 5):**  
  Number of folds for K-Fold cross-validation.
- **`--device` (Optional, default: "cpu"):**  
  Device on which to run the training.

---

## Analysis Mode (`analyze`)

- **`--landmarks` (Required):**  
  CSV file containing the landmark outputs from DeepLabCut.
- **`--coarse_model` (Required):**  
  Path to the trained coarse model file.
- **`--screen_width` (Optional, default: 1024):**  
  Width of the target screen in pixels.
- **`--screen_height` (Optional, default: 768):**  
  Height of the target screen in pixels.
- **`--output_csv` (Optional):**  
  Destination CSV file to save the gaze predictions.
- **`--device` (Optional, default: "cpu"):**  
  Device on which to run the analysis.
- **`--screenshot_folder` (Optional):**  
  Folder containing screenshot images for overlay visualization.
- **`--screenshot_output_folder` (Optional):**  
  Folder where annotated screenshots will be saved.
- **`--detailed` (Optional flag):**  
  Enables detailed logging output during analysis.
- **`--normalize` (Optional flag):**  
  Applies normalization to the input data.

---

## Evaluate Mode (`evaluate`)

- **`--data_csv` (Required):**  
  CSV file containing the labeled evaluation data.
- **`--coarse_model` (Required):**  
  Path to the trained coarse model file.
- **`--fine_model_dir` (Required):**  
  Directory containing the fine-tuned model files.
- **`--device` (Optional, default: "cpu"):**  
  Device on which to run the evaluation.
- **`--confusion` (Optional flag):**  
  If specified, outputs a confusion matrix and additional performance metrics.
