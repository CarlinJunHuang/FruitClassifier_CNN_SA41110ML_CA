# Change Logs

## v1.1

Only change filepaths and channel.

## v1.2

Change the input size to 64 by 64.
Increase the epochs to 10.
Add a mis-prediction output result.

## v1.3

Global hyperparameters (Epochs, Batch Size, Image Size) extracted for easy modification.
Set the epoches to 10, the batch size to 32, and the image size to 64.
Changed training data source to 'data_final'.
Structured output: Unique timestamped folders for each run in 'outputs/'.
Added visualization: Loss/Accuracy curves and Confusion Matrix saved as images.
Cleaned up legacy comments.
Added CUDA support (auto-detect GPU).
Added intra-epoch progress printing.
Improved plot labeling with folder name and test accuracy.

**OPTIMIZATION**
Pre-load all images to RAM/GPU at start to eliminate disk I/O bottleneck.

## v1.4

Update Testing Timer to print training time.

- Edit Global hyper parameter TRAINING_TIMER to True/False to enable/disable it. (Enabled by default)
  Minor update to print statement formatting for more clarity.

## v1.5

**Key Feature: "Helper of Tuning"**

- **Global Configuration**: All key parameters (Epochs, Batch, LR, Image Size, Filters, Neurons, Pooling, Dropout) are now global variables at the top of the file for instant tuning.
- **Validation Split**: Added a `VALIDATION_SPLIT` parameter (default 0.2) to automatically create a validation set and track overfitting.
- **Dynamic Architecture**: The model now automatically adjusts to changes in Image Size or filter counts without manual code changes (borrowed logic from Maha's pooling script).
- **Enhanced Visualization**:
  - Added a parameter summary box to the training plots so you know exactly which settings produced the result.
  - Added "Validation" curves (Red/Magenta) alongside Training curves.
- **New Features**:
  - Added `POOLING_TYPE` ("Max" or "Avg").
  - Added `DROPOUT_RATE` for regularization.
