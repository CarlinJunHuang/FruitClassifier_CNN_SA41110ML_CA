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
