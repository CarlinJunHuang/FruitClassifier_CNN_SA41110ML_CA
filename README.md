# Fruit Image Classifier (SA4110 ML)

This is a repository of SA4110 course assignment, created by SA61 Group3. 

PyTorch CNN pipeline for classifying fruit images (`apple`, `banana`, `orange`, `mixed`). The latest script (`codes/train_v1.5.py`) exposes all key hyperparameters at the top for fast experiment tuning and produces training/validation metrics plus optional test evaluation.

## Repository Layout
- `codes/train_v1.5.py`: Main training pipeline with configurable hyperparameters, schedulers, and automatic train/val split.
- `codes/train_v1.5 copy.py`: Same pipeline with a preset "best_para" configuration (TEAM_MEMBER_NAME = HJ).
- `codes/data_augumentation.py`: Simple offline augmentation helper (edit paths before use).
- `codes/train_v1.1`~`v1.4`, `codes/train_CW_copy.py`: Earlier baselines kept for reference.
- `data_final/`: Default training set (filenames must start with the class name, e.g., `apple_*.jpg`).
- `test/`: Optional hold-out set evaluated automatically if present.
- `outputs/`: Run artifacts (`model.pth`, `training_metrics.png`, `confusion_matrix.png`).

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate   # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```
PyTorch should match your CUDA/cuDNN stack if you plan to use GPU acceleration.

## Data Preparation
- Place training images in `data_final/` (or change `DATA_SOURCE_DIR` in the script).
- Each image is labeled via filename prefix: `apple_`, `banana_`, `orange_`, `mixed_`.
- Default input resolution is `64x64`; adjust `IMAGE_SIZE` if you change the dataset.

## Running Training
1) Open `codes/train_v1.5.py` and set:
   - `TEAM_MEMBER_NAME`, `TRIAL_DESCRIPTION`
   - `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `WEIGHT_DECAY`
   - `CONV_ARCH`, `FC_ARCH`, `POOLING_TYPE`, `DROPOUT_RATE`
   - `LR_SCHEDULER_TYPE` (0=None, 1=StepLR, 2=ExpLR, 3=Cosine, 4=CosineWarmRestarts) plus scheduler params
   - `VALIDATION_SPLIT` (e.g., `0.2` for 80/20 train/val)
2) Run from the repo root:
```bash
python codes/train_v1.5.py
```
The script will auto-detect CUDA, pre-load the dataset to RAM/GPU, and create `outputs/<TEAM>_<TRIAL>_<timestamp>/`.

## Outputs
- `model.pth`: Trained weights.
- `training_metrics.png`: Loss/accuracy curves with parameter summary and optional validation curves.
- `confusion_matrix.png`: Saved when a `test/` set is found.
- Console log includes per-epoch metrics and test accuracy (if applicable).

## Augmentation (Optional)
Edit paths in `codes/data_augumentation.py` (`DATA_DIR`, `OUTPUT_DIR`, `NUM_IMAGES_PER_CLASS`) and run:
```bash
python codes/data_augumentation.py
```

## Tips
- Keep filenames clean (`<class>_<anything>.jpg/png/bmp`) so labels parse correctly.
- If you hit CUDA OOM, lower `BATCH_SIZE` or `IMAGE_SIZE`.
- Validation can be disabled by setting `VALIDATION_SPLIT = 0`.
