#!/usr/bin/env bash
# This script trains K-Fold models for the Global Wheat Detection project.
# It iterates through 5 folds, training a YOLOv11x model for each.

# Exit immediately if a command exits with a non-zero status.
set -e

# ── Global Training Parameters ─────────────────────────────────────────────────
# Base model weights. Assumes 'yolov11x.pt' is in the current working directory
# or a path recognizable by 'yolo train' (e.g., pre-downloaded to cache or official name).
# If it's a local file, 'yolo train' will use it.
MODEL_WEIGHTS="yolov11x.pt"

IMAGE_SIZE=1024       # Input image size for training
NUM_EPOCHS=60         # Total number of training epochs
BATCH_SIZE=6          # Batch size (chosen to fit 11GB VRAM of RTX 2080 Ti)
NUM_WORKERS=4         # Number of data loader workers
OUTPUT_PROJECT="gw11" # Main project directory for saving runs (e.g., gw11/gwd_fold0)

# Other YOLO train parameters
CLOSE_MOSAIC_EPOCHS=10 # Disable mosaic augmentation for the last N epochs
DETERMINISTIC_RUN=True # For reproducibility
RANDOM_SEED=0          # Seed for deterministic runs

# ── K-Fold Training Loop (Folds 0 to 4) ───────────────────────────────────────
echo "Starting K-Fold training process..."

for FOLD_INDEX in {0..4}; do
  echo -e "\n──────────────────────────────────────────────────────────────"
  echo   "🚀  Training FOLD $FOLD_INDEX  (Epochs: $NUM_EPOCHS, Image Size: $IMAGE_SIZE, Batch: $BATCH_SIZE)"
  echo   "──────────────────────────────────────────────────────────────"

  # Define the name for this specific training run
  RUN_NAME="gwd_fold${FOLD_INDEX}"
  # Define the path to the dataset configuration YAML for this fold
  # Assumes dataset_yamls/fold0.yaml, etc., are in CWD/dataset_yamls/
  DATA_YAML="dataset_yamls/fold${FOLD_INDEX}.yaml"

  # Set GPU to be used (e.g., first GPU)
  # Ensure this is correctly set for your environment if you have multiple GPUs.
  export CUDA_VISIBLE_DEVICES=0 
  
  # Launch YOLO training for the current fold
  yolo train \
      model=$MODEL_WEIGHTS \
      data=$DATA_YAML \
      imgsz=$IMAGE_SIZE \
      batch=$BATCH_SIZE \
      epochs=$NUM_EPOCHS \
      workers=$NUM_WORKERS \
      close_mosaic=$CLOSE_MOSAIC_EPOCHS \
      deterministic=$DETERMINISTIC_RUN seed=$RANDOM_SEED \
      name=$RUN_NAME \
      project=$OUTPUT_PROJECT
      # Other parameters like optimizer, lr0, lrf, loss gains, specific augmentations
      # are using the defaults from Ultralytics YOLO / YOLOv11x for this training.
done

echo -e "\n✓ All K-Fold training runs completed."
echo "Check results in subdirectories under '${OUTPUT_PROJECT}/', e.g., ${OUTPUT_PROJECT}/gwd_fold0/results.csv"
