I'll create a comprehensive README based on your code and requirements.

# Enhanced ResNeXt with Channel Attention for 100-Class Image Classification

## NYCU Computer Vision 2025 Spring HW1
**StudentID:** 111550196  
**Name:** 狄豪飛 Jorge Tyrakowski

## Introduction

This project implements an enhanced image classification model for a 100-class classification challenge. The core approach uses ResNeXt101_32X8D as a backbone with added channel attention mechanisms to improve feature representation and address class imbalance.

Key features:
- Channel attention (Squeeze-and-Excitation) module 
- Custom dataset-specific normalization values 
- CutMix data augmentation for regularization
- Focal Loss with label smoothing to handle class imbalance
- Test-Time Augmentation for robust inference

## How to install

### Dependencies

```bash
pip install -r requirements.txt
```

### Directory Structure

```
.
├── config.py       # Configuration parameters
├── dataset.py      # Dataset classes and transformations
├── losses.py       # Loss function implementations
├── main.py         # Main entry point
├── models.py       # Model architecture definitions
├── inference.py    # Test prediction code
├── train.py        # Training and validation routines
├── utils.py        # Visualization and helper functions
├── requirements.txt # Project dependencies
└── data/           # Dataset directory
    ├── train/      # Training images (100 classes)
    ├── val/        # Validation images
    └── test/       # Test images
```

## Usage

### Training

```bash
python main.py train --train_data_dir data/train --val_data_dir data/val --save_dir ./results --cutmix --weighted_loss
```

Additional training options:
- `--num_epochs 20`: Set number of training epochs (default: 20)
- `--batch_size 10`: Change batch size (default: 10)
- `--learning_rate 1e-5`: Adjust learning rate (default: 1e-5)
- `--criterion focal`: Select loss function, options: "focal" or "cross_entropy" (default: "focal")
- `--nodropout`: Disable dropout (default: dropout enabled with p=0.5)
- `--seed 42`: Set random seed for reproducibility (default: 42)
- `--device cuda`: Select device for training (default: "cuda")
- `--weighted_loss`: Enable class weighting in loss function

### Inference

```bash
python main.py inference --test_data_dir data/test --model_path ./results/best_model.pth --save_dir ./results --tta
```

Options:
- `--test_data_dir data/test`: Directory containing test images (default: "./data/test")
- `--model_path`: Path to the trained model weights (required)
- `--save_dir ./results`: Directory to save prediction results (default: "./results")
- `--tta`: Enable Test-Time Augmentation for improved accuracy
- `--batch_size 10`: Adjust batch size for inference (default: 10)
- `--nodropout`: Disable dropout (should match training configuration)
- `--device cuda`: Select device for inference (default: "cuda")

## Model Architecture

The model enhances a ResNeXt101_32X8D backbone with a Squeeze-and-Excitation channel attention mechanism:

1. **Feature Extraction**: The ResNeXt101_32X8D backbone processes the input image (512×512 pixels) to extract 2048-channel feature maps
2. **Global Average Pooling**: Reduces spatial dimensions to create channel descriptors
3. **Channel Attention**: A Squeeze-and-Excitation module with reduction ratio 16 recalibrates feature importance
4. **Classification Head**: A classifier with optional dropout (p=0.5) produces the final prediction across 100 classes

The implementation uses mixed precision training for efficiency and includes early stopping to prevent overfitting.

## Performance snapshot

- Validation accuracy: 92.3%
- Public test data accuracy: 96%
- Parameters: 89.1M (within competition constraint of 100M)

![image](https://github.com/user-attachments/assets/9b3865ff-0032-469e-8676-e21e3fb029fc)

