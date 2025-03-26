# Enhanced ResNeXt with Channel Attention for 100-Class Image Classification

## NYCU Computer Vision 2025 Spring HW1
**StudentID:** 111550196  
**Name:** 狄豪飛 Jorge Tyrakowski

## Introduction

This project implements an enhanced image classification model for a 100-class classification challenge. The core approach uses ResNeXt101_32X8D as a backbone with added channel attention mechanisms to improve feature representation and address class imbalance.

Key features:
- Channel attention (Squeeze-and-Excitation) module for better feature discrimination
- Custom dataset-specific normalization values
- CutMix data augmentation for regularization
- Focal Loss with label smoothing to handle class imbalance
- Test-Time Augmentation for robust inference

## How to install

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
├── images/         # Results visualization
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
- `--num_epochs 30`: Set number of training epochs
- `--batch_size 16`: Change batch size
- `--learning_rate 5e-5`: Adjust learning rate
- `--criterion cross_entropy`: Use Cross-Entropy loss instead of Focal Loss
- `--nodropout`: Disable dropout
- `--patience 7`: Adjust early stopping patience

### Inference

```bash
python main.py inference --test_data_dir data/test --model_path ./results/best_model.pth --save_dir ./results --tta
```

Options:
- `--tta`: Enable Test-Time Augmentation
- `--device cpu`: Use CPU instead of GPU
- `--nodropout`: Should match training configuration

## Model Architecture

The model enhances a ResNeXt101_32X8D backbone with a Squeeze-and-Excitation channel attention mechanism:

1. The backbone processes the input image to extract features
2. Global Average Pooling reduces spatial dimensions
3. Channel attention recalibrates feature importance
4. A classifier head with dropout produces the final prediction

## Key Improvements

1. **Dataset-specific normalization**: Mean=[0.4575, 0.4705, 0.3730], Std=[0.1975, 0.1955, 0.2001]
2. **Input size optimization**: 512×512 pixels for better detail preservation
3. **CutMix augmentation**: Creates diverse training samples by combining images
4. **Focal Loss**: Focuses on hard examples with label smoothing for better generalization
5. **Channel attention**: Dynamically emphasizes important features

## Performance snapshot

![image](https://github.com/user-attachments/assets/8aedce01-a0e4-4bb8-8f16-66afcd5d184d)
