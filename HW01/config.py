"""
Configuration parameters for image classification.

This module contains shared parameters and configuration values.
"""

# Dataset-specific normalization values
MEAN = [0.4575, 0.4705, 0.3730]
STD = [0.1975, 0.1955, 0.2001]

# Image sizes
IMAGE_SIZE = 512  # Standard size for input images

# Training parameters
BATCH_SIZE = 10
NUM_EPOCHS = 20
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-2
DROPOUT_PROB = 0.5
PATIENCE = 5  # Early stopping patience

# Data augmentation parameters
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.2
COLOR_JITTER = 0.2
ROTATION_DEGREES = 15

# Model parameters
NUM_CLASSES = 100

# Paths
DEFAULT_TRAIN_DIR = "./data/train"
DEFAULT_VAL_DIR = "./data/val"
DEFAULT_TEST_DIR = "./data/test"
DEFAULT_SAVE_DIR = "./results"

# TTA parameters
TTA_NUM_AUGS = 5
