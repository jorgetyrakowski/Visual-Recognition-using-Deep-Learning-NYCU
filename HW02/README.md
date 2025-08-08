## NYCU Visual Recognition using Deep Learning 2025 HW2
**StudentID:** 111550196  
**Name:** 狄豪飛 Jorge Tyrakowski

## Introduction

This repository contains my implementation for Homework 2 of the Visual Recognition using Deep Learning course. The project focuses on digit recognition using the Street View House Numbers (SVHN) dataset with an improved Faster R-CNN architecture.

![image](https://github.com/user-attachments/assets/019cc6d3-5ead-4146-a1d2-d3d069a24224)

The solution implements a highly optimized Faster R-CNN model with a ResNet50-FPN-V2 backbone, specifically tuned for the digit recognition task. Key enhancements include:

- Customized Region Proposal Network (RPN) parameters to better handle the varying scales and aspect ratios of street number digits
- Optimized Non-Maximum Suppression (NMS) thresholds for improved digit separation
- Implementation of gradient clipping and learning rate strategies for training stability
- A sophisticated post-processing algorithm for the number recognition task that considers spatial relationships between detected digits

The implementation achieves a performance with 0.382 mAP for Task 1 (digit detection) and 0.845 accuracy for Task 2 (number recognition), significantly exceeding the competition baselines. The code is organized into modular components following PEP8 standards for better readability and maintainability.

The task consists of two parts:
1. Task 1 (Object Detection): Detect individual digits in images, providing class and bounding box for each digit.
2. Task 2 (Number Recognition): Identify the complete number in each image by combining detected digits in their correct order.

## How to install

### Requirements

- Python 3.9 or higher
- PyTorch 1.10+
- torchvision
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- PIL
- tqdm

### How to install dependencies

Install required packages using pip:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn pillow tqdm
````

Or using the requirements file:

```bash
pip install -r requirements.txt
```

## Project Structure

The project is organized into modular components:

- `data_processing.py`: Dataset class and data loading utilities
- `model.py`: Model definitions and architecture configurations
- `train.py`: Training and evaluation functions
- `utils.py`: Utility functions for metrics, visualization, and post-processing
- `main.py`: Main script that orchestrates the training process
- `inference.py`: Script for making predictions on test data

## Usage

### Training

To train the model from scratch:

```bash
python main.py
```

This will:

1. Load and preprocess the SVHN dataset
2. Train the Faster R-CNN model with the specified backbone 
3. Evaluate the model on the validation set
4. Save model checkpoints, training curves, and confusion matrices

### Inference

To run inference on test data with a pre-trained model (you can modify the .pth in inference.py):

```bash
python inference.py
```

This generates:

- `pred.json`: Task 1 predictions (digit detection)
- `pred.csv`: Task 2 predictions (whole number recognition)

## Methods

### Model Architecture

I experimented with two different backbones:

1. **ResNet50-FPN-V2**: A powerful feature extractor with Feature Pyramid Network
2. **MobileNetV3-Large-FPN**: A lightweight alternative with comparable performance

For the improved model, I optimized the Region Proposal Network (RPN) parameters:

- Increased `rpn_pre_nms_top_n_train` from 2000 to 3000
- Increased `rpn_post_nms_top_n_train` from 1000 to 1500
- Adjusted `rpn_nms_thresh` from 0.7 to 0.75
- Set `box_score_thresh` to 0.4 and `box_nms_thresh` to 0.45

Training stability improvements:

- Applied gradient clipping with `max_norm=1.0`
- Reduced learning rate slightly (0.004 instead of 0.005)
- Used a more gradual learning rate decay (gamma=0.2 instead of 0.1)

### Task 2 Approach

For Task 2 (number recognition), I implemented a post-processing algorithm that:

1. Groups detections by image ID
2. Filters detections with confidence > 0.6
3. Sorts digits from left to right based on bounding box coordinates
4. Combines digit classes into complete numbers

## Performance snapshot

![model1_training_curves_epoch_10](https://github.com/user-attachments/assets/246a4931-5e57-4f5f-971b-0ae42e623484)

### Experimental Results

|Model / Experiment|Task 1 (mAP)|Task 2 (Accuracy)|
|---|---|---|
|MobileNet-V3-Large-FPN|0.372|0.81|
|ResNet50-FPN-V2 (Baseline)|0.382|0.837|
|ResNet50-FPN-V2 (Improved)|0.382|0.845|

### Training Curves

![model1_confusion_matrix_epoch_6](https://github.com/user-attachments/assets/f6f8b691-8222-4679-9228-9f9c0a6329f9)

![CurrentModel_training_curves_epoch_10](https://github.com/user-attachments/assets/92be8fce-857c-4704-a5fa-26a7842c0824)

### Confusion Matrix

![model1_confusion_matrix_epoch_6](https://github.com/user-attachments/assets/ed3f4a26-95ea-4063-8226-451ceb41853d)

![CurrentModel_confusion_matrix_epoch_4](https://github.com/user-attachments/assets/687da12a-9774-4605-bc29-91ac0cc118f1)

## Conclusions

The experiments demonstrated that:

1. ResNet50-FPN-V2 backbone provides better performance than MobileNetV3-Large-FPN
2. Optimizing RPN parameters leads to improved detection performance
3. Enhanced post-processing for Task 2 significantly improves number recognition accuracy

Both models exceed the competition baselines (0.35 for Task 1 and 0.70 for Task 2), with the improved ResNet50-FPN-V2 model achieving the best overall performance (0.382 mAP for Task 1 and 0.845 accuracy for Task 2).

## Performance Snapshot

![image](https://github.com/user-attachments/assets/b879b989-52f3-4f0d-a86a-e57d6ecbc29f)



