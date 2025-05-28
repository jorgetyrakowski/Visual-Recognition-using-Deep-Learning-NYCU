# Cell Instance Segmentation

## NYCU Visual Recognition using Deep Learning 2025 HW1
**StudentID:** 111550196  
**Name:** 狄豪飛 Jorge Tyrakowski

## Introduction

This repository contains a deep learning solution for cell instance segmentation in biomedical images. The goal is to develop a model that can accurately identify and segmentate individual cell instances across four different cell classes (class1, class2, class3, class4) in microscopy images.

Instance segmentation of cells presents several unique challenges:
- Cells often have irregular shapes with diffuse boundaries
- Dense cell clusters with significant overlap
- High variability in size and appearance within the same class

Our approach is based on the Cascade Mask R-CNN architecture with multiple refinement stages, which helps to distinguish individual cells in clusters and provides more accurate boundary delineation.

![image](https://github.com/user-attachments/assets/ff430826-fe1a-4b64-bbff-12895d5d1458)
![sample_7_iou_0 46](https://github.com/user-attachments/assets/4a78d191-cec6-4c67-9183-31611a7158a6)

*Example of model predictions (right) versus ground truth (left)*

## Features

- Cascade Mask R-CNN with multi-stage detection refinement
- Multiple backbone options (ResNet50/101/152)
- Dice Loss for mask optimization
- Advanced data augmentation pipelines tailored for cell images
- Multi-scale training for improved robustness
- Test-time augmentation (TTA) for better inference
- Large image handling with tiling
- Quantitative evaluation using COCO metrics (AP, AP50, AP75)

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- OpenCV
- pycocotools
- scikit-image
- matplotlib
- numpy

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/cell-instance-segmentation.git
cd cell-instance-segmentation
```

2. Create and activate a conda environment:
```bash
conda create -n cell-seg python=3.8
conda activate cell-seg
```

3. Install the required packages:

   **Option 1**: Using requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

   **Option 2**: Installing packages manually:
   ```bash
   pip install torch torchvision
   pip install opencv-python pycocotools scikit-image matplotlib numpy
   ```

4. Prepare your data:
   - Place your data in the `data` directory with the structure described in the dataset section
   - Make sure you have the test_image_name_to_ids.json file for inference

## Dataset Structure

The dataset should be organized as follows:

```
data/
  ├── train/
  │    ├── [image_folder_1]/
  │    │    ├── image.tif (original image)
  │    │    ├── class1.tif (mask for cells type 1)
  │    │    ├── class2.tif (mask for cells type 2)
  │    │    ├── class3.tif (mask for cells type 3)
  │    │    └── class4.tif (mask for cells type 4)
  │    └── [image_folder_2]/
  │         └── ...
  ├── test_release/
  │    ├── [image_name_1].tif
  │    ├── [image_name_2].tif
  │    └── ...
  └── test_image_name_to_ids.json
```

In each mask file, each unique pixel value (except 0) represents an individual cell instance.

## Usage

### Training

1. First, create a train/validation split:

```bash
python train.py --data_root data --make_split 0.1
```

2. Train the model:

```bash
python train.py --mode train \
    --data_root data \
    --epochs 30 \
    --batch_size 1 \
    --accum_steps 8 \
    --lr 5e-4 \
    --backbone resnet101 \
    --val_every 2 \
    --aug 1 \
    --multi_scale \
    --amp \
    --class_weights \
    --out_dir outputs
```

You can adjust the hyperparameters as needed. Important options:

- `--backbone`: Choose from "resnet50", "resnet101", or "resnet152"
- `--multi_scale`: Enable multi-scale training for better robustness
- `--amp`: Use mixed precision training for faster training
- `--class_weights`: Apply class weighting based on instance frequency
- `--val_every`: Validation frequency in epochs

### Inference

To generate predictions on the test set:

```bash
python train.py --mode infer \
    --data_root data \
    --backbone resnet101 \
    --ckpt outputs/best.pth \
    --out_file submission.json
```

The generated `submission.json` file will contain predictions in COCO format.

## Results

Our enhanced Cascade Mask R-CNN model with ResNet101 backbone achieves the following performance metrics:

| Model | AP | AP50 | AP75 |
|-------|----|----|-----|
| Mask R-CNN (Baseline) | 0.340 | 0.523 | 0.412 |
| Cascade Mask R-CNN (Ours) | 0.490 | 0.684 | 0.552 |

![class_performance_metrics](https://github.com/user-attachments/assets/e2505a14-3d6d-4284-bf52-71b0a023d8ce)

*Performance metrics by cell class*

Performance varies considerably across cell types, with class3 and class4 achieving F1 scores of 0.844 and 0.800 respectively, while class1 and class2 prove more challenging with F1 scores of 0.526 and 0.646.

## Analysis

Several key improvements in our approach contributed to the performance gain:

1. **Cascade Mask R-CNN Architecture**: The multi-stage detection refinement process significantly improves the model's ability to distinguish individual cells in clusters.

2. **Dice Loss Implementation**: By directly optimizing for region overlap, Dice Loss improves mask quality, particularly for cells with irregular shapes.

3. **Advanced Data Augmentation**: The sophisticated augmentation pipeline, especially the elastic deformations, helps the model generalize to varying cell appearances.

4. **Custom Anchor Generation**: Modified anchor sizes and aspect ratios to better match cell dimensions led to improved recall.

5. **Test-Time Augmentation**: Using both original and horizontally flipped images during inference improved robustness.

## Code Structure

The project is organized into multiple modules for improved readability and maintainability:

- `models.py`: Model architecture definitions (Cascade Mask R-CNN, DiceLoss)
- `datasets.py`: Dataset class and data processing utilities
- `utils.py`: Utility functions, evaluation metrics, and logger
- `inference.py`: Inference utilities and test-time augmentation
- `train.py`: Main script for training and inference

## Acknowledgements

The implementation is based on PyTorch and torchvision's Mask R-CNN implementation with substantial enhancements for cell segmentation tasks.

## References

1. K. He, G. Gkioxari, P. Dollár, and R. Girshick, "Mask R-CNN," in IEEE International Conference on Computer Vision (ICCV), 2017.
2. Z. Cai and N. Vasconcelos, "Cascade R-CNN: Delving into High Quality Object Detection," in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
3. T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, "Feature Pyramid Networks for Object Detection," in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
4. F. Milletari, N. Navab, and S.-A. Ahmadi, "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation," in Fourth International Conference on 3D Vision (3DV), 2016.

## Performance Snapshot

![Screenshot from 2025-05-06 17-25-37](https://github.com/user-attachments/assets/f843f0ea-5541-4fd7-8ae5-056571d3f881)


