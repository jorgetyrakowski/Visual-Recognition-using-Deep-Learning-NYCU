# Enhanced Image Restoration with PromptIR-V4

## NYCU Computer Vision 2025 Spring HW4
**StudentID:** YOUR_STUDENT_ID_HERE 
**Name:** YOUR_NAME_HERE

## Introduction
This project presents PromptIR-V4, an advanced model for image restoration, specifically targeting the removal of degradation effects like rain and snow. The model builds upon the principles of PromptIR, incorporating a U-Net architecture with Transformer blocks and a novel dynamic spatial prompting mechanism (V4) to adaptively process image features. This work aims to achieve state-of-the-art performance in restoring clean images from degraded inputs.

![image](https://github.com/user-attachments/assets/560afd6e-0feb-4cae-8a93-0d065e380cb8)

This repository contains the source code for the PromptIR-V4 model, training and prediction scripts, and utilities for dataset handling and metric calculation.

## How to Install

1.  **Clone the repository:**
    ```bash
    git clone <your_repository_url>
    cd <repository_name>
    ```

2.  **Create a Conda environment (recommended):**
    The file `Implementacion/env.yml` (from the original PromptIR repository) can be used as a reference, though some packages might differ or need updates. A minimal setup would involve:
    ```bash
    conda create -n promptir_env python=3.9  # Or your preferred Python version
    conda activate promptir_env
    ```

3.  **Install PyTorch:**
    Visit [pytorch.org](https://pytorch.org/) for instructions specific to your CUDA version (if using GPU) or for CPU-only installation. For example:
    ```bash
    # Example for CUDA 11.8
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # For CPU-only
    # pip install torch torchvision torchaudio
    ```

4.  **Install other dependencies:**
    ```bash
    pip install numpy Pillow matplotlib seaborn scikit-image
    # Add any other specific packages used if not covered by the above
    ```

## How to Train

1.  **Prepare your dataset:**
    Organize your training data as follows:
    ```
    Data/
    └── train/
        ├── clean/
        │   ├── rain_clean-1.png
        │   ├── snow_clean-1.png
        │   └── ...
        └── degraded/
            ├── rain-1.png
            ├── snow-1.png
            └── ...
    ```
    The `Data/test/degraded/` directory should contain the test images (e.g., `0.png`, `1.png`, ... `99.png`).

2.  **Configure training parameters:**
    Open `scripts/train_v4.py` and adjust parameters such as `LEARNING_RATE`, `BATCH_SIZE`, `NUM_EPOCHS`, model architecture details (if changed from defaults), and loss weights as needed.

3.  **Run the training script:**
    ```bash
    python scripts/train_v4.py
    ```
    Trained model checkpoints will be saved in the `trained_models_v4/` directory. Logs will be printed to the console and can be saved to a file (e.g., `logs.txt`).

## How to Test/Predict

1.  **Ensure you have a trained model checkpoint.**
    The prediction script defaults to using `trained_models_v4/promptir_v4_best.pth`. Update `TRAINED_MODEL_PATH` in `scripts/predict_v4.py` if you want to use a different checkpoint.

2.  **Ensure your test images are in `Data/test/degraded/`** (e.g., `0.png`, `1.png`, ..., `99.png`).

3.  **Run the prediction script:**
    ```bash
    python scripts/predict_v4.py
    ```
    This will generate a compressed NumPy file named `pred.npz` in the project root directory. This file contains the restored images as uint8 NumPy arrays (CHW format), keyed by their original filenames.

## Model Architecture (PromptIR-V4)
The PromptIR-V4 model is a U-Net based architecture enhanced with Transformer blocks and a specialized dynamic spatial prompting mechanism.
-   **Backbone**: A U-Net structure with multiple levels of encoder and decoder blocks.
-   **Transformer Blocks**: Used within each level of the U-Net and in the bottleneck for powerful feature extraction and contextual understanding. These blocks include Multi-DConv Head Transposed Self-Attention (MDTA) and Gated-DConv Feed-Forward Networks (GDFN).
-   **Prompt Generation Modules (PGM-V4)**: Dynamically generate spatial prompts at multiple decoder stages. These prompts are conditioned on the decoder features at that stage and are formed by a weighted combination of learnable spatial prompt components.
-   **Prompt Interaction Modules (PIM-V4)**: Integrate the generated spatial prompts with the decoder features using a dedicated Transformer block, allowing the model to adaptively modulate features based on the learned prompts.
-   **Global Residual Connection**: The final output is added to the original input image.

This design allows the model to learn and apply context-specific restoration strategies.

**Visualizations:**

![Uploading collage_val.png…]()
![comparition](https://github.com/user-attachments/assets/5f546fe7-e1b8-43e1-971e-630d1a5a2a21)


**Training Curves:**

![plot_val_psnr_v4](https://github.com/user-attachments/assets/3903df0f-94d7-4e65-a84a-f5c162a7ba97)
![plot_train_losses_v4](https://github.com/user-attachments/assets/03b05926-cf3f-443f-b55d-0a69b4c35dbf)

## Performance Snapshot
![Screenshot from 2025-05-28 18-51-24](https://github.com/user-attachments/assets/056ae35a-e3f0-431f-9ca5-5ba122e15bff)

