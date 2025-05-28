"""
Prediction script for the PromptIR-V4 model.

This script loads a trained PromptIR-V4 model checkpoint and performs inference
on the test dataset (`Data/test/degraded/`). The restored images are saved
in CHW format (3, H, W) as uint8 NumPy arrays into a compressed .npz file
named `pred.npz`. The keys in the .npz file correspond to the input filenames
(e.g., "0.png", "1.png").
"""
import os
import sys
import glob
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T

# Add project root to Python path for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.promptir_v4_model import PromptIR_V4

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IMAGE_SIZE = (256, 256)  # Target image size for processing

# Path to the trained V4 model checkpoint
# Ensure this points to the desired model (e.g., best performing or latest)
TRAINED_MODEL_PATH = "trained_models_v4/promptir_v4_best.pth"
# Example for a specific epoch:
# TRAINED_MODEL_PATH = "trained_models_v4/promptir_v4_last_epoch141.pth"

# Model V4 Parameters (must match the architecture of the loaded TRAINED_MODEL_PATH)
MODEL_BASE_DIM = 48
MODEL_NUM_BLOCKS_PER_LEVEL = [3, 4, 4, 6]
MODEL_NUM_REFINEMENT_BLOCKS = 4
MODEL_NUM_PROMPT_COMPONENTS = 5
MODEL_PG_PROMPT_DIM_MAP = {0: 256, 1: 128, 2: 64}
MODEL_PG_BASE_HW_MAP = {
    0: IMAGE_SIZE[0] // 16,
    1: IMAGE_SIZE[0] // 8,
    2: IMAGE_SIZE[0] // 4
}
MODEL_BACKBONE_ATTN_HEADS = 8
MODEL_PROMPT_INTERACTION_ATTN_HEADS = 8

# Path to the test dataset (degraded images)
TEST_DEGRADED_DIR = "Data/test/degraded"

# Output file name for predictions
OUTPUT_NPZ_FILE = "pred.npz"


class TestImageDataset(Dataset):
    """
    Dataset class for loading test images.
    Sorts images numerically by filename (e.g., 0.png, 1.png, ...).
    """
    def __init__(self, test_degraded_dir, image_size=(256, 256)):
        """
        Args:
            test_degraded_dir (str): Directory containing degraded test images.
            image_size (tuple): Target (height, width) to resize images to.
        """
        self.test_degraded_dir = test_degraded_dir
        # Sort files numerically based on their names (e.g., "0.png", "1.png")
        self.image_files = sorted(
            glob.glob(os.path.join(test_degraded_dir, "*.png")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),  # Converts PIL Image (HWC, [0,255]) to Tensor (CHW, [0,1])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        filename_key = os.path.basename(img_path)  # e.g., "0.png"
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image: {img_path}. Error: {e}")
            # Return a dummy tensor and a special key if loading fails
            dummy_tensor = torch.zeros(
                (3, self.transform.transforms[0].size[0], self.transform.transforms[0].size[1])
            )
            return dummy_tensor, "-1"  # Special key to indicate error

        return image, filename_key


def predict_v4():
    """
    Main function to perform inference using the PromptIR-V4 model.
    Loads test data, initializes the model, loads trained weights,
    runs predictions, and saves them to an .npz file.
    """
    # Validate paths
    if not os.path.exists(TRAINED_MODEL_PATH):
        print(f"Error: Trained model not found at {TRAINED_MODEL_PATH}")
        return
    if not os.path.isdir(TEST_DEGRADED_DIR):
        print(f"Error: Test data directory not found at {TEST_DEGRADED_DIR}")
        return

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = TestImageDataset(
        test_degraded_dir=TEST_DEGRADED_DIR,
        image_size=IMAGE_SIZE
    )
    if len(test_dataset) == 0:
        print(f"No images found in {TEST_DEGRADED_DIR}. Exiting.")
        return
    
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,  # Batch size 1 for individual processing
        num_workers=4, pin_memory=True
    )
    print(f"Test dataset size: {len(test_dataset)}")

    # Initialize PromptIR_V4 model
    print("Initializing PromptIR_V4 model...")
    model = PromptIR_V4(
        in_channels=3, out_channels=3, base_dim=MODEL_BASE_DIM,
        num_blocks_per_level=MODEL_NUM_BLOCKS_PER_LEVEL,
        num_refinement_blocks=MODEL_NUM_REFINEMENT_BLOCKS,
        num_prompt_components=MODEL_NUM_PROMPT_COMPONENTS,
        pg_prompt_dim_map=MODEL_PG_PROMPT_DIM_MAP,
        pg_base_hw_map=MODEL_PG_BASE_HW_MAP,
        backbone_num_attn_heads=MODEL_BACKBONE_ATTN_HEADS,
        prompt_interaction_num_attn_heads=MODEL_PROMPT_INTERACTION_ATTN_HEADS,
        bias=False
    )
    
    # Load trained model weights
    print(f"Loading trained model weights from {TRAINED_MODEL_PATH}...")
    state_dict = torch.load(TRAINED_MODEL_PATH, map_location=DEVICE)
    
    # Remove 'module.' prefix if model was saved using nn.DataParallel
    if all(key.startswith('module.') for key in state_dict.keys()):
        print("Removing 'module.' prefix from DataParallel checkpoint keys...")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("Ensure model parameters in script match the saved checkpoint.")
        return

    model.to(DEVICE)
    model.eval()  # Set model to evaluation mode

    # Perform inference
    print("Performing inference on test images...")
    predictions_dict = {}
    processed_count = 0
    
    with torch.no_grad():  # Disable gradient calculations for inference
        for degraded_imgs, filename_keys in test_loader:
            current_key = filename_keys[0]  # filename_keys is a tuple of strings
            
            if current_key == "-1":  # Check for dummy key from dataset error
                print(f"Skipping a problematic image load (identified by dataset).")
                # Placeholder will be added later if this key is missing from 0-99.png
                continue

            degraded_imgs = degraded_imgs.to(DEVICE)
            
            try:
                restored_imgs_tensor = model(degraded_imgs)
                # Process output: clamp, move to CPU, convert to NumPy, scale, change type
                restored_img = torch.clamp(restored_imgs_tensor[0], 0.0, 1.0) # Assuming batch size 1
                # Convert to NumPy array (C, H, W) and scale to [0, 255], uint8
                restored_img_np = restored_img.cpu().numpy() * 255.0  # Keep as CHW
                restored_img_np = restored_img_np.astype(np.uint8)  # Shape: (3, H, W)
                predictions_dict[current_key] = restored_img_np
            except Exception as e:
                print(f"Error during model inference for key {current_key}: {e}")
                # Add placeholder if inference fails for this specific image
                placeholder_img = np.zeros(
                    (3, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.uint8  # CHW format
                )
                predictions_dict[current_key] = placeholder_img
            
            processed_count += 1
            if processed_count % 10 == 0: # Log progress every 10 images
                print(f"Processed {processed_count}/{len(test_dataset)} images.")
    
    # Ensure all 100 expected keys ('0.png' through '99.png') are present for submission
    final_predictions_for_npz = {}
    for i in range(100):  # Assuming test set has 100 images named 0.png to 99.png
        key = f"{i}.png"
        if key in predictions_dict:
            final_predictions_for_npz[key] = predictions_dict[key]
        else:
            print(f"Warning: Prediction for key '{key}' was missing. Adding zeros placeholder.")
            final_predictions_for_npz[key] = np.zeros(
                (3, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.uint8  # CHW format
            )

    # Save predictions to .npz file
    print(f"Saving predictions to {OUTPUT_NPZ_FILE}...")
    try:
        np.savez_compressed(OUTPUT_NPZ_FILE, **final_predictions_for_npz)
        print(f"Predictions saved successfully to {OUTPUT_NPZ_FILE}. "
              f"Total images in file: {len(final_predictions_for_npz)}")
    except Exception as e:
        print(f"Error saving .npz file: {e}")


if __name__ == "__main__":
    predict_v4()
