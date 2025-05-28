import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
import sys
import glob
import torchvision.transforms as T

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.promptir_baseline_model import PromptIRBaseline

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IMAGE_SIZE = (256, 256) # Should match training image size

# Path to the trained model
# Update this to the specific model checkpoint you want to use for prediction
# e.g., "trained_models/promptir_baseline_best.pth" or a specific epoch's model
TRAINED_MODEL_PATH = "trained_models/promptir_baseline_best.pth" 
# Or use: TRAINED_MODEL_PATH = "trained_models/promptir_baseline_last_epoch50.pth" # Example

# Model parameters (should match the parameters of the loaded model)
# These were the parameters used in train_baseline.py with reduced complexity for 11GB GPU
MODEL_BASE_DIM = 32
MODEL_NUM_BLOCKS_PER_LEVEL = [2, 3, 3, 4]
MODEL_NUM_PROMPT_COMPONENTS = 4
MODEL_PROMPT_COMPONENT_DIM = MODEL_BASE_DIM
MODEL_NUM_ATTN_HEADS = 4

# Path to the test dataset
TEST_DEGRADED_DIR = "Data/test/degraded"

# Output file name
OUTPUT_NPZ_FILE = "pred.npz"

# --- Test Dataset Class ---
class TestImageDataset(Dataset):
    def __init__(self, test_degraded_dir, transform=None, image_size=(256, 256)):
        self.test_degraded_dir = test_degraded_dir
        self.image_files = sorted(glob.glob(os.path.join(test_degraded_dir, "*.png")))
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize(image_size),
                T.ToTensor(), # Converts to [0,1] range
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {img_path}. Error: {e}")
            raise e
            
        if self.transform:
            image = self.transform(image)
        
        filename = os.path.basename(img_path)
        return image, filename

# --- Main Prediction Function ---
def predict():
    if not os.path.exists(TRAINED_MODEL_PATH):
        print(f"Error: Trained model not found at {TRAINED_MODEL_PATH}")
        return

    if not os.path.isdir(TEST_DEGRADED_DIR):
        print(f"Error: Test data directory not found at {TEST_DEGRADED_DIR}")
        return

    # 1. Load Test Dataset
    print("Loading test dataset...")
    test_dataset = TestImageDataset(
        test_degraded_dir=TEST_DEGRADED_DIR, 
        image_size=IMAGE_SIZE
    )
    if len(test_dataset) == 0:
        print(f"No images found in {TEST_DEGRADED_DIR}. Exiting.")
        return
        
    # Using a batch size of 1 for prediction is usually safest,
    # but can be increased if GPU memory allows and model supports batch processing for inference.
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f"Test dataset size: {len(test_dataset)}")

    # 2. Initialize and Load Model
    print("Initializing model...")
    model = PromptIRBaseline(
        in_channels=3,
        out_channels=3,
        base_dim=MODEL_BASE_DIM,
        num_blocks_per_level=MODEL_NUM_BLOCKS_PER_LEVEL,
        num_prompt_components=MODEL_NUM_PROMPT_COMPONENTS,
        prompt_component_dim=MODEL_PROMPT_COMPONENT_DIM,
        num_attn_heads=MODEL_NUM_ATTN_HEADS,
        bias=False
    ).to(DEVICE)

    print(f"Loading trained model weights from {TRAINED_MODEL_PATH}...")
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=DEVICE))
    model.eval() # Set model to evaluation mode

    # 3. Perform Inference and Store Results
    print("Performing inference...")
    predictions_dict = {}
    with torch.no_grad():
        for batch_idx, (degraded_imgs, filenames) in enumerate(test_loader):
            degraded_imgs = degraded_imgs.to(DEVICE)
            
            restored_imgs_tensor = model(degraded_imgs) # Output is [0,1] float tensor

            # Process each image in the batch (though batch_size is likely 1)
            for i in range(restored_imgs_tensor.size(0)):
                filename = filenames[i]
                restored_img = restored_imgs_tensor[i] # Shape: [C, H, W], range [0,1]

                # Convert to NumPy array, scale to [0, 255], change to uint8, and transpose to (3, H, W)
                # The output format requires (3, H, W) and uint8
                # Current restored_img is already (C,H,W) i.e. (3,H,W)
                
                # Clamp values to ensure they are within [0,1] before scaling
                restored_img = torch.clamp(restored_img, 0.0, 1.0)
                
                # Scale to 0-255 and convert to uint8
                restored_img_np = restored_img.cpu().numpy() * 255.0
                restored_img_np = restored_img_np.astype(np.uint8) # Shape: (3, H, W)
                
                predictions_dict[filename] = restored_img_np
                
                if (batch_idx * test_loader.batch_size + i + 1) % 10 == 0:
                    print(f"Processed {batch_idx * test_loader.batch_size + i + 1}/{len(test_dataset)} images.")

    # 4. Save to .npz file
    print(f"Saving predictions to {OUTPUT_NPZ_FILE}...")
    np.savez_compressed(OUTPUT_NPZ_FILE, **predictions_dict) # Use savez_compressed for smaller file size
    print(f"Predictions saved successfully. Total images: {len(predictions_dict)}")

if __name__ == "__main__":
    predict()
