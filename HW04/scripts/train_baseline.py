import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import sys

# Add project root to Python path to allow importing from models and data_loader
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.promptir_baseline_model import PromptIRBaseline
from data_loader.dataset import ImageRestorationDataset
from utils.metrics import calculate_psnr

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

LEARNING_RATE = 1e-4 # Reduced from 2e-4 to combat NaN
BATCH_SIZE = 1 # Adjusted based on user feedback for 11GB GPU
NUM_EPOCHS = 50 # Start with a moderate number of epochs
IMAGE_SIZE = (256, 256)

# Model parameters (adjust based on GPU memory and desired complexity)
# PromptIR paper uses base_dim=48, num_blocks_per_level=[4,6,6,8]
# For an 11GB GPU, this might be too large with prompts.
# Let's start with a slightly smaller configuration.
MODEL_BASE_DIM = 32 # Reduced from 48
MODEL_NUM_BLOCKS_PER_LEVEL = [2, 3, 3, 4] # Reduced from [4,6,6,8]
MODEL_NUM_PROMPT_COMPONENTS = 4 # N
MODEL_PROMPT_COMPONENT_DIM = MODEL_BASE_DIM # C_prompt, can be same as base_dim or different
MODEL_NUM_ATTN_HEADS = 4 # Reduced from 8

# Data paths
TRAIN_DEGRADED_DIR = "Data/train/degraded"
TRAIN_CLEAN_DIR = "Data/train/clean"
# VAL_DEGRADED_DIR = "Data/val/degraded" # Assuming a split later
# VAL_CLEAN_DIR = "Data/val/clean"

MODEL_SAVE_DIR = "trained_models"
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "promptir_baseline_best.pth")
LAST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "promptir_baseline_last.pth")


# --- Main Training Function ---
def train_model():
    torch.autograd.set_detect_anomaly(True) # Enable anomaly detection
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    # 1. Dataset and DataLoader
    print("Loading dataset...")
    full_dataset = ImageRestorationDataset(
        degraded_base_dir=TRAIN_DEGRADED_DIR,
        clean_base_dir=TRAIN_CLEAN_DIR,
        image_size=IMAGE_SIZE
    )
    
    # Splitting dataset (e.g., 90% train, 10% val)
    val_split_ratio = 0.1
    val_size = int(val_split_ratio * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    if val_size == 0 and len(full_dataset) > 0: # Ensure val_size is at least 1 if dataset is not empty
        if train_size > 1 : # Ensure train_size is also at least 1
             val_size = 1
             train_size = len(full_dataset) - val_size
        else: # Cannot split, use all for training, no validation
            print("Dataset too small to split into train/val. Using full dataset for training.")
            train_dataset = full_dataset
            val_dataset = None # Or a copy of a small part of train_dataset if needed for code flow
    elif len(full_dataset) == 0:
        print("Dataset is empty. Exiting.")
        return
    else:
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Training dataset size: {len(train_dataset)}")

    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        print(f"Validation dataset size: {len(val_dataset)}")
    else:
        val_loader = None
        print("No validation dataset.")


    # 2. Model, Optimizer, Loss
    print("Initializing model...")
    model = PromptIRBaseline(
        in_channels=3,
        out_channels=3,
        base_dim=MODEL_BASE_DIM,
        num_blocks_per_level=MODEL_NUM_BLOCKS_PER_LEVEL,
        num_prompt_components=MODEL_NUM_PROMPT_COMPONENTS,
        prompt_component_dim=MODEL_PROMPT_COMPONENT_DIM,
        num_attn_heads=MODEL_NUM_ATTN_HEADS,
        bias=False # PromptIR often uses bias=False in convs
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.L1Loss() # L1 loss as used in PromptIR

    # (Optional) Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7) # Adjusted eta_min

    # 3. Training Loop
    print("Starting training...")
    best_val_psnr = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (degraded_imgs, clean_imgs) in enumerate(train_loader):
            degraded_imgs = degraded_imgs.to(DEVICE)
            clean_imgs = clean_imgs.to(DEVICE)

            optimizer.zero_grad()
            restored_imgs = model(degraded_imgs)
            loss = criterion(restored_imgs, clean_imgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 20 == 0: # Log every 20 batches
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        print(f"--- Epoch [{epoch+1}/{NUM_EPOCHS}] Average Training Loss: {epoch_loss:.4f} ---")

        # Validation step
        if val_loader:
            model.eval()
            val_psnr_total = 0.0
            val_loss_total = 0.0
            num_val_batches = 0
            with torch.no_grad():
                for val_degraded, val_clean in val_loader:
                    val_degraded = val_degraded.to(DEVICE)
                    val_clean = val_clean.to(DEVICE)
                    val_restored = model(val_degraded)
                    
                    val_loss = criterion(val_restored, val_clean)
                    val_loss_total += val_loss.item()

                    # Ensure images are in [0,1] for PSNR if ToTensor() normalizes them this way
                    # Our ToTensor() does not normalize beyond [0,1] by default.
                    for i in range(val_restored.size(0)):
                        psnr = calculate_psnr(val_restored[i], val_clean[i], max_val=1.0)
                        if psnr != float('inf'): # Avoid issues with perfect matches if any
                             val_psnr_total += psnr
                    num_val_batches += val_restored.size(0) # Count individual images for PSNR averaging

            avg_val_loss = val_loss_total / len(val_loader)
            avg_val_psnr = val_psnr_total / num_val_batches if num_val_batches > 0 else 0.0
            print(f"--- Epoch [{epoch+1}/{NUM_EPOCHS}] Validation Loss: {avg_val_loss:.4f}, Validation PSNR: {avg_val_psnr:.2f} dB ---")

            if avg_val_psnr > best_val_psnr:
                best_val_psnr = avg_val_psnr
                print(f"New best validation PSNR: {best_val_psnr:.2f} dB. Saving model...")
                torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
                print(f"Best model saved to {BEST_MODEL_SAVE_PATH}")
        
        # Save last model checkpoint
        if (epoch + 1) % 10 == 0 or epoch == NUM_EPOCHS - 1 : # Save last model periodically and at the end
            torch.save(model.state_dict(), LAST_MODEL_SAVE_PATH.replace(".pth", f"_epoch{epoch+1}.pth"))
            print(f"Last model checkpoint saved to {LAST_MODEL_SAVE_PATH.replace('.pth', f'_epoch{epoch+1}.pth')}")


        scheduler.step()

    print("Training finished.")
    if val_loader:
        print(f"Best validation PSNR achieved: {best_val_psnr:.2f} dB")
    print(f"Final model (last epoch) saved to {LAST_MODEL_SAVE_PATH.replace('.pth', f'_epoch{NUM_EPOCHS}.pth')}")

if __name__ == "__main__":
    train_model()
