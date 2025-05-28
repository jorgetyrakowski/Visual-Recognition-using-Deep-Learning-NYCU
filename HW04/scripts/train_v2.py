import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import sys
import time

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.promptir_v2_model import PromptIR_V2
from data_loader.dataset import ImageRestorationDataset
from utils.metrics import calculate_psnr
from utils.losses import DirectionalDecoupledLoss

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Experiment V2 Config
LEARNING_RATE = 1.5e-4 # Can be tuned
BATCH_SIZE = 2 # Confirmed by user for 11GB GPU
NUM_EPOCHS = 100 # Might need more for this complex model
IMAGE_SIZE = (256, 256)

# Model V2 parameters
MODEL_BASE_DIM = 40 # Increased capacity
MODEL_NUM_BLOCKS_PER_LEVEL = [2, 3, 3, 4] # Kept from baseline, can be increased later
MODEL_NUM_DEGRADATIONS = 2 # Rain, Snow
MODEL_DEGRADATION_PROMPT_DIM = 64
MODEL_BASIC_PROMPT_C = 64
MODEL_BASIC_PROMPT_HW = 16 # Spatial dim of basic prompt
MODEL_UNIVERSAL_PROMPT_DIM = 64
MODEL_P2P_HEADS = 4
MODEL_P2F_HEADS = 4
MODEL_BACKBONE_ATTN_HEADS = 4 # Reduced from 8 for baseline, keep for V2 for now

# Loss parameters
L1_LOSS_WEIGHT = 1.0
DDL_LOSS_WEIGHT = 0.002 # Alpha from PIP paper
DDL_THRESHOLD_DEGREES = 90.0

# Data paths
TRAIN_DEGRADED_DIR = "Data/train/degraded"
TRAIN_CLEAN_DIR = "Data/train/clean"

MODEL_SAVE_DIR = "trained_models_v2"
BEST_MODEL_SAVE_PATH_V2 = os.path.join(MODEL_SAVE_DIR, "promptir_v2_best.pth")
LAST_MODEL_SAVE_PATH_V2 = os.path.join(MODEL_SAVE_DIR, "promptir_v2_last.pth")

# --- Main Training Function ---
def train_model_v2():
    torch.autograd.set_detect_anomaly(True)
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    # 1. Dataset and DataLoader
    print("Loading dataset for V2 training...")
    full_dataset = ImageRestorationDataset(
        degraded_base_dir=TRAIN_DEGRADED_DIR,
        clean_base_dir=TRAIN_CLEAN_DIR,
        image_size=IMAGE_SIZE
    )
    
    val_split_ratio = 0.1
    val_size = int(val_split_ratio * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    if val_size == 0 and len(full_dataset) > 0:
        if train_size > 1:
             val_size = 1
             train_size = len(full_dataset) - val_size
        else:
            print("Dataset too small to split. Using full dataset for training.")
            train_dataset = full_dataset
            val_dataset = None
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
    print("Initializing PromptIR_V2 model...")
    model = PromptIR_V2(
        in_channels=3,
        out_channels=3,
        base_dim=MODEL_BASE_DIM,
        num_blocks_per_level=MODEL_NUM_BLOCKS_PER_LEVEL,
        num_degradations=MODEL_NUM_DEGRADATIONS,
        degradation_prompt_dim=MODEL_DEGRADATION_PROMPT_DIM,
        basic_prompt_c=MODEL_BASIC_PROMPT_C,
        basic_prompt_hw=MODEL_BASIC_PROMPT_HW,
        universal_prompt_dim=MODEL_UNIVERSAL_PROMPT_DIM,
        p2p_heads=MODEL_P2P_HEADS,
        p2f_heads=MODEL_P2F_HEADS,
        num_attn_heads=MODEL_BACKBONE_ATTN_HEADS,
        bias=False
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion_l1 = nn.L1Loss()
    criterion_ddl = DirectionalDecoupledLoss(threshold_angle_degrees=DDL_THRESHOLD_DEGREES).to(DEVICE)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

    # 3. Training Loop
    print("Starting V2 training...")
    best_val_psnr = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_start_time = time.time()
        running_l1_loss = 0.0
        running_ddl_loss = 0.0
        running_total_loss = 0.0
        
        for batch_idx, (degraded_imgs, clean_imgs, degrad_labels) in enumerate(train_loader):
            degraded_imgs = degraded_imgs.to(DEVICE)
            clean_imgs = clean_imgs.to(DEVICE)
            degrad_labels = degrad_labels.to(DEVICE)

            optimizer.zero_grad()
            
            # Forward pass - model now takes degradation_label
            restored_imgs = model(degraded_imgs, degrad_labels)
            
            l1_loss = criterion_l1(restored_imgs, clean_imgs)
            
            # Get degradation_aware_prompts for DDL loss
            # The nn.Embedding layer itself holds these prompts
            degrad_aware_prompt_vectors = model.degradation_aware_prompts.weight 
            ddl_loss = criterion_ddl(degrad_aware_prompt_vectors)
            
            total_loss = L1_LOSS_WEIGHT * l1_loss + DDL_LOSS_WEIGHT * ddl_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_l1_loss += l1_loss.item()
            running_ddl_loss += ddl_loss.item()
            running_total_loss += total_loss.item()

            if (batch_idx + 1) % 50 == 0: # Log more frequently for V2
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"L1: {l1_loss.item():.4f}, DDL: {ddl_loss.item():.4f}, Total: {total_loss.item():.4f}")
        
        avg_l1_loss = running_l1_loss / len(train_loader)
        avg_ddl_loss = running_ddl_loss / len(train_loader)
        avg_total_loss = running_total_loss / len(train_loader)
        print(f"--- Epoch [{epoch+1}/{NUM_EPOCHS}] Avg Train L1: {avg_l1_loss:.4f}, Avg DDL: {avg_ddl_loss:.4f}, Avg Total: {avg_total_loss:.4f} ---")

        if val_loader:
            model.eval()
            val_psnr_total = 0.0
            val_l1_loss_total = 0.0
            num_val_samples = 0
            with torch.no_grad():
                for val_degraded, val_clean, val_degrad_labels in val_loader:
                    val_degraded = val_degraded.to(DEVICE)
                    val_clean = val_clean.to(DEVICE)
                    val_degrad_labels = val_degrad_labels.to(DEVICE)
                    
                    val_restored = model(val_degraded, val_degrad_labels)
                    val_l1 = criterion_l1(val_restored, val_clean)
                    val_l1_loss_total += val_l1.item() * val_degraded.size(0)

                    for i in range(val_restored.size(0)):
                        psnr = calculate_psnr(val_restored[i], val_clean[i], max_val=1.0)
                        if psnr != float('inf'):
                             val_psnr_total += psnr
                    num_val_samples += val_restored.size(0)

            avg_val_l1_loss = val_l1_loss_total / num_val_samples if num_val_samples > 0 else 0
            avg_val_psnr = val_psnr_total / num_val_samples if num_val_samples > 0 else 0.0
            print(f"--- Epoch [{epoch+1}/{NUM_EPOCHS}] Val L1: {avg_val_l1_loss:.4f}, Val PSNR: {avg_val_psnr:.2f} dB ---")

            if avg_val_psnr > best_val_psnr:
                best_val_psnr = avg_val_psnr
                print(f"New best validation PSNR: {best_val_psnr:.2f} dB. Saving model...")
                torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH_V2)
                print(f"Best V2 model saved to {BEST_MODEL_SAVE_PATH_V2}")
        
        torch.save(model.state_dict(), LAST_MODEL_SAVE_PATH_V2.replace(".pth", f"_epoch{epoch+1}.pth"))
        print(f"Last V2 model checkpoint saved for epoch {epoch+1}")
        
        scheduler.step()
        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} took {epoch_end_time - epoch_start_time:.2f} seconds.")


    print("V2 Training finished.")
    if val_loader:
        print(f"Best validation PSNR for V2 model: {best_val_psnr:.2f} dB")
    print(f"Final V2 model (last epoch) saved to {LAST_MODEL_SAVE_PATH_V2.replace('.pth', f'_epoch{NUM_EPOCHS}.pth')}")

if __name__ == "__main__":
    train_model_v2()
