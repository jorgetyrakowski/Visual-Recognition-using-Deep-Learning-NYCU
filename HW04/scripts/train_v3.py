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

from models.promptir_v3_model import PromptIR_V3 # Using V3 model class
from data_loader.dataset import ImageRestorationDataset
from utils.metrics import calculate_psnr
from utils.losses import DirectionalDecoupledLoss, VGGPerceptualLoss

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Experiment V3 Config
LEARNING_RATE = 1e-4 # Start with a common LR for larger models
BATCH_SIZE = 2 
NUM_EPOCHS = 100 
IMAGE_SIZE = (256, 256)

# Model V3 parameters (Increased Capacity)
MODEL_BASE_DIM = 48
MODEL_NUM_BLOCKS_PER_LEVEL = [4, 6, 6, 8] 
MODEL_NUM_DEGRADATIONS = 2 # Rain, Snow
MODEL_DEGRADATION_PROMPT_DIM = 64
MODEL_BASIC_PROMPT_C = 64 
MODEL_BASIC_PROMPT_HW = 16 
MODEL_UNIVERSAL_PROMPT_DIM = 64
MODEL_P2P_HEADS = 4 
MODEL_P2F_HEADS = 4 
MODEL_BACKBONE_ATTN_HEADS = 8 # Increased for backbone

# Loss parameters
L1_LOSS_WEIGHT = 1.0
DDL_LOSS_WEIGHT = 0.002 
DDL_THRESHOLD_DEGREES = 90.0
PERCEPTUAL_LOSS_WEIGHT = 0.05 # Starting weight for perceptual loss

# Data paths
TRAIN_DEGRADED_DIR = "Data/train/degraded"
TRAIN_CLEAN_DIR = "Data/train/clean"

MODEL_SAVE_DIR = "trained_models_v3"
BEST_MODEL_SAVE_PATH_V3 = os.path.join(MODEL_SAVE_DIR, "promptir_v3_best.pth")
LAST_MODEL_SAVE_PATH_V3 = os.path.join(MODEL_SAVE_DIR, "promptir_v3_last.pth")

# --- Main Training Function ---
def train_model_v3():
    torch.autograd.set_detect_anomaly(True)
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    print("Loading dataset for V3 training...")
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
             val_size = 1; train_size = len(full_dataset) - val_size
        else:
            print("Dataset too small. Using full for training."); train_dataset = full_dataset; val_dataset = None
    elif len(full_dataset) == 0: print("Dataset empty."); return
    else: train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Training dataset size: {len(train_dataset)}")

    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        print(f"Validation dataset size: {len(val_dataset)}")
    else: val_loader = None; print("No validation dataset.")

    print("Initializing PromptIR_V3 model...")
    model = PromptIR_V3(
        in_channels=3, out_channels=3, base_dim=MODEL_BASE_DIM,
        num_blocks_per_level=MODEL_NUM_BLOCKS_PER_LEVEL,
        num_degradations=MODEL_NUM_DEGRADATIONS,
        degradation_prompt_dim=MODEL_DEGRADATION_PROMPT_DIM,
        basic_prompt_c=MODEL_BASIC_PROMPT_C, basic_prompt_hw=MODEL_BASIC_PROMPT_HW,
        universal_prompt_dim=MODEL_UNIVERSAL_PROMPT_DIM,
        p2p_heads=MODEL_P2P_HEADS, p2f_heads=MODEL_P2F_HEADS,
        num_attn_heads=MODEL_BACKBONE_ATTN_HEADS, bias=False
    ) #.to(DEVICE) # Move .to(DEVICE) after DataParallel wrapping

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion_l1 = nn.L1Loss().to(DEVICE)
    criterion_ddl = DirectionalDecoupledLoss(DDL_THRESHOLD_DEGREES).to(DEVICE)
    criterion_perceptual = VGGPerceptualLoss().to(DEVICE) # Initialize Perceptual Loss
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

    print("Starting V3 training...")
    best_val_psnr = 0.0

    # Memory check: Run a few batches first
    # print("Performing initial memory check (1 epoch, 5 batches)...")
    # model.train()
    # for _tmp_batch_idx, (_tmp_degraded, _tmp_clean, _tmp_labels) in enumerate(train_loader):
    #     if _tmp_batch_idx >= 5: break
    #     _tmp_degraded, _tmp_clean, _tmp_labels = _tmp_degraded.to(DEVICE), _tmp_clean.to(DEVICE), _tmp_labels.to(DEVICE)
    #     optimizer.zero_grad()
    #     _tmp_restored = model(_tmp_degraded, _tmp_labels)
    #     _tmp_l1 = criterion_l1(_tmp_restored, _tmp_clean)
    #     _tmp_percep = criterion_perceptual(_tmp_restored, _tmp_clean)
    #     _tmp_ddl = criterion_ddl(model.degradation_aware_prompts.weight)
    #     _tmp_total_loss = L1_LOSS_WEIGHT * _tmp_l1 + PERCEPTUAL_LOSS_WEIGHT * _tmp_percep + DDL_LOSS_WEIGHT * _tmp_ddl
    #     _tmp_total_loss.backward()
    #     optimizer.step()
    #     print(f"Memory check batch {_tmp_batch_idx+1} passed.")
    # print("Initial memory check complete. Proceeding with full training if no OOM.")
    # torch.cuda.empty_cache() # Clear cache before full loop

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_start_time = time.time()
        running_l1_loss, running_ddl_loss, running_perceptual_loss, running_total_loss = 0.0, 0.0, 0.0, 0.0
        
        for batch_idx, (degraded_imgs, clean_imgs, degrad_labels) in enumerate(train_loader):
            degraded_imgs, clean_imgs, degrad_labels = degraded_imgs.to(DEVICE), clean_imgs.to(DEVICE), degrad_labels.to(DEVICE)
            optimizer.zero_grad()
            restored_imgs = model(degraded_imgs, degrad_labels)
            
            l1_loss = criterion_l1(restored_imgs, clean_imgs)
            perceptual_loss = criterion_perceptual(restored_imgs, clean_imgs)
            
            # Accessing model's direct attributes needs .module if wrapped in DataParallel
            if isinstance(model, nn.DataParallel):
                degrad_aware_prompt_vectors = model.module.degradation_aware_prompts.weight
            else:
                degrad_aware_prompt_vectors = model.degradation_aware_prompts.weight
            
            ddl_loss = criterion_ddl(degrad_aware_prompt_vectors)
            
            total_loss = (L1_LOSS_WEIGHT * l1_loss + 
                          PERCEPTUAL_LOSS_WEIGHT * perceptual_loss + 
                          DDL_LOSS_WEIGHT * ddl_loss)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_l1_loss += l1_loss.item()
            running_perceptual_loss += perceptual_loss.item()
            running_ddl_loss += ddl_loss.item()
            running_total_loss += total_loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"E[{epoch+1}/{NUM_EPOCHS}], B[{batch_idx+1}/{len(train_loader)}], "
                      f"L1: {l1_loss.item():.4f}, Pcp: {perceptual_loss.item():.4f}, DDL: {ddl_loss.item():.4f}, Tot: {total_loss.item():.4f}")
        
        avg_l1 = running_l1_loss / len(train_loader)
        avg_perceptual = running_perceptual_loss / len(train_loader)
        avg_ddl = running_ddl_loss / len(train_loader)
        avg_total = running_total_loss / len(train_loader)
        print(f"--- E[{epoch+1}] Avg Train L1: {avg_l1:.4f}, Pcp: {avg_perceptual:.4f}, DDL: {avg_ddl:.4f}, Tot: {avg_total:.4f} ---")

        if val_loader:
            model.eval()
            val_psnr_total, val_l1_total, num_val_samples = 0.0, 0.0, 0
            with torch.no_grad():
                for val_degraded, val_clean, val_degrad_labels in val_loader:
                    val_degraded, val_clean, val_degrad_labels = val_degraded.to(DEVICE), val_clean.to(DEVICE), val_degrad_labels.to(DEVICE)
                    val_restored = model(val_degraded, val_degrad_labels)
                    val_l1 = criterion_l1(val_restored, val_clean)
                    val_l1_total += val_l1.item() * val_degraded.size(0)

                    for i in range(val_restored.size(0)):
                        psnr = calculate_psnr(val_restored[i], val_clean[i]); val_psnr_total += psnr if psnr!=float('inf') else 0
                    num_val_samples += val_restored.size(0)

            avg_val_l1 = val_l1_total / num_val_samples if num_val_samples > 0 else 0
            avg_val_psnr = val_psnr_total / num_val_samples if num_val_samples > 0 else 0.0
            print(f"--- E[{epoch+1}] Val L1: {avg_val_l1:.4f}, Val PSNR: {avg_val_psnr:.2f} dB ---")

            if avg_val_psnr > best_val_psnr:
                best_val_psnr = avg_val_psnr
                print(f"New best val PSNR: {best_val_psnr:.2f} dB. Saving model to {BEST_MODEL_SAVE_PATH_V3}")
                torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH_V3)
        
        torch.save(model.state_dict(), LAST_MODEL_SAVE_PATH_V3.replace(".pth", f"_epoch{epoch+1}.pth"))
        scheduler.step()
        print(f"Epoch {epoch+1} took {time.time() - epoch_start_time:.2f}s. Last model saved.")

    print("V3 Training finished.")
    if val_loader: print(f"Best V3 val PSNR: {best_val_psnr:.2f} dB")
    print(f"Final V3 model saved to {LAST_MODEL_SAVE_PATH_V3.replace('.pth', f'_epoch{NUM_EPOCHS}.pth')}")

if __name__ == "__main__":
    train_model_v3()
