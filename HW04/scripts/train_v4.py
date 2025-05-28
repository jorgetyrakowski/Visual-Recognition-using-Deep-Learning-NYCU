"""
Main training script for the PromptIR-V4 model.

This script handles:
- Configuration of model parameters, training hyperparameters, and data paths.
- Dataset loading and splitting into training and validation sets.
- Model initialization (PromptIR_V4).
- Definition of loss functions (L1, SSIM, Charbonnier, Perceptual).
- The main training loop, including:
    - Forward and backward passes.
    - Optimization steps.
    - Learning rate scheduling.
    - In-loop data augmentation (flips).
    - Logging of training and validation metrics.
    - Saving of the best and last model checkpoints.
- Optional Automatic Mixed Precision (AMP) support.
"""
import os
import sys
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torch.cuda.amp import GradScaler, autocast  # For AMP

# Add project root to Python path for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.promptir_v4_model import PromptIR_V4
from data_loader.dataset import ImageRestorationDataset
from utils.metrics import calculate_psnr
from utils.losses import CharbonnierLoss, SSIMLoss, VGGPerceptualLoss

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Training Hyperparameters
LEARNING_RATE = 2e-4
BATCH_SIZE = 2  # Total batch size, will be split if using DataParallel
NUM_EPOCHS = 150
IMAGE_SIZE = (256, 256)  # Target image size for training and validation
AMP_ENABLED = False  # Automatic Mixed Precision (disabled due to previous NaN issues)

# Model V4 Parameters (must match the architecture defined in PromptIR_V4)
MODEL_BASE_DIM = 48  # Base dimension for model channels
MODEL_NUM_BLOCKS_PER_LEVEL = [4, 6, 6, 8]  # Num Transformer blocks per U-Net level
MODEL_NUM_REFINEMENT_BLOCKS = 4  # Num blocks in the refinement stage (bottleneck)
MODEL_NUM_PROMPT_COMPONENTS = 5  # Num of learnable spatial prompt components

# Prompt Generation (PGM) configuration per decoder stage (0=deep, 1=mid, 2=shallow)
MODEL_PG_PROMPT_DIM_MAP = {0: 256, 1: 128, 2: 64}  # Output channel dim of PGM
MODEL_PG_BASE_HW_MAP = {
    0: IMAGE_SIZE[0] // 16,  # Base H,W for prompt components at deepest PGM stage
    1: IMAGE_SIZE[0] // 8,   # Base H,W for prompt components at mid PGM stage
    2: IMAGE_SIZE[0] // 4    # Base H,W for prompt components at shallowest PGM stage
}
MODEL_BACKBONE_ATTN_HEADS = 8  # Num attention heads in backbone Transformer blocks
MODEL_PROMPT_INTERACTION_ATTN_HEADS = 8  # Num attention heads in prompt interaction blocks

# Loss Function Weights
L1_LOSS_WEIGHT = 0.7
SSIM_LOSS_WEIGHT = 0.15
CHARBONNIER_LOSS_WEIGHT = 0.05
PERCEPTUAL_LOSS_WEIGHT = 0.10

# Data Paths
TRAIN_DEGRADED_DIR = "Data/train/degraded"
TRAIN_CLEAN_DIR = "Data/train/clean"

# Model Saving Paths
MODEL_SAVE_DIR = "trained_models_v4"
BEST_MODEL_SAVE_PATH_V4 = os.path.join(MODEL_SAVE_DIR, "promptir_v4_best.pth")
LAST_MODEL_SAVE_PATH_V4 = os.path.join(MODEL_SAVE_DIR, "promptir_v4_last.pth")


class CustomAugmentations:
    """
    A callable class for applying custom augmentations to pairs of PIL images.
    Note: This class is defined but not actively used in the final train_model_v4,
    as augmentations are applied directly to tensors in the training loop.
    It's kept for potential future use or reference.
    """
    def __init__(self):
        self.rotations = [0, 90, 180, 270]

    def __call__(self, img1, img2):
        """
        Applies random horizontal flips, vertical flips, and 90-degree rotations.
        Args:
            img1 (PIL.Image): The first image (e.g., degraded).
            img2 (PIL.Image): The second image (e.g., clean).
        Returns:
            tuple: A tuple containing the augmented (img1, img2).
        """
        # Random horizontal flip
        if random.random() > 0.5:
            img1 = T.functional.hflip(img1)
            img2 = T.functional.hflip(img2)
        # Random vertical flip
        if random.random() > 0.5:
            img1 = T.functional.vflip(img1)
            img2 = T.functional.vflip(img2)
        # Random 90-degree rotation (requires PIL Image)
        # angle = random.choice(self.rotations)
        # img1 = T.functional.rotate(img1, angle)
        # img2 = T.functional.rotate(img2, angle)
        return img1, img2


def train_model_v4():
    """
    Main function to train the PromptIR-V4 model.
    """
    if AMP_ENABLED:
        print("AMP Enabled for V4 training.")
        scaler = GradScaler()
    
    # Helps in debugging NaN issues if they occur
    # torch.autograd.set_detect_anomaly(True) # Can be slow, use if needed

    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        print(f"Created directory: {MODEL_SAVE_DIR}")

    print("Loading dataset for V4 training...")
    full_dataset = ImageRestorationDataset(
        degraded_base_dir=TRAIN_DEGRADED_DIR,
        clean_base_dir=TRAIN_CLEAN_DIR,
        patch_size=IMAGE_SIZE[0],
        is_train=True  # Dataset handles its own train/val mode logic for transforms
    )
    
    # Splitting dataset into training and validation
    val_split_ratio = 0.1
    num_total_samples = len(full_dataset)
    val_size = int(val_split_ratio * num_total_samples)
    train_size = num_total_samples - val_size
    
    # Ensure loaders are not empty, especially with small datasets/large batch sizes
    if val_size < BATCH_SIZE and val_size > 0 : 
        print(f"Warning: Validation size ({val_size}) is less than batch size ({BATCH_SIZE}). Adjusting.")
        # This case might still lead to issues if val_size is 0 after adjustment.
        # A more robust way is to ensure val_size is at least 1 if val_split_ratio > 0
    if train_size < BATCH_SIZE:
        print("Error: Dataset too small for train/val split with current batch size.")
        print("Consider using the full dataset for training or reducing batch size.")
        return # Exit if training set is too small
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42) # For reproducible splits
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True
    )
    print(f"Training dataset size: {len(train_dataset)}")

    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=False
        )
        print(f"Validation dataset size: {len(val_dataset)}")
    else:
        val_loader = None
        print("No validation dataset or validation set is empty.")

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
        bias=False  # Typically False for models with normalization layers
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via nn.DataParallel.")
        model = nn.DataParallel(model)
    model.to(DEVICE)

    # Optimizer and Loss Functions
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion_l1 = nn.L1Loss().to(DEVICE)
    criterion_ssim = SSIMLoss(data_range=1.0, channel=3).to(DEVICE)
    criterion_char = CharbonnierLoss(eps=1e-3).to(DEVICE)
    criterion_perceptual = VGGPerceptualLoss().to(DEVICE)
    print("Losses: L1, SSIM, Charbonnier, Perceptual (VGG-based).")
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-7
    )

    print(f"Starting V4 training for {NUM_EPOCHS} epochs...")
    best_val_psnr = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_start_time = time.time()
        running_losses = {
            'l1': 0.0, 'ssim': 0.0, 'char': 0.0, 
            'percep': 0.0, 'total': 0.0
        }
        
        for batch_idx, batch_data in enumerate(train_loader):
            degraded_imgs, clean_imgs, _ = batch_data # Label not used by V4
            degraded_imgs = degraded_imgs.to(DEVICE)
            clean_imgs = clean_imgs.to(DEVICE)
            
            # In-loop tensor augmentations
            if random.random() > 0.5:  # Horizontal Flip
                degraded_imgs = T.functional.hflip(degraded_imgs)
                clean_imgs = T.functional.hflip(clean_imgs)
            if random.random() > 0.5:  # Vertical Flip
                degraded_imgs = T.functional.vflip(degraded_imgs)
                clean_imgs = T.functional.vflip(clean_imgs)

            optimizer.zero_grad()
            
            with autocast(enabled=AMP_ENABLED):
                restored_imgs = model(degraded_imgs)
                
                l1_loss = criterion_l1(restored_imgs, clean_imgs)
                ssim_loss = criterion_ssim(restored_imgs, clean_imgs)
                char_loss = criterion_char(restored_imgs, clean_imgs)
                perceptual_loss = criterion_perceptual(restored_imgs, clean_imgs)
                
                total_loss = (L1_LOSS_WEIGHT * l1_loss +
                              SSIM_LOSS_WEIGHT * ssim_loss +
                              CHARBONNIER_LOSS_WEIGHT * char_loss +
                              PERCEPTUAL_LOSS_WEIGHT * perceptual_loss)
            
            if AMP_ENABLED:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)  # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Accumulate losses for epoch average
            running_losses['l1'] += l1_loss.item()
            running_losses['ssim'] += ssim_loss.item()
            running_losses['char'] += char_loss.item()
            running_losses['percep'] += perceptual_loss.item() # .item() for scalar
            running_losses['total'] += total_loss.item()

            if (batch_idx + 1) % 50 == 0: # Log every 50 batches
                print(f"E[{epoch+1}/{NUM_EPOCHS}], B[{batch_idx+1}/{len(train_loader)}], "
                      f"L1: {l1_loss.item():.4f}, SSIM: {ssim_loss.item():.4f}, "
                      f"Char: {char_loss.item():.4f}, Pcp: {perceptual_loss.item():.4f}, "
                      f"Tot: {total_loss.item():.4f}")
        
        # Calculate and print average losses for the epoch
        num_batches = len(train_loader)
        print(f"--- E[{epoch+1}] Avg Train: "
              f"L1: {running_losses['l1']/num_batches:.4f}, "
              f"SSIM: {running_losses['ssim']/num_batches:.4f}, "
              f"Char: {running_losses['char']/num_batches:.4f}, "
              f"Pcp: {running_losses['percep']/num_batches:.4f}, "
              f"Tot: {running_losses['total']/num_batches:.4f} ---")

        # Validation phase
        if val_loader:
            model.eval()
            val_psnr_epoch_total = 0.0
            val_l1_epoch_total = 0.0
            num_val_samples_processed = 0
            
            with torch.no_grad():
                for val_degraded, val_clean, _ in val_loader:
                    val_degraded = val_degraded.to(DEVICE)
                    val_clean = val_clean.to(DEVICE)
                    
                    with autocast(enabled=AMP_ENABLED):
                        val_restored = model(val_degraded)
                    
                    val_l1 = criterion_l1(val_restored, val_clean)
                    val_l1_epoch_total += val_l1.item() * val_degraded.size(0)

                    for i in range(val_restored.size(0)):
                        # Ensure tensors are detached, on CPU, and in correct format for PSNR
                        restored_img_psnr = val_restored[i].detach()
                        clean_img_psnr = val_clean[i].detach()
                        psnr = calculate_psnr(restored_img_psnr, clean_img_psnr)
                        val_psnr_epoch_total += psnr if psnr != float('inf') else 35.0 # Cap inf
                    num_val_samples_processed += val_degraded.size(0)

            avg_val_l1 = val_l1_epoch_total / num_val_samples_processed if num_val_samples_processed > 0 else 0
            avg_val_psnr = val_psnr_epoch_total / num_val_samples_processed if num_val_samples_processed > 0 else 0.0
            print(f"--- E[{epoch+1}] Val L1: {avg_val_l1:.4f}, Val PSNR: {avg_val_psnr:.2f} dB ---")

            if avg_val_psnr > best_val_psnr:
                best_val_psnr = avg_val_psnr
                print(f"New best val PSNR: {best_val_psnr:.2f} dB. Saving model to {BEST_MODEL_SAVE_PATH_V4}")
                torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH_V4)
        
        # Save model checkpoint at the end of each epoch (or less frequently if desired)
        epoch_save_path = LAST_MODEL_SAVE_PATH_V4.replace(".pth", f"_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_save_path)
        
        scheduler.step() # Step the scheduler
        print(f"Epoch {epoch+1} took {time.time() - epoch_start_time:.2f}s. Model saved to {epoch_save_path}")

    print("V4 Training finished.")
    if val_loader:
        print(f"Best V4 validation PSNR achieved: {best_val_psnr:.2f} dB")
    
    final_model_path = LAST_MODEL_SAVE_PATH_V4.replace(".pth", f"_epoch{NUM_EPOCHS}.pth")
    print(f"Final V4 model (epoch {NUM_EPOCHS}) saved to {final_model_path}")


if __name__ == "__main__":
    train_model_v4()
