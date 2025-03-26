#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training and validation routines for image classification models.
"""

import os
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from utils import (
    plot_confusion_matrix,
    plot_class_accuracy,
    plot_training_curves,
    cutmix_data
)


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device,
                epoch, use_cutmix=True):
    """
    Train model for one epoch with potential CutMix augmentation.

    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer for updating weights
        scheduler: Learning rate scheduler
        device (torch.device): Device to use for training
        epoch (int): Current epoch number
        use_cutmix (bool): Whether to apply CutMix augmentation

    Returns:
        tuple: (average loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    scaler = GradScaler('cuda')

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training", unit="batch")

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Apply CutMix with 20% probability
        cutmix_applied = False
        if use_cutmix and random.random() < 0.2:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets)
            cutmix_applied = True

        # Mixed precision training
        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(inputs)

            if cutmix_applied:
                loss = lam * criterion(outputs, targets_a) + \
                       (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, targets)

        # Backward and optimize with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += targets.size(0)

        if cutmix_applied:
            # Approximate accuracy for mixed samples
            correct += (lam * predicted.eq(targets_a).sum().float() +
                      (1 - lam) * predicted.eq(targets_b).sum().float()).item()
        else:
            correct += predicted.eq(targets).sum().item()

        # Update loss
        running_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total,
            'lr': optimizer.param_groups[0]['lr']
        })

    # Update LR scheduler (if it's not a OneCycleLR which updates per step)
    if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # This will be updated after validation
            pass
        else:
            scheduler.step()

    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device, epoch):
    """
    Validate the model on validation set.

    Args:
        model (nn.Module): Model to validate
        dataloader (DataLoader): Validation data loader
        criterion: Loss function
        device (torch.device): Device to use for validation
        epoch (int): Current epoch number

    Returns:
        tuple: (average loss, accuracy, per-class accuracy, confusion matrix)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # For per-class accuracy tracking and confusion matrix
    class_correct = {}
    class_total = {}
    all_targets = []
    all_predictions = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Validation", unit="batch")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Record loss
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Store targets and predictions for confusion matrix
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Update per-class metrics
            for i in range(targets.size(0)):
                label = targets[i].item()
                pred = predicted[i].item()

                # Initialize counters if needed
                if label not in class_total:
                    class_total[label] = 0
                    class_correct[label] = 0

                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

    # Calculate per-class accuracy
    class_acc = {}
    for class_idx in class_total:
        class_acc[class_idx] = 100. * class_correct[class_idx] / class_total[class_idx]

    # Print problematic classes (accuracy < 70%)
    print("\nClasses with accuracy < 70%:")
    problem_classes = {cls: acc for cls, acc in class_acc.items() if acc < 70.0}
    for cls, acc in sorted(problem_classes.items(), key=lambda x: x[1]):
        print(f"  Class {cls}: {acc:.2f}%")

    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions, labels=range(100))

    return running_loss / len(dataloader), 100. * correct / total, class_acc, cm


def train_and_validate(model, train_loader, val_loader, criterion, optimizer,
                       scheduler, device, args):
    """
    Main training and validation function.

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        optimizer: Optimizer for updating weights
        scheduler: Learning rate scheduler
        device (torch.device): Device for training
        args: Command line arguments

    Returns:
        float: Best validation accuracy
    """
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize variables for training
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Early stopping parameters
    patience = args.patience if hasattr(args, 'patience') else 5
    patience_counter = 0
    
    # Training loop
    for epoch in range(args.num_epochs):
        print(f"\n{'='*20} Epoch {epoch+1}/{args.num_epochs} {'='*20}")
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, None, device, 
            epoch, use_cutmix=args.cutmix
        )
        
        # Validate
        val_loss, val_acc, class_acc, cm = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print summary
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Generate and save visualizations
        if (epoch + 1) % 5 == 0 or epoch == 0 or val_acc > best_val_acc:
            # Save confusion matrix
            plot_confusion_matrix(
                cm, 
                class_names=range(100), 
                save_path=os.path.join(args.save_dir, f"confusion_matrix_epoch_{epoch+1}.png")
            )
            
            # Save per-class accuracy
            plot_class_accuracy(
                {k: v/100 for k, v in class_acc.items()},  # Convert to [0,1] range
                save_path=os.path.join(args.save_dir, f"class_accuracy_epoch_{epoch+1}.png")
            )
            
            # Save training curves
            history = {
                'train_loss': train_losses,
                'val_loss': val_losses,
                'train_acc': train_accs,
                'val_acc': val_accs
            }
            plot_training_curves(
                history,
                save_path=os.path.join(args.save_dir, f"training_curves_epoch_{epoch+1}.png")
            )
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs (patience: {patience})")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.save_dir, "final_model.pth"))
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }
    np.save(os.path.join(args.save_dir, "training_history.npy"), history)
    
    # Generate final visualizations
    plot_training_curves(
        history,
        save_path=os.path.join(args.save_dir, "final_training_curves.png")
    )
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    return best_val_acc