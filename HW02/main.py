import os
import json
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Import from our modules
from data_processing import SVHNDataset, collate_fn
from model import get_improved_model
from train import train_one_epoch, evaluate, predict_test_set
from utils import (
    calculate_metrics,
    plot_losses,
    plot_confusion_matrix,
    create_improved_task2_predictions,
    visualize_predictions,
)


def main():
    """Main function to train and evaluate the model."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Path configuration
    BASE_DIR = os.path.join(os.getcwd(), "nycu-hw2-data")
    TRAIN_JSON = os.path.join(BASE_DIR, "train.json")
    VALID_JSON = os.path.join(BASE_DIR, "valid.json")
    TRAIN_DIR = os.path.join(BASE_DIR, "train")
    VALID_DIR = os.path.join(BASE_DIR, "valid")
    TEST_DIR = os.path.join(BASE_DIR, "test")

    # Output directory
    OUTPUT_DIR = "improved_model"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)

    # Initialize data transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_dataset = SVHNDataset(TRAIN_JSON, TRAIN_DIR, transform=transform)
    valid_dataset = SVHNDataset(VALID_JSON, VALID_DIR, transform=transform)
    test_dataset = SVHNDataset(None, TEST_DIR, transform=transform, is_test=True)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=5, shuffle=True, collate_fn=collate_fn, num_workers=4
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=5, shuffle=False, collate_fn=collate_fn, num_workers=4
    )

    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False, num_workers=4)

    # Create improved model - 11 classes: background + 10 digits
    model = get_improved_model(num_classes=11)
    model.to(device)

    # Create optimizer with slightly lower learning rate for stability
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.004, momentum=0.9, weight_decay=0.0005)

    # Create learning rate scheduler with more gradual decay
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)

    # Track metrics
    train_losses = []
    val_metrics = []

    # Training loop
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        print(f"\nStarting Epoch {epoch}/{num_epochs}")

        # Train for one epoch
        epoch_losses, train_preds, train_targets = train_one_epoch(
            model, optimizer, train_loader, device, epoch
        )
        train_losses.append(epoch_losses)

        # Update learning rate
        lr_scheduler.step()

        # Evaluate on training set
        train_metrics = calculate_metrics(train_preds, train_targets)
        print(f"Training Metrics:")
        print(f"  Precision: {train_metrics['precision']:.4f}")
        print(f"  Recall: {train_metrics['recall']:.4f}")
        print(f"  F1 Score: {train_metrics['f1']:.4f}")

        # Evaluate on validation set
        val_preds, val_targets = evaluate(model, valid_loader, device)
        metrics = calculate_metrics(val_preds, val_targets)
        val_metrics.append(metrics)

        print(f"Validation Metrics:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")

        # Plot confusion matrix
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            os.path.join(OUTPUT_DIR, f"confusion_matrix_epoch_{epoch}.png"),
        )

        # Plot training curves
        plot_losses(
            train_losses,
            val_metrics,
            os.path.join(OUTPUT_DIR, f"training_curves_epoch_{epoch}.png"),
        )

        # Save model
        torch.save(
            model.state_dict(),
            os.path.join(OUTPUT_DIR, "models", f"model_epoch_{epoch}.pth"),
        )

        # Visualize predictions every few epochs
        if epoch % 5 == 0 or epoch == num_epochs:
            visualize_predictions(
                model,
                valid_dataset,
                device,
                num_samples=5,
                save_dir=os.path.join(OUTPUT_DIR, f"visualizations_epoch_{epoch}"),
            )

    print("Training complete!")

    # Generate test predictions (Task 1)
    print("Generating test predictions...")
    test_preds = predict_test_set(model, test_loader, device)

    # Save Task 1 predictions
    pred_json_path = os.path.join(OUTPUT_DIR, "pred.json")
    with open(pred_json_path, "w") as f:
        json.dump(test_preds, f)

    # Generate Task 2 predictions
    print("Generating Task 2 predictions...")
    task2_preds = create_improved_task2_predictions(test_preds)

    # Save Task 2 predictions
    pred_csv_path = os.path.join(OUTPUT_DIR, "pred.csv")
    df = pd.DataFrame(task2_preds)
    df.to_csv(pred_csv_path, index=False)

    print(f"Predictions saved to {pred_json_path} and {pred_csv_path}")

    # Save final plots
    final_confusion_path = os.path.join(OUTPUT_DIR, "final_confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], final_confusion_path)

    final_curves_path = os.path.join(OUTPUT_DIR, "final_training_curves.png")
    plot_losses(train_losses, val_metrics, final_curves_path)

    # Save experiment summary
    summary_path = os.path.join(OUTPUT_DIR, "experiment_summary.txt")
    with open(summary_path, "w") as f:
        f.write("SVHN Digit Detection - Experiment Summary\n")
        f.write("=======================================\n\n")
        f.write(f"Total epochs: {num_epochs}\n")
        f.write(f"Final validation metrics:\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall: {metrics['recall']:.4f}\n")
        f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"  True Positives: {metrics['true_positives']}\n")
        f.write(f"  False Positives: {metrics['false_positives']}\n")
        f.write(f"  False Negatives: {metrics['false_negatives']}\n\n")
        f.write("Model improvements:\n")
        f.write("- Modified RPN parameters to improve detection\n")
        f.write("- Adjusted NMS thresholds for better digit separation\n")
        f.write("- Implemented gradient clipping for training stability\n")
        f.write("- Enhanced Task 2 digit ordering algorithm\n")
        f.write("- Used higher confidence threshold (0.6) for Task 2\n")
        f.write("- Applied more gradual learning rate decay (gamma=0.2)\n")

    print(f"Experiment summary saved to {summary_path}")


if __name__ == "__main__":
    main()
