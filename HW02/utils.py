import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
from torchvision.ops import box_iou
from PIL import ImageDraw
import torchvision
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def calculate_metrics(predictions, targets, iou_threshold=0.5, score_threshold=0.5):
    """
    Calculate precision, recall, F1 and confusion matrix.

    Args:
        predictions (list): List of prediction dictionaries
        targets (list): List of target dictionaries
        iou_threshold (float): IoU threshold for considering a detection a match
        score_threshold (float): Confidence threshold for filtering predictions

    Returns:
        dict: Dictionary containing metrics
    """
    # Map image_id to predictions and targets
    pred_by_image = {p["image_id"]: p for p in predictions}
    target_by_image = {t["image_id"]: t for t in targets}

    # Collect common image IDs
    common_ids = set(pred_by_image.keys()).intersection(set(target_by_image.keys()))

    # Prepare for confusion matrix
    all_true_labels = []
    all_pred_labels = []

    # Counters for metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for img_id in common_ids:
        pred = pred_by_image[img_id]
        target = target_by_image[img_id]

        # Filter predictions by score
        mask = pred["scores"] >= score_threshold
        pred_boxes = pred["boxes"][mask]
        pred_labels = pred["labels"][mask]

        target_boxes = target["boxes"]
        target_labels = target["labels"]

        # Calculate IoU between each pred box and target box
        if len(pred_boxes) > 0 and len(target_boxes) > 0:
            iou_matrix = box_iou(pred_boxes, target_boxes)

            # For each prediction, find the best matching target
            for i in range(len(pred_boxes)):
                # Find target with highest IoU
                max_iou, max_idx = torch.max(iou_matrix[i], dim=0)

                if max_iou >= iou_threshold:
                    # We have a match
                    if pred_labels[i] == target_labels[max_idx]:
                        # Correct classification
                        true_positives += 1
                        all_true_labels.append(target_labels[max_idx].item())
                        all_pred_labels.append(pred_labels[i].item())
                    else:
                        # Wrong classification
                        false_positives += 1
                        all_true_labels.append(target_labels[max_idx].item())
                        all_pred_labels.append(pred_labels[i].item())

                    # Remove this target to avoid double counting
                    iou_matrix[:, max_idx] = -1
                else:
                    # No match found (IoU too low)
                    false_positives += 1

            # Count unmatched targets as false negatives
            unmatched = torch.all(iou_matrix == -1, dim=0).logical_not()
            false_negatives += unmatched.sum().item()
        else:
            # If there are no predictions but there are targets, count as false negatives
            if len(target_boxes) > 0:
                false_negatives += len(target_boxes)
            # If there are predictions but no targets, count as false positives
            if len(pred_boxes) > 0:
                false_positives += len(pred_boxes)

    # Calculate metrics
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    # Create confusion matrix
    if len(all_true_labels) > 0 and len(all_pred_labels) > 0:
        cm = confusion_matrix(all_true_labels, all_pred_labels, labels=range(1, 11))
    else:
        cm = np.zeros((10, 10))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "confusion_matrix": cm,
    }


def plot_losses(train_losses, val_metrics, save_path):
    """
    Plot training losses and validation metrics.

    Args:
        train_losses (list): List of loss dictionaries
        val_metrics (list): List of validation metric dictionaries
        save_path (str): Path to save the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # Plot training losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, [loss["loss"] for loss in train_losses], "b-", label="Total Loss")
    ax1.plot(
        epochs,
        [loss["loss_classifier"] for loss in train_losses],
        "g-",
        label="Classifier Loss",
    )
    ax1.plot(
        epochs,
        [loss["loss_box_reg"] for loss in train_losses],
        "r-",
        label="Box Reg Loss",
    )
    ax1.set_title("Training Losses")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot validation metrics
    ax2.plot(epochs, [m["precision"] for m in val_metrics], "b-", label="Precision")
    ax2.plot(epochs, [m["recall"] for m in val_metrics], "g-", label="Recall")
    ax2.plot(epochs, [m["f1"] for m in val_metrics], "r-", label="F1 Score")
    ax2.set_title("Validation Metrics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True)

    # Plot TP, FP, FN
    ax3.plot(
        epochs, [m["true_positives"] for m in val_metrics], "g-", label="True Positives"
    )
    ax3.plot(
        epochs,
        [m["false_positives"] for m in val_metrics],
        "r-",
        label="False Positives",
    )
    ax3.plot(
        epochs,
        [m["false_negatives"] for m in val_metrics],
        "b-",
        label="False Negatives",
    )
    ax3.set_title("TP, FP, FN Counts")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Count")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(cm, save_path):
    """
    Plot confusion matrix.

    Args:
        cm (numpy.ndarray): Confusion matrix
        save_path (str): Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Map class indices (1-10) to digits (0-9)
    digit_labels = [str(i - 1) for i in range(1, 11)]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digit_labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical")

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def create_improved_task2_predictions(predictions):
    """
    Generate task 2 predictions (whole number recognition) with improved handling
    of digit ordering and confidence thresholds.

    Args:
        predictions (list): List of task 1 predictions

    Returns:
        list: List of task 2 predictions
    """
    # Group predictions by image_id
    predictions_by_image = defaultdict(list)
    for pred in predictions:
        predictions_by_image[pred["image_id"]].append(pred)

    # Process each image
    results = []
    for img_id, preds in predictions_by_image.items():
        # Use higher confidence threshold for better precision in Task 2
        preds = [p for p in preds if p["score"] > 0.6]

        if len(preds) == 0:
            # No digits detected
            results.append({"image_id": img_id, "pred_label": -1})
            continue

        # Improved sorting to handle digit arrangement
        # For multi-digit numbers, digits should be read from left to right
        def sort_key(pred):
            x, y, w, h = pred["bbox"]
            # Use x-coordinate for horizontal ordering
            # With a small weight for y to handle vertically aligned digits
            return x + 0.01 * y

        # Sort digits from left to right
        preds.sort(key=sort_key)

        # Combine digits to form the whole number
        # Convert category_id (1-10) to digit (0-9)
        digits = [str(p["category_id"] - 1) for p in preds]
        num = int("".join(digits))

        results.append({"image_id": img_id, "pred_label": num})

    return results


def visualize_predictions(
    model, dataset, device, indices=None, num_samples=5, save_dir=None
):
    """
    Visualize model predictions on dataset samples.

    Args:
        model (torch.nn.Module): The trained model
        dataset (torch.utils.data.Dataset): The dataset to visualize from
        device (torch.device): Device to use for inference
        indices (list, optional): List of specific indices to visualize
        num_samples (int): Number of random samples to visualize if indices not provided
        save_dir (str): Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    if indices is None:
        # Random indices
        indices = np.random.choice(len(dataset), num_samples, replace=False)

    # Process each sample
    for i, idx in enumerate(indices):
        image, target = dataset[idx]
        image_id = target["image_id"].item()

        # Move to device
        image_tensor = image.unsqueeze(0).to(device)

        # Get predictions
        with torch.no_grad():
            predictions = model(image_tensor)
            prediction = predictions[0]

        # Convert image from tensor to PIL
        image_pil = torchvision.transforms.functional.to_pil_image(image)
        draw = ImageDraw.Draw(image_pil)

        # Draw ground truth boxes in green
        for box, label in zip(target["boxes"], target["labels"]):
            box = box.numpy()
            draw.rectangle(
                [(box[0], box[1]), (box[2], box[3])], outline="green", width=2
            )
            draw.text((box[0], box[1] - 10), f"{label.item()-1}", fill="green")

        # Draw predicted boxes in red
        for box, label, score in zip(
            prediction["boxes"], prediction["labels"], prediction["scores"]
        ):
            if score > 0.5:  # Only draw high confidence predictions
                box = box.cpu().numpy()
                draw.rectangle(
                    [(box[0], box[1]), (box[2], box[3])], outline="red", width=2
                )
                draw.text(
                    (box[0], box[1] - 10), f"{label.item()-1} ({score:.2f})", fill="red"
                )

        # Save the image
        image_pil.save(os.path.join(save_dir, f"pred_img{image_id}.png"))

    print(f"Saved {len(indices)} visualization images to {save_dir}")
