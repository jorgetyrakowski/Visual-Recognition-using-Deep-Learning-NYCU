import time
import torch
from tqdm import tqdm


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train
        optimizer (torch.optim.Optimizer): The optimizer to use
        data_loader (torch.utils.data.DataLoader): DataLoader with training data
        device (torch.device): Device to use for training
        epoch (int): Current epoch number
        print_freq (int): Frequency of printing training metrics

    Returns:
        tuple: A tuple containing (loss_dict, all_predictions, all_targets)
    """
    model.train()

    # Metrics
    total_loss = 0
    loss_classifier = 0
    loss_box_reg = 0
    loss_objectness = 0
    loss_rpn_box_reg = 0

    # Track predictions for evaluation
    all_preds = []
    all_targets = []

    start_time = time.time()

    for i, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        # Move to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)

        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())

        # Apply gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update metrics
        total_loss += losses.item()
        loss_classifier += loss_dict["loss_classifier"].item()
        loss_box_reg += loss_dict["loss_box_reg"].item()
        loss_objectness += loss_dict["loss_objectness"].item()
        loss_rpn_box_reg += loss_dict["loss_rpn_box_reg"].item()

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Store predictions and targets for evaluation
        with torch.no_grad():
            model.eval()
            preds = model(images)
            model.train()

            for j, pred in enumerate(preds):
                if len(pred["boxes"]) > 0:
                    all_preds.append(
                        {
                            "image_id": targets[j]["image_id"].item(),
                            "boxes": pred["boxes"].cpu(),
                            "labels": pred["labels"].cpu(),
                            "scores": pred["scores"].cpu(),
                        }
                    )
                    all_targets.append(
                        {
                            "image_id": targets[j]["image_id"].item(),
                            "boxes": targets[j]["boxes"].cpu(),
                            "labels": targets[j]["labels"].cpu(),
                        }
                    )

        # Print metrics
        if (i + 1) % print_freq == 0:
            batch_time = time.time() - start_time
            print(
                f"Epoch: {epoch}, Batch: {i+1}/{len(data_loader)}, "
                f"Loss: {total_loss/(i+1):.4f}, "
                f"Time: {batch_time:.2f}s"
            )

    # Calculate average losses
    num_batches = len(data_loader)
    avg_loss = total_loss / num_batches
    avg_loss_classifier = loss_classifier / num_batches
    avg_loss_box_reg = loss_box_reg / num_batches
    avg_loss_objectness = loss_objectness / num_batches
    avg_loss_rpn_box_reg = loss_rpn_box_reg / num_batches

    losses_dict = {
        "loss": avg_loss,
        "loss_classifier": avg_loss_classifier,
        "loss_box_reg": avg_loss_box_reg,
        "loss_objectness": avg_loss_objectness,
        "loss_rpn_box_reg": avg_loss_rpn_box_reg,
    }

    return losses_dict, all_preds, all_targets


def evaluate(model, data_loader, device):
    """
    Evaluate the model on the given dataset.

    Args:
        model (torch.nn.Module): The model to evaluate
        data_loader (torch.utils.data.DataLoader): DataLoader with validation data
        device (torch.device): Device to use for evaluation

    Returns:
        tuple: A tuple containing (all_predictions, all_targets)
    """
    model.eval()

    # Track predictions for evaluation
    all_preds = []
    all_targets = []

    for images, targets in tqdm(data_loader, desc="Evaluating"):
        # Move to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Perform prediction
        with torch.no_grad():
            preds = model(images)

        # Store predictions and targets
        for i, pred in enumerate(preds):
            if len(pred["boxes"]) > 0:
                all_preds.append(
                    {
                        "image_id": targets[i]["image_id"].item(),
                        "boxes": pred["boxes"].cpu(),
                        "labels": pred["labels"].cpu(),
                        "scores": pred["scores"].cpu(),
                    }
                )
                all_targets.append(
                    {
                        "image_id": targets[i]["image_id"].item(),
                        "boxes": targets[i]["boxes"].cpu(),
                        "labels": targets[i]["labels"].cpu(),
                    }
                )

    return all_preds, all_targets


def predict_test_set(model, test_loader, device):
    """
    Generate predictions for the test set.

    Args:
        model (torch.nn.Module): The trained model to use
        test_loader (torch.utils.data.DataLoader): DataLoader with test data
        device (torch.device): Device to use for inference

    Returns:
        list: List of predictions in COCO format
    """
    model.eval()
    predictions = []

    for batch in tqdm(test_loader, desc="Predicting on test set"):
        images = batch["image"]
        image_ids = batch["image_id"]

        # Move images to device
        images = list(image.to(device) for image in images)

        # Perform prediction
        with torch.no_grad():
            outputs = model(images)

        # Store predictions
        for i, output in enumerate(outputs):
            # Apply improved score threshold for better precision
            keep = output["scores"] > 0.5

            # Extract predictions
            boxes = output["boxes"][keep].cpu().numpy()
            labels = output["labels"][keep].cpu().numpy()
            scores = output["scores"][keep].cpu().numpy()

            # Add to predictions list
            for j in range(len(boxes)):
                # Convert box to COCO format [x, y, width, height]
                x1, y1, x2, y2 = boxes[j]
                width = x2 - x1
                height = y2 - y1

                predictions.append(
                    {
                        "image_id": int(image_ids[i]),
                        "bbox": [float(x1), float(y1), float(width), float(height)],
                        "score": float(scores[j]),
                        "category_id": int(labels[j]),
                    }
                )

    return predictions
