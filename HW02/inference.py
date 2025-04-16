import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path configuration
BASE_DIR = os.path.join(os.getcwd(), "nycu-hw2-data")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Model configuration
MODEL_PATH = "models/model_epoch_6.pth"  # Change this to use a different epoch
MODEL_NAME = "resnet50_fpn_v2"
NUM_CLASSES = 11  # background + 10 digits


def get_model(num_classes):
    """
    Create a Faster R-CNN model with ResNet50-FPN-v2 backbone
    """
    # Create model without pre-trained weights
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def load_test_images():
    """
    Load all test images and their IDs
    """
    test_images = []
    for img_name in sorted(os.listdir(TEST_DIR)):
        if img_name.endswith(".png"):
            image_id = int(os.path.splitext(img_name)[0])
            img_path = os.path.join(TEST_DIR, img_name)
            test_images.append((image_id, img_path))

    return test_images


def predict_single_image(model, image_path, device, transform):
    """
    Make predictions for a single image
    """
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get predictions
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)

    return predictions[0]


def predict_test_set(model, test_images, device):
    """
    Generate predictions for the test set
    """
    # Transform for test images
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Initialize list to store predictions
    predictions = []

    # Process each image
    for image_id, image_path in tqdm(test_images, desc="Predicting on test set"):
        # Get prediction
        output = predict_single_image(model, image_path, device, transform)

        # Apply score threshold
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
                    "image_id": int(image_id),
                    "bbox": [float(x1), float(y1), float(width), float(height)],
                    "score": float(scores[j]),
                    "category_id": int(labels[j]),
                }
            )

    return predictions


def create_task2_predictions(predictions):
    """
    Generate task 2 predictions (whole number recognition) based on task 1 predictions
    """
    # Group predictions by image_id
    predictions_by_image = defaultdict(list)
    for pred in predictions:
        predictions_by_image[pred["image_id"]].append(pred)

    # Process each image
    results = []
    for img_id, preds in predictions_by_image.items():
        # Filter predictions by score
        preds = [p for p in preds if p["score"] > 0.5]

        if len(preds) == 0:
            # No digits detected
            results.append({"image_id": img_id, "pred_label": -1})
            continue

        # Sort digits from left to right based on x-coordinate of bbox
        preds.sort(key=lambda x: x["bbox"][0])

        # Combine digits to form the whole number
        # Convert category_id (1-10) to digit (0-9)
        digits = [str(p["category_id"] - 1) for p in preds]
        num = int("".join(digits))

        results.append({"image_id": img_id, "pred_label": num})

    return results


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load model
    model = get_model(NUM_CLASSES)
    model.to(device)

    # Load state dict
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Loaded model from {MODEL_PATH}")

    # Load test images
    test_images = load_test_images()
    print(f"Found {len(test_images)} test images")

    # Generate test predictions (Task 1)
    print("Generating test predictions...")
    test_preds = predict_test_set(model, test_images, device)

    # Save Task 1 predictions
    task1_output = f"{MODEL_NAME}_pred.json"
    with open(task1_output, "w") as f:
        json.dump(test_preds, f)
    print(f"Task 1 predictions saved to {task1_output}")

    # Generate Task 2 predictions
    print("Generating Task 2 predictions...")
    task2_preds = create_task2_predictions(test_preds)

    # Save Task 2 predictions
    task2_output = f"{MODEL_NAME}_pred.csv"
    df = pd.DataFrame(task2_preds)
    df.to_csv(task2_output, index=False)
    print(f"Task 2 predictions saved to {task2_output}")


if __name__ == "__main__":
    main()
