import os
import json
import torch
import torch.utils.data as data
from collections import defaultdict
from PIL import Image


class SVHNDataset(data.Dataset):
    """
    Dataset class for loading and processing SVHN (Street View House Numbers) data.

    Handles both training/validation data with annotations and test data without labels.
    For training and validation, loads annotations from COCO-format JSON files.
    For test data, loads images without annotations.
    """

    def __init__(self, json_file, img_dir, transform=None, is_test=False):
        """
        Initialize SVHN dataset.

        Args:
            json_file (str): Path to the JSON annotation file (None for test set)
            img_dir (str): Directory containing the images
            transform (callable, optional): Optional transform to be applied to images
            is_test (bool): Whether this is a test dataset (without annotations)
        """
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

        # For test set, just load image paths
        if is_test:
            self.images = [f for f in os.listdir(img_dir) if f.endswith(".png")]
            return

        # Load annotations
        with open(json_file, "r") as f:
            self.dataset = json.load(f)

        # Create image_id to file_name mapping
        self.id_to_filename = {
            img["id"]: img["file_name"] for img in self.dataset["images"]
        }

        # Group annotations by image_id
        self.annotations_by_image = defaultdict(list)
        for anno in self.dataset["annotations"]:
            self.annotations_by_image[anno["image_id"]].append(anno)

        # Get list of all image IDs
        self.image_ids = list(self.id_to_filename.keys())

    def __len__(self):
        """Return the number of images in the dataset."""
        if self.is_test:
            return len(self.images)
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve

        Returns:
            For test data: dict with 'image' and 'image_id'
            For train/val data: tuple of (image, target)
        """
        if self.is_test:
            img_name = self.images[idx]
            image_id = int(os.path.splitext(img_name)[0])
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return {"image": image, "image_id": image_id}

        # Get image ID and filename
        image_id = self.image_ids[idx]
        img_name = self.id_to_filename[image_id]
        img_path = os.path.join(self.img_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Get annotations for this image
        annotations = self.annotations_by_image[image_id]

        # Initialize targets
        boxes = []
        labels = []

        # Process annotations
        for anno in annotations:
            # Extract bounding box
            bbox = anno["bbox"]
            # Convert from [x, y, width, height] to [x1, y1, x2, y2]
            x, y, width, height = bbox
            boxes.append([x, y, x + width, y + height])

            # Extract category - note that we keep the original category_id (1-10)
            # The model will handle mapping internally
            labels.append(anno["category_id"])

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
        }

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, target


def collate_fn(batch):
    """
    Custom collate function for the DataLoader to handle variable-sized images and targets.

    Args:
        batch: A list of tuples (image, target) or dicts

    Returns:
        A tuple of batched images and targets, or a batch of dicts for test data
    """
    if isinstance(batch[0], tuple):
        return tuple(zip(*batch))
    else:
        # For test data
        return batch
