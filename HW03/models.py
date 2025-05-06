#!/usr/bin/env python3
"""
models.py - Model architecture definitions for Cell Instance Segmentation
=======================================================================
This module contains the implementation of the Cascade Mask R-CNN architecture
and custom loss functions used for cell segmentation tasks.

Key components:
- CascadeMaskRCNN class: Multi-stage instance segmentation model
- DiceLoss: Loss function optimizing segmentation overlap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNN


# Global configuration
NUM_CLASSES = 5  # background + 4 cell classes


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation masks.
    
    This loss function directly optimizes the overlap between predicted 
    and ground truth masks, which is closely aligned with IoU metrics used
    in evaluation. It inherently handles class imbalance between foreground 
    and background pixels.
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Calculate Dice Loss between predicted masks and target masks.
        
        Args:
            pred: (N, 1, H, W) predicted mask logits
            target: (N, H, W) ground truth binary masks
            
        Returns:
            dice_loss: Scalar tensor with the dice loss value
        """
        pred = torch.sigmoid(pred)  # Convert logits to probabilities
        
        # Flatten the tensors
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1).float()
        
        # Calculate Dice coefficient
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        # Dice loss
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss


class CascadeMaskRCNN(nn.Module):
    """
    Cascade Mask R-CNN with multiple detection heads at different IoU thresholds.
    
    This advanced architecture addresses the mismatch between IoU thresholds used
    for training and evaluation. It employs a sequence of detectors trained with 
    increasing IoU thresholds, where each stage refines the predictions of the
    previous stage.
    
    Key improvements:
    - Progressive refinement of object detections
    - Multi-stage box regression with increasing quality requirements
    - Better handling of challenging cases (overlapping cells, irregular shapes)
    
    Args:
        num_classes (int): Number of classes (including background)
        backbone (str): Backbone architecture ('resnet50', 'resnet101', or 'resnet152')
    """
    def __init__(self, num_classes, backbone="resnet50"):
        super(CascadeMaskRCNN, self).__init__()
        
        # Initialize base model based on specified backbone
        if backbone == "resnet50":
            self.base_model = maskrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
                box_detections_per_img=2000,
                rpn_post_nms_top_n_train=2000,
                rpn_post_nms_top_n_test=2000,
                trainable_backbone_layers=5,
            )
        elif backbone == "resnet101":
            # Create ResNet101 backbone with FPN
            backbone_model = resnet_fpn_backbone(
                'resnet101',
                weights=ResNet101_Weights.IMAGENET1K_V2,
                trainable_layers=5
            )
            
            # Create Mask R-CNN model with the custom backbone
            self.base_model = MaskRCNN(
                backbone_model,
                num_classes=num_classes,
                box_detections_per_img=2000,
                rpn_post_nms_top_n_train=2000,
                rpn_post_nms_top_n_test=2000
            )
        
        elif backbone == "resnet152":
            # Create ResNet152 backbone with FPN
            backbone_model = resnet_fpn_backbone(
                'resnet152',
                weights=ResNet152_Weights.IMAGENET1K_V2,
                trainable_layers=5
            )
            
            # Create Mask R-CNN model with the custom backbone
            self.base_model = MaskRCNN(
                backbone_model,
                num_classes=num_classes,
                box_detections_per_img=2000,
                rpn_post_nms_top_n_train=2000,
                rpn_post_nms_top_n_test=2000
            )
        
        # Modify the anchor generator for better cell detection
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        self.base_model.rpn.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes, 
            aspect_ratios=aspect_ratios
        )
        
        # Increase NMS threshold for better recall
        self.base_model.rpn.nms_thresh = 0.7
        
        # Cascade stages with increasing IoU thresholds
        self.iou_thresholds = [0.5, 0.6, 0.7]
        
        # Shared feature extractor and RPN
        self.backbone = self.base_model.backbone
        self.rpn = self.base_model.rpn
        self.roi_heads = self.base_model.roi_heads
        self.transform = self.base_model.transform
        
        # Replace box predictor for each cascade stage
        in_features = self.roi_heads.box_predictor.cls_score.in_features
        self.box_predictors = nn.ModuleList([
            FastRCNNPredictor(in_features, num_classes) for _ in range(len(self.iou_thresholds))
        ])
        
        # Replace mask predictor with enhanced version
        in_channels = self.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.mask_predictor = MaskRCNNPredictor(in_channels, hidden_layer, num_classes)
        
        # Set min size for handling smaller images
        self.transform.min_size = (512,)
        self.transform.max_size = 1333
        
        # Replace the original predictor with ours
        self.roi_heads.box_predictor = self.box_predictors[0]  # Default is first stage
        self.roi_heads.mask_predictor = self.mask_predictor
        
        # Dice Loss for mask refinement
        self.dice_loss = DiceLoss()
        
    def forward(self, images, targets=None):
        """
        Forward pass with cascade refinement.
        
        Args:
            images (List[Tensor]): Input images
            targets (List[Dict], optional): Ground truth boxes, labels and masks
            
        Returns:
            During training: Dict[str, Tensor] containing the losses
            During inference: List[Dict[str, Tensor]] with detection results
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
            
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
            
        # Transform images
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        
        if targets is None:
            # Inference mode
            proposals, _ = self.rpn(images, features, targets)
            
            # Process through cascade stages
            detections = None
            for i, box_predictor in enumerate(self.box_predictors):
                # Replace predictor temporarily
                self.roi_heads.box_predictor = box_predictor
                
                if i == 0:
                    # First stage uses RPN proposals
                    detections, _ = self.roi_heads(features, proposals, images.image_sizes, targets)
                else:
                    # Subsequent stages use refined boxes
                    self.roi_heads.proposal_matcher.iou_threshold = self.iou_thresholds[i]
                    detections, _ = self.roi_heads(features, [det["boxes"] for det in detections], 
                                                  images.image_sizes, targets)
            
            # Transform back to original image space
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            return detections
        else:
            # Training mode
            proposals, proposal_losses = self.rpn(images, features, targets)
            
            # Process through cascade stages
            all_box_losses = {}
            for i, box_predictor in enumerate(self.box_predictors):
                # Replace predictor temporarily
                self.roi_heads.box_predictor = box_predictor
                self.roi_heads.proposal_matcher.iou_threshold = self.iou_thresholds[i]
                
                # Get predictions and losses
                _, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
                
                # Store box losses with stage prefix
                if i == 0:
                    # Use losses from first stage as base
                    all_losses = detector_losses
                    all_box_losses[f"loss_classifier_s{i}"] = detector_losses["loss_classifier"]
                    all_box_losses[f"loss_box_reg_s{i}"] = detector_losses["loss_box_reg"]
                else:
                    # Add weighted losses from other stages
                    weight = 1.0 / (i + 1)  # Decrease weight for later stages
                    all_box_losses[f"loss_classifier_s{i}"] = detector_losses["loss_classifier"] * weight
                    all_box_losses[f"loss_box_reg_s{i}"] = detector_losses["loss_box_reg"] * weight
                   
            # Consolidate cascade box losses
            all_losses["loss_classifier"] = sum(loss for name, loss in all_box_losses.items() 
                                             if "classifier" in name)
            all_losses["loss_box_reg"] = sum(loss for name, loss in all_box_losses.items() 
                                          if "box_reg" in name)
            
            # Combine with RPN losses
            losses = {}
            losses.update(all_losses)
            losses.update(proposal_losses)
            
            return losses