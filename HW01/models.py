#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model architectures for image classification.

This module contains model definitions including enhanced ResNeXt101
with channel attention mechanism.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNeXt101_32X8D_Weights


class EnhancedResNeXt101(nn.Module):
    """
    Enhanced ResNeXt101_32X8D with channel attention mechanism.

    This model extends the base ResNeXt101_32X8D architecture with
    Squeeze-and-Excitation style channel attention for improved performance.
    """

    def __init__(self, num_classes=100, dropout_prob=0.5):
        """
        Initialize the model.

        Args:
            num_classes (int): Number of output classes
            dropout_prob (float): Dropout probability for regularization
        """
        super(EnhancedResNeXt101, self).__init__()
        # Load pretrained ResNeXt101_32X8D_Weights
        base_model = models.resnext101_32x8d(
            weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2)

        # Calculate model size
        total_params = sum(p.numel() for p in base_model.parameters())
        print(f"Base ResNeXt101_32X8D parameters: {total_params:,} "
              f"({total_params/1e6:.2f}M)")

        # Remove the final FC layer
        self.features = nn.Sequential(*list(base_model.children())[:-2])

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Channel attention module (Squeeze-and-Excitation style)
        self.channel_attention = nn.Sequential(
            nn.Linear(2048, 2048 // 16),
            nn.ReLU(inplace=True),
            nn.Linear(2048 // 16, 2048),
            nn.Sigmoid()
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(2048, num_classes)
        )

        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Enhanced ResNeXt101_32X8D parameters: {total_params:,} "
              f"({total_params/1e6:.2f}M)")

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights for the attention and classifier modules."""
        for m in self.channel_attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]

        Returns:
            torch.Tensor: Output tensor of shape [B, num_classes]
        """
        # Feature extraction
        x = self.features(x)

        # Global average pooling
        x_pool = self.avg_pool(x).view(x.size(0), -1)

        # Apply channel attention
        att = self.channel_attention(x_pool)
        x_att = x_pool * att

        # Classification
        out = self.classifier(x_att)

        return out


def create_model(model_type="resnext101", num_classes=100, dropout_prob=0.5):
    """
    Factory function to create a model.

    Args:
        model_type (str): Model architecture to use
        num_classes (int): Number of output classes
        dropout_prob (float): Dropout probability

    Returns:
        nn.Module: Instantiated model
    """
    if model_type == "resnext101":
        return EnhancedResNeXt101(
            num_classes=num_classes,
            dropout_prob=dropout_prob
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
