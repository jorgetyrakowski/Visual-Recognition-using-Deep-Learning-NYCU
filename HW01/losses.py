"""
Loss functions for training image classification models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance."""

    def __init__(self, gamma=2, alpha=None, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Apply label smoothing
        num_classes = inputs.size(-1)
        if self.label_smoothing > 0:
            smoothed_targets = torch.zeros_like(inputs).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            smoothed_targets = (
                smoothed_targets * (1 - self.label_smoothing)
                + self.label_smoothing / num_classes
            )
            log_probs = F.log_softmax(inputs, dim=1)
            loss = -(smoothed_targets * log_probs).sum(dim=1)
        else:
            # Standard cross entropy
            ce_loss = F.cross_entropy(inputs, targets, reduction="none")
            loss = ce_loss

        # Apply focal scaling
        pt = torch.exp(-loss)
        focal_loss = ((1 - pt) ** self.gamma) * loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            focal_loss = alpha_weight * focal_loss

        return focal_loss.mean()


def get_class_weights(dataset):
    """
    Calculate class weights to handle imbalanced data.

    Args:
        dataset: Dataset with get_class_distribution method

    Returns:
        torch.FloatTensor: Tensor of class weights
    """
    class_counts = dataset.get_class_distribution()
    max_count = max(class_counts.values())
    weights = torch.zeros(max(class_counts.keys()) + 1)

    # Compute inverse square root weights
    for label, count in class_counts.items():
        weights[label] = torch.sqrt(torch.tensor(max_count / count))

    # Normalize weights
    weights = weights / weights.mean()
    return weights
