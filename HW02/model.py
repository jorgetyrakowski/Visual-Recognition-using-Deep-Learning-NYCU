import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_improved_model(num_classes):
    """
    Create an improved Faster R-CNN model with ResNet50-FPN-v2 backbone
    with optimized detection parameters for the SVHN dataset.

    Args:
        num_classes (int): Number of classes (including background)

    Returns:
        torch.nn.Module: The initialized Faster R-CNN model
    """
    # Load pre-trained weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT

    # Create model with pre-trained weights and modified parameters
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        # Improved parameters for SVHN
        rpn_pre_nms_top_n_train=3000,  # Default: 2000 - Consider more region proposals
        rpn_post_nms_top_n_train=1500,  # Default: 1000 - Keep more proposals after NMS
        rpn_pre_nms_top_n_test=1500,  # Default: 1000
        rpn_post_nms_top_n_test=1000,  # Default: 1000
        rpn_nms_thresh=0.75,  # Default: 0.7 - Increased to retain more potential digits
        rpn_score_thresh=0.0,  # Default: 0.0
        # Customize ROI parameters
        box_score_thresh=0.4,  # Default: 0.05 - Higher threshold for precision
        box_nms_thresh=0.45,  # Default: 0.5 - More aggressive NMS
        box_detections_per_img=100,  # Default: 100
    )

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_mobilenet_model(num_classes):
    """
    Create a Faster R-CNN model with MobileNetV3-Large-FPN backbone.

    Args:
        num_classes (int): Number of classes (including background)

    Returns:
        torch.nn.Module: The initialized Faster R-CNN model
    """
    # Create model with pre-trained weights
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
        rpn_score_thresh=0.0,
        box_score_thresh=0.4,
        box_nms_thresh=0.45,
    )

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
