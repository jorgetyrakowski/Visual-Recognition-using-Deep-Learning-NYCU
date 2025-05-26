# Ultimo intento/pipeline_scripts/config_and_utils.py
"""
Configuration and Core Utility Functions for the GWD YOLOv11x Pipeline.

This module centralizes:
- File paths and directory configurations for Kaggle environment.
- Model and training hyperparameters.
- TTA (Test-Time Augmentation) and WBF (Weighted Boxes Fusion) parameters.
- Core, reusable functions for seeding, image transformation (TTA),
  bounding box rotation, YOLOv11x inference on single images/arrays,
  and WBF application.
"""

import os
import pandas as pd
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import time # For unique filenames if ever needed, though TTA now avoids temp files
from pathlib import Path

# ==================== FILE & DIRECTORY PATHS (Kaggle Environment) ====================
KAGGLE_INPUT_DIR = "/kaggle/input"
GWD_DATASET_PATH = os.path.join(KAGGLE_INPUT_DIR, "global-wheat-detection")

# Assumes a Kaggle dataset named "gw11-weights" contains:
# - best_fold0.pt, ..., best_fold4.pt (trained K-Fold models)
# - yolov11x.pt (base COCO pre-trained model for PL re-training)
MODEL_WEIGHTS_KAGGLE_DATASET_NAME = "gw11-weights" # User's dataset name on Kaggle
MODEL_WEIGHTS_KAGGLE_PATH = os.path.join(KAGGLE_INPUT_DIR, MODEL_WEIGHTS_KAGGLE_DATASET_NAME)

# Paths to the 5 pre-trained K-Fold YOLOv11x models
# These are expected to be in the MODEL_WEIGHTS_KAGGLE_PATH dataset
MODEL_PATHS_KFold = [
    os.path.join(MODEL_WEIGHTS_KAGGLE_PATH, "best_fold0.pt"),
    os.path.join(MODEL_WEIGHTS_KAGGLE_PATH, "best_fold1.pt"),
    os.path.join(MODEL_WEIGHTS_KAGGLE_PATH, "best_fold2.pt"),
    os.path.join(MODEL_WEIGHTS_KAGGLE_PATH, "best_fold3.pt"),
    os.path.join(MODEL_WEIGHTS_KAGGLE_PATH, "best_fold4.pt")
]
# Base YOLOv11x model for PL re-training (COCO pre-trained)
BASE_YOLOV11x_FOR_PL_RETRAIN = os.path.join(MODEL_WEIGHTS_KAGGLE_PATH, "yolov11x.pt") 

# Original competition data paths
ORIGINAL_TRAIN_CSV_PATH = os.path.join(GWD_DATASET_PATH, "train.csv")
ORIGINAL_TRAIN_IMAGES_DIR_PATH = os.path.join(GWD_DATASET_PATH, "train")
TEST_IMAGES_DIR_PATH = os.path.join(GWD_DATASET_PATH, "test")
SAMPLE_SUBMISSION_CSV_PATH = os.path.join(GWD_DATASET_PATH, "sample_submission.csv")

# Working directory paths for generated data (within /kaggle/working/)
KAGGLE_WORKING_DIR = Path("/kaggle/working")

# Pseudo-Labeling (PL) Dataset Configuration
PL_DATA_BASE_DIR = KAGGLE_WORKING_DIR / "pseudo_labeled_data"
PL_IMAGES_TRAIN_SUBDIR = PL_DATA_BASE_DIR / "images" / "train"
PL_LABELS_TRAIN_SUBDIR = PL_DATA_BASE_DIR / "labels" / "train"
PL_IMAGES_VAL_SUBDIR = PL_DATA_BASE_DIR / "images" / "val"
PL_LABELS_VAL_SUBDIR = PL_DATA_BASE_DIR / "labels" / "val"
PL_DATASET_YAML_PATH = PL_DATA_BASE_DIR / "dataset_pl.yaml" # YAML for PL re-training

# PL Re-training Output Configuration
PL_RETRAIN_OUTPUT_PROJECT_DIR = KAGGLE_WORKING_DIR / "runs" / "detect" 
PL_RETRAIN_RUN_NAME = "pseudo_labeled_wheat_model"
# Path to the best model produced by PL re-training
FINAL_MODEL_AFTER_PL_PATH = PL_RETRAIN_OUTPUT_PROJECT_DIR / PL_RETRAIN_RUN_NAME / "weights" / "best.pt"

# Final Submission File Path
SUBMISSION_CSV_OUTPUT_PATH = KAGGLE_WORKING_DIR / "submission.csv"

# Path for saving optimized parameters from OOF (e.g., as a JSON file)
OPTIMIZED_PARAMS_JSON_PATH = KAGGLE_WORKING_DIR / "optimized_oof_params.json"


# ==================== GENERAL SETTINGS ====================
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 1024 # Input image size for models (height and width)

# ==================== TTA & WBF PARAMETERS ====================
# These WBF parameters are adopted from successful public notebooks and kept fixed.
WBF_IOU_THR = 0.6
WBF_SKIP_BOX_THR = 0.43 
# Uniform weights for ensembling the 5 K-Fold models during Pseudo-Label generation.
# If fewer models are valid, this list will be sliced.
ENSEMBLE_MODEL_TTA_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0] 

# ==================== OOF & PL PARAMETERS ====================
# Low confidence threshold for initial predictions during TTA passes in OOF and PL stages.
# This allows WBF to consider more boxes.
INITIAL_CONF_THR_FOR_TTA_PASSES = 0.05 

# Placeholder for parameters optimized via OOF.
# 'conf_threshold' is the final score threshold applied *after* TTA and WBF.
# This dictionary will be updated by the OOF optimization step.
OPTIMIZED_POSTPROCESSING_PARAMS = { 
    'final_score_threshold': 0.4, # Default/Fallback if OOF fails. This is optimized.
    'wbf_iou_threshold': WBF_IOU_THR,     # Fixed for WBF.
    'wbf_skip_box_threshold': WBF_SKIP_BOX_THR, # Fixed for WBF.
    'oof_map_score': 0.0 # To store the mAP achieved on OOF set with these params.
}

# ==================== PL RE-TRAINING PARAMETERS ====================
PL_RETRAIN_NUM_EPOCHS = 15
PL_RETRAIN_BATCH_SZ = 2 
PL_RETRAIN_INITIAL_LR = 0.001
PL_RETRAIN_FINAL_LR_FACTOR = 0.01 # lrf = lr0 * PL_RETRAIN_FINAL_LR_FACTOR
PL_RETRAIN_WARMUP_EPOCHS = 1
PL_RETRAIN_PATIENCE_EARLY_STOP = 10 # Early stopping patience
PL_RETRAIN_OPTIMIZER = 'AdamW'
PL_RETRAIN_AUGMENT_FLAG = True # Enable standard Ultralytics augmentations
PL_RETRAIN_MIXUP_FRACTION = 0.1
PL_RETRAIN_COPYPASTE_FRACTION = 0.1

# ==================== FINAL INFERENCE SETTINGS ====================
# Low confidence for TTA passes before WBF and final thresholding during final submission generation.
FINAL_INFERENCE_INITIAL_CONF_THR_TTA = 0.05 


# ==================== CORE UTILITY FUNCTIONS ====================

def set_seed_all(seed_value=SEED):
    """Sets random seeds for Python, NumPy, and PyTorch for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # For multi-GPU.
        # Potentially add for cudNN determinism, though can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False 
    print(f"Seed set to {seed_value} for random, numpy, and torch.")

set_seed_all() # Set seed at the beginning of script execution

def predict_on_numpy_image(img_array_bgr, model_instance, 
                           conf_thr=0.25, iou_thr_nms=0.45, use_augment=False):
    """
    Performs YOLO inference on a NumPy image array.
    Args:
        img_array_bgr: Image as a NumPy array (BGR format), assumed to be already resized to IMAGE_SIZE.
        model_instance: Loaded YOLO model instance.
        conf_thr: Confidence threshold for predictions.
        iou_thr_nms: IoU threshold for NMS within model.predict().
        use_augment: Boolean, whether to use model.predict(augment=True).
    Returns:
        boxes_abs_xyxy: NumPy array of detected boxes [x1, y1, x2, y2] in absolute pixel coordinates.
        scores: NumPy array of confidence scores.
    """
    try:
        results = model_instance.predict(
            source=img_array_bgr,
            conf=conf_thr,
            iou=iou_thr_nms,
            augment=use_augment,
            verbose=False,
            save=False, # Do not save prediction images/labels to disk
            device=DEVICE,
            imgsz=IMAGE_SIZE # Ensure model predicts on consistent size
        )
        # result = results[0] # model.predict can return a list of Results objects
        if not results or not results[0].boxes:
            return np.array([]).reshape(0, 4), np.array([])
        
        # .xyxy provides absolute pixel coordinates relative to the input image size (IMAGE_SIZE x IMAGE_SIZE)
        boxes_abs_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = results[0].boxes.conf.cpu().numpy().astype(np.float32)
        return boxes_abs_xyxy, scores
        
    except Exception as e:
        img_shape_info = img_array_bgr.shape if isinstance(img_array_bgr, np.ndarray) else 'Unknown'
        print(f"Error in predict_on_numpy_image for image shape {img_shape_info}: {e}")
        return np.array([]).reshape(0, 4), np.array([])

def apply_tta_rotation(image_np_bgr_resized, tta_rotation_idx):
    """
    Applies geometric rotation TTA to a resized NumPy image.
    Args:
        image_np_bgr_resized: Image as NumPy array (BGR), resized to IMAGE_SIZE.
        tta_rotation_idx: 0 for 90° clockwise, 1 for 180°, 2 for 270° clockwise (90° CCW), 3 for original.
    Returns:
        Rotated NumPy image array.
    """
    if tta_rotation_idx == 0: return cv2.rotate(image_np_bgr_resized, cv2.ROTATE_90_CLOCKWISE)
    if tta_rotation_idx == 1: return cv2.rotate(image_np_bgr_resized, cv2.ROTATE_180)
    if tta_rotation_idx == 2: return cv2.rotate(image_np_bgr_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if tta_rotation_idx == 3: return image_np_bgr_resized.copy() # Original
    return image_np_bgr_resized.copy() # Fallback, should not be reached if tta_rotation_idx is 0-3

def de_rotate_boxes_90ccw_multiple(boxes_abs_xyxy, original_img_width, original_img_height, num_90deg_ccw_rotations):
    """
    De-rotates bounding boxes that were predicted on a TTA-rotated image.
    This function applies `num_90deg_ccw_rotations` of 90-degree counter-clockwise
    transformations to the box coordinates.
    Args:
        boxes_abs_xyxy: Absolute pixel coordinates [N, 4] in [x1, y1, x2, y2] format,
                        relative to the TTA-transformed image dimensions.
        original_img_width: Width of the image *before* it was TTA-rotated for prediction.
        original_img_height: Height of the image *before* it was TTA-rotated for prediction.
        num_90deg_ccw_rotations: Number of 90° CCW rotations to apply to boxes (0, 1, 2, or 3).
    Returns:
        De-rotated boxes in [x1, y1, x2, y2] absolute pixel format, relative to original image dimensions.
    """
    if len(boxes_abs_xyxy) == 0 or num_90deg_ccw_rotations == 0:
        return boxes_abs_xyxy.astype(np.float32)

    current_boxes = boxes_abs_xyxy.astype(np.float32)
    # Dimensions of the coordinate system the boxes are currently in.
    # After each CCW rotation, width and height swap.
    current_coord_sys_w, current_coord_sys_h = original_img_width, original_img_height
    if num_90deg_ccw_rotations % 2 == 1: # 1 or 3 rotations means dimensions are swapped
        current_coord_sys_w, current_coord_sys_h = original_img_height, original_img_width
        
    for _ in range(num_90deg_ccw_rotations):
        de_rotated_boxes_temp = []
        for box in current_boxes:
            x1, y1, x2, y2 = box
            # Apply one 90deg CCW rotation to box coordinates:
            # new_x1 = old_y1
            # new_y1 = H_current_coord_sys - 1 - old_x2 (using H of the system *before* this CCW turn)
            # new_x2 = old_y2
            # new_y2 = H_current_coord_sys - 1 - old_x1
            nx1 = y1
            ny1 = (current_coord_sys_h - 1) - x2 
            nx2 = y2
            ny2 = (current_coord_sys_h - 1) - x1
            
            de_rotated_boxes_temp.append([min(nx1,nx2), min(ny1,ny2), max(nx1,nx2), max(ny1,ny2)])
        current_boxes = np.array(de_rotated_boxes_temp).astype(np.float32)
        # After one CCW rotation, the coordinate system's effective W and H swap
        current_coord_sys_w, current_coord_sys_h = current_coord_sys_h, current_coord_sys_w 
        
    return current_boxes

def apply_wbf(list_of_boxes_abs_xyxy, list_of_scores, 
              image_ref_width=IMAGE_SIZE, image_ref_height=IMAGE_SIZE, 
              wbf_iou_threshold=WBF_IOU_THR, wbf_skip_box_threshold=WBF_SKIP_BOX_THR, 
              ensemble_weights_list=None):
    """Applies Weighted Boxes Fusion to a list of box sets and score sets."""
    if not any(len(b) > 0 for b in list_of_boxes_abs_xyxy):
        return np.array([]).reshape(0,4), np.array([]), np.array([])

    norm_factor = np.array([image_ref_width, image_ref_height, image_ref_width, image_ref_height], dtype=np.float32)
    
    # Filter out empty box lists and corresponding scores before normalization
    # Ensure boxes are float32 for division
    active_boxes_lists = [b.astype(np.float32) for b in list_of_boxes_abs_xyxy if len(b) > 0]
    active_scores_lists = [s.astype(np.float32) for s, b in zip(list_of_scores, list_of_boxes_abs_xyxy) if len(b) > 0]

    if not active_boxes_lists: # If all TTA passes had no detections
        return np.array([]).reshape(0,4), np.array([]), np.array([])

    normalized_boxes_list = [b / norm_factor for b in active_boxes_lists]
    # All detections are class 0 (wheat)
    class_labels_list = [np.zeros(len(s), dtype=int) for s in active_scores_lists] 
    
    effective_ensemble_weights = None
    if ensemble_weights_list is not None:
        if len(ensemble_weights_list) == len(normalized_boxes_list):
            effective_ensemble_weights = ensemble_weights_list
        else:
            print(f"Warning: Length of ensemble_weights_list ({len(ensemble_weights_list)}) "
                  f"does not match number of non-empty box lists ({len(normalized_boxes_list)}). "
                  f"Using uniform weights for WBF.")
    
    fused_boxes_norm, fused_scores, fused_labels = weighted_boxes_fusion(
        normalized_boxes_list, active_scores_lists, class_labels_list,
        weights=effective_ensemble_weights, 
        iou_thr=wbf_iou_threshold, 
        skip_box_thr=wbf_skip_box_threshold
    )
    
    # De-normalize boxes back to absolute pixel coordinates
    fused_boxes_abs_xyxy = fused_boxes_norm * norm_factor
    return fused_boxes_abs_xyxy.astype(np.float32), fused_scores.astype(np.float32), fused_labels


def predict_with_tta_for_single_model(
    image_path_or_np_array, model_instance, 
    initial_conf_thr=INITIAL_CONF_THR_FOR_TTA_PASSES, 
    nms_iou_thr_in_predict=0.6, # NMS IoU for individual model.predict calls
    apply_rotation_tta=True, apply_augment_flag_tta=True
):
    """
    Performs TTA for a single model on a single image.
    Returns fused boxes (absolute xyxy) and scores, BEFORE final confidence thresholding.
    """
    if isinstance(image_path_or_np_array, str): 
        img_bgr_full_size = cv2.imread(image_path_or_np_array)
        if img_bgr_full_size is None:
            print(f"Error: Could not read image {image_path_or_np_array}")
            return np.array([]).reshape(0,4), np.array([])
    else: 
        img_bgr_full_size = image_path_or_np_array.copy()

    # Resize once at the beginning to the model's expected input size
    img_bgr_resized_for_model = cv2.resize(img_bgr_full_size, (IMAGE_SIZE, IMAGE_SIZE))
    
    all_tta_pass_boxes_abs = [] 
    all_tta_pass_scores = []

    # 1. Rotational TTA (4 passes: Original, Rot90CW, Rot180, Rot270CW/90CCW)
    if apply_rotation_tta:
        for tta_rot_idx in range(4): 
            # Apply rotation to the *resized* image
            img_tta_rotated_np = apply_tta_rotation(img_bgr_resized_for_model, tta_rot_idx)
            
            # Predict on the TTA-transformed NumPy array
            boxes_pred_abs_on_tta_img, scores_pred_on_tta_img = predict_on_numpy_image(
                img_tta_rotated_np, model_instance, initial_conf_thr, nms_iou_thr_in_predict, use_augment=False
            )
            
            if len(boxes_pred_abs_on_tta_img) > 0:
                # De-rotate boxes back to original image orientation (of img_bgr_resized)
                num_ccw_rotations_to_revert = 0
                if tta_rot_idx == 0: num_ccw_rotations_to_revert = 1 # Rot90CW on image -> 1x Rot90CCW on boxes
                elif tta_rot_idx == 1: num_ccw_rotations_to_revert = 2 # Rot180 on image -> 2x Rot90CCW on boxes
                elif tta_rot_idx == 2: num_ccw_rotations_to_revert = 3 # Rot270CW on image -> 3x Rot90CCW on boxes
                # if tta_rot_idx == 3 (Original), num_ccw_rotations_to_revert = 0
                
                boxes_derotated_abs = de_rotate_boxes_90ccw_multiple(
                    boxes_pred_abs_on_tta_img, IMAGE_SIZE, IMAGE_SIZE, num_ccw_rotations_to_revert
                )
                all_tta_pass_boxes_abs.append(boxes_derotated_abs)
                all_tta_pass_scores.append(scores_pred_on_tta_img)

    # 2. YOLO's built-in `augment=True` TTA (on original resized image)
    if apply_augment_flag_tta:
        boxes_aug_abs, scores_aug = predict_on_numpy_image(
            img_bgr_resized_for_model, model_instance, initial_conf_thr, nms_iou_thr_in_predict, use_augment=True
        )
        if len(boxes_aug_abs) > 0:
            all_tta_pass_boxes_abs.append(boxes_aug_abs)
            all_tta_pass_scores.append(scores_aug)

    # 3. If no TTA was used at all, predict on the plain resized image
    if not apply_rotation_tta and not apply_augment_flag_tta:
        boxes_plain_abs, scores_plain = predict_on_numpy_image(
            img_bgr_resized_for_model, model_instance, initial_conf_thr, nms_iou_thr_in_predict, use_augment=False
        )
        if len(boxes_plain_abs) > 0:
            all_tta_pass_boxes_abs.append(boxes_plain_abs)
            all_tta_pass_scores.append(scores_plain)
        
    if not all_tta_pass_boxes_abs: # If no TTA passes yielded any boxes at all
        return np.array([]).reshape(0,4), np.array([])

    # 4. Apply WBF to all collected TTA predictions for this single model
    # WBF parameters (WBF_IOU_THR, WBF_SKIP_BOX_THR) are taken from global config
    fused_boxes_abs, fused_scores, _ = apply_wbf(
        all_tta_pass_boxes_abs, all_tta_pass_scores,
        image_ref_width=IMAGE_SIZE, image_ref_height=IMAGE_SIZE
        # Default WBF_IOU_THR and WBF_SKIP_BOX_THR from global config will be used by apply_wbf
    )
    # Return boxes as int32 for compatibility if downstream functions expect int, but float is generally better for precision.
    # The user's original script cast to int32 here.
    return fused_boxes_abs.astype(np.int32), fused_scores 

# Placeholder for the rest of the script (OOF, PL, Main pipeline, etc.)
# ...
