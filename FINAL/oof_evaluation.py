# Ultimo intento/pipeline_scripts/oof_evaluation.py
"""
Out-of-Fold (OOF) Evaluation and Score Threshold Optimization.

This script performs OOF evaluation using the K-Fold models
and optimizes the final confidence score threshold based on the
competition's mAP metric.
"""
import os
import pandas as pd
import numpy as np
import json # For saving/loading optimized params
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import numba

# Import from config_and_utils module (now expected to be in the same directory)
from config_and_utils import (
    ORIGINAL_TRAIN_CSV_PATH, ORIGINAL_TRAIN_IMAGES_DIR_PATH, SEED, IMAGE_SIZE,
    MODEL_PATHS_KFold, DEVICE, YOLO, # YOLO needed if models are re-loaded here
    OOF_CONF_THRESHOLD_LOW, COMPETITION_IOU_THRESHOLDS_NUMBA,
    OPTIMIZED_PARAMS_JSON_PATH, OPTIMIZED_POSTPROCESSING_PARAMS, # Using OPTIMIZED_POSTPROCESSING_PARAMS as default
    predict_with_tta_for_single_model # This is the key function for getting preds
)
# Note: The Numba metric functions are defined below. If they were in config_and_utils, they'd be imported.


# ==================== OOF HELPER FUNCTIONS (METRICS) ====================
# Numba JIT compiled functions for fast metric calculation, adapted from user's script.
# These should be identical to those in config_and_utils if they were also needed there,
# or defined once and imported. For modularity, defining them here if specific to OOF.
# However, these are general metric functions. Let's assume they are defined once
# in config_and_utils.py and imported. If not, they need to be defined here.
# For this example, I'll re-define them here for clarity of this module's dependencies.

@numba.jit(nopython=True)
def _calculate_iou_numba_oof(gt_box_xyxy, pred_box_xyxy):
    """Calculates IoU between two boxes [x1,y1,x2,y2]."""
    ixmin = max(gt_box_xyxy[0], pred_box_xyxy[0])
    iymin = max(gt_box_xyxy[1], pred_box_xyxy[1])
    ixmax = min(gt_box_xyxy[2], pred_box_xyxy[2])
    iymax = min(gt_box_xyxy[3], pred_box_xyxy[3])
    iw = max(ixmax - ixmin, 0.0)
    ih = max(iymax - iymin, 0.0)
    intersection = iw * ih
    gt_area = (gt_box_xyxy[2] - gt_box_xyxy[0]) * (gt_box_xyxy[3] - gt_box_xyxy[1])
    pred_area = (pred_box_xyxy[2] - pred_box_xyxy[0]) * (pred_box_xyxy[3] - pred_box_xyxy[1])
    union = gt_area + pred_area - intersection
    if union < 1e-6: return 0.0
    return intersection / union

@numba.jit(nopython=True)
def _calculate_precision_for_iou_thr_numba_oof(gt_boxes_img_xyxy, pred_boxes_img_xyxy, iou_threshold):
    """Calculates precision for a single image at a single IoU threshold."""
    num_gt = len(gt_boxes_img_xyxy)
    num_pred = len(pred_boxes_img_xyxy)
    
    if num_pred == 0: return 1.0 if num_gt == 0 else 0.0
    if num_gt == 0: return 0.0

    tp = 0
    fp = 0
    gt_already_matched = np.zeros(num_gt, dtype=numba.boolean)

    for i_pred in range(num_pred):
        best_iou_for_this_pred = -1.0
        best_gt_match_idx = -1
        for i_gt in range(num_gt):
            if gt_already_matched[i_gt]:
                continue
            iou = _calculate_iou_numba_oof(gt_boxes_img_xyxy[i_gt], pred_boxes_img_xyxy[i_pred])
            if iou > best_iou_for_this_pred:
                best_iou_for_this_pred = iou
                best_gt_match_idx = i_gt
        
        if best_iou_for_this_pred >= iou_threshold:
            if not gt_already_matched[best_gt_match_idx]:
                tp += 1
                gt_already_matched[best_gt_match_idx] = True
            else:
                fp += 1 
        else:
            fp += 1
            
    if (tp + fp) == 0: return 1.0 
    return tp / (tp + fp)

@numba.jit(nopython=True)
def calculate_image_map_score_numba_oof(gt_boxes_img_xyxy, pred_boxes_img_xyxy, iou_thresholds_list):
    """Calculates competition mAP for a single image."""
    if len(gt_boxes_img_xyxy) == 0:
        return 1.0 if len(pred_boxes_img_xyxy) == 0 else 0.0
    if len(pred_boxes_img_xyxy) == 0:
        return 0.0

    avg_precision = 0.0
    for iou_thr in iou_thresholds_list:
        avg_precision += _calculate_precision_for_iou_thr_numba_oof(gt_boxes_img_xyxy.copy(), pred_boxes_img_xyxy, iou_thr)
    return avg_precision / len(iou_thresholds_list)

# ==================== OOF EVALUATION FUNCTIONS ====================

def create_stratified_kfold_splits_for_oof(train_csv_path=ORIGINAL_TRAIN_CSV_PATH, num_splits=5, random_seed=SEED):
    """
    Creates stratified K-Fold splits of image_ids from train.csv based on box counts per image.
    This is used for defining validation sets for OOF evaluation.
    Args:
        train_csv_path: Path to the original train.csv.
        num_splits: Number of folds.
        random_seed: Seed for reproducibility.
    Returns:
        folds_data: Dict mapping fold_idx to {'train_image_ids': [], 'val_image_ids': []}.
        marking_df: Processed DataFrame with 'x', 'y', 'w', 'h' columns.
    """
    print(f"📁 Creating Stratified {num_splits}-Fold splits for OOF evaluation from {train_csv_path}...")
    marking_df = pd.read_csv(train_csv_path)
    
    # Parse 'bbox' column: "[x,y,w,h]" string to list of floats
    # Ensure robust parsing, e.g. if it's already parsed or format varies slightly
    if isinstance(marking_df['bbox'].iloc[0], str):
        try: # Assuming bbox is like "[x, y, w, h]"
            marking_df['bbox_list'] = marking_df['bbox'].apply(lambda x: json.loads(x))
        except json.JSONDecodeError: # Fallback for simpler string like "x, y, w, h"
            print("Warning: json.loads failed for bbox. Trying string split (less robust).")
            marking_df['bbox_list'] = marking_df['bbox'].apply(lambda x: [float(v.strip()) for v in x.strip('[]').split(',')])
    else: # If 'bbox' column might already contain lists (e.g., if notebook re-run)
        marking_df['bbox_list'] = marking_df['bbox']
        
    temp_bbox_df = pd.DataFrame(marking_df['bbox_list'].tolist(), columns=['x', 'y', 'w', 'h'])
    # Drop original 'bbox' and 'bbox_list' if it exists, then concat
    marking_df = pd.concat([marking_df.drop(columns=['bbox', 'bbox_list'], errors='ignore'), temp_bbox_df], axis=1)
    
    boxes_per_image = marking_df.groupby('image_id').size()
    unique_image_ids = boxes_per_image.index.values
    box_counts_for_stratify = boxes_per_image.values
    
    num_strat_bins = 5 
    if len(np.unique(box_counts_for_stratify)) > num_strat_bins:
        try:
            _, bins = pd.qcut(box_counts_for_stratify, q=num_strat_bins, retbins=True, duplicates='drop')
            bins = np.unique(bins) # Ensure unique bin edges
            if len(bins) <= 1:
                 bins = np.linspace(box_counts_for_stratify.min(), box_counts_for_stratify.max(), num_strat_bins + 1)
        except ValueError: 
             bins = np.linspace(box_counts_for_stratify.min(), box_counts_for_stratify.max(), num_strat_bins + 1)
    else: 
        bins = np.unique(np.sort(box_counts_for_stratify))
        if len(bins) > 1: bins = np.append(bins, bins[-1] + 1e-3) # Ensure last bin covers max, add epsilon for digitize
        elif len(bins) == 1: bins = np.array([bins[0] - 1e-3, bins[0] + 1e-3]) # Create a bin around the single value
        else: bins = np.array([0, 1]) # Fallback if no boxes

    image_id_bins = np.digitize(box_counts_for_stratify, bins[:-1] if len(bins) > 1 else bins)

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_seed)
    
    folds_data_map = {}
    for fold_idx, (_, val_indices) in enumerate(skf.split(unique_image_ids, image_id_bins)):
        folds_data_map[fold_idx] = {
            'train_image_ids': unique_image_ids[train_indices], # Not strictly needed for OOF eval but good to have
            'val_image_ids': unique_image_ids[val_indices]
        }
        print(f"  Fold {fold_idx} (OOF val set): {len(unique_image_ids[val_indices])} images")
    
    print(f"✅ Stratified {num_splits}-fold splits created for OOF. Total unique images: {len(unique_image_ids)}")
    return folds_data_map, marking_df

def run_oof_predictions_for_threshold_optim(
    loaded_kfold_models_map, # Dict: {fold_idx: YOLO_model_instance}
    oof_fold_definitions,    # Dict: {fold_idx: {'val_image_ids': [...]}}
    gt_marking_dataframe     # Processed DataFrame with x,y,w,h cols
):
    """
    Generates Out-of-Fold predictions using the K-Fold models on their respective validation sets.
    Each model's predictions are processed with TTA and WBF (using fixed global WBF params).
    """
    print("🎯 Running Out-of-Fold (OOF) Prediction Generation...")
    all_oof_predictions_for_thresholding = [] 

    for fold_idx, model_instance in loaded_kfold_models_map.items():
        if model_instance is None:
            print(f"Warning: Model for fold {fold_idx} not loaded. Skipping OOF for this fold.")
            continue

        print(f"\n--- Generating OOF predictions for Fold {fold_idx} ---")
        validation_image_ids_for_this_fold = oof_fold_definitions[fold_idx]['val_image_ids']
        
        for image_id_str in tqdm(validation_image_ids_for_this_fold, desc=f"OOF Preds Fold {fold_idx}"):
            image_file_path = os.path.join(ORIGINAL_TRAIN_IMAGES_DIR_PATH, f"{image_id_str}.jpg")
            if not os.path.exists(image_file_path):
                print(f"Warning: Image {image_file_path} not found for OOF. Skipping.")
                continue

            # Get Ground Truth boxes for this image
            gt_records_for_img = gt_marking_dataframe[gt_marking_dataframe['image_id'] == image_id_str]
            gt_boxes_abs_xyxy_np = np.array([]).reshape(0,4) # Default to empty
            if not gt_records_for_img.empty:
                gt_boxes_abs_xywh_np = gt_records_for_img[['x', 'y', 'w', 'h']].values.astype(np.float32)
                gt_boxes_abs_xyxy_np = gt_boxes_abs_xywh_np.copy()
                gt_boxes_abs_xyxy_np[:, 2] = gt_boxes_abs_xywh_np[:, 0] + gt_boxes_abs_xywh_np[:, 2] # x2 = x1 + w
                gt_boxes_abs_xyxy_np[:, 3] = gt_boxes_abs_xywh_np[:, 1] + gt_boxes_abs_xywh_np[:, 3] # y2 = y1 + h
            
            # Get predictions using TTA for this single model on this OOF image
            # predict_with_tta_for_single_model applies TTA and then WBF internally
            pred_boxes_abs_xyxy_np, pred_scores_np = predict_with_tta_for_single_model(
                image_file_path, model_instance, 
                initial_conf_thr=OOF_CONF_THRESHOLD_LOW, # Use low initial conf for TTA passes
                # WBF params inside predict_with_tta_for_single_model use global WBF_IOU_THR, WBF_SKIP_BOX_THR
            )
            
            all_oof_predictions_for_thresholding.append({
                'image_id': image_id_str,
                'pred_boxes': pred_boxes_abs_xyxy_np, 
                'pred_scores': pred_scores_np,
                'gt_boxes': gt_boxes_abs_xyxy_np
            })
            
    print(f"✅ OOF prediction generation completed. Collected predictions for {len(all_oof_predictions_for_thresholding)} OOF image instances.")
    return all_oof_predictions_for_thresholding

def optimize_final_score_threshold_from_oof_data(oof_predictions_data_list):
    """
    Optimizes the final score_threshold to be applied *after* TTA and WBF.
    Uses OOF predictions where TTA and WBF (with fixed params) have already been applied per model.
    """
    print("⚙️ Optimizing Final Score Threshold using OOF predictions...")
    
    candidate_final_score_thresholds = np.arange(0.05, 0.76, 0.05) # e.g., [0.05, 0.10, ..., 0.75]
    best_overall_map_score = -1.0
    optimized_score_threshold = OPTIMIZED_PARAMS['final_score_threshold'] # Fallback to default

    for score_thr_candidate_val in tqdm(candidate_final_score_thresholds, desc="Optimizing Final Score Thr (OOF)"):
        current_eval_image_map_scores = []
        for oof_pred_item_data in oof_predictions_data_list:
            gt_boxes_for_img = oof_pred_item_data['gt_boxes']
            # These pred_boxes are already post-TTA & post-WBF (with fixed WBF params)
            pred_boxes_after_tta_wbf = oof_pred_item_data['pred_boxes']
            pred_scores_after_tta_wbf = oof_pred_item_data['pred_scores']

            if len(pred_boxes_after_tta_wbf) == 0: # No boxes predicted by TTA+WBF
                img_map_score = 1.0 if len(gt_boxes_for_img) == 0 else 0.0
                current_eval_image_map_scores.append(img_map_score)
                continue
            
            # Apply the candidate final score threshold
            qualifying_indices = pred_scores_after_tta_wbf >= score_thr_candidate_val
            final_pred_boxes_for_this_eval = pred_boxes_after_tta_wbf[qualifying_indices]
            
            # Calculate competition mAP for this image with this threshold
            img_map_score = calculate_image_map_score_numba_oof(
                gt_boxes_for_img, final_pred_boxes_for_this_eval, COMPETITION_IOU_THRESHOLDS_NUMBA
            )
            current_eval_image_map_scores.append(img_map_score)
        
        mean_map_for_this_score_thr = np.mean(current_eval_image_map_scores) if current_eval_image_map_scores else 0.0
        print(f"  Final Score Threshold Candidate {score_thr_candidate_val:.2f} -> OOF mAP: {mean_map_for_this_score_thr:.4f}")
        
        if mean_map_for_this_score_thr > best_overall_map_score:
            best_overall_map_score = mean_map_for_this_score_thr
            optimized_score_threshold = score_thr_candidate_val
            
    print(f"\n✅ Optimal Final Score Threshold (from OOF): {optimized_score_threshold:.2f} (Yielded OOF mAP: {best_overall_map_score:.4f})")
    
    # Store and return the full set of parameters (WBF params are fixed, score_threshold is optimized)
    final_optimized_params = {
        'final_score_threshold': optimized_score_threshold, 
        'wbf_iou_threshold': WBF_IOU_THR,      
        'wbf_skip_box_threshold': WBF_SKIP_BOX_THR, 
        'oof_map_score': best_overall_map_score
    }
    
    # Save optimized params to a JSON file for later use by other scripts
    try:
        with open(OPTIMIZED_PARAMS_JSON_PATH, 'w') as f_json:
            json.dump(final_optimized_params, f_json, indent=4)
        print(f"  Optimized parameters saved to: {OPTIMIZED_PARAMS_JSON_PATH}")
    except Exception as e_json:
        print(f"Error saving optimized params to JSON: {e_json}")
        
    return final_optimized_params

# Placeholder for subsequent pipeline stages (PL Generation, PL Re-training, Final Submission)
# ...
