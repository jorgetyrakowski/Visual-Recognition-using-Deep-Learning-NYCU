# Ultimo intento/pipeline_scripts/pseudo_labeler.py
"""
Pseudo-Label Generation and Dataset Preparation.

This script takes the K-Fold models and optimized OOF parameters to:
1. Generate pseudo-labels on the test set using a multi-model TTA & WBF ensemble.
2. Prepare the combined dataset (original training data + pseudo-labels)
   for the pseudo-label re-training stage.
"""
import os
import pandas as pd
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from tqdm import tqdm
import shutil
import json
from pathlib import Path
import yaml

# Import from config_and_utils module (now expected to be in the same directory)
from config_and_utils import (
    TEST_IMAGES_DIR_PATH, MODEL_PATHS_KFold, DEVICE, IMAGE_SIZE,
    OPTIMIZED_PARAMS_JSON_PATH, OPTIMIZED_POSTPROCESSING_PARAMS, # For default/fallback
    INITIAL_CONF_THR_FOR_TTA_PASSES, # Renamed from OOF_CONF_THRESHOLD_LOW for clarity
    WBF_IOU_THR, WBF_SKIP_BOX_THR, ENSEMBLE_MODEL_TTA_WEIGHTS,
    PL_OUTPUT_DIR, PL_IMAGES_TRAIN_SUBDIR, PL_LABELS_TRAIN_SUBDIR,
    PL_IMAGES_VAL_SUBDIR, PL_LABELS_VAL_SUBDIR, PL_DATASET_YAML_PATH,
    ORIGINAL_TRAIN_CSV_PATH, ORIGINAL_TRAIN_IMAGES_DIR_PATH, SEED,
    predict_with_tta_for_single_model, # For ensembling
    apply_wbf # For the final ensemble WBF
)

# ==================== MULTI-MODEL ENSEMBLE PREDICTION FOR PL ====================

def ensemble_predict_with_tta_for_pl(
    image_path_or_np_array, 
    list_of_kfold_models, 
    initial_conf_for_each_model_tta=OOF_CONF_THRESHOLD_LOW,
    ensemble_wbf_iou_thr=WBF_IOU_THR,
    ensemble_wbf_skip_thr=WBF_SKIP_BOX_THR,
    ensemble_wbf_weights=ENSEMBLE_MODEL_TTA_WEIGHTS
):
    """
    Generates predictions by ensembling multiple K-Fold models, each using TTA.
    This is used for generating pseudo-label candidates.
    Args:
        image_path_or_np_array: Path to image or NumPy BGR image array.
        list_of_kfold_models: List of loaded YOLOv11 K-Fold model instances.
        initial_conf_for_each_model_tta: Confidence for individual model's TTA passes.
        ensemble_wbf_iou_thr, ensemble_wbf_skip_thr: Parameters for the final WBF across models.
        ensemble_wbf_weights: Weights for WBF across models.
    Returns:
        ensembled_boxes_abs_xyxy, ensembled_scores (absolute pixel coordinates)
    """
    all_models_predictions_boxes = [] # List of [N,4] box arrays from each model (after its own TTA+WBF)
    all_models_predictions_scores = []# List of [N,] score arrays from each model

    for model_idx, model_inst in enumerate(list_of_kfold_models):
        if model_inst is None:
            print(f"Warning: Model {model_idx} is None, skipping for PL generation on this image.")
            continue
        
        try:
            # predict_with_tta_for_single_model applies TTA and then WBF to that single model's TTA outputs
            model_fused_boxes_abs, model_fused_scores = predict_with_tta_for_single_model(
                image_path_or_np_array, model_inst,
                conf_threshold_tta_pass=initial_conf_for_each_model_tta,
                # WBF params inside predict_with_tta_for_single_model use global defaults
            )
            
            if len(model_fused_boxes_abs) > 0:
                all_models_predictions_boxes.append(model_fused_boxes_abs)
                all_models_predictions_scores.append(model_fused_scores)
                
        except Exception as e_model_pred:
            print(f"Error during TTA prediction for model {model_idx} on image: {e_model_pred}")
            continue # Skip this model for this image if an error occurs
    
    if not all_models_predictions_boxes: # No model produced any predictions
        return np.array([]).reshape(0,4), np.array([])

    # Apply a second level of WBF to ensemble the outputs of the (TTA'd+WBF'd) individual models
    final_ensembled_boxes_abs, final_ensembled_scores, _ = apply_wbf(
        all_models_predictions_boxes, all_models_predictions_scores,
        image_ref_width=IMAGE_SIZE, image_ref_height=IMAGE_SIZE, # All boxes are in IMAGE_SIZE space
        iou_thr=ensemble_wbf_iou_thr,
        skip_box_thr=ensemble_wbf_skip_thr,
        weights=ensemble_wbf_weights[:len(all_models_predictions_boxes)] # Adjust weights if some models failed
    )
    return final_ensembled_boxes_abs.astype(np.int32), final_ensembled_scores


def generate_pseudo_labels_from_ensemble(
    list_of_kfold_models, 
    optimized_postproc_params, # Dict containing 'final_score_threshold'
    test_images_root_dir=TEST_IMAGES_DIR_PATH
):
    """
    Generates pseudo-labels for the test set using an ensemble of K-Fold models.
    Filters predictions using the OOF-optimized final score threshold.
    """
    print(f"🏷️ Generating pseudo-labels using {len(list_of_kfold_models)}-Fold ensemble...")
    
    if not os.path.exists(test_images_root_dir):
        print(f"❌ Test images directory not found: {test_images_root_dir}")
        return []
    
    test_image_filenames_list = [f for f in os.listdir(test_images_root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not test_image_filenames_list:
        print(f"❌ No images found in test directory: {test_images_root_dir}")
        return []
        
    print(f"  Found {len(test_image_filenames_list)} test images for pseudo-labeling.")
    
    pseudo_labels_data_collected = [] # List of dicts: {'image_id', 'boxes', 'scores', 'image_path'}
    
    # The score threshold to filter PLs comes from OOF optimization
    pl_filtering_score_threshold = optimized_postproc_params['final_score_threshold']
    print(f"  Using final score threshold for PL filtering: {pl_filtering_score_threshold:.2f}")

    for img_fname in tqdm(test_image_filenames_list, desc="Generating Pseudo-Labels"):
        img_full_path_str = os.path.join(test_images_root_dir, img_fname)
        image_id_str_stem = Path(img_fname).stem
        
        try:
            # Get ensembled predictions (post-TTA from each model, post-multi-model-WBF)
            ensembled_boxes_abs, ensembled_scores = ensemble_predict_with_tta_for_pl(
                img_full_path_str, list_of_kfold_models,
                initial_conf_for_each_model_tta=INITIAL_CONF_THR_FOR_TTA_PASSES,
                # WBF params for ensembling are from global config (WBF_IOU_THR, WBF_SKIP_BOX_THR)
                # as optimized_params only contains the final_score_threshold
                ensemble_wbf_iou_thr=optimized_postproc_params['wbf_iou_threshold'],
                ensemble_wbf_skip_thr=optimized_postproc_params['wbf_skip_box_threshold']
            )
            
            # Filter these ensembled predictions by the OOF-optimized final score threshold
            if len(ensembled_boxes_abs) > 0:
                high_confidence_indices = ensembled_scores >= pl_filtering_score_threshold
                
                if high_confidence_indices.sum() > 0:
                    final_pl_boxes_abs = ensembled_boxes_abs[high_confidence_indices]
                    final_pl_scores = ensembled_scores[high_confidence_indices]
                    
                    pseudo_labels_data_collected.append({
                        'image_id': image_id_str_stem,
                        'boxes': final_pl_boxes_abs, # Absolute xyxy
                        'scores': final_pl_scores, 
                        'image_path': img_full_path_str 
                    })
        except Exception as e_pl_gen:
            print(f"Error processing {image_id_str_stem} for pseudo-labeling: {e_pl_gen}")
            continue
            
    print(f"✅ Pseudo-label candidates generated for {len(pseudo_labels_data_collected)} images "
          f"(before saving to YOLO format).")
    return pseudo_labels_data_collected

# ==================== PL DATASET PREPARATION FUNCTIONS ====================

def create_pl_dataset_directories(base_pl_dir=PL_OUTPUT_DIR):
    """Creates directory structure for the pseudo-labeling dataset if they don't exist."""
    print(f"📁 Creating Pseudo-Label dataset directory structure under: {base_pl_dir}")
    for subdir_name_str in ["images/train", "images/val", "labels/train", "labels/val"]:
        Path(base_pl_dir / subdir_name_str).mkdir(parents=True, exist_ok=True)
    print("  ✅ PL Dataset directory structure verified/created.")

def copy_gt_train_data_to_pl_dataset(
    original_csv_path=ORIGINAL_TRAIN_CSV_PATH,
    original_img_dir=ORIGINAL_TRAIN_IMAGES_DIR_PATH,
    pl_img_train_dir=PL_IMAGES_TRAIN_SUBDIR,
    pl_lbl_train_dir=PL_LABELS_TRAIN_SUBDIR,
    img_ref_size=IMAGE_SIZE
):
    """Copies original training images and converts their GT labels to YOLO format into PL dataset dirs."""
    print(f"🚚 Copying original GT training data to PL directories...")
    gt_marking_df = pd.read_csv(original_csv_path)
    if isinstance(gt_marking_df['bbox'].iloc[0], str):
        gt_marking_df['bbox_list'] = gt_marking_df['bbox'].apply(lambda x: json.loads(x))
    else:
        gt_marking_df['bbox_list'] = gt_marking_df['bbox']
    temp_bbox_df_gt = pd.DataFrame(gt_marking_df['bbox_list'].tolist(), columns=['x', 'y', 'w', 'h'])
    gt_marking_df = pd.concat([gt_marking_df.drop(columns=['bbox', 'bbox_list'], errors='ignore'), temp_bbox_df_gt], axis=1)

    gt_processed_count = 0
    for img_id_str, group_df in tqdm(gt_marking_df.groupby('image_id'), desc="Processing GT Training Data"):
        src_img_file_path = Path(original_img_dir) / f"{img_id_str}.jpg"
        dst_img_file_path = Path(pl_img_train_dir) / f"{img_id_str}.jpg"
        dst_label_file_path = Path(pl_lbl_train_dir) / f"{img_id_str}.txt"

        if not src_img_file_path.exists(): continue
        shutil.copy2(str(src_img_file_path), str(dst_img_file_path))
        
        img_w_ref, img_h_ref = img_ref_size, img_ref_size 
        yolo_gt_annotations = []
        for _, row in group_df.iterrows():
            x_abs, y_abs, w_abs, h_abs_bbox = row['x'], row['y'], row['w'], row['h']
            cx_n = (x_abs + w_abs / 2) / img_w_ref
            cy_n = (y_abs + h_abs_bbox / 2) / img_h_ref
            w_n = w_abs / img_w_ref
            h_n = h_abs_bbox / img_h_ref
            yolo_gt_annotations.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
            
        with open(dst_label_file_path, 'w') as f: f.write('\n'.join(yolo_gt_annotations))
        gt_processed_count += 1
    print(f"  ✅ Original GT training data: {gt_processed_count} images copied with labels.")
    return gt_processed_count

def save_pseudo_labels_in_yolo_format(
    generated_pl_data_list, # List of dicts from generate_multi_model_pseudo_labels
    pl_img_train_dir=PL_IMAGES_TRAIN_SUBDIR,
    pl_lbl_train_dir=PL_LABELS_TRAIN_SUBDIR,
    img_ref_size=IMAGE_SIZE
):
    """Saves generated pseudo-labels (for test images) to the PL dataset structure in YOLO format."""
    print(f"💾 Saving {len(generated_pl_data_list)} test images with their pseudo-labels...")
    pl_saved_count = 0
    for pl_item_data in tqdm(generated_pl_data_list, desc="Saving Pseudo-Labels"):
        img_id_str = pl_item_data['image_id']
        src_test_img_file_path = Path(pl_item_data['image_path'])
        
        dst_pl_img_file_path = Path(pl_img_train_dir) / f"{img_id_str}.jpg"
        dst_pl_label_file_path = Path(pl_lbl_train_dir) / f"{img_id_str}.txt"

        if not src_test_img_file_path.exists(): continue
        shutil.copy2(str(src_test_img_file_path), str(dst_pl_img_file_path))

        boxes_abs_xyxy_for_pl = pl_item_data['boxes'] 
        yolo_pl_annotations = []
        img_w_ref, img_h_ref = img_ref_size, img_ref_size 

        for box_xyxy in boxes_abs_xyxy_for_pl:
            x1, y1, x2, y2 = box_xyxy
            w_abs, h_abs = x2 - x1, y2 - y1
            cx_abs, cy_abs = x1 + w_abs / 2, y1 + h_abs / 2
            cx_n, cy_n = np.clip(cx_abs / img_w_ref, 0.0, 1.0), np.clip(cy_abs / img_h_ref, 0.0, 1.0)
            w_n, h_n = np.clip(w_abs / img_w_ref, 0.0, 1.0), np.clip(h_abs / img_h_ref, 0.0, 1.0)
            yolo_pl_annotations.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
            
        if yolo_pl_annotations:
            with open(dst_pl_label_file_path, 'w') as f: f.write('\n'.join(yolo_pl_annotations))
            pl_saved_count += 1
    print(f"  ✅ Pseudo-labels saved for {pl_saved_count} test images.")
    return pl_saved_count

def create_pl_dataset_yaml(dataset_root_abs_path=PL_OUTPUT_DIR, yaml_file_out_path=PL_DATASET_YAML_PATH):
    """Creates the dataset.yaml file for the combined (GT + PL) dataset."""
    print(f"📝 Creating dataset YAML for PL re-training: {yaml_file_out_path}")
    yaml_content = {
        'path': str(dataset_root_abs_path.resolve()), 
        'train': 'images/train', 'val': 'images/val',               
        'nc': 1, 'names': ['wheat']
    }
    with open(yaml_file_out_path, 'w') as f: yaml.safe_dump(yaml_content, f, sort_keys=False)
    print(f"  ✅ PL Dataset YAML created: {yaml_file_out_path}")

def create_validation_split_from_pl_train(
    pl_img_train_dir=PL_IMAGES_TRAIN_SUBDIR, 
    pl_lbl_train_dir=PL_LABELS_TRAIN_SUBDIR,
    pl_img_val_dir=PL_IMAGES_VAL_SUBDIR,
    pl_lbl_val_dir=PL_LABELS_VAL_SUBDIR,
    val_percentage=0.20
):
    """Creates a validation split from the combined training data for PL re-training."""
    print(f"📊 Creating validation split for PL dataset (validation: {val_percentage*100:.0f}%)...")
    all_img_files_in_pl_train_dir = sorted([f for f in os.listdir(pl_img_train_dir) if f.endswith('.jpg')])
    
    current_rng_state = np.random.get_state()
    np.random.seed(SEED) 
    np.random.shuffle(all_img_files_in_pl_train_dir)
    np.random.set_state(current_rng_state)

    num_val_files_to_move = int(len(all_img_files_in_pl_train_dir) * val_percentage)
    if num_val_files_to_move == 0 and len(all_img_files_in_pl_train_dir) > 1: num_val_files_to_move = 1
    
    val_filenames_to_move_list = all_img_files_in_pl_train_dir[:num_val_files_to_move]
    print(f"  Moving {len(val_filenames_to_move_list)} images (and their labels) to PL validation set.")

    moved_pairs_count = 0
    for img_fname_str in val_filenames_to_move_list:
        img_basename_str = Path(img_fname_str).stem
        
        src_img = pl_img_train_dir / img_fname_str
        dst_img = pl_img_val_dir / img_fname_str
        src_lbl = pl_lbl_train_dir / f"{img_basename_str}.txt"
        dst_lbl = pl_lbl_val_dir / f"{img_basename_str}.txt"

        if src_img.exists(): shutil.move(str(src_img), str(dst_img))
        if src_lbl.exists(): shutil.move(str(src_lbl), str(dst_lbl))
        moved_pairs_count +=1
        
    print(f"  ✅ {moved_pairs_count} image-label pairs moved to PL validation.")
    final_pl_train_img_count = len(list(pl_img_train_dir.glob('*.jpg')))
    final_pl_val_img_count = len(list(pl_img_val_dir.glob('*.jpg')))
    print(f"  Final PL dataset split: {final_pl_train_img_count} train, {final_pl_val_img_count} val.")

# Placeholder for train_model_with_pseudo_labels, generate_final_submission, main_pipeline
# ...
