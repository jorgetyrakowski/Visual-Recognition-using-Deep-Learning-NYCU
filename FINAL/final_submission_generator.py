# Ultimo intento/pipeline_scripts/final_submission_generator.py
"""
Final Submission Generation Script.

This script loads the pseudo-label re-trained model (or falls back to
an ensemble of K-Fold models if PL model is not available), applies
Test-Time Augmentation (TTA) and Weighted Boxes Fusion (WBF) using
optimized parameters, and generates the final submission.csv file.
"""
import os
import pandas as pd
import numpy as np
import cv2 # For image reading if paths are passed directly, though utils handle it
import torch
from ultralytics import YOLO
from tqdm import tqdm
import json
from pathlib import Path

# Import from config_and_utils module (now expected to be in the same directory)
from config_and_utils import (
    TEST_IMAGES_DIR_PATH, SAMPLE_SUBMISSION_CSV_PATH, SUBMISSION_CSV_OUTPUT_PATH,
    FINAL_MODEL_AFTER_PL_PATH, MODEL_PATHS_KFold, # For fallback ensemble KFold model paths
    OPTIMIZED_PARAMS_JSON_PATH, OPTIMIZED_POSTPROCESSING_PARAMS, # For optimized thresholds
    IMAGE_SIZE, DEVICE,
    FINAL_INFERENCE_INITIAL_CONF_THR_TTA, # Low conf for TTA passes
    predict_with_tta_for_single_model # TTA for single model
)
# Import the multi-model ensemble prediction function from pseudo_labeler.py for fallback
try:
    from .pseudo_labeler import ensemble_predict_with_tta_for_pl
except ImportError:
    from pseudo_labeler import ensemble_predict_with_tta_for_pl


def generate_submission_file(
    model_path_to_use, # Path to the single PL-retrained model
    kfold_model_instances_for_fallback, # List of loaded K-Fold models for fallback
    optimized_parameters, # Dict with 'final_score_threshold', 'wbf_iou_threshold', 'wbf_skip_box_threshold'
    test_img_dir=TEST_IMAGES_DIR_PATH,
    sample_csv=SAMPLE_SUBMISSION_CSV_PATH,
    output_csv=SUBMISSION_CSV_OUTPUT_PATH
):
    """
    Generates the final submission.csv file.
    Uses the PL-retrained model if available, otherwise falls back to K-Fold ensemble.
    """
    print(f"📄 Generating Final Submission CSV: {output_csv}")

    submission_model_instance = None
    use_ensemble_fallback = False

    if model_path_to_use and Path(model_path_to_use).exists():
        print(f"  Using PL-Retrained Model: {model_path_to_use}")
        try:
            submission_model_instance = YOLO(str(model_path_to_use))
            submission_model_instance.to(DEVICE) # Ensure model is on correct device
        except Exception as e_load_pl:
            print(f"  ❌ Error loading PL-Retrained Model {model_path_to_use}: {e_load_pl}")
            submission_model_instance = None # Force fallback
    
    if submission_model_instance is None:
        if kfold_model_instances_for_fallback and any(m is not None for m in kfold_model_instances_for_fallback):
            print(f"  ⚠️ PL-Retrained model not available or failed to load. Using K-Fold ensemble as fallback.")
            use_ensemble_fallback = True
            # kfold_model_instances_for_fallback are already loaded YOLO objects
        else:
            print(f"❌ ERROR: No model available for final submission (PL model failed, no K-Fold models for fallback).")
            pd.DataFrame(columns=['image_id', 'PredictionString']).to_csv(output_csv, index=False)
            print(f"  Created empty submission file: {output_csv}")
            return None

    # Load sample submission to get all test image IDs in order
    if os.path.exists(sample_csv):
        submission_template_df = pd.read_csv(sample_csv)
    else: 
        print(f"Warning: {sample_csv} not found. Generating image list from {test_img_dir}")
        test_img_filenames_list = [f.replace('.jpg','') for f in os.listdir(test_img_dir) if f.lower().endswith('.jpg')]
        submission_template_df = pd.DataFrame({'image_id': sorted(test_img_filenames_list)})

    submission_data_list = []
    final_score_threshold_to_apply = optimized_parameters['final_score_threshold']
    
    for _, row_data in tqdm(submission_template_df.iterrows(), total=len(submission_template_df), desc="Generating Final Predictions"):
        current_img_id = row_data['image_id']
        current_img_file_path = os.path.join(test_img_dir, f"{current_img_id}.jpg")
        prediction_str_parts = []

        if os.path.exists(current_img_file_path):
            try:
                if use_ensemble_fallback:
                    # Use the K-Fold ensemble for prediction
                    # multi_model_predict_with_tta_for_pl_gen already applies WBF across models
                    # and uses optimized_params for its internal WBF and initial conf.
                    # It returns boxes *before* the final score threshold.
                    pred_boxes_abs, pred_scores = multi_model_predict_with_tta_for_pl_gen(
                        current_img_file_path, 
                        kfold_model_instances_for_fallback,
                        initial_conf_for_each_model_tta=FINAL_INFERENCE_INITIAL_CONF_THR_TTA,
                        ensemble_wbf_iou_thr=optimized_parameters['wbf_iou_threshold'],
                        ensemble_wbf_skip_thr=optimized_parameters['wbf_skip_box_threshold']
                    )
                else: # Use the single PL-retrained model
                    # predict_with_tta_for_single_model applies TTA and WBF for that single model.
                    pred_boxes_abs, pred_scores = predict_with_tta_for_single_model(
                        current_img_file_path, submission_model_instance,
                        initial_conf_thr=FINAL_INFERENCE_INITIAL_CONF_THR_TTA
                        # WBF params inside predict_with_tta_for_single_model use global defaults from config_and_utils
                    )
                
                # Apply the final OOF-optimized score threshold
                if len(pred_boxes_abs) > 0:
                    qualifying_indices = pred_scores >= final_score_threshold_to_apply
                    final_boxes_for_submission_abs = pred_boxes_abs[qualifying_indices]
                    final_scores_for_submission = pred_scores[qualifying_indices]

                    for box_coords_abs, score_val in zip(final_boxes_for_submission_abs, final_scores_for_submission):
                        x1, y1, x2, y2 = box_coords_abs.astype(int) 
                        w_box = max(1, x2 - x1)
                        h_box = max(1, y2 - y1)
                        x1_box = max(0, x1) 
                        y1_box = max(0, y1)
                        
                        # Clip to image boundaries (IMAGE_SIZE)
                        if x1_box + w_box > IMAGE_SIZE: w_box = int(IMAGE_SIZE - x1_box)
                        if y1_box + h_box > IMAGE_SIZE: h_box = int(IMAGE_SIZE - y1_box)
                        w_box = max(1, w_box); h_box = max(1, h_box) 
                        x1_box = min(x1_box, int(IMAGE_SIZE) -1); y1_box = min(y1_box, int(IMAGE_SIZE) -1)
                        x1_box = max(0, x1_box); y1_box = max(0, y1_box) # Re-ensure non-negative after clipping w/h

                        prediction_str_parts.append(f"{score_val:.4f} {x1_box} {y1_box} {w_box} {h_box}")
            except Exception as e_pred_final:
                print(f"Error processing {current_img_id} for final submission: {e_pred_final}")
        else:
            print(f"Warning: Test image {current_img_file_path} not found (likely private test set). Empty prediction.")

        submission_data_list.append({
            'image_id': current_img_id,
            'PredictionString': " ".join(prediction_str_parts)
        })

    final_submission_df = pd.DataFrame(submission_data_list)
    final_submission_df.to_csv(output_csv, index=False)
    
    num_imgs_with_preds = (final_submission_df['PredictionString'] != '').sum()
    print(f"✅ Final Submission saved: {output_csv}")
    print(f"   Images with predictions: {num_imgs_with_preds} / {len(final_submission_df)}")
    return output_csv


if __name__ == "__main__":
    print("Executing Final Submission Generation Script...")
    
    # This script assumes:
    # 1. K-Fold models are available at paths specified in MODEL_PATHS_KFold (for fallback).
    # 2. The PL-retrained model is available at FINAL_MODEL_AFTER_PL_PATH.
    # 3. Optimized parameters (esp. final_score_threshold) are in OPTIMIZED_PARAMS_JSON_PATH.

    # Load optimized parameters
    loaded_optimized_params = OPTIMIZED_POSTPROCESSING_PARAMS # Default
    if Path(OPTIMIZED_PARAMS_JSON_PATH).exists():
        try:
            with open(OPTIMIZED_PARAMS_JSON_PATH, 'r') as f_json_params:
                loaded_optimized_params = json.load(f_json_params)
            print(f"  Successfully loaded optimized parameters from {OPTIMIZED_PARAMS_JSON_PATH}")
        except Exception as e_load_json:
            print(f"  Error loading optimized params from JSON: {e_load_json}. Using defaults.")
    else:
        print(f"  Optimized parameters JSON not found at {OPTIMIZED_PARAMS_JSON_PATH}. Using default parameters.")
    print(f"  Using parameters for submission: {loaded_optimized_params}")

    # Load K-Fold models for fallback if PL model is missing
    kfold_models_for_fallback = []
    print("  Loading K-Fold models for potential fallback...")
    for i, model_f_path_str in enumerate(MODEL_PATHS_KFold):
        if Path(model_f_path_str).exists():
            try:
                kfold_models_for_fallback.append(YOLO(model_f_path_str))
                print(f"    Fold Model {i} loaded for fallback.")
            except Exception as e_load_f_model:
                print(f"    Error loading Fold Model {i} ({model_f_path_str}) for fallback: {e_load_f_model}")
                kfold_models_for_fallback.append(None)
        else:
            print(f"    Fold Model {i} path ({model_f_path_str}) not found for fallback.")
            kfold_models_for_fallback.append(None)
            
    submission_file = generate_submission_file(
        model_path_to_use=FINAL_MODEL_AFTER_PL_PATH,
        kfold_model_instances_for_fallback=kfold_models_for_fallback,
        optimized_parameters=loaded_optimized_params
    )

    if submission_file:
        print(f"\n🎉 Submission generation successful: {submission_file}")
    else:
        print("\n❌ Submission generation failed.")
