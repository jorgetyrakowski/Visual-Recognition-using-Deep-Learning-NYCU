# Ultimo intento/pipeline_scripts/train_pl_model.py
"""
Script for Re-training/Fine-tuning a YOLOv11m model with Pseudo-Labels.

This script takes a base YOLOv11m model and fine-tunes it on the
combined dataset consisting of original ground truth training data and
high-confidence pseudo-labels generated from the test set.
"""
import os
from pathlib import Path
from ultralytics import YOLO

# Import from config_and_utils module (now expected to be in the same directory)
from config_and_utils import (
    PL_DATASET_YAML_PATH, BASE_YOLOV11x_FOR_PL_RETRAIN,
    PL_RETRAIN_NUM_EPOCHS, PL_RETRAIN_BATCH_SZ, IMAGE_SIZE, DEVICE,
    PL_RETRAIN_OUTPUT_PROJECT_DIR, PL_RETRAIN_RUN_NAME,
    PL_RETRAIN_INITIAL_LR, PL_RETRAIN_FINAL_LR_FACTOR,
    PL_RETRAIN_WARMUP_EPOCHS, PL_RETRAIN_PATIENCE_EARLY_STOP,
    PL_RETRAIN_OPTIMIZER, PL_RETRAIN_AUGMENT_FLAG,
    PL_RETRAIN_MIXUP_FRACTION, PL_RETRAIN_COPYPASTE_FRACTION,
    FINAL_MODEL_AFTER_PL_PATH # To confirm where the model is saved
)

def retrain_model_with_pseudo_labels(
    dataset_yaml_file=PL_DATASET_YAML_PATH,
    base_model_weights_path=BASE_YOLOV11x_FOR_PL_RETRAIN,
    num_epochs=PL_RETRAIN_NUM_EPOCHS,
    batch_size=PL_RETRAIN_BATCH_SZ,
    img_size_for_train=IMAGE_SIZE,
    device_to_use=DEVICE,
    output_project_base_dir=PL_RETRAIN_OUTPUT_PROJECT_DIR,
    output_run_name_pl=PL_RETRAIN_RUN_NAME,
    initial_lr=PL_RETRAIN_INITIAL_LR,
    final_lr_factor=PL_RETRAIN_FINAL_LR_FACTOR,
    warmup_epochs_pl=PL_RETRAIN_WARMUP_EPOCHS,
    early_stop_patience=PL_RETRAIN_PATIENCE_EARLY_STOP,
    optimizer_name=PL_RETRAIN_OPTIMIZER,
    use_augment=PL_RETRAIN_AUGMENT_FLAG,
    mixup_val=PL_RETRAIN_MIXUP_FRACTION,
    copypaste_val=PL_RETRAIN_COPYPASTE_FRACTION
):
    """
    Trains/fine-tunes a YOLOv11x model on the dataset including pseudo-labels.
    Args:
        dataset_yaml_file: Path to the YAML file defining the PL-augmented dataset.
        base_model_weights_path: Path to the base model weights (e.g., COCO pre-trained yolov11x.pt).
        num_epochs, batch_size, etc.: Training hyperparameters.
    Returns:
        Path to the best model checkpoint (.pt file) from this training run, or None if failed.
    """
    print(f"🚀 Starting Pseudo-Label Re-training Stage...")
    print(f"  Dataset YAML: {dataset_yaml_file}")
    print(f"  Base Model for PL Re-train: {base_model_weights_path}")
    print(f"  Epochs: {num_epochs}, Batch: {batch_size}, ImgSize: {img_size_for_train}")

    if not Path(base_model_weights_path).exists():
        print(f"❌ ERROR: Base model for PL re-training not found: {base_model_weights_path}")
        print(f"  Please ensure '{Path(base_model_weights_path).name}' is in your Kaggle input dataset: "
              f"{Path(base_model_weights_path).parent}")
        return None

    if not Path(dataset_yaml_file).exists():
        print(f"❌ ERROR: PL Dataset YAML file not found: {dataset_yaml_file}")
        return None

    # Initialize the model with base weights
    pl_model = YOLO(str(base_model_weights_path)) 
    
    print(f"  Starting training on {device_to_use}...")
    try:
        pl_model.train(
            data=str(dataset_yaml_file),
            epochs=num_epochs,
            imgsz=img_size_for_train,
            batch=batch_size,
            device=device_to_use,
            project=str(output_project_base_dir), 
            name=output_run_name_pl,      
            exist_ok=True, # Allow overwriting if run name exists from a previous attempt
            
            optimizer=optimizer_name,
            lr0=initial_lr, 
            lrf=final_lr_factor,           
            warmup_epochs=warmup_epochs_pl,    
            patience=early_stop_patience,        
            
            augment=use_augment,       
            mixup=mixup_val,          
            copy_paste=copypaste_val,     
            
            val=True, # Perform validation during training using the 'val' split in dataset_yaml_file
            save=True,          
            save_period=5, # Save checkpoint every 5 epochs (or adjust as needed)     
            plots=True, # Generate training plots (e.g., mAP curves, loss curves)         
            verbose=True # Show detailed training progress
        )
    except Exception as e_train:
        print(f"❌ ERROR during PL model training: {e_train}")
        return None
    
    print(f"✅ Pseudo-Label Re-training completed.")
    
    # Construct the expected path to the best model weights
    # This path is defined in config_and_utils.py as FINAL_MODEL_AFTER_PL_PATH
    expected_best_model_path = FINAL_MODEL_AFTER_PL_PATH 
    
    if expected_best_model_path.exists():
        print(f"   Final PL-retrained model saved to: {expected_best_model_path}")
        return expected_best_model_path
    else:
        # Check for last.pt if best.pt is not found (e.g. if training was interrupted before final save)
        last_model_path = Path(output_project_base_dir) / output_run_name_pl / 'weights' / 'last.pt'
        if last_model_path.exists():
            print(f"Warning: best.pt not found, but last.pt exists at {last_model_path}. Using last.pt.")
            return last_model_path
        print(f"❌ ERROR: Best model from PL re-training not found at expected path: {expected_best_model_path}")
        return None

if __name__ == "__main__":
    print("Executing Pseudo-Label Model Re-training Script...")
    
    # This script assumes that the PL dataset (images, labels, YAML) has already been prepared
    # by running the pseudo_labeler.py script or its equivalent functions.
    
    # Ensure the base model for re-training exists
    if not Path(BASE_YOLOV11x_FOR_PL_RETRAIN).exists():
        print(f"CRITICAL: Base model {BASE_YOLOV11x_FOR_PL_RETRAIN} for PL re-training not found. Exiting.")
    elif not Path(PL_DATASET_YAML_PATH).exists():
        print(f"CRITICAL: PL Dataset YAML {PL_DATASET_YAML_PATH} not found. Run pseudo_labeler.py first. Exiting.")
    else:
        trained_pl_model_path = retrain_model_with_pseudo_labels()
        if trained_pl_model_path and trained_pl_model_path.exists():
            print(f"\nSuccessfully re-trained model with pseudo-labels.")
            print(f"Final model checkpoint: {trained_pl_model_path}")
        else:
            print("\nPL Model re-training failed or model was not saved correctly.")
