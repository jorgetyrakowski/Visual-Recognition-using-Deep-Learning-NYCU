#!/bin/bash
# Orchestrator script for the full YOLOv11x GWD pipeline.
# This script runs the modularized Python scripts in the correct sequence.

# Exit on any error
set -e

echo "🚀 STARTING FULL PIPELINE ORCHESTRATION 🚀"
# Assumes this script is in 'Ultimo intento/' and the Python pipeline scripts
# (oof_evaluation.py, pseudo_labeler.py, etc.) are also in 'Ultimo intento/'.

# The Python scripts save outputs (like JSON, datasets, models) to /kaggle/working/,
# which are absolute paths, so CWD for python script execution is less critical for outputs.
# However, imports within python scripts (e.g. `from config_and_utils import ...`)
# assume all pipeline python scripts are in the same directory.

# Step 1: OOF Evaluation and Score Threshold Optimization
echo -e "\n--- Running Step 1: OOF Evaluation & Threshold Optimization ---"
python oof_evaluation.py
echo "✅ Step 1 completed."

# Step 2: Pseudo-Label Generation and PL Dataset Preparation
echo -e "\n--- Running Step 2: Pseudo-Label Generation & Dataset Prep ---"
python pseudo_labeler.py
echo "✅ Step 2 completed."

# Step 3: Re-train Model with Pseudo-Labels
echo -e "\n--- Running Step 3: Re-training Model with Pseudo-Labels ---"
python train_pl_model.py
echo "✅ Step 3 completed."

# Step 4: Generate Final Submission CSV
echo -e "\n--- Running Step 4: Generating Final Submission CSV ---"
python final_submission_generator.py
echo "✅ Step 4 completed."

echo -e "\n🎉🎉🎉 FULL PIPELINE EXECUTION FINISHED! 🎉🎉🎉"
echo "Final submission.csv should be in /kaggle/working/submission.csv"
echo "Optimized parameters JSON: /kaggle/working/optimized_oof_params.json"
echo "Pseudo-label dataset: /kaggle/working/pseudo_labeled_data/"
echo "PL Re-trained model: /kaggle/working/runs/detect/pseudo_labeled_wheat_model/weights/best.pt"
