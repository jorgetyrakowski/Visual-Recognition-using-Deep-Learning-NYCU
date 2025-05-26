#!/usr/bin/env python3
"""
Script to create K-Fold splits for YOLO object detection training.

This script takes a directory of images and a CSV file (typically train.csv from
the Global Wheat Detection competition), generates K-Fold splits of the image data,
and creates corresponding .txt files listing image paths for each train/validation
fold. It also generates .yaml configuration files for each fold, suitable for
use with the Ultralytics YOLO framework.

The splits are created using GroupKFold, where each unique image ID forms a group,
effectively behaving like KFold for unique images.
"""
import argparse
import pandas as pd
import yaml # PyYAML
from pathlib import Path
from sklearn.model_selection import GroupKFold

def save_list_to_file(image_paths, output_filename):
    """Saves a list of image paths to a text file, one path per line."""
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True) # Ensure parent directory (e.g., 'lists/') exists
    with open(output_filename, "w") as f:
        f.write("\n".join(image_paths))
    print(f"  ✓ List saved: {output_filename}")

def main(args):
    """Main function to generate K-Fold splits and YAML files."""
    print(f"Starting K-Fold split generation for {args.folds} folds...")
    print(f"Reading image data from: {args.images}")
    print(f"Using CSV for image IDs: {args.csv}")

    # train.csv is read but primarily image IDs are derived from the --images directory content.
    # This ensures only images actually present are used.
    # csv_data = pd.read_csv(args.csv) # Not strictly needed if IDs come from image dir

    image_files = sorted(Path(args.images).glob("*.jpg"))
    if not image_files:
        print(f"Error: No .jpg images found in directory: {args.images}")
        return

    image_ids = [p.stem for p in image_files]
    # Store absolute paths for robustness, as YAMLs will be in CWD
    # and yolo train might be run from a different CWD if not careful.
    # However, if yolo train is run from same dir as YAMLs, and YAML path: '.',
    # then absolute paths in lists are fine.
    image_id_to_path = {p.stem: str(p.resolve()) for p in image_files} 

    gkf = GroupKFold(n_splits=args.folds)

    # Define the target directory for YAML files
    yaml_output_dir = Path.cwd() / "dataset_yamls"
    yaml_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the directory for list files (relative to CWD where script is run)
    list_files_dir_from_cwd = Path.cwd() / "lists"

    print(f"\nGenerating fold files (YAMLs in '{yaml_output_dir}', lists in '{list_files_dir_from_cwd}'):")
    for fold_index, (_, val_indices) in enumerate(gkf.split(image_ids, groups=image_ids)):
        print(f"\nProcessing Fold {fold_index}:")
        
        validation_image_ids = [image_ids[j] for j in val_indices]
        training_image_ids = [img_id for img_id in image_ids if img_id not in validation_image_ids]

        # List files are saved in CWD/lists/
        train_list_path = list_files_dir_from_cwd / f"train_fold{fold_index}.txt"
        val_list_path = list_files_dir_from_cwd / f"val_fold{fold_index}.txt"

        save_list_to_file([image_id_to_path[img_id] for img_id in training_image_ids], str(train_list_path))
        save_list_to_file([image_id_to_path[img_id] for img_id in validation_image_ids], str(val_list_path))

        # YAML files are saved in CWD/dataset_yamls/
        yaml_file_path = yaml_output_dir / f"fold{fold_index}.yaml"
        
        # Paths in YAML:
        # 'path' is the dataset root. Since .txt files contain absolute image paths,
        # YOLO will use them directly. 'path' can be '.' if yolo train is run from where YAMLs are,
        # or it can be the root of the project if labels need to be found relative to it.
        # For maximum robustness with absolute paths in lists, path: '.' is fine.
        # The train/val list paths need to be relative to the YAML file itself.
        # If YAML is in 'dataset_yamls/' and lists are in 'lists/', path is '../lists/'.
        yaml_content = {
            "path": "..", # Assumes YAML is in 'dataset_yamls/', lists are in 'lists/', images are absolute paths
                         # and yolo train is run from 'Ultimo intento/'.
                         # 'path: ..' means relative paths in train/val are relative to parent of 'dataset_yamls', i.e. 'Ultimo intento'
                         # So, train: 'lists/train_foldX.txt' would resolve to 'Ultimo intento/lists/train_foldX.txt'
            "train": f"../lists/train_fold{fold_index}.txt", 
            "val":   f"../lists/val_fold{fold_index}.txt",   
            "nc": 1,
            "names": ["wheat"]
        }
        
        with open(yaml_file_path, "w") as f:
            yaml.safe_dump(yaml_content, f, sort_keys=False)
        print(f"  ✓ YAML created: {yaml_file_path.resolve()}")

    print(f"\n✓ K-Fold split generation complete for {args.folds} folds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create K-Fold splits and YAML configuration files for YOLO training.")
    parser.add_argument("--images", required=True, help="Path to the directory containing all training image files (e.g., data/train_images).")
    parser.add_argument("--labels", required=True, help="Path to the directory containing corresponding YOLO label files. This script assumes labels exist and are co-located or findable by YOLO relative to image paths.")
    parser.add_argument("--csv", required=True, help="Path to the main training CSV file (e.g., train.csv from the competition, primarily used for ensuring consistency or can be used for more complex splitting if needed).")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds to create (default: 5).")
    
    args = parser.parse_args()
    main(args)
