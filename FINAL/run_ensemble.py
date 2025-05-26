#!/usr/bin/env python3
"""
Create pseudo-labels for Global Wheat Detection test set by ensembling
the five fold checkpoints with Weighted Box Fusion (WBF).
Outputs go to data/train/labels/  (will be appended to GT labels)
"""

from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from pathlib import Path
import numpy as np
import cv2, os, tqdm

# ------------------------------------------------------------------
# Configuration
MODEL_PATHS = [
    'gw11/gwd_fold0/weights/best.pt',
    'gw11/gwd_fold1/weights/best.pt',
    'gw11/gwd_fold2/weights/best.pt',
    'gw11/gwd_fold3/weights/best.pt',
    'gw11/gwd_fold4/weights/best.pt',
]
SOURCE_DIR   = Path('data/test/images')
CONF_THRES   = 0.70       # high-precision harvest
IMG_SIZE     = 1024
IOU_WBF      = 0.55
OUT_DIR      = Path('data/train/labels')   # append here
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ------------------------------------------------------------------

print('Loading 5 fold models …')
models = [YOLO(p) for p in MODEL_PATHS]

# helper: run one model → list of (boxes, scores) per image
def predict_one(model):
    return model.predict(
        source=str(SOURCE_DIR),
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=0.7,
        save=False,
        verbose=False,
        max_det=300
    )

# run all five models
print('Running inference …')
all_preds = [predict_one(m) for m in models]

# index images once
img_paths = sorted(SOURCE_DIR.glob('*.jpg'))
n_images  = len(img_paths)

for idx, img_path in enumerate(tqdm.tqdm(img_paths, total=n_images, unit='img')):
    boxes_list, scores_list = [], []
    for fold_preds in all_preds:
        # Ultralytics Results list is aligned with img order
        r = fold_preds[idx]
        dl = r.boxes.data.cpu().numpy()      # (n, 6)  x1 y1 x2 y2 conf cls
        if dl.size:
            boxes_list.append(dl[:, :4] / IMG_SIZE)   # normalise 0-1
            scores_list.append(dl[:, 4])
        else:
            boxes_list.append(np.zeros((0,4)))
            scores_list.append(np.zeros((0,)))

    # perform WBF across 5 lists
    if any(len(b) for b in boxes_list):
        boxes, scores, _ = weighted_boxes_fusion(
            boxes_list, scores_list,
            labels_list=[np.zeros_like(s) for s in scores_list],
            iou_thr=IOU_WBF, skip_box_thr=CONF_THRES
        )
        # Save to txt (class 0)
        txt_path = OUT_DIR / f'{img_path.stem}.txt'
        with open(txt_path, 'w') as f:
            for b, s in zip(boxes, scores):
                x1, y1, x2, y2 = b
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w/2, y1 + h/2
                f.write(f'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {s:.4f}\n')

print('✓ Pseudo-labels written to', OUT_DIR)
