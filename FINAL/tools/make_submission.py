#!/usr/bin/env python3
"""
Assemble YOLO txt predictions into submission.csv (optionally WBF fused).
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from ensemble_boxes import weighted_boxes_fusion

def load_one(txt):
    b, s = [], []
    for line in Path(txt).read_text().strip().splitlines():
        cls, cx, cy, w, h, conf = map(float, line.split())
        x1, y1 = cx - w/2, cy - h/2
        b.append([x1, y1, x1 + w, y1 + h]); s.append(conf)
    return np.array(b), np.array(s)

def xywh(box):
    x1, y1, x2, y2 = box
    return x1, y1, x2 - x1, y2 - y1

def main(a):
    rows=[]
    for txt in Path(a.pred_dir).glob("*.txt"):
        boxes, scores = load_one(txt)
        if a.wbf and len(boxes):
            boxes, scores, _ = weighted_boxes_fusion(
                [boxes], [scores], [np.zeros_like(scores)],
                iou_thr=0.55)
        pred=""
        for b,s in zip(boxes, scores):
            x,y,w,h = xywh(b)
            pred += f"{s:.4f} {x:.4f} {y:.4f} {w:.4f} {h:.4f} "
        rows.append({"image_id": txt.stem, "PredictionString": pred.strip()})
    pd.DataFrame(rows).to_csv(a.output, index=False)
    print(f"✓ {a.output} written ({len(rows)} rows)")

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--output",  default="submission.csv")
    p.add_argument("--wbf", action="store_true")
    main(p.parse_args())
