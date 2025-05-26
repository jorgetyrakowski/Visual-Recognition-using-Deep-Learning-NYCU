#!/usr/bin/env python3
"""
Convert Global-Wheat train.csv to YOLO txt labels (class=0).
Each txt line: 0 cx cy w h  (normalised).
"""
import argparse, pandas as pd, re, cv2
from pathlib import Path
from tqdm import tqdm

def parse_bbox(b):
    # "[x, y, w, h]"  → list[float]
    return list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", b)))

def main(a):
    csv = pd.read_csv(a.csv)
    out = Path(a.out_labels); out.mkdir(parents=True, exist_ok=True)
    img_dir = Path(a.img_dir)
    wh_cache = {}
    for _, r in tqdm(csv.iterrows(), total=len(csv), desc="Converting"):
        img_id, bbox = r.image_id, parse_bbox(r.bbox)
        x, y, w, h = bbox
        if img_id not in wh_cache:
            img = cv2.imread(str(img_dir / f"{img_id}.jpg"))
            wh_cache[img_id] = (img.shape[1], img.shape[0])  # W,H
        W, H = wh_cache[img_id]
        cx, cy = x + w/2, y + h/2
        with open(out / f"{img_id}.txt", "a") as f:
            f.write(f"0 {cx/W:.6f} {cy/H:.6f} {w/W:.6f} {h/H:.6f}\n")
    print(f"✓ {len(csv)} boxes written to {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--img_dir", required=True)
    p.add_argument("--out_labels", required=True)
    main(p.parse_args())
