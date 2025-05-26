#!/usr/bin/env python3
"""
Sweep confidence thresholds for each fold model, return the best global value.
"""
import argparse, numpy as np, re
from pathlib import Path
from ultralytics import YOLO

def sweep(model, data_yaml, thr):
    m = YOLO(str(model))
    res = [m.val(data=str(data_yaml), conf=t,
                iou=0.5, plots=False, verbose=False).box.map for t in thr]

    return res

def main(a):
    thr = np.arange(0.05, 0.96, 0.05)
    scores = []
    for d in a.fold_dirs:
        d = Path(d)
        model = next((d / "weights").glob("best*.pt"))
        idx   = int(re.search(r"fold(\d+)", d.name).group(1))
        scores.append(sweep(model, f"fold{idx}.yaml", thr))
    m = np.stack(scores).mean(0)
    best = thr[m.argmax()]
    for t, s in zip(thr, m):
        print(f"thr {t:.2f}  mAP50-95 {s:.4f}")
    print(f"\n>>> BEST {best:.2f}")
    Path("best_threshold.txt").write_text(str(best))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dirs", nargs="+", required=True)
    main(ap.parse_args())
