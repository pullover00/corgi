#!/usr/bin/env python3
"""Multi-class evaluation of change_pipeline's ChangeSim output against real
per-class ground truth. GT-decoding logic (raw AirSim segmentation IDs ->
our Label enum) ported from change_detect's src/ocmask/changesim.py
(CHANGESIM_RGB_TO_RAW palette, verified there against the official
ChangeSim script/utils/idx2color.txt) -- not re-derived here.

Unlike PASLCD (binary changed/unchanged only) and SceneDiff (presence-only,
no class distinction), ChangeSim gives exact per-pixel class identity, so
this can check whether a MOVED prediction lands on a GT MOVED pixel
specifically, not just "any change."
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ocmask_pipeline.types import Label  # noqa: E402

CHANGESIM_RGB_TO_RAW = {
    (0, 0, 0): 0,
    (81, 38, 0): 1,
    (41, 36, 132): 2,
    (25, 48, 16): 3,
    (131, 192, 13): 4,
}
RAW_TO_LABEL = {
    0: int(Label.UNCHANGED),
    1: int(Label.ADDED),
    2: int(Label.REMOVED),
    3: int(Label.MOVED),
    4: int(Label.REPLACED),
}
LABEL_NAMES = {int(l): l.name for l in Label}


def decode_target_array(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target)
    if target.ndim == 2:
        if not np.issubdtype(target.dtype, np.integer):
            raise ValueError(f"Scalar ChangeSim labels must have integer dtype, got {target.dtype}")
        unknown = np.setdiff1d(np.unique(target), np.arange(5))
        if len(unknown):
            raise ValueError(f"Unknown scalar ChangeSim labels: {unknown.tolist()}")
        return target.astype(np.uint8, copy=False)
    if target.ndim != 3 or target.shape[2] < 3:
        raise ValueError(f"Expected HW or HWC ChangeSim target, got {target.shape}")
    rgb = target[..., :3]
    raw = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for color, label in CHANGESIM_RGB_TO_RAW.items():
        raw[np.all(rgb == color, axis=2)] = label
    if np.any(raw == 255):
        unknown = np.unique(rgb[raw == 255].reshape(-1, 3), axis=0)
        raise ValueError(f"Unknown ChangeSim RGB colors: {unknown[:20].tolist()}")
    return raw


def normalize_target(path: Path) -> np.ndarray:
    raw = decode_target_array(np.asarray(Image.open(path)))
    output = np.zeros_like(raw)
    for r, canonical in RAW_TO_LABEL.items():
        output[raw == r] = canonical
    return output


def resize_labels(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if labels.shape[:2] == shape:
        return labels
    return cv2.resize(labels, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    OUT_ROOT = REPO / "results/changesim_test20"
    query_meta = json.loads((OUT_ROOT / "query_meta.json").read_text())

    agg_confusion = defaultdict(lambda: {"pixels": 0, "tp": 0, "fp": 0})
    binary_tp = binary_fp = binary_fn = binary_tn = 0
    per_pair = {}
    failed = []

    for pair_id, meta in query_meta.items():
        try:
            gt = normalize_target(Path(meta["target"]))
            pred = cv2.imread(str(REPO / meta["output_dir"] / "labels.png"), cv2.IMREAD_GRAYSCALE)
            if pred is None:
                raise FileNotFoundError(meta["output_dir"])
            pred = resize_labels(pred, gt.shape)
        except Exception as e:
            print(f"[{pair_id}] FAILED: {e}")
            failed.append(pair_id)
            continue

        pair_conf = {}
        total_fp_pair = 0
        for value, name in LABEL_NAMES.items():
            if value == 0:
                continue
            mask = pred == value
            n = int(mask.sum())
            tp = int((mask & (gt == value)).sum())
            fp = n - tp
            pair_conf[name] = {"pixels": n, "tp": tp, "fp": fp}
            agg_confusion[name]["pixels"] += n
            agg_confusion[name]["tp"] += tp
            agg_confusion[name]["fp"] += fp
            total_fp_pair += fp

        pred_changed = pred != 0
        gt_changed = gt != 0
        binary_tp += int((pred_changed & gt_changed).sum())
        binary_fp += int((pred_changed & ~gt_changed).sum())
        binary_fn += int((~pred_changed & gt_changed).sum())
        binary_tn += int((~pred_changed & ~gt_changed).sum())

        per_pair[pair_id] = pair_conf
        print(f"[{pair_id}] " + " ".join(f"{k}:tp={v['tp']} fp={v['fp']}" for k, v in pair_conf.items() if v["pixels"] > 0))

    print("=" * 78)
    print(f"n_pairs succeeded: {len(per_pair)}/{len(query_meta)} (failed: {failed})")
    print("=" * 78)
    print(f"{'class':10s} {'pixels':>10s} {'tp':>10s} {'fp':>10s} {'precision':>10s} {'share_of_fp':>12s}")
    total_fp = sum(d["fp"] for d in agg_confusion.values())
    for name in ["ADDED", "REMOVED", "MOVED", "WARPED", "REPLACED"]:
        d = agg_confusion[name]
        prec = d["tp"] / d["pixels"] if d["pixels"] else float("nan")
        share = d["fp"] / total_fp if total_fp else float("nan")
        print(f"{name:10s} {d['pixels']:10d} {d['tp']:10d} {d['fp']:10d} {prec:10.4f} {share:12.4f}")

    binary_union = binary_tp + binary_fp + binary_fn
    iou = binary_tp / binary_union if binary_union else 0.0
    precision = binary_tp / (binary_tp + binary_fp) if (binary_tp + binary_fp) else 0.0
    recall = binary_tp / (binary_tp + binary_fn) if (binary_tp + binary_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"\nBINARY: iou={iou:.4f} f1={f1:.4f} precision={precision:.4f} recall={recall:.4f}")
    print(f"binary tp={binary_tp} fp={binary_fp} fn={binary_fn} tn={binary_tn}")

    out = {
        "per_pair": per_pair, "failed": failed,
        "agg_confusion": dict(agg_confusion),
        "binary": {"tp": binary_tp, "fp": binary_fp, "fn": binary_fn, "tn": binary_tn,
                   "iou": iou, "f1": f1, "precision": precision, "recall": recall},
    }
    out_path = OUT_ROOT / "changesim_eval_20.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
