from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .types import Label


@dataclass
class ClassMetrics:
    support: int
    precision: float
    recall: float
    f1: float
    iou: float


def class_metrics(prediction: np.ndarray, target: np.ndarray, label: int) -> ClassMetrics:
    """Compute one-vs-rest segmentation metrics for a single label."""
    pred = prediction == label
    true = target == label
    tp = int(np.logical_and(pred, true).sum())
    fp = int(np.logical_and(pred, ~true).sum())
    fn = int(np.logical_and(~pred, true).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return ClassMetrics(int(true.sum()), precision, recall, f1, iou)


def evaluate_arrays(prediction: np.ndarray, target: np.ndarray) -> dict:
    """Compute the binary and multiclass protocols reported for ChangeSim."""
    labels = [Label.UNCHANGED, Label.ADDED, Label.REMOVED, Label.MOVED, Label.REPLACED]
    multiclass = {label.name.lower(): asdict(class_metrics(prediction, target, int(label))) for label in labels}
    pred_binary = prediction != Label.UNCHANGED
    target_binary = target != Label.UNCHANGED
    binary = {
        "unchanged": asdict(class_metrics(pred_binary.astype(np.uint8), target_binary.astype(np.uint8), 0)),
        "changed": asdict(class_metrics(pred_binary.astype(np.uint8), target_binary.astype(np.uint8), 1)),
    }
    multiclass_present = [v for v in multiclass.values() if v["support"] > 0]
    return {
        "multiclass": multiclass,
        "multiclass_miou": float(np.mean([v["iou"] for v in multiclass_present])),
        "multiclass_macro_f1": float(np.mean([v["f1"] for v in multiclass_present])),
        "binary": binary,
        "binary_miou": float(np.mean([v["iou"] for v in binary.values()])),
        "binary_macro_f1": float(np.mean([v["f1"] for v in binary.values()])),
    }
