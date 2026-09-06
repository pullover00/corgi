"""Binary change-mask metrics matching PASLCD's own evaluation convention,
not a generic library default.

Reference implementation: ``evaluate.py`` /
``evaluate_segmentation``/``calculate_metrics_torch`` in the PASLCD paper's
own released code (MV3DCD, github.com/Chumsy0725/MV3DCD, CVPR 2025 --
``Multi-View Pose-Agnostic Change Localization with Zero Labels``), which
``run.sh`` calls once per scene instance as
``evaluate.py --gt data/PASLCD/<scene>/<instance>/gt_mask --pred_binary
output/.../binary_masks/``. That script:

  1. Loads each GT mask grayscale and thresholds it at 127 -> {0, 1}.
     PASLCD's gt_mask/*.png are already (almost) binary uint8 {0, 255} --
     verified by inspecting all 500 masks in this copy of the dataset: the
     only values present are 0, 254, and 255, with 254 appearing on a
     handful of stray pixels (PNG resize antialiasing, not a real third
     class). There is no separate ignore/invalid-region encoding anywhere
     in the dataset -- every pixel is scored, which is why there is no
     "valid pixel" mask here.
  2. Loads the predicted mask grayscale; if its shape differs from the GT
     mask's shape, resizes the *prediction* to the GT shape with
     cv2.INTER_NEAREST (nearest-neighbor, since this is a label map, not a
     photo) -- never the other way around. The GT mask's own resolution
     (downsampled from the raw ~4000px photos to a ~1600px longest side) is
     therefore the authoritative evaluation resolution.
  3. Thresholds the (possibly resized) prediction at 127 -> {0, 1} the same
     way.
  4. Computes mIoU/F1 with torchmetrics' ``JaccardIndex(task="binary")`` /
     ``F1Score(task="binary")``, which are the standard confusion-matrix
     formulas IoU = TP/(TP+FP+FN), F1 = 2*TP/(2*TP+FP+FN) -- verified
     against a live torchmetrics 1.5.2 instance in the mv3dcd conda env,
     including its zero_division=0 convention when TP+FP+FN == 0 (an
     all-background GT and an all-background prediction score 0.0, not
     1.0 or NaN). ``compute_binary_metrics`` below reproduces that
     convention exactly, and extends it with Precision/Recall (not part of
     the original script, using the same TP/FP/FN and the same
     zero-division convention) since the SceneDiff pipeline task requested
     them too.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    iou: float
    f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    tn: int

    def as_dict(self) -> dict:
        return asdict(self)


def load_paslcd_gt(path: str) -> np.ndarray:
    """Load and binarize a PASLCD gt_mask/*.png exactly as the official
    evaluate.py does. Returns a {0, 1} uint8 array at the mask's native
    (already-downsampled) resolution."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"could not read GT mask: {path}")
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return (binary // 255).astype(np.uint8)


def binarize_prediction(prediction: np.ndarray, gt_shape: tuple[int, int]) -> np.ndarray:
    """Binarize a predicted change mask and, if needed, resize it to the GT
    mask's own resolution with nearest-neighbor interpolation -- matching
    evaluate.py's "always resize the prediction into the GT's shape"
    convention. ``prediction`` may be a boolean array, a pipeline label
    raster (any nonzero pixel = changed), or an already-binary 0/255 mask.
    """
    pred = np.asarray(prediction)
    if pred.dtype != np.uint8:
        pred = np.where(pred != 0, 255, 0).astype(np.uint8)
    if pred.shape != gt_shape:
        pred = cv2.resize(pred, (gt_shape[1], gt_shape[0]), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(pred, 127, 255, cv2.THRESH_BINARY)
    return (binary // 255).astype(np.uint8)


def compute_binary_metrics(gt: np.ndarray, pred: np.ndarray) -> BinaryMetrics:
    """IoU/F1/Precision/Recall from a {0,1} GT and a {0,1} prediction of the
    same shape, with torchmetrics' zero_division=0 convention: a score is
    0.0 (not 1.0 or NaN) whenever its denominator is zero."""
    if gt.shape != pred.shape:
        raise ValueError(f"gt/pred shape mismatch: {gt.shape} vs {pred.shape}")
    gt_bool, pred_bool = gt.astype(bool), pred.astype(bool)
    tp = int(np.logical_and(gt_bool, pred_bool).sum())
    fp = int(np.logical_and(~gt_bool, pred_bool).sum())
    fn = int(np.logical_and(gt_bool, ~pred_bool).sum())
    tn = int(np.logical_and(~gt_bool, ~pred_bool).sum())

    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0
    f1_denom = 2 * tp + fp + fn
    f1 = (2 * tp) / f1_denom if f1_denom > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return BinaryMetrics(iou=iou, f1=f1, precision=precision, recall=recall, tp=tp, fp=fp, fn=fn, tn=tn)


def tp_fp_fn_visualization(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """TP green / FP red / FN blue / TN black -- the standard
    change-detection error-visualization convention (also used by e.g.
    build_paslcd_gallery.py-style debug tools)."""
    gt_bool, pred_bool = gt.astype(bool), pred.astype(bool)
    vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
    vis[gt_bool & pred_bool] = (0, 200, 0)
    vis[~gt_bool & pred_bool] = (220, 0, 0)
    vis[gt_bool & ~pred_bool] = (0, 80, 220)
    return vis
