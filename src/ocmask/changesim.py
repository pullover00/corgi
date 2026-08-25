from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .io import save_json
from .types import Label

# Official ChangeSim ``script/utils/idx2color.txt`` palette. The first five
# AirSim segmentation IDs encode static, new, missing, rotated, and replaced.
CHANGESIM_RGB_TO_RAW = {
    (0, 0, 0): 0,
    (81, 38, 0): 1,
    (41, 36, 132): 2,
    (25, 48, 16): 3,
    (131, 192, 13): 4,
}


@dataclass(frozen=True)
class ChangeSimPair:
    pair_id: str
    image0: Path
    image1: Path
    target: Path
    stratum: tuple[int, ...]


def decode_target_array(target: np.ndarray) -> np.ndarray:
    """Decode either official RGB palette labels or pre-decoded scalar labels."""
    target = np.asarray(target)
    if target.ndim == 2:
        if not np.issubdtype(target.dtype, np.integer):
            raise ValueError(
                f"Scalar ChangeSim labels must have an integer dtype, got {target.dtype}"
            )
        # Validate before narrowing to uint8: otherwise 256 wraps to 0 and a
        # negative value wraps into the valid-looking byte range.
        unknown = np.setdiff1d(np.unique(target), np.arange(5))
        if len(unknown):
            raise ValueError(f"Unknown scalar ChangeSim labels: {unknown.tolist()}")
        return target.astype(np.uint8, copy=False)
    if target.ndim != 3 or target.shape[2] < 3:
        raise ValueError(f"Expected an HW or HWC ChangeSim target, got {target.shape}")

    rgb = target[..., :3]
    raw = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for color, label in CHANGESIM_RGB_TO_RAW.items():
        raw[np.all(rgb == color, axis=2)] = label
    if np.any(raw == 255):
        unknown = np.unique(rgb[raw == 255].reshape(-1, 3), axis=0)
        raise ValueError(f"Unknown ChangeSim RGB colors: {unknown[:20].tolist()}")
    return raw


def load_manifest(
    path: str | Path, *, require_declared_classes: bool = False
) -> list[ChangeSimPair]:
    """Load explicit pairs, optionally forbidding GT reads for missing strata."""
    base = Path(path).resolve().parent
    pairs = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        resolve = lambda value: (base / value).resolve() if not Path(value).is_absolute() else Path(value)
        try:
            target = resolve(row["target"])
            # Reading class presence from ground truth is used only to choose a
            # representative evaluation subset; labels never enter inference.
            if "classes" in row:
                stratum = tuple(sorted(int(value) for value in row["classes"] if int(value) != 0))
            else:
                if require_declared_classes:
                    raise ValueError(
                        f"Manifest line {line_number} has no precomputed classes; "
                        "refusing to open ground truth before prediction freeze"
                    )
                raw = decode_target_array(np.asarray(Image.open(target)))
                stratum = tuple(int(value) for value in np.unique(raw) if int(value) != 0)
            pairs.append(ChangeSimPair(row["id"], resolve(row["image0"]), resolve(row["image1"]), target, stratum))
        except KeyError as exc:
            raise ValueError(f"Manifest line {line_number} lacks {exc.args[0]}") from exc
    return pairs


def deterministic_subset(pairs: list[ChangeSimPair], fraction: float, seed: int) -> list[ChangeSimPair]:
    """Select a seeded subset while preserving class-combination proportions."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    target_count = max(1, round(len(pairs) * fraction))
    groups: dict[tuple[int, ...], list[ChangeSimPair]] = {}
    for pair in pairs:
        groups.setdefault(pair.stratum, []).append(pair)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    # Largest-remainder allocation produces exactly target_count examples while
    # approximating each stratum's prevalence in the full manifest.
    exact = {key: len(group) * target_count / len(pairs) for key, group in groups.items()}
    counts = {key: min(len(groups[key]), int(value)) for key, value in exact.items()}
    remaining = target_count - sum(counts.values())
    order = sorted(groups, key=lambda key: (-(exact[key] - counts[key]), key))
    while remaining:
        progressed = False
        for key in order:
            if counts[key] < len(groups[key]):
                counts[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            break
    selected = [pair for key in sorted(groups) for pair in groups[key][: counts[key]]]
    rng.shuffle(selected)
    return selected


def normalize_target(path: str | Path, mapping: dict[int, int] | None = None) -> np.ndarray:
    """Translate raw ChangeSim values into the reproduction's canonical labels."""
    target = decode_target_array(np.asarray(Image.open(path)))
    # Default manifest contract: 0 static, 1 new, 2 missing, 3 rotated, 4 replaced.
    mapping = mapping or {
        0: int(Label.UNCHANGED),
        1: int(Label.ADDED),
        2: int(Label.REMOVED),
        3: int(Label.MOVED),
        4: int(Label.REPLACED),
    }
    output = np.full(target.shape, 255, dtype=np.uint8)
    for raw, canonical in mapping.items():
        output[target == raw] = canonical
    unknown = np.unique(target[output == 255])
    if len(unknown):
        raise ValueError(f"Unknown target label values in {path}: {unknown.tolist()}")
    return output


class MetricAccumulator:
    """Accumulate a fixed-size confusion matrix instead of retaining every image."""

    def __init__(self):
        self.confusion = np.zeros((6, 6), dtype=np.int64)
        self.count = 0

    def add(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """Add one prediction whose shape exactly matches the ground truth."""
        prediction = np.asarray(prediction)
        target = np.asarray(target)
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target shapes must match exactly: "
                f"prediction={prediction.shape}, target={target.shape}"
            )
        for name, values in (("prediction", prediction), ("target", target)):
            if not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"{name} labels must have an integer dtype, got {values.dtype}")
            invalid = np.unique(values[(values < 0) | (values >= 6)])
            if len(invalid):
                raise ValueError(
                    f"{name} contains labels outside the canonical 0..5 range: "
                    f"{invalid.tolist()}"
                )
        prediction = prediction.astype(np.uint8, copy=False)
        target = target.astype(np.uint8, copy=False)
        self.confusion += np.bincount(
            (target.astype(np.int64) * 6 + prediction).ravel(),
            minlength=36,
        ).reshape(6, 6)
        self.count += 1

    def add_confusion(self, confusion: list[list[int]] | np.ndarray) -> None:
        """Restore one completed pair from its compact progress checkpoint."""
        matrix = np.asarray(confusion, dtype=np.int64)
        if matrix.shape != (6, 6):
            raise ValueError(f"Expected a 6x6 confusion matrix, got {matrix.shape}")
        self.confusion += matrix
        self.count += 1

    def compute(self) -> dict:
        """Compute the paper's one-vs-rest metrics from aggregate pixel counts."""
        if not self.count:
            raise ValueError("No successful predictions to evaluate")

        def scores(tp: int, fp: int, fn: int, support: int) -> dict:
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            return {
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            }

        labels = [Label.UNCHANGED, Label.ADDED, Label.REMOVED, Label.MOVED, Label.REPLACED]
        multiclass = {}
        for label in labels:
            index = int(label)
            tp = int(self.confusion[index, index])
            fp = int(self.confusion[:, index].sum() - tp)
            fn = int(self.confusion[index, :].sum() - tp)
            support = int(self.confusion[index, :].sum())
            multiclass[label.name.lower()] = scores(tp, fp, fn, support)

        unchanged_tp = int(self.confusion[0, 0])
        unchanged_fp = int(self.confusion[1:, 0].sum())
        unchanged_fn = int(self.confusion[0, 1:].sum())
        changed_tp = int(self.confusion[1:, 1:].sum())
        changed_fp = unchanged_fn
        changed_fn = unchanged_fp
        binary = {
            "unchanged": scores(
                unchanged_tp, unchanged_fp, unchanged_fn, int(self.confusion[0, :].sum())
            ),
            "changed": scores(
                changed_tp, changed_fp, changed_fn, int(self.confusion[1:, :].sum())
            ),
        }
        present = [value for value in multiclass.values() if value["support"] > 0]
        return {
            "multiclass": multiclass,
            "multiclass_miou": float(np.mean([value["iou"] for value in present])),
            "multiclass_macro_f1": float(np.mean([value["f1"] for value in present])),
            "binary": binary,
            "binary_miou": float(np.mean([value["iou"] for value in binary.values()])),
            "binary_macro_f1": float(np.mean([value["f1"] for value in binary.values()])),
        }
