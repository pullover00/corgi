#!/usr/bin/env python3
"""Object-level TP/FP/FN and per-object stage-survival tracing.

ChangeSim's ground truth is a single per-pixel semantic label map -- there is
no instance/object ground truth. This script derives objects by connected-
component-izing that semantic map (and, symmetrically, the pipeline's own
predicted label map) so that object-level detection quality (not just
aggregate pixel IoU) can be reported: for each GT changed object, was there
*any* predicted object of the same class overlapping it enough to count as a
detection, and if not, why not.

It is a post-hoc, read-only analysis over one evaluation run's frozen output
(``ocmask evaluate changesim --full-pipeline --save-stage-artifacts``). It
never touches inference, never opens ground truth before predictions exist
on disk (the run it reads has already frozen every prediction), and computes
nothing that affects any prediction pixel.

Stage-survival tracing answers "for this GT object, at which stage does the
pipeline's own intermediate label map first stop agreeing with ground truth
inside this object's footprint" by re-reading the per-stage label rasters
``--save-stage-artifacts`` wrote (03 baseline -> 07 after fusion -> 08 after
veto -> 10 after association -> 11 final). This is necessarily a pixel-
footprint match, not a proposal-ID match: the pipeline's internal proposal
IDs are not stable object identities across stages (a moved object is a
different proposal ID in the source and target images), so "the same GT
object" is defined the only way ground truth allows -- by its GT pixel mask,
re-scored against each stage's own raster at those same pixels.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from ocmask.changesim import load_manifest, normalize_target  # noqa: E402
from ocmask.io import save_json  # noqa: E402
from ocmask.reproducibility import pair_output_directory  # noqa: E402
from ocmask.types import Label  # noqa: E402

CHANGED_LABELS = (Label.ADDED, Label.REMOVED, Label.MOVED, Label.REPLACED)

# In pipeline order. Each entry is (stage_name, path_relative_to_pair_dir).
# Only stages `--save-stage-artifacts` actually wrote are used; earlier
# stages are skipped for a pair frozen without that flag (see `_stage_paths`).
STAGE_LABEL_FILES = (
    ("03_baseline", "03_tracking/labels.png"),
    ("07_after_fusion", "07_evidence_fusion/labels_after_fusion.png"),
    ("08_after_veto", "08_feature_veto_gate/labels_after_veto.png"),
    ("10_after_association", "10_association_resolution/labels_after_association.png"),
    ("11_final_guarded", "labels_guarded.png"),
    ("11_final_full", "labels.png"),
)


@dataclass
class ObjectMatch:
    gt_component_id: int
    gt_class: str
    gt_area: int
    matched_component_id: int | None
    matched_class: str | None
    iou: float
    outcome: str  # "true_positive" | "misclassified" | "false_negative"


@dataclass
class PairObjectReport:
    pair_id: str
    gt_objects: list[ObjectMatch]
    predicted_only: list[dict[str, Any]]  # unmatched predicted components (false positives)
    object_confusion: dict[str, dict[str, int]]  # gt_class -> {predicted_class_or_none: count}
    stage_survival: list[dict[str, Any]]


def _connected_components(label_map: np.ndarray, classes: tuple[Label, ...]) -> list[dict[str, Any]]:
    """Split ``label_map`` into per-class 4-connected components."""

    components = []
    for label in classes:
        mask = label_map == int(label)
        if not mask.any():
            continue
        component_ids, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
        for component_id in range(1, count + 1):
            component_mask = component_ids == component_id
            components.append(
                {
                    "class": label.name.lower(),
                    "mask": component_mask,
                    "area": int(component_mask.sum()),
                }
            )
    return components


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = int(np.logical_and(a, b).sum())
    if intersection == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return intersection / union if union else 0.0


def match_objects(
    gt_components: list[dict[str, Any]],
    predicted_components: list[dict[str, Any]],
    *,
    minimum_iou: float,
) -> tuple[list[ObjectMatch], list[int]]:
    """Greedily match each GT object to its best-overlapping predicted object.

    Greedy-by-descending-IoU (not Hungarian): ChangeSim GT components can
    heavily outnumber predicted components (or vice versa) per pair, and the
    question here is per-GT-object detection quality, not a one-to-one
    global assignment. A predicted component may be consumed by at most one
    GT object; any predicted component of the *same class* consumed first by
    a higher-IoU GT match is unavailable to a later, lower-IoU one.
    """

    candidates = []
    for gt_index, gt in enumerate(gt_components):
        for pred_index, pred in enumerate(predicted_components):
            iou = _iou(gt["mask"], pred["mask"])
            if iou > 0:
                candidates.append((iou, gt_index, pred_index))
    candidates.sort(key=lambda item: -item[0])

    matched_gt: dict[int, tuple[int, float]] = {}
    consumed_pred: set[int] = set()
    for iou, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in consumed_pred:
            continue
        matched_gt[gt_index] = (pred_index, iou)
        consumed_pred.add(pred_index)

    results = []
    for gt_index, gt in enumerate(gt_components):
        if gt_index in matched_gt:
            pred_index, iou = matched_gt[gt_index]
            pred = predicted_components[pred_index]
            if iou >= minimum_iou:
                outcome = "true_positive" if pred["class"] == gt["class"] else "misclassified"
                results.append(
                    ObjectMatch(
                        gt_component_id=gt_index,
                        gt_class=gt["class"],
                        gt_area=gt["area"],
                        matched_component_id=pred_index,
                        matched_class=pred["class"],
                        iou=iou,
                        outcome=outcome,
                    )
                )
                continue
        results.append(
            ObjectMatch(
                gt_component_id=gt_index,
                gt_class=gt["class"],
                gt_area=gt["area"],
                matched_component_id=None,
                matched_class=None,
                iou=0.0,
                outcome="false_negative",
            )
        )
    matched_pred_indices = {pred_index for pred_index, _ in matched_gt.values()}
    unmatched_pred = [index for index in range(len(predicted_components)) if index not in matched_pred_indices]
    return results, unmatched_pred


def _majority_class(label_map: np.ndarray, mask: np.ndarray) -> str:
    values, counts = np.unique(label_map[mask], return_counts=True)
    winner = int(values[np.argmax(counts)])
    return Label(winner).name.lower()


def trace_stage_survival(
    pair_dir: Path, gt_components: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """For each GT changed object, find each saved stage's majority verdict.

    Returns one record per GT object with the majority predicted class at
    every stage that was actually saved for this pair, plus
    ``first_wrong_stage``/``first_recovered_stage`` -- the first stage (in
    pipeline order) whose majority verdict disagrees/agrees with GT, or
    ``None`` if every saved stage agrees/none do.
    """

    available_stages = [
        (name, pair_dir / relative) for name, relative in STAGE_LABEL_FILES if (pair_dir / relative).is_file()
    ]
    if not available_stages:
        return []
    stage_maps = {name: np.asarray(Image.open(path)) for name, path in available_stages}

    records = []
    for gt_index, gt in enumerate(gt_components):
        per_stage = {}
        first_wrong = None
        first_recovered = None
        was_ever_right = False
        for name, _ in available_stages:
            majority = _majority_class(stage_maps[name], gt["mask"])
            correct = majority == gt["class"]
            per_stage[name] = {"majority_class": majority, "correct": correct}
            if correct:
                was_ever_right = True
                if first_wrong is not None and first_recovered is None:
                    first_recovered = name
            elif first_wrong is None:
                first_wrong = name
        records.append(
            {
                "gt_component_id": gt_index,
                "gt_class": gt["class"],
                "gt_area": gt["area"],
                "per_stage": per_stage,
                "first_wrong_stage": first_wrong,
                "first_recovered_stage": first_recovered,
                "never_correct": not was_ever_right,
            }
        )
    return records


def analyze_pair(
    pair_id: str,
    target_path: Path,
    prediction_path: Path,
    pair_dir: Path,
    *,
    minimum_iou: float,
) -> PairObjectReport:
    target = normalize_target(target_path)
    with Image.open(prediction_path) as image:
        prediction = np.asarray(image, dtype=np.uint8).copy()
    if prediction.shape != target.shape:
        raise ValueError(f"{pair_id}: prediction/target shape mismatch {prediction.shape} vs {target.shape}")

    gt_components = _connected_components(target, CHANGED_LABELS)
    predicted_components = _connected_components(prediction, CHANGED_LABELS)
    matches, unmatched_pred = match_objects(gt_components, predicted_components, minimum_iou=minimum_iou)

    confusion: dict[str, dict[str, int]] = {}
    for match in matches:
        row = confusion.setdefault(match.gt_class, {})
        key = match.matched_class or "none"
        row[key] = row.get(key, 0) + 1

    predicted_only = [
        {"component_id": index, "class": predicted_components[index]["class"], "area": predicted_components[index]["area"]}
        for index in unmatched_pred
    ]
    stage_survival = trace_stage_survival(pair_dir, gt_components)
    return PairObjectReport(
        pair_id=pair_id,
        gt_objects=matches,
        predicted_only=predicted_only,
        object_confusion=confusion,
        stage_survival=stage_survival,
    )


def summarize(reports: list[PairObjectReport]) -> dict[str, Any]:
    per_class: dict[str, dict[str, int]] = {}
    for report in reports:
        for match in report.gt_objects:
            row = per_class.setdefault(match.gt_class, {"tp": 0, "misclassified": 0, "fn": 0})
            if match.outcome == "true_positive":
                row["tp"] += 1
            elif match.outcome == "misclassified":
                row["misclassified"] += 1
            else:
                row["fn"] += 1
        for entry in report.predicted_only:
            row = per_class.setdefault(entry["class"], {"tp": 0, "misclassified": 0, "fn": 0, "fp": 0})
            row["fp"] = row.get("fp", 0) + 1
    for row in per_class.values():
        row.setdefault("fp", 0)
        tp = row["tp"]
        fp = row["fp"] + row["misclassified"]
        fn = row["fn"] + row["misclassified"]
        row["precision"] = tp / (tp + fp) if (tp + fp) else 0.0
        row["recall"] = tp / (tp + fn) if (tp + fn) else 0.0
    never_correct_by_class: dict[str, int] = {}
    first_wrong_stage_histogram: dict[str, int] = {}
    for report in reports:
        for record in report.stage_survival:
            if record["never_correct"]:
                never_correct_by_class[record["gt_class"]] = never_correct_by_class.get(record["gt_class"], 0) + 1
            stage = record["first_wrong_stage"] or "always_correct"
            first_wrong_stage_histogram[stage] = first_wrong_stage_histogram.get(stage, 0) + 1
    return {
        "pairs_analyzed": len(reports),
        "object_level_per_class": per_class,
        "stage_survival": {
            "never_correct_by_class": never_correct_by_class,
            "first_wrong_stage_histogram": first_wrong_stage_histogram,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", required=True, help="an evaluate-changesim run directory (contains pairs/, report.json)")
    parser.add_argument("--manifest", required=True, help="the same ChangeSim manifest the run used")
    parser.add_argument("--prediction-variant", choices=["guarded", "full"], default="guarded")
    parser.add_argument("--minimum-iou", type=float, default=0.10, help="minimum GT/predicted object IoU counted as a detection")
    parser.add_argument("--output", default=None, help="output JSON path (default: <run-output>/object_level_analysis.json)")
    parser.add_argument("--limit", type=int, default=None, help="analyze only the first N pairs (for a quick check)")
    args = parser.parse_args(argv)

    run_output = Path(args.run_output).resolve()
    pairs = load_manifest(Path(args.manifest).resolve(), require_declared_classes=True)
    if args.limit:
        pairs = pairs[: args.limit]

    prediction_filename = "labels_guarded.png" if args.prediction_variant == "guarded" else "labels.png"
    reports: list[PairObjectReport] = []
    skipped: list[dict[str, str]] = []
    for pair in pairs:
        pair_dir = pair_output_directory(run_output / "pairs", pair.pair_id)
        prediction_path = pair_dir / prediction_filename
        if not prediction_path.is_file():
            skipped.append({"id": pair.pair_id, "reason": "no_frozen_prediction"})
            continue
        report = analyze_pair(
            pair.pair_id, pair.target, prediction_path, pair_dir, minimum_iou=args.minimum_iou
        )
        reports.append(report)

    summary = summarize(reports)
    output_path = Path(args.output) if args.output else run_output / "object_level_analysis.json"
    save_json(
        output_path,
        {
            "prediction_variant": args.prediction_variant,
            "minimum_iou": args.minimum_iou,
            "summary": summary,
            "skipped": skipped,
            "pairs": [
                {
                    "pair_id": report.pair_id,
                    "gt_objects": [asdict(match) for match in report.gt_objects],
                    "predicted_only": report.predicted_only,
                    "object_confusion": report.object_confusion,
                    "stage_survival": report.stage_survival,
                }
                for report in reports
            ],
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
