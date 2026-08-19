#!/usr/bin/env python3
"""Run the isolated DINOv2 identity + aligned-location ten-pair experiment.

A drop-in backbone-swap ablation of run_sam3_identity_location_experiment.py:
every calibration, pairing, and classification function is imported
unchanged from ocmask.stages.sam3_identity_location (all of it is
backbone-agnostic once given a dense C x H x W feature map). Only the
feature extractor differs (DINOv2 full-image patch embedding instead of
SAM3 image embedding). The on-disk cache filename (sam3_features.npz/json)
is deliberately unchanged so run_sam3_feature_veto_gate_experiment.py can
consume this experiment's output as a drop-in identity_evaluation root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ocmask.changesim import MetricAccumulator, load_manifest, normalize_target
from ocmask.config import load_config
from ocmask.adapters.dinov2 import Dinov2FeatureExtractor
from ocmask.stages.sam3_identity_location import (
    calibrate_identity_threshold,
    classify_identity_location,
    compose_identity_labels,
    cosine_similarity_matrix,
    mask_descriptors,
    pairwise_mask_iou,
)
from ocmask.stages.sam3_pairwise import (
    load_cached_inputs,
    load_proposal_cache,
    proposals_to_objects,
)
from ocmask.io import load_rgb, save_image, save_json
from ocmask.masks import filter_visible
from ocmask.types import Label, ObjectMask
from ocmask.visualization import colorize, overlay


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/stages/changesim-dinov2-identity-location-no-splat.yaml"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benchmark-dinov2-identity-location-no-splat"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pair-id", action="append", default=[])
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Keep reusable features/decisions/labels but omit qualitative images.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate immutable inputs without loading SAM3 or reading ground truth.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _implementation_hash() -> str:
    repository = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        repository / "src/ocmask/stages/sam3_identity_location.py",
        repository / "src/ocmask/stages/sam3_pairwise.py",
        repository / "src/ocmask/masks.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(repository)).encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _table3(metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        "binary": {
            "changed": 100.0 * metrics["binary"]["changed"]["iou"],
            "unchanged": 100.0 * metrics["binary"]["unchanged"]["iou"],
            "miou": 100.0 * metrics["binary_miou"],
        },
        "multiclass": {
            **{
                name: 100.0 * metrics["multiclass"][name]["iou"]
                for name in ("added", "removed", "moved", "replaced", "unchanged")
            },
            "miou": 100.0 * metrics["multiclass_miou"],
        },
    }


def _validate_parent(path: Path) -> tuple[dict, list[str], dict[str, dict]]:
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    selection = json.loads((path / "selection.json").read_text(encoding="utf-8"))
    selected = list(selection["ids"])
    records = {record["id"]: record for record in report["pairs"]}
    if report.get("failures"):
        raise RuntimeError("the frozen SAM3+SAM2 parent contains failures")
    if report["protocol"]["pairs_succeeded"] != len(selected):
        raise RuntimeError("the frozen SAM3+SAM2 parent is incomplete")
    if set(selected) != set(records):
        raise RuntimeError("parent selection and report contain different pairs")
    return report, selected, records


def _proposal_paths(config: dict, pair_id: str) -> tuple[Path, Path]:
    root = Path(config["proposal_cache_parent"]).resolve() / "pairs" / pair_id
    cache = root / "proposal_cache"
    return cache / "source.npz", cache / "target.npz"


def _gate_flags(
    parent_artifact: Path,
    stage: str,
    objects: list[ObjectMask],
) -> np.ndarray:
    attempts = json.loads(
        (parent_artifact / "tracking_attempts.json").read_text(encoding="utf-8")
    )["stages"][stage]["attempts"]
    expected_ids = [obj.metadata.get("automatic_proposal_id") for obj in objects]
    attempt_ids = [record.get("proposal_id") for record in attempts]
    if expected_ids != attempt_ids:
        raise RuntimeError(
            f"{parent_artifact.name}/{stage}: visible proposal order differs from frozen gate"
        )
    # True means the clean-render tracker/gateway rejected this proposal and
    # the baseline therefore promoted it to a change candidate.
    return np.asarray(
        [not bool(record["post_consistency_gate_accepted"]) for record in attempts],
        dtype=bool,
    )


def _pair_inputs(config: dict, parent_record: dict) -> dict[str, Any]:
    parent_artifact = Path(parent_record["artifacts"])
    geometry_artifact = Path(parent_record["parent_artifacts"])
    inputs = load_cached_inputs(geometry_artifact)
    source_path, target_path = _proposal_paths(config, parent_record["id"])
    source = proposals_to_objects(load_proposal_cache(source_path))
    target = proposals_to_objects(load_proposal_cache(target_path))
    classification = config["classification"]
    source = filter_visible(
        source,
        inputs.cross_coverage,
        float(classification["visibility_alpha"]),
        int(classification["minimum_mask_area"]),
    )
    target = filter_visible(
        target,
        inputs.cross_coverage,
        float(classification["visibility_alpha"]),
        int(classification["minimum_mask_area"]),
    )
    return {
        "parent_artifact": parent_artifact,
        "geometry_artifact": geometry_artifact,
        "inputs": inputs,
        "source_objects": source,
        "target_objects": target,
        "source_changed": _gate_flags(parent_artifact, "source_to_clean", source),
        "target_changed": _gate_flags(parent_artifact, "target_to_clean", target),
        "source_proposal_path": source_path,
        "target_proposal_path": target_path,
    }


def _pair_fingerprint(context: dict[str, Any], config_hash: str) -> str:
    digest = hashlib.sha256(bytes.fromhex(config_hash))
    paths = (
        context["source_proposal_path"],
        context["target_proposal_path"],
        context["parent_artifact"] / "tracking_attempts.json",
        context["geometry_artifact"] / "render_0_to_1.png",
        context["geometry_artifact"] / "reconstruction.npz",
        context["geometry_artifact"] / "geometry.npz",
    )
    for path in paths:
        digest.update(str(path.name).encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _feature_cache_valid(path: Path, fingerprint: str, checkpoint_hash: str) -> bool:
    metadata = path.with_suffix(".json")
    if not path.is_file() or not metadata.is_file():
        return False
    value = json.loads(metadata.read_text(encoding="utf-8"))
    return (
        value.get("pair_fingerprint_sha256") == fingerprint
        and value.get("checkpoint_sha256") == checkpoint_hash
    )


def _save_feature_cache(
    path: Path,
    source: np.ndarray,
    target: np.ndarray,
    *,
    fingerprint: str,
    checkpoint_hash: str,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, source=source, target=target)
    save_json(
        path.with_suffix(".json"),
        {
            "pair_fingerprint_sha256": fingerprint,
            "checkpoint_sha256": checkpoint_hash,
            "source_shape": list(source.shape),
            "target_shape": list(target.shape),
            "dtype": str(source.dtype),
            "elapsed_seconds": elapsed_seconds,
            "ground_truth_used": False,
        },
    )


def _load_feature_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as cache:
        return (
            np.asarray(cache["source"], np.float32),
            np.asarray(cache["target"], np.float32),
        )


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return (
        (float(xs.mean()), float(ys.mean())) if len(xs) else (0.0, 0.0)
    )


def _identity_visualization(
    source_image: np.ndarray,
    target_image: np.ndarray,
    source_objects: list[ObjectMask],
    target_objects: list[ObjectMask],
    records: list[dict[str, Any]],
) -> np.ndarray:
    """Show moved/replaced associations with semantic colors, not all statics."""

    from scipy.ndimage import binary_erosion

    selected = [record for record in records if record["decision"] != "unchanged"]
    # A no-change pair still receives a small sanity view of its strongest
    # unchanged identities rather than an empty diagnostic.
    if not selected:
        selected = sorted(
            records, key=lambda record: -float(record.get("cosine", 0.0))
        )[:5]
    canvas = np.concatenate((source_image, target_image), axis=1).copy()
    offset = source_image.shape[1]
    colors = {
        "unchanged": (100, 116, 139),
        "moved": (59, 130, 246),
        "replaced": (245, 158, 11),
    }
    for record in selected:
        color = np.asarray(colors.get(str(record["decision"]), (255, 255, 255)))
        source_mask = np.asarray(
            source_objects[int(record["source_index"])].mask, dtype=bool
        )
        target_mask = np.asarray(
            target_objects[int(record["target_index"])].mask, dtype=bool
        )
        for mask, x_offset in ((source_mask, 0), (target_mask, offset)):
            region = canvas[:, x_offset : x_offset + mask.shape[1]]
            region[mask] = np.round(0.58 * region[mask] + 0.42 * color).astype(
                np.uint8
            )
            boundary = mask & ~binary_erosion(mask)
            region[boundary] = color
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    for record in selected:
        source_index = int(record["source_index"])
        target_index = int(record["target_index"])
        start = _centroid(source_objects[source_index].mask)
        end = _centroid(target_objects[target_index].mask)
        end = (end[0] + offset, end[1])
        color = colors.get(str(record["decision"]), (255, 255, 255))
        draw.line((start, end), fill=color, width=3)
        draw.ellipse(
            (start[0] - 4, start[1] - 4, start[0] + 4, start[1] + 4),
            fill=color,
        )
        draw.ellipse(
            (end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4),
            fill=color,
        )
        label = (
            f"S{record.get('source_proposal_id')}→"
            f"T{record.get('target_proposal_id')} {record['decision']}"
        )
        draw.rectangle(
            (end[0] + 5, end[1] - 9, end[0] + 5 + 6 * len(label), end[1] + 8),
            fill=(12, 16, 24),
        )
        draw.text((end[0] + 7, end[1] - 7), label, fill=color)
    return np.asarray(image)


def _error_map(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    output = np.full((*prediction.shape, 3), (34, 42, 55), dtype=np.uint8)
    pred_changed = prediction != int(Label.UNCHANGED)
    target_changed = target != int(Label.UNCHANGED)
    correct = prediction == target
    output[correct] = (65, 150, 90)
    output[pred_changed & ~target_changed] = (235, 55, 55)
    output[~pred_changed & target_changed] = (245, 180, 40)
    output[pred_changed & target_changed & ~correct] = (190, 70, 220)
    return output


def main() -> int:
    args = _arguments()
    config = load_config(args.config)
    output = args.output.resolve()
    parent_path = Path(config["parent_evaluation"]).resolve()
    parent_report, selected_ids, parent_records = _validate_parent(parent_path)
    if args.pair_id and args.limit is not None:
        raise ValueError("--pair-id and --limit cannot be combined")
    if args.pair_id:
        unknown = [pair_id for pair_id in args.pair_id if pair_id not in selected_ids]
        if unknown:
            raise ValueError(f"pair IDs are outside the frozen selection: {unknown}")
        selected_ids = list(dict.fromkeys(args.pair_id))
    elif args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        selected_ids = selected_ids[: args.limit]

    dinov2 = config["dinov2"]
    source_path = Path(dinov2["source"]).resolve()
    checkpoint = Path(dinov2["checkpoint"]).resolve()
    if _git_commit(source_path) != dinov2["source_commit"]:
        raise RuntimeError("DINOv2 source commit differs from the frozen config")
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != dinov2["checkpoint_sha256"]:
        raise RuntimeError("DINOv2 checkpoint SHA-256 differs from the frozen config")

    config_hash = _json_hash(config)
    implementation_hash = _implementation_hash()
    parent_hash = _sha256(parent_path / "report.json")
    execution_hash = _json_hash(
        {
            "config_sha256": config_hash,
            "implementation_sha256": implementation_hash,
            "parent_report_sha256": parent_hash,
            "dinov2_checkpoint_sha256": checkpoint_hash,
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "pairs").mkdir(exist_ok=True)
    save_json(
        output / "experiment.json",
        {
            "experiment_id": config["experiment_id"],
            "config": str(args.config.resolve()),
            "config_sha256": config_hash,
            "implementation_sha256": implementation_hash,
            "execution_sha256": execution_hash,
            "parent_evaluation": str(parent_path),
            "parent_report_sha256": parent_hash,
            "selection": selected_ids,
            "ground_truth_used_in_inference": False,
        },
    )
    save_json(output / "selection.json", {"ids": selected_ids})

    contexts: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, str] = {}
    for number, pair_id in enumerate(selected_ids, 1):
        context = _pair_inputs(config, parent_records[pair_id])
        fingerprint = _pair_fingerprint(context, config_hash)
        contexts[pair_id] = context
        fingerprints[pair_id] = fingerprint
        print(
            f"[validate {number}/{len(selected_ids)}] {pair_id}: "
            f"{len(context['source_objects'])} source, "
            f"{len(context['target_objects'])} target visible proposals",
            flush=True,
        )
    if args.validate_only:
        save_json(
            output / "input_validation.json",
            {
                "passed": True,
                "execution_sha256": execution_hash,
                "pairs": [
                    {
                        "id": pair_id,
                        "pair_fingerprint_sha256": fingerprints[pair_id],
                        "source_visible": len(contexts[pair_id]["source_objects"]),
                        "target_visible": len(contexts[pair_id]["target_objects"]),
                    }
                    for pair_id in selected_ids
                ],
            },
        )
        print("Input validation passed without GPU inference.", flush=True)
        return 0

    # Stage 1: extract all full-image feature maps while SAM3 occupies the GPU.
    missing_features = [
        pair_id
        for pair_id in selected_ids
        if not _feature_cache_valid(
            output / "pairs" / pair_id / "sam3_features.npz",
            fingerprints[pair_id],
            checkpoint_hash,
        )
    ]
    if missing_features:
        extractor = Dinov2FeatureExtractor(config)
        try:
            for number, pair_id in enumerate(missing_features, 1):
                started = time.perf_counter()
                context = contexts[pair_id]
                source_features = extractor.feature_map(context["inputs"].source_render)
                target_features = extractor.feature_map(context["inputs"].target_image)
                cache_path = output / "pairs" / pair_id / "sam3_features.npz"
                _save_feature_cache(
                    cache_path,
                    source_features,
                    target_features,
                    fingerprint=fingerprints[pair_id],
                    checkpoint_hash=checkpoint_hash,
                    elapsed_seconds=time.perf_counter() - started,
                )
                print(
                    f"[features {number}/{len(missing_features)}] {pair_id}: "
                    f"{tuple(source_features.shape)}",
                    flush=True,
                )
        finally:
            extractor.release()
    else:
        print("[features] all SAM3 feature maps cached", flush=True)

    manifest = {pair.pair_id: pair for pair in load_manifest(Path(config["manifest"]))}
    missing_manifest = [pair_id for pair_id in selected_ids if pair_id not in manifest]
    if missing_manifest:
        raise RuntimeError(f"pairs missing from manifest: {missing_manifest}")
    accumulator = MetricAccumulator()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    for number, pair_id in enumerate(selected_ids, 1):
        pair_started = time.perf_counter()
        pair_output = output / "pairs" / pair_id
        pair_output.mkdir(parents=True, exist_ok=True)
        result_path = pair_output / "result.json"
        if result_path.is_file():
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                cached.get("execution_sha256") == execution_hash
                and cached.get("pair_fingerprint_sha256") == fingerprints[pair_id]
            ):
                accumulator.add_confusion(cached["confusion"])
                records.append(cached)
                print(f"[classify {number}/{len(selected_ids)}] {pair_id} (cached)", flush=True)
                continue
        try:
            context = contexts[pair_id]
            source_map, target_map = _load_feature_cache(
                pair_output / "sam3_features.npz"
            )
            source_descriptors = mask_descriptors(
                source_map,
                context["source_objects"],
                minimum_feature_cells=float(dinov2["minimum_feature_cells"]),
            )
            target_descriptors = mask_descriptors(
                target_map,
                context["target_objects"],
                minimum_feature_cells=float(dinov2["minimum_feature_cells"]),
            )
            spatial_iou = pairwise_mask_iou(
                context["source_objects"], context["target_objects"]
            )
            similarity = cosine_similarity_matrix(
                source_descriptors, target_descriptors
            )
            matching = config["matching"]
            calibration = calibrate_identity_threshold(
                similarity,
                spatial_iou,
                ~context["source_changed"],
                ~context["target_changed"],
                source_descriptors.valid,
                target_descriptors.valid,
                fallback_threshold=float(matching["fallback_minimum_cosine"]),
                control_iou=float(matching["calibration_control_iou"]),
                negative_iou=float(matching["calibration_negative_iou"]),
                maximum_negative_acceptance=float(
                    matching["calibration_maximum_negative_acceptance"]
                ),
                minimum_positive_acceptance=float(
                    matching["calibration_minimum_positive_acceptance"]
                ),
                minimum_positive_count=int(
                    matching["calibration_minimum_positive_count"]
                ),
                minimum_negative_count=int(
                    matching["calibration_minimum_negative_count"]
                ),
            )
            classification = classify_identity_location(
                context["source_objects"],
                context["target_objects"],
                source_descriptors,
                target_descriptors,
                context["source_changed"],
                context["target_changed"],
                calibration,
                config,
                spatial_iou=spatial_iou,
            )
            labels, final_counts = compose_identity_labels(
                context["geometry_artifact"], classification, config
            )

            # Ground truth is deliberately loaded only after labels are fixed.
            target = normalize_target(manifest[pair_id].target)
            pair_accumulator = MetricAccumulator()
            pair_accumulator.add(labels, target)
            accumulator.add_confusion(pair_accumulator.confusion)
            target_rgb = load_rgb(manifest[pair_id].image1, labels.shape[::-1])
            save_image(pair_output / "labels.png", labels)
            if not args.cache_only:
                save_image(pair_output / "labels_color.png", colorize(labels))
                save_image(pair_output / "ground_truth.png", colorize(target))
                save_image(pair_output / "overlay.png", overlay(target_rgb, labels))
                save_image(pair_output / "error.png", _error_map(labels, target))
                save_image(
                    pair_output / "identity_matches.png",
                    _identity_visualization(
                        context["inputs"].source_render,
                        context["inputs"].target_image,
                        context["source_objects"],
                        context["target_objects"],
                        classification.match_records,
                    ),
                )
            save_json(
                pair_output / "decisions.json",
                {
                    "diagnostics": classification.diagnostics,
                    "final_object_counts": final_counts,
                    "matches": classification.match_records,
                },
            )
            record = {
                "id": pair_id,
                "status": "success",
                "artifacts": str(pair_output.resolve()),
                "parent_artifacts": str(context["parent_artifact"].resolve()),
                "geometry_artifacts": str(context["geometry_artifact"].resolve()),
                "execution_sha256": execution_hash,
                "pair_fingerprint_sha256": fingerprints[pair_id],
                "confusion": pair_accumulator.confusion.tolist(),
                "diagnostics": {
                    **classification.diagnostics,
                    "source_visible_count": len(context["source_objects"]),
                    "target_visible_count": len(context["target_objects"]),
                    "source_feature_grid": list(source_map.shape),
                    "target_feature_grid": list(target_map.shape),
                    "final_object_counts": final_counts,
                },
                "elapsed_seconds": time.perf_counter() - pair_started,
            }
            save_json(result_path, record)
            records.append(record)
            print(
                f"[classify {number}/{len(selected_ids)}] {pair_id}: "
                f"identity={classification.diagnostics['identity_match_count']}, "
                f"moved={final_counts['moved']}, replaced={final_counts['direct_replaced']}",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "id": pair_id,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            save_json(pair_output / "failure.json", failure)
            failures.append(failure)
            print(f"[classify {number}/{len(selected_ids)}] {pair_id}: FAILED: {exc}", flush=True)

    if not records:
        raise RuntimeError("all identity/location pairs failed")
    metrics = accumulator.compute()
    report = {
        "protocol": {
            "dataset": "ChangeSim",
            "experiment": config["experiment_id"],
            "pairs_selected": len(selected_ids),
            "pairs_succeeded": len(records),
            "fraction": parent_report["protocol"]["parent_fraction"]
            if "parent_fraction" in parent_report["protocol"]
            else parent_report["protocol"].get("fraction"),
            "seed": parent_report["protocol"]["seed"],
        },
        "metrics": metrics,
        "table3_iou_percent": _table3(metrics),
        "parent_table3_iou_percent": parent_report["table3_iou_percent"],
        "pairs": records,
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - run_started,
        "provenance": {
            "config_sha256": config_hash,
            "implementation_sha256": implementation_hash,
            "execution_sha256": execution_hash,
            "parent_report_sha256": parent_hash,
            "dinov2_source_commit": dinov2["source_commit"],
            "dinov2_checkpoint_sha256": checkpoint_hash,
            "ground_truth_used_in_inference": False,
            "tuned_on_test": False,
        },
    }
    save_json(output / "report.json", report)
    print(json.dumps(report["table3_iou_percent"], indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
