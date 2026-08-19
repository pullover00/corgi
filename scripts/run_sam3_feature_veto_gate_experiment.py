#!/usr/bin/env python3
"""Run the fixed-ten SAM3 feature veto of the frozen IoU-area clean gate.

This is an isolated experiment.  It never edits the production pipeline or
any parent result.  The expensive SAM3 proposals and feature maps are reused
from immutable caches.  Only proposals newly promoted by the feature veto are
propagated with SAM2, and those small directional track sets are cached here.

The script has two strictly ordered phases:

1. build all five predictions and hash them without opening ChangeSim labels;
2. verify that freeze, then load ground truth and compute Table-3 metrics.

The usual ten pairs are a development ablation, not a held-out benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from ocmask.changesim import MetricAccumulator, normalize_target
from ocmask.config import load_config
from ocmask.stages.sam2_tracking_backend import Sam2MaskTracker
from ocmask.stages.sam3_feature_veto import (
    apply_direct_semantics,
    direct_replacement_mask,
    merge_objects_with_parent,
    ordinary_promoted_objects,
    pair_and_classify_gate_features,
)
from ocmask.stages.sam3_identity_location import mask_descriptors
from ocmask.stages.sam3_pairwise import (
    load_cached_inputs,
    load_proposal_cache,
    proposals_to_objects,
)
from ocmask.io import save_image, save_json
from ocmask.masks import filter_visible
from ocmask.types import Label, ObjectMask


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY
    / "configs/stages/changesim-sam3-conservative-a3-fixed10-densegrid96.yaml"
)
VARIANTS = (
    "a0_baseline_replay",
    "a1_hard_feature_veto",
    "a2_guarded_feature_veto",
    "a3_guarded_direct_replacement",
    "a4_guarded_replacement_moved_reasoning",
)
VARIANT_DESCRIPTIONS = {
    VARIANTS[0]: "Frozen best IoU-area-gate parent, replayed byte-for-byte.",
    VARIANTS[1]: "Hard veto: every valid same-place cosine below T_same is changed.",
    VARIANTS[2]: "Guarded veto: only cosine at or below T_same - 0.10 is changed.",
    VARIANTS[3]: "Guarded veto plus direct replacement on same-place disagreement.",
    VARIANTS[4]: "Direct replacement plus frozen displaced-identity moved reasoning.",
}

# The earlier A3/A4 experiment used the combined guarded-hybrid semantic map
# as A0.  Keep that historical default, but allow a new config to replay a
# different already-frozen semantic arm without changing any tracking input.
DEFAULT_A0_PARENT_VARIANT = "combined_guarded_hybrid"
FROZEN_PARENT_VARIANTS = (
    "replacement_only",
    "moved_verification",
    "combined_guarded_hybrid",
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to recommended_output in the experiment config.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate every frozen cache and decision without loading SAM2 or GT.",
    )
    parser.add_argument(
        "--predictions-only",
        action="store_true",
        help="Track, compose, and freeze predictions without opening GT.",
    )
    parser.add_argument(
        "--force-retrack",
        action="store_true",
        help="Overwrite only this experiment's promoted-mask tracking caches.",
    )
    parser.add_argument(
        "--skip-html",
        action="store_true",
        help="Write report.json but do not build index.html.",
    )
    parser.add_argument(
        "--record-audit",
        type=Path,
        help=(
            "Write the GT-free feature-audit counts and stop before comparing "
            "them or opening ground truth. Used to bootstrap a new frozen shard."
        ),
    )
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _implementation_hash() -> str:
    """Fingerprint every local rule that can change a prediction."""

    paths = (
        Path(__file__).resolve(),
        REPOSITORY / "src/ocmask/stages/sam3_feature_veto.py",
        REPOSITORY / "src/ocmask/stages/sam3_identity_location.py",
        REPOSITORY / "src/ocmask/stages/sam3_pairwise.py",
        REPOSITORY / "src/ocmask/stages/sam2_tracking_backend.py",
        REPOSITORY / "src/ocmask/adapters/sam2.py",
        REPOSITORY / "src/ocmask/masks.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPOSITORY)).encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _table3(metrics: dict) -> dict[str, dict[str, float]]:
    """Format the exact classes used by Table 3 as percentages."""

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


def _load_report_root(path: Path, name: str) -> tuple[dict, list[str]]:
    report_path = path / "report.json"
    selection_path = path / "selection.json"
    if not report_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError(f"{name} report/selection is incomplete: {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selection = list(json.loads(selection_path.read_text(encoding="utf-8"))["ids"])
    if report.get("failures"):
        raise RuntimeError(f"{name} contains failed pairs")
    if not selection or len(set(selection)) != len(selection):
        raise RuntimeError(f"{name} must contain unique, non-empty pairs")
    return report, selection


def _load_proposal_cache_root(path: Path, name: str) -> tuple[dict, list[str]]:
    """Like ``_load_report_root``, but also accepts a proposals-only cache.

    ``run_sam3_pairwise_experiment.py --proposals-only`` (used by
    ``a3_overnight.py`` for this stage) intentionally exits before computing
    metrics, so it writes ``proposal_cache_complete.json`` instead of
    ``report.json``. This root only ever needs the cached proposal masks and
    the pair-id list -- never the metrics -- so a bare completion marker is
    just as valid a signal here as a full report.
    """
    report_path = path / "report.json"
    if report_path.is_file():
        return _load_report_root(path, name)

    marker_path = path / "proposal_cache_complete.json"
    selection_path = path / "selection.json"
    if not marker_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError(f"{name} report/selection is incomplete: {path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    selected = list(json.loads(selection_path.read_text(encoding="utf-8"))["ids"])
    if not selected or len(set(selected)) != len(selected):
        raise RuntimeError(f"{name} must contain unique, non-empty pairs")
    if set(marker.get("pairs", [])) != set(selected):
        raise RuntimeError(f"{name} completion marker disagrees with its own selection")
    return {"failures": []}, selected


def _validate_roots(config: dict) -> dict[str, Any]:
    roots = {
        "gate": (REPOSITORY / config["parent_evaluation"]).resolve(),
        "best": (REPOSITORY / config["best_parent_evaluation"]).resolve(),
        "identity": (REPOSITORY / config["identity_evaluation"]).resolve(),
        "proposals": (REPOSITORY / config["proposal_cache_parent"]).resolve(),
        "old_tracks": (REPOSITORY / config["moved_track_cache_parent"]).resolve(),
    }
    promoted_cache_parent = config.get("promoted_tracking_cache_parent")
    if promoted_cache_parent:
        roots["promoted_tracks"] = (REPOSITORY / promoted_cache_parent).resolve()
    reports: dict[str, dict] = {}
    selection: list[str] | None = None
    for name, root in roots.items():
        loader = _load_proposal_cache_root if name == "proposals" else _load_report_root
        report, current = loader(root, name)
        reports[name] = report
        if selection is None:
            selection = current
        elif current != selection:
            raise RuntimeError(f"{name} pair order differs from the frozen gate parent")
    assert selection is not None
    gate_records = {str(row["id"]): row for row in reports["gate"]["pairs"]}
    if set(gate_records) != set(selection):
        raise RuntimeError("gate report and selection disagree")

    # The A0 raster was itself frozen in the previous guarded-hybrid run.
    best_freeze_path = roots["best"] / "predictions_frozen.json"
    best_freeze = json.loads(best_freeze_path.read_text(encoding="utf-8"))
    if best_freeze.get("selection") != selection:
        raise RuntimeError("best-parent freeze order differs from selection")
    if best_freeze.get("variants") != list(FROZEN_PARENT_VARIANTS):
        raise RuntimeError("best-parent frozen variants changed")
    parent_variant = str(
        config.get("a0_parent_variant", DEFAULT_A0_PARENT_VARIANT)
    )
    if parent_variant not in FROZEN_PARENT_VARIANTS:
        raise ValueError(
            "a0_parent_variant must name one of the frozen parent variants: "
            f"{list(FROZEN_PARENT_VARIANTS)}"
        )
    frozen_ids = [str(row["id"]) for row in best_freeze["pairs"]]
    if frozen_ids != selection or len(set(frozen_ids)) != len(frozen_ids):
        raise RuntimeError("best-parent freeze pair rows are missing or duplicated")
    best_frozen = {
        str(row["id"]): row["predictions"][parent_variant]
        for row in best_freeze["pairs"]
    }
    if set(best_frozen) != set(selection):
        raise RuntimeError("best-parent prediction freeze differs from selection")
    return {
        "roots": roots,
        "reports": reports,
        "selection": selection,
        "gate_records": gate_records,
        "best_frozen": best_frozen,
        "best_freeze_path": best_freeze_path,
        "a0_parent_variant": parent_variant,
    }


def _validate_parent_protocol(config: dict, context: dict) -> dict:
    """Prove every gate pair used the settings declared by this ablation."""

    config_paths = [
        Path(context["gate_records"][pair_id]["parent_artifacts"]) / "config.json"
        for pair_id in context["selection"]
    ]
    hashes = {_sha256_file(path) for path in config_paths}
    if len(hashes) != 1:
        raise RuntimeError("the ten parent pairs do not share one baseline config")
    baseline = json.loads(config_paths[0].read_text(encoding="utf-8"))
    tracking = baseline["tracking"]
    gate = config["clean_gate"]
    downstream = config["changed_object_pipeline"]
    checks = {
        "minimum_track_score": (
            float(tracking["minimum_track_score"]),
            float(gate["minimum_track_score"]),
        ),
        "minimum_track_iou": (
            float(tracking["minimum_track_iou"]),
            float(gate["minimum_track_iou"]),
        ),
        "track_area_ratio_bounds": (
            list(map(float, tracking["track_area_ratio_bounds"])),
            list(map(float, gate["track_area_ratio_bounds"])),
        ),
        "visibility_alpha": (
            float(tracking["visibility_alpha"]),
            float(downstream["visibility_alpha"]),
        ),
        "minimum_mask_area": (
            int(tracking["minimum_mask_area"]),
            int(downstream["minimum_mask_area"]),
        ),
        "replacement_overlap_iou": (
            float(tracking["replacement_overlap_iou"]),
            float(downstream["replacement_overlap_iou"]),
        ),
        "ssim_enabled": (
            bool(baseline["ssim"]["enabled"]),
            bool(downstream["ssim_enabled"]),
        ),
    }
    changed = [name for name, (actual, declared) in checks.items() if actual != declared]
    if changed:
        raise RuntimeError(f"declared experiment protocol differs from parent: {changed}")
    expected_priority = ["warped", "moved", "removed", "added", "unchanged"]
    if list(downstream["label_priority"]) != expected_priority:
        raise RuntimeError("changed-object label priority differs from implementation")
    return baseline


def _validate_identity_protocol(config: dict, context: dict) -> None:
    """Ensure cached A4 records were produced with the declared constants."""

    experiment = json.loads(
        (context["roots"]["identity"] / "experiment.json").read_text(encoding="utf-8")
    )
    identity_config = load_config(Path(experiment["config"]))
    matching = identity_config["matching"]
    classification = identity_config["classification"]
    moved = config["moved_reasoning"]
    feature = config["feature_thresholds"]
    comparisons = {
        "minimum_bidirectional_margin": (
            float(matching["minimum_bidirectional_margin"]),
            float(moved["minimum_bidirectional_margin"]),
        ),
        "require_mutual_nearest": (
            bool(matching["require_mutual_nearest"]),
            bool(moved["require_mutual_nearest"]),
        ),
        "area_ratio_bounds": (
            list(map(float, matching["area_ratio_bounds"])),
            list(map(float, moved["area_ratio_bounds"])),
        ),
        "same_location_bonus": (
            float(matching["same_location_bonus"]),
            float(moved["association_same_location_bonus"]),
        ),
        "same_location_iou": (
            float(classification["unchanged_mask_iou"]),
            float(moved["same_location_iou"]),
        ),
        "same_location_area_ratio_bounds": (
            list(map(float, classification["unchanged_area_ratio_bounds"])),
            list(map(float, moved["same_location_area_ratio_bounds"])),
        ),
        "orientation_anisotropy": (
            float(classification["orientation_anisotropy"]),
            float(moved["orientation_anisotropy"]),
        ),
        "orientation_change_degrees": (
            float(classification["orientation_change_degrees"]),
            float(moved["orientation_change_degrees"]),
        ),
        "fallback_same_identity_cosine": (
            float(matching["fallback_minimum_cosine"]),
            float(feature["fallback_same_identity_cosine"]),
        ),
    }
    changed = [name for name, values in comparisons.items() if values[0] != values[1]]
    if changed:
        raise RuntimeError(f"cached identity protocol differs from A4 config: {changed}")


def _prepare_output(output: Path, config_hash: str) -> None:
    marker = output / "experiment.json"
    if output.exists() and any(output.iterdir()):
        if not marker.is_file():
            raise RuntimeError(f"refusing non-empty unowned output: {output}")
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("config_sha256") != config_hash:
            raise RuntimeError("output belongs to a different experiment config")
    output.mkdir(parents=True, exist_ok=True)
    (output / "pairs").mkdir(exist_ok=True)


def _validate_output(output: Path, roots: Mapping[str, Path]) -> None:
    output = output.resolve()
    for root in roots.values():
        root = root.resolve()
        if output == root or root in output.parents:
            raise ValueError(f"output must be outside immutable parent {root}")


def _gate_flags(
    ledger: dict, stage: str, objects: Sequence[ObjectMask]
) -> tuple[np.ndarray, np.ndarray]:
    """Return accepted/rejected flags after proving the proposal order."""

    rows = ledger["stages"][stage]["attempts"]
    ids = [int(obj.metadata["automatic_proposal_id"]) for obj in objects]
    row_ids = [int(row["proposal_id"]) for row in rows]
    if ids != row_ids:
        raise RuntimeError(f"{stage}: visible proposal order differs from gate ledger")
    for obj, row in zip(objects, rows, strict=True):
        if int(np.asarray(obj.mask, bool).sum()) != int(row["source_area"]):
            raise RuntimeError(f"{stage}: proposal area differs from gate ledger")
    accepted = np.asarray(
        [bool(row["post_consistency_gate_accepted"]) for row in rows], bool
    )
    return accepted, ~accepted


def _objects_by_id(objects: Sequence[ObjectMask]) -> dict[int, ObjectMask]:
    output = {
        int(obj.metadata["automatic_proposal_id"]): obj for obj in objects
    }
    if len(output) != len(objects):
        raise RuntimeError("proposal IDs are not unique")
    return output


def _pair_paths(context: dict, pair_id: str) -> dict[str, Path]:
    roots = context["roots"]
    geometry = Path(context["gate_records"][pair_id]["parent_artifacts"])
    return {
        "best_labels": roots["best"]
        / "pairs"
        / pair_id
        / context["a0_parent_variant"]
        / "labels.png",
        "best_decisions": roots["best"] / "pairs" / pair_id / "decisions.json",
        "gate_ledger": roots["gate"] / "pairs" / pair_id / "tracking_attempts.json",
        "source_proposals": roots["proposals"]
        / "pairs"
        / pair_id
        / "proposal_cache/source.npz",
        "target_proposals": roots["proposals"]
        / "pairs"
        / pair_id
        / "proposal_cache/target.npz",
        "feature_arrays": roots["identity"]
        / "pairs"
        / pair_id
        / "sam3_features.npz",
        "feature_metadata": roots["identity"]
        / "pairs"
        / pair_id
        / "sam3_features.json",
        "identity_decisions": roots["identity"]
        / "pairs"
        / pair_id
        / "decisions.json",
        "identity_result": roots["identity"] / "pairs" / pair_id / "result.json",
        "render": geometry / "render_0_to_1.png",
        "reconstruction": geometry / "reconstruction.npz",
        "geometry": geometry / "geometry.npz",
        "baseline_config": geometry / "config.json",
    }


def _pair_fingerprint(paths: Mapping[str, Path], config_hash: str) -> str:
    """Hash every inference input; ChangeSim target files are absent by design."""

    digest = hashlib.sha256(bytes.fromhex(config_hash))
    for name, path in sorted(paths.items()):
        if not path.is_file():
            raise FileNotFoundError(f"missing feature-veto input {name}: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _load_pair(config: dict, context: dict, pair_id: str) -> dict[str, Any]:
    """Load and cross-check one pair without retaining it across iterations."""

    paths = _pair_paths(context, pair_id)
    gate_record = context["gate_records"][pair_id]
    geometry_artifact = Path(gate_record["parent_artifacts"])
    inputs = load_cached_inputs(geometry_artifact)
    source = proposals_to_objects(load_proposal_cache(paths["source_proposals"]))
    target = proposals_to_objects(load_proposal_cache(paths["target_proposals"]))
    classification = config["changed_object_pipeline"]
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
    ledger = json.loads(paths["gate_ledger"].read_text(encoding="utf-8"))
    source_accepted, source_changed = _gate_flags(
        ledger, "source_to_clean", source
    )
    target_accepted, target_changed = _gate_flags(
        ledger, "target_to_clean", target
    )

    metadata = json.loads(paths["feature_metadata"].read_text(encoding="utf-8"))
    if metadata.get("ground_truth_used"):
        raise RuntimeError(f"{pair_id}: feature cache claims ground-truth use")
    if metadata.get("checkpoint_sha256") != config["sam3"]["checkpoint_sha256"]:
        raise RuntimeError(f"{pair_id}: SAM3 feature checkpoint differs")
    with np.load(paths["feature_arrays"]) as cache:
        source_map = np.asarray(cache["source"])
        target_map = np.asarray(cache["target"])
    if list(source_map.shape) != list(metadata["source_shape"]):
        raise RuntimeError(f"{pair_id}: source feature shape metadata differs")
    if list(target_map.shape) != list(metadata["target_shape"]):
        raise RuntimeError(f"{pair_id}: target feature shape metadata differs")
    if str(source_map.dtype) != metadata["dtype"] or str(target_map.dtype) != metadata["dtype"]:
        raise RuntimeError(f"{pair_id}: feature dtype metadata differs")
    identity_result = json.loads(paths["identity_result"].read_text(encoding="utf-8"))
    if metadata["pair_fingerprint_sha256"] != identity_result["pair_fingerprint_sha256"]:
        raise RuntimeError(f"{pair_id}: feature/result fingerprints differ")
    minimum_cells = float(config["sam3"]["minimum_feature_cells"])
    source_features = mask_descriptors(
        source_map, source, minimum_feature_cells=minimum_cells
    )
    target_features = mask_descriptors(
        target_map, target, minimum_feature_cells=minimum_cells
    )
    identity = json.loads(paths["identity_decisions"].read_text(encoding="utf-8"))
    calibration = identity["diagnostics"]["calibration"]
    if calibration["valid"]:
        same_threshold = float(calibration["threshold"])
        threshold_origin = "frozen_pair_calibration"
    else:
        same_threshold = float(
            config["feature_thresholds"]["fallback_same_identity_cosine"]
        )
        threshold_origin = "declared_fallback"
    different_margin = float(config["feature_thresholds"]["different_identity_margin"])
    # Optional static floor: raises the bar for calling two objects the same
    # identity (and therefore vetoing a change candidate back to unchanged),
    # regardless of where the per-pair threshold came from. Absent by default
    # so existing configs are unaffected; only ever tightens, never relaxes,
    # matching this file's existing "static controls may tighten but never
    # relax" convention for fallback_same_identity_cosine.
    #
    # different_threshold = same_threshold - different_margin, so naively
    # raising same_threshold drags different_threshold up with it -- which
    # makes it *easier*, not harder, to call a pair "confidently different"
    # (the promote-to-changed/REPLACED band), the opposite of what a tighter
    # veto is supposed to do. Widening different_margin by the same amount
    # keeps different_threshold pinned at its original value, so raising the
    # floor tightens only the similar/veto boundary as intended.
    tighten_floor = config["feature_thresholds"].get(
        "tighten_minimum_same_identity_cosine"
    )
    if tighten_floor is not None and float(tighten_floor) > same_threshold:
        original_different_threshold = same_threshold - different_margin
        same_threshold = float(tighten_floor)
        different_margin = same_threshold - original_different_threshold
        threshold_origin = f"{threshold_origin}+tightened_floor_veto_only"
    pairing = config["same_place_pairing"]
    # Optional fourth component: a CIE Lab a*/b* chrominance comparison that
    # can additionally promote a pair to "different" when cosine alone reads
    # "similar"/"uncertain" but the object's own pixels changed color. Absent
    # by default (the config key is None/missing) so existing configs are
    # unaffected -- see pair_and_classify_gate_features's docstring.
    color_different_threshold = config["feature_thresholds"].get(
        "color_different_threshold"
    )
    color_method = config["feature_thresholds"].get("color_method")
    color_scales = tuple(
        config["feature_thresholds"].get("color_component_scales", [0.15, 0.20, 0.20])
    )
    color_center = tuple(
        config["feature_thresholds"].get("color_component_center", [0.0, 0.0, 0.0])
    )
    decision = pair_and_classify_gate_features(
        source,
        target,
        source_features,
        target_features,
        source_accepted,
        target_accepted,
        same_threshold=same_threshold,
        different_margin=float(different_margin),
        minimum_spatial_iou=float(pairing["minimum_spatial_iou"]),
        maximum_centroid_distance=float(
            pairing["maximum_normalized_centroid_distance"]
        ),
        area_ratio_bounds=tuple(pairing["area_ratio_bounds"]),
        source_image=inputs.source_render if color_different_threshold is not None else None,
        target_image=inputs.target_image if color_different_threshold is not None else None,
        color_different_threshold=color_different_threshold,
        color_method=color_method or "chroma",
        color_coverage=inputs.cross_coverage,
        color_center=color_center,
        color_scales=color_scales,
        color_minimum_mask_pixels=int(
            config["feature_thresholds"].get("color_minimum_mask_pixels", 64)
        ),
        color_minimum_context_pixels=int(
            config["feature_thresholds"].get("color_minimum_context_pixels", 4096)
        ),
        color_maximum_clipped_fraction=float(
            config["feature_thresholds"].get(
                "color_maximum_clipped_fraction", 0.25
            )
        ),
        color_apply_to_embedding_bands=config["feature_thresholds"].get(
            "color_apply_to_embedding_bands", ["similar", "uncertain"]
        ),
    )

    baseline = np.asarray(Image.open(paths["best_labels"]), np.uint8)
    best_record = context["best_frozen"][pair_id]
    if _sha256_file(paths["best_labels"]) != best_record["file_sha256"]:
        raise RuntimeError(f"{pair_id}: best-parent labels differ from their freeze")
    if _sha256_array(baseline) != best_record["array_sha256"]:
        raise RuntimeError(f"{pair_id}: best-parent label array differs from freeze")
    return {
        "paths": paths,
        "fingerprint": _pair_fingerprint(paths, context["config_hash"]),
        "geometry_artifact": geometry_artifact,
        "inputs": inputs,
        "source": source,
        "target": target,
        "source_by_id": _objects_by_id(source),
        "target_by_id": _objects_by_id(target),
        "source_accepted": source_accepted,
        "target_accepted": target_accepted,
        "source_changed": source_changed,
        "target_changed": target_changed,
        "decision": decision,
        "same_threshold": same_threshold,
        "threshold_origin": threshold_origin,
        "calibration": calibration,
        "identity_matches": list(identity["matches"]),
        "baseline": baseline,
    }


def _tracking_identity(baseline_config: dict) -> dict[str, Any]:
    """Fingerprint the model and local adapter that produce cached tracks."""

    checkpoint = Path(baseline_config["sam2"]["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = (REPOSITORY / checkpoint).resolve()
    spec = importlib.util.find_spec("sam2")
    package_root: Path | None = None
    if spec is not None and spec.submodule_search_locations:
        package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    if package_root is None:
        raise RuntimeError("installed SAM2 package could not be resolved")
    package_sources = sorted(package_root.rglob("*.py"))
    if not package_sources:
        raise RuntimeError("installed SAM2 package has no Python sources")
    package_digest = hashlib.sha256()
    for path in package_sources:
        package_digest.update(str(path.relative_to(package_root)).encode("utf-8"))
        package_digest.update(bytes.fromhex(_sha256_file(path)))
    model_cfg = Path(baseline_config["sam2"]["model_cfg"])
    candidates = (
        model_cfg,
        REPOSITORY / model_cfg,
        REPOSITORY / "src/mast3r" / model_cfg,
        package_root / model_cfg,
    )
    resolved_model_cfg = next((path.resolve() for path in candidates if path.is_file()), None)
    if resolved_model_cfg is None:
        raise FileNotFoundError(f"SAM2 model config is unavailable: {model_cfg}")
    import torch

    identity = {
        "backend": "sam2",
        "baseline_config_sha256": _sha256_json(baseline_config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "model_cfg": str(resolved_model_cfg),
        "model_cfg_sha256": _sha256_file(resolved_model_cfg),
        "adapter_sha256": _sha256_file(
            REPOSITORY / "src/ocmask/adapters/sam2.py"
        ),
        "wrapper_sha256": _sha256_file(
            REPOSITORY / "src/ocmask/stages/sam2_tracking_backend.py"
        ),
        "sam2_distribution_version": importlib.metadata.version("sam-2"),
        "sam2_package_root": str(package_root),
        "sam2_python_source_count": len(package_sources),
        "sam2_python_tree_sha256": package_digest.hexdigest(),
        "torch_version": torch.__version__,
        "torch_compiled_cuda": torch.version.cuda,
        "batching": "one shared hard-superset state per direction and pair",
    }
    identity["tracking_protocol_sha256"] = _sha256_json(identity)
    return identity


def _object_subset_hash(
    ids: Sequence[int], objects: Mapping[int, ObjectMask]
) -> str:
    digest = hashlib.sha256()
    for proposal_id in ids:
        obj = objects[int(proposal_id)]
        digest.update(str(int(proposal_id)).encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_array(np.asarray(obj.mask, bool))))
    return digest.hexdigest()


def _tracking_input_hash(pair: dict, tracking_protocol_hash: str) -> str:
    decision = pair["decision"]
    return _sha256_json(
        {
            "source_rgb": _sha256_array(pair["inputs"].source_render),
            "target_rgb": _sha256_array(pair["inputs"].target_image),
            "hard_source_ids": list(decision.hard_source_ids),
            "hard_target_ids": list(decision.hard_target_ids),
            "hard_source_masks": _object_subset_hash(
                decision.hard_source_ids, pair["source_by_id"]
            ),
            "hard_target_masks": _object_subset_hash(
                decision.hard_target_ids, pair["target_by_id"]
            ),
            "feature_pairs": [item.to_dict() for item in decision.pairs],
            "tracking_protocol_sha256": tracking_protocol_hash,
        }
    )


def _packed_masks(masks: Sequence[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if not masks:
        return np.empty((0, shape[0], (shape[1] + 7) // 8), np.uint8)
    stacked = np.stack([np.asarray(mask, bool) for mask in masks])
    if stacked.shape[1:] != shape:
        raise ValueError("tracked masks have inconsistent shapes")
    return np.packbits(stacked, axis=2)


def _save_tracking_cache(
    cache_dir: Path,
    forward_attempts,
    reverse_attempts,
    *,
    shape: tuple[int, int],
    input_hash: str,
    source_ids: Sequence[int],
    target_ids: Sequence[int],
) -> None:
    """Persist all attempts, including rejected raw masks, for exact resume."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = cache_dir / "tracks.npz"
    temporary = cache_dir / "tracks.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            height=np.int32(shape[0]),
            width=np.int32(shape[1]),
            forward_masks_packed=_packed_masks(
                [item.mask for item in forward_attempts], shape
            ),
            reverse_masks_packed=_packed_masks(
                [item.mask for item in reverse_attempts], shape
            ),
            forward_accepted=np.asarray(
                [item.accepted for item in forward_attempts], bool
            ),
            reverse_accepted=np.asarray(
                [item.accepted for item in reverse_attempts], bool
            ),
        )
    temporary.replace(arrays_path)

    def rows(ids: Sequence[int], attempts) -> list[dict[str, Any]]:
        return [
            {
                "proposal_id": int(proposal_id),
                "accepted": bool(attempt.accepted),
                "object_score_logit": (
                    None
                    if attempt.object_score_logit is None
                    else float(attempt.object_score_logit)
                ),
                "rejection_reasons": list(attempt.rejection_reasons),
                "area": int(np.asarray(attempt.mask, bool).sum()),
                "mask_sha256": _sha256_array(np.asarray(attempt.mask, bool)),
            }
            for proposal_id, attempt in zip(ids, attempts, strict=True)
        ]

    save_json(
        cache_dir / "metadata.json",
        {
            "schema_version": 1,
            "input_sha256": input_hash,
            "arrays_sha256": _sha256_file(arrays_path),
            "ground_truth_used": False,
            "batch_context": "hard-superset shared by hard and guarded variants",
            "source_to_target": rows(source_ids, forward_attempts),
            "target_to_source": rows(target_ids, reverse_attempts),
        },
    )


def _load_tracking_cache(
    cache_dir: Path, *, input_hash: str
) -> tuple[dict[int, np.ndarray | None], dict[int, np.ndarray | None], dict]:
    metadata_path = cache_dir / "metadata.json"
    arrays_path = cache_dir / "tracks.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError("promoted tracking cache is incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("ground_truth_used"):
        raise RuntimeError("invalid promoted tracking cache protocol")
    if metadata.get("input_sha256") != input_hash:
        raise RuntimeError("promoted tracking input hash changed")
    if metadata.get("arrays_sha256") != _sha256_file(arrays_path):
        raise RuntimeError("promoted tracking array hash changed")
    with np.load(arrays_path) as cache:
        height, width = int(cache["height"]), int(cache["width"])

        def restore(prefix: str, stage: str) -> dict[int, np.ndarray | None]:
            packed = np.asarray(cache[f"{prefix}_masks_packed"], np.uint8)
            masks = np.unpackbits(packed, axis=2, count=width).astype(bool)
            masks = masks[:, :height, :width]
            accepted = np.asarray(cache[f"{prefix}_accepted"], bool)
            rows = metadata[stage]
            if not (len(masks) == len(accepted) == len(rows)):
                raise RuntimeError(f"{stage}: cache row counts differ")
            output: dict[int, np.ndarray | None] = {}
            for mask, keep, row in zip(masks, accepted, rows, strict=True):
                proposal_id = int(row["proposal_id"])
                if proposal_id in output:
                    raise RuntimeError(f"{stage}: duplicate proposal ID")
                if bool(keep) != bool(row["accepted"]):
                    raise RuntimeError(f"{stage}: acceptance metadata differs")
                if _sha256_array(mask) != row["mask_sha256"]:
                    raise RuntimeError(f"{stage}: raw mask hash differs")
                if int(mask.sum()) != int(row["area"]):
                    raise RuntimeError(f"{stage}: raw mask area differs")
                output[proposal_id] = mask.copy() if keep else None
            return output

        return (
            restore("forward", "source_to_target"),
            restore("reverse", "target_to_source"),
            metadata,
        )


def _validate_tracking_cache_for_plan(
    cache_dir: Path,
    *,
    input_hash: str,
    source_ids: Sequence[int],
    target_ids: Sequence[int],
) -> tuple[dict[int, np.ndarray | None], dict[int, np.ndarray | None], dict]:
    """Deep-check a cache against the ordered proposal IDs in this run."""

    forward, reverse, metadata = _load_tracking_cache(
        cache_dir, input_hash=input_hash
    )
    if list(forward) != list(map(int, source_ids)):
        raise RuntimeError("source-to-target cache proposal order changed")
    if list(reverse) != list(map(int, target_ids)):
        raise RuntimeError("target-to-source cache proposal order changed")
    return forward, reverse, metadata


def _import_tracking_cache(
    source_dir: Path,
    destination_dir: Path,
    *,
    input_hash: str,
    source_ids: Sequence[int],
    target_ids: Sequence[int],
    tracking_protocol_hash: str,
    source_experiment: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy an independently valid, parent-invariant SAM2 tracking cache.

    Tracking is performed before semantic composition.  Therefore a cache may
    be shared by experiments with different A0 label maps, but only when its
    complete image/mask/feature/protocol fingerprint and proposal order match.
    The copy gives the new experiment ownership of its artifacts; no symlink or
    hardlink can let a later edit mutate both runs.
    """

    source_protocol = source_experiment.get("tracking_identity", {}).get(
        "tracking_protocol_sha256"
    )
    if source_protocol != tracking_protocol_hash:
        raise RuntimeError("external promoted-track protocol differs")
    _validate_tracking_cache_for_plan(
        source_dir,
        input_hash=input_hash,
        source_ids=source_ids,
        target_ids=target_ids,
    )

    destination_dir.mkdir(parents=True, exist_ok=True)
    source_files = {
        name: source_dir / name for name in ("tracks.npz", "metadata.json")
    }
    for name, source in source_files.items():
        temporary = destination_dir / f".{name}.importing"
        shutil.copy2(source, temporary)
        temporary.replace(destination_dir / name)

    _validate_tracking_cache_for_plan(
        destination_dir,
        input_hash=input_hash,
        source_ids=source_ids,
        target_ids=target_ids,
    )
    provenance = {
        "schema_version": 1,
        "cache_reused": True,
        "ground_truth_used": False,
        "source": str(source_dir.resolve()),
        "source_experiment_id": source_experiment.get("experiment_id"),
        "source_execution_sha256": source_experiment.get("execution_sha256"),
        "tracking_input_sha256": input_hash,
        "tracking_protocol_sha256": tracking_protocol_hash,
        "source_file_sha256": {
            name: _sha256_file(path) for name, path in source_files.items()
        },
        "destination_file_sha256": {
            name: _sha256_file(destination_dir / name) for name in source_files
        },
        "copy_mode": "independent_byte_copy",
    }
    if provenance["source_file_sha256"] != provenance["destination_file_sha256"]:
        raise RuntimeError("external tracking cache changed during import")
    save_json(destination_dir / "reuse_provenance.json", provenance)
    return provenance


def _aggregate_audit(plans: Sequence[dict]) -> dict[str, int]:
    output = {
        "reciprocal_same_place_pairs": 0,
        "similar_pairs": 0,
        "uncertain_pairs": 0,
        "confidently_different_pairs": 0,
        "hard_source_promotions": 0,
        "hard_target_promotions": 0,
        "guarded_source_promotions": 0,
        "guarded_target_promotions": 0,
    }
    for plan in plans:
        pairs = plan["funnel"]["pairs"]
        output["reciprocal_same_place_pairs"] += int(
            pairs["reciprocal_same_place"]
        )
        output["similar_pairs"] += int(pairs["similar"])
        output["uncertain_pairs"] += int(pairs["uncertain"])
        output["confidently_different_pairs"] += int(pairs["different"])
        for key in (
            "hard_source_promotions",
            "hard_target_promotions",
            "guarded_source_promotions",
            "guarded_target_promotions",
        ):
            output[key] += int(plan[key])
    return output


def _sum_nested_funnels(plans: Sequence[dict]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"pairs": {}, "source": {}, "target": {}}
    for section in aggregate:
        keys = plans[0]["funnel"][section] if plans else {}
        aggregate[section] = {
            key: int(sum(int(plan["funnel"][section][key]) for plan in plans))
            for key in keys
        }
    return aggregate


def _aggregate_color_audit(plans: Sequence[dict]) -> dict[str, Any]:
    """Summarize illumination evidence before any ground-truth access."""

    records = [record for plan in plans for record in plan["pairs"]]
    status: dict[str, int] = {}
    for record in records:
        key = str(record.get("color_evidence_status") or "not_evaluated")
        status[key] = status.get(key, 0) + 1
    distances = [
        float(record["color_distance"])
        for record in records
        if record.get("color_distance") is not None
    ]
    color_promoted = [
        record for record in records if record.get("band_source") == "color"
    ]
    return {
        "status": status,
        "valid_distance_count": len(distances),
        "color_promoted_pairs": len(color_promoted),
        "color_promoted_from_embedding_similar_or_uncertain": len(color_promoted),
        "distance_quantiles": (
            {
                name: float(np.quantile(distances, quantile))
                for name, quantile in (
                    ("q50", 0.50),
                    ("q90", 0.90),
                    ("q95", 0.95),
                    ("q99", 0.99),
                    ("max", 1.00),
                )
            }
            if distances
            else {}
        ),
    }


def _moved_reasoning_masks(
    pair: dict, guarded_source_ids: Sequence[int], guarded_target_ids: Sequence[int]
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Select frozen displaced-identity matches not reserved as replacements."""

    different = [item for item in pair["decision"].pairs if item.feature_band == "different"]
    reserved_source = {item.source_proposal_id for item in different}
    reserved_target = {item.target_proposal_id for item in different}
    source_changed_ids = {
        int(obj.metadata["automatic_proposal_id"])
        for obj, changed in zip(pair["source"], pair["source_changed"], strict=True)
        if changed
    } | set(map(int, guarded_source_ids))
    target_changed_ids = {
        int(obj.metadata["automatic_proposal_id"])
        for obj, changed in zip(pair["target"], pair["target_changed"], strict=True)
        if changed
    } | set(map(int, guarded_target_ids))
    masks: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for record in pair["identity_matches"]:
        if record.get("decision") != "moved":
            continue
        source_index = int(record["source_index"])
        target_index = int(record["target_index"])
        source_id = int(record["source_proposal_id"])
        target_id = int(record["target_proposal_id"])
        if int(pair["source"][source_index].metadata["automatic_proposal_id"]) != source_id:
            raise RuntimeError("frozen moved source index/ID is stale")
        if int(pair["target"][target_index].metadata["automatic_proposal_id"]) != target_id:
            raise RuntimeError("frozen moved target index/ID is stale")
        if source_id in reserved_source or target_id in reserved_target:
            continue
        if source_id not in source_changed_ids and target_id not in target_changed_ids:
            continue
        native = np.asarray(
            Image.fromarray(np.asarray(pair["target"][target_index].mask, np.uint8)).resize(
                pair["baseline"].shape[::-1], Image.Resampling.NEAREST
            ),
            bool,
        )
        masks.append(native)
        records.append(
            {
                "source_proposal_id": source_id,
                "target_proposal_id": target_id,
                "cosine": float(record["cosine"]),
                "spatial_iou": float(record["spatial_iou"]),
            }
        )
    return masks, records


def _transition(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.bincount(
        np.asarray(first, np.int64).reshape(-1) * 6
        + np.asarray(second, np.int64).reshape(-1),
        minlength=36,
    ).reshape(6, 6)


def _manifest_targets(path: Path) -> dict[str, Path]:
    """Resolve GT paths only after all predictions have been frozen."""

    base = path.resolve().parent
    output: dict[str, Path] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        target = Path(row["target"])
        output[str(row["id"])] = target if target.is_absolute() else (base / target).resolve()
    return output


def _verify_freeze(freeze: dict, output: Path) -> None:
    for pair in freeze["pairs"]:
        for variant, record in pair["predictions"].items():
            path = output / record["relative_path"]
            if _sha256_file(path) != record["file_sha256"]:
                raise RuntimeError(f"prediction changed after freeze: {pair['id']}/{variant}")
            labels = np.asarray(Image.open(path), np.uint8)
            if _sha256_array(labels) != record["array_sha256"]:
                raise RuntimeError(f"prediction array changed: {pair['id']}/{variant}")


def _build_html(output: Path) -> None:
    builder = REPOSITORY / "scripts/build_sam3_feature_veto_gate_report.py"
    subprocess.run(
        [
            sys.executable,
            str(builder),
            "--root",
            str(output),
            "--output",
            str(output / "index.html"),
            "--force",
        ],
        cwd=REPOSITORY,
        check=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    experiment_started = time.perf_counter()
    args = _arguments(argv)
    config = load_config(args.config)
    if tuple(config["variants"]) != VARIANTS:
        raise ValueError("config must declare the frozen A0-A4 order exactly")
    context = _validate_roots(config)
    _validate_identity_protocol(config, context)
    context["config_hash"] = _sha256_json(config)
    output = (
        args.output.resolve()
        if args.output
        else (REPOSITORY / config["recommended_output"]).resolve()
    )
    _validate_output(output, context["roots"])
    _prepare_output(output, context["config_hash"])

    baseline_config = _validate_parent_protocol(config, context)
    tracking_identity = _tracking_identity(baseline_config)
    implementation_hash = _implementation_hash()
    parent_hashes = {
        name: _sha256_file(
            root / "report.json"
            if (root / "report.json").is_file()
            else root / "proposal_cache_complete.json"
        )
        for name, root in context["roots"].items()
    }
    execution_hash = _sha256_json(
        {
            "config_sha256": context["config_hash"],
            "implementation_sha256": implementation_hash,
            "parent_report_sha256": parent_hashes,
            "tracking_protocol_sha256": tracking_identity[
                "tracking_protocol_sha256"
            ],
        }
    )
    save_json(
        output / "experiment.json",
        {
            "experiment_id": config["experiment_id"],
            "config": str(args.config.resolve()),
            "config_sha256": context["config_hash"],
            "implementation_sha256": implementation_hash,
            "execution_sha256": execution_hash,
            "parent_report_sha256": parent_hashes,
            "tracking_identity": tracking_identity,
            "a0_parent_variant": context["a0_parent_variant"],
            "ground_truth_used_in_inference": False,
            "selection_role": config["protocol"]["selection_role"],
        },
    )
    save_json(output / "selection.json", {"ids": context["selection"]})

    # First compute all feature decisions on CPU.  This pre-GT audit is also a
    # regression oracle for proposal ordering and threshold conventions.
    plans: list[dict[str, Any]] = []
    for number, pair_id in enumerate(context["selection"], 1):
        pair = _load_pair(config, context, pair_id)
        decision = pair["decision"]
        if not set(decision.guarded_source_ids).issubset(decision.hard_source_ids):
            raise AssertionError(f"{pair_id}: guarded source IDs are not a hard subset")
        if not set(decision.guarded_target_ids).issubset(decision.hard_target_ids):
            raise AssertionError(f"{pair_id}: guarded target IDs are not a hard subset")
        tracking_hash = _tracking_input_hash(
            pair, tracking_identity["tracking_protocol_sha256"]
        )
        pair_plan = {
            "id": pair_id,
            "pair_fingerprint_sha256": pair["fingerprint"],
            "tracking_input_sha256": tracking_hash,
            "same_threshold": pair["same_threshold"],
            "threshold_origin": pair["threshold_origin"],
            "calibration": pair["calibration"],
            "funnel": decision.funnel,
            "hard_source_ids": list(decision.hard_source_ids),
            "hard_target_ids": list(decision.hard_target_ids),
            "guarded_source_ids": list(decision.guarded_source_ids),
            "guarded_target_ids": list(decision.guarded_target_ids),
            "hard_source_promotions": len(decision.hard_source_ids),
            "hard_target_promotions": len(decision.hard_target_ids),
            "guarded_source_promotions": len(decision.guarded_source_ids),
            "guarded_target_promotions": len(decision.guarded_target_ids),
            "pairs": [item.to_dict() for item in decision.pairs],
        }
        plans.append(pair_plan)
        pair_dir = output / "pairs" / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        save_json(pair_dir / "feature_decisions.json", pair_plan)
        print(
            f"[audit {number}/{len(context['selection'])}] {pair_id}: "
            f"pairs={len(decision.pairs)}, hard="
            f"{len(decision.hard_source_ids)}/{len(decision.hard_target_ids)}, "
            f"guarded={len(decision.guarded_source_ids)}/{len(decision.guarded_target_ids)}",
            flush=True,
        )

    audit = _aggregate_audit(plans)
    color_audit = _aggregate_color_audit(plans)
    if args.record_audit:
        save_json(args.record_audit.resolve(), audit)
        print(f"[audit freeze] wrote {args.record_audit.resolve()}", flush=True)
        return 0
    expected = {key: int(value) for key, value in config["pre_ground_truth_audit_expected"].items()}
    if audit != expected:
        raise AssertionError(f"pre-GT feature audit changed: {audit} != {expected}")
    save_json(
        output / "input_validation.json",
        {
            "passed": True,
            "selection": context["selection"],
            "audit": audit,
            "expected": expected,
            "ground_truth_opened": False,
        },
    )
    if args.validate_only:
        print("All frozen inputs and the pre-GT audit are valid.", flush=True)
        return 0

    # Track the hard union once.  Guarded is a strict subset and deliberately
    # reuses the same multiplex state, keeping the comparison controlled.
    missing: list[str] = []
    imported: list[str] = []
    external_cache_root = context["roots"].get("promoted_tracks")
    external_experiment: dict[str, Any] | None = None
    if external_cache_root is not None:
        external_experiment = json.loads(
            (external_cache_root / "experiment.json").read_text(encoding="utf-8")
        )
    for plan in plans:
        cache_dir = output / "pairs" / plan["id"] / "promoted_tracking_cache"
        if args.force_retrack:
            missing.append(plan["id"])
            continue
        try:
            _validate_tracking_cache_for_plan(
                cache_dir,
                input_hash=plan["tracking_input_sha256"],
                source_ids=plan["hard_source_ids"],
                target_ids=plan["hard_target_ids"],
            )
        except (FileNotFoundError, RuntimeError, KeyError, ValueError):
            if external_cache_root is None or external_experiment is None:
                missing.append(plan["id"])
                continue
            source_dir = (
                external_cache_root
                / "pairs"
                / plan["id"]
                / "promoted_tracking_cache"
            )
            try:
                _import_tracking_cache(
                    source_dir,
                    cache_dir,
                    input_hash=plan["tracking_input_sha256"],
                    source_ids=plan["hard_source_ids"],
                    target_ids=plan["hard_target_ids"],
                    tracking_protocol_hash=tracking_identity[
                        "tracking_protocol_sha256"
                    ],
                    source_experiment=external_experiment,
                )
                imported.append(plan["id"])
            except (FileNotFoundError, RuntimeError, KeyError, ValueError) as error:
                print(
                    f"[tracking cache rejected] {plan['id']}: {error}",
                    flush=True,
                )
                missing.append(plan["id"])
    print(
        f"[tracking] imported={len(imported)}, generated={len(missing)}, "
        f"local={len(plans) - len(imported) - len(missing)}",
        flush=True,
    )
    if missing:
        tracker = Sam2MaskTracker(baseline_config)
        try:
            plan_by_id = {plan["id"]: plan for plan in plans}
            for number, pair_id in enumerate(missing, 1):
                pair = _load_pair(config, context, pair_id)
                plan = plan_by_id[pair_id]
                started = time.perf_counter()
                source_ids = plan["hard_source_ids"]
                target_ids = plan["hard_target_ids"]
                forward = tracker.track(
                    [pair["source_by_id"][item].mask for item in source_ids],
                    pair["inputs"].source_render,
                    pair["inputs"].target_image,
                )
                reverse = tracker.track(
                    [pair["target_by_id"][item].mask for item in target_ids],
                    pair["inputs"].target_image,
                    pair["inputs"].source_render,
                )
                if len(forward) != len(source_ids) or len(reverse) != len(target_ids):
                    raise RuntimeError(f"{pair_id}: SAM2 returned incomplete tracks")
                _save_tracking_cache(
                    output / "pairs" / pair_id / "promoted_tracking_cache",
                    forward,
                    reverse,
                    shape=pair["inputs"].cross_coverage.shape,
                    input_hash=plan["tracking_input_sha256"],
                    source_ids=source_ids,
                    target_ids=target_ids,
                )
                _load_tracking_cache(
                    output / "pairs" / pair_id / "promoted_tracking_cache",
                    input_hash=plan["tracking_input_sha256"],
                )
                print(
                    f"[tracking {number}/{len(missing)}] {pair_id}: "
                    f"{len(source_ids)} forward + {len(target_ids)} reverse, "
                    f"{time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        finally:
            tracker.release()

    # Compose all predictions without any target-label access.
    plan_by_id = {plan["id"]: plan for plan in plans}
    frozen_pairs: list[dict[str, Any]] = []
    for number, pair_id in enumerate(context["selection"], 1):
        pair = _load_pair(config, context, pair_id)
        plan = plan_by_id[pair_id]
        if pair["fingerprint"] != plan["pair_fingerprint_sha256"]:
            raise RuntimeError(f"{pair_id}: input changed between audit and compose")
        tracking_cache_dir = output / "pairs" / pair_id / "promoted_tracking_cache"
        forward, reverse, track_metadata = _validate_tracking_cache_for_plan(
            tracking_cache_dir,
            input_hash=plan["tracking_input_sha256"],
            source_ids=plan["hard_source_ids"],
            target_ids=plan["hard_target_ids"],
        )
        decision = pair["decision"]
        hard_objects, hard_counts = ordinary_promoted_objects(
            pair["source_by_id"],
            pair["target_by_id"],
            decision.hard_source_ids,
            decision.hard_target_ids,
            forward,
            reverse,
        )
        hard_labels, hard_retained = merge_objects_with_parent(
            pair["baseline"],
            hard_objects,
            pair["inputs"].cross_coverage,
            visibility_alpha=float(config["changed_object_pipeline"]["visibility_alpha"]),
            minimum_mask_area=int(config["changed_object_pipeline"]["minimum_mask_area"]),
            replacement_overlap_iou=float(
                config["changed_object_pipeline"]["replacement_overlap_iou"]
            ),
        )
        guarded_objects, guarded_counts = ordinary_promoted_objects(
            pair["source_by_id"],
            pair["target_by_id"],
            decision.guarded_source_ids,
            decision.guarded_target_ids,
            forward,
            reverse,
        )
        guarded_labels, guarded_retained = merge_objects_with_parent(
            pair["baseline"],
            guarded_objects,
            pair["inputs"].cross_coverage,
            visibility_alpha=float(config["changed_object_pipeline"]["visibility_alpha"]),
            minimum_mask_area=int(config["changed_object_pipeline"]["minimum_mask_area"]),
            replacement_overlap_iou=float(
                config["changed_object_pipeline"]["replacement_overlap_iou"]
            ),
        )
        replacement = direct_replacement_mask(
            decision.pairs,
            pair["source_by_id"],
            pair["target_by_id"],
            pair["baseline"].shape,
            allowed_band_sources=config["direct_replacement"].get(
                "allowed_band_sources", ["embedding", "color"]
            ),
        )
        direct_labels = apply_direct_semantics(guarded_labels, replacement, [])
        moved_masks, moved_records = _moved_reasoning_masks(
            pair, decision.guarded_source_ids, decision.guarded_target_ids
        )
        full_labels = apply_direct_semantics(
            guarded_labels, replacement, moved_masks
        )
        labels_by_variant = {
            VARIANTS[0]: pair["baseline"],
            VARIANTS[1]: hard_labels,
            VARIANTS[2]: guarded_labels,
            VARIANTS[3]: direct_labels,
            VARIANTS[4]: full_labels,
        }
        if not np.array_equal(direct_labels != 0, guarded_labels != 0):
            raise AssertionError(f"{pair_id}: A3 changed A2 binary support")
        if not np.array_equal(full_labels != 0, guarded_labels != 0):
            raise AssertionError(f"{pair_id}: A4 changed A2 binary support")
        if np.any((guarded_labels != 0) & (pair["baseline"] == 0) & (hard_labels == 0)):
            raise AssertionError(f"{pair_id}: A2 new support is not a subset of A1")
        for variant, labels in labels_by_variant.items():
            if np.any((pair["baseline"] != 0) & (labels == 0)):
                raise AssertionError(f"{pair_id}/{variant}: changed pixels became unchanged")

        predictions: dict[str, dict[str, Any]] = {}
        pair_dir = output / "pairs" / pair_id
        for variant, labels in labels_by_variant.items():
            variant_dir = pair_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            path = variant_dir / "labels.png"
            save_image(path, labels)
            predictions[variant] = {
                "relative_path": str(path.relative_to(output)),
                "file_sha256": _sha256_file(path),
                "array_sha256": _sha256_array(labels),
                "binary_array_sha256": _sha256_array(labels != 0),
            }
        record = {
            "id": pair_id,
            "pair_fingerprint_sha256": pair["fingerprint"],
            "tracking_input_sha256": plan["tracking_input_sha256"],
            "tracking_arrays_sha256": track_metadata["arrays_sha256"],
            "tracking_metadata_sha256": _sha256_file(
                tracking_cache_dir / "metadata.json"
            ),
            "tracking_cache_reused_from_parent": (
                tracking_cache_dir / "reuse_provenance.json"
            ).is_file(),
            "feature_funnel": decision.funnel,
            "promotion": {
                "hard": {
                    "source_ids": list(decision.hard_source_ids),
                    "target_ids": list(decision.hard_target_ids),
                    "classification": hard_counts,
                    "retained_objects": hard_retained,
                },
                "guarded": {
                    "source_ids": list(decision.guarded_source_ids),
                    "target_ids": list(decision.guarded_target_ids),
                    "classification": guarded_counts,
                    "retained_objects": guarded_retained,
                },
            },
            "direct_replacement_pair_count": sum(
                item.feature_band == "different"
                and item.band_source
                in set(
                    config["direct_replacement"].get(
                        "allowed_band_sources", ["embedding", "color"]
                    )
                )
                for item in decision.pairs
            ),
            "direct_replacement_candidate_pixels": int(replacement.sum()),
            "moved_reasoning_records": moved_records,
            "predictions": predictions,
            "transitions_from_a0": {
                variant: _transition(pair["baseline"], labels).tolist()
                for variant, labels in labels_by_variant.items()
            },
            "ground_truth_used": False,
        }
        save_json(pair_dir / "decisions.json", record)
        frozen_pairs.append(record)
        print(
            f"[predict {number}/{len(context['selection'])}] {pair_id}: "
            f"A1 +{int(np.count_nonzero((hard_labels != 0) & (pair['baseline'] == 0)))}, "
            f"A2 +{int(np.count_nonzero((guarded_labels != 0) & (pair['baseline'] == 0)))}, "
            f"replace={int(replacement.sum())}, moved_matches={len(moved_records)}",
            flush=True,
        )

    freeze = {
        "schema_version": 1,
        "execution_sha256": execution_hash,
        "selection": context["selection"],
        "variants": list(VARIANTS),
        "prediction_count": len(frozen_pairs) * len(VARIANTS),
        "ground_truth_opened_before_freeze": False,
        "pairs": frozen_pairs,
    }
    save_json(output / "predictions_frozen.json", freeze)
    _verify_freeze(freeze, output)
    print(f"[freeze] {freeze['prediction_count']} predictions hashed before GT", flush=True)
    if args.predictions_only:
        return 0

    # Evaluation begins here.  The complete freeze is rechecked before every
    # target read, so later report code cannot silently mutate a prediction.
    targets = _manifest_targets((REPOSITORY / config["manifest"]).resolve())
    accumulators = {variant: MetricAccumulator() for variant in VARIANTS}
    pair_reports: list[dict[str, Any]] = []
    aggregate_transitions = {
        variant: np.zeros((6, 6), np.int64) for variant in VARIANTS
    }
    for number, frozen in enumerate(frozen_pairs, 1):
        _verify_freeze(freeze, output)
        pair_id = frozen["id"]
        if pair_id not in targets:
            raise RuntimeError(f"{pair_id}: missing from manifest")
        target = normalize_target(targets[pair_id])
        predictions: dict[str, np.ndarray] = {}
        confusions: dict[str, list[list[int]]] = {}
        for variant in VARIANTS:
            record = frozen["predictions"][variant]
            labels = np.asarray(Image.open(output / record["relative_path"]), np.uint8)
            predictions[variant] = labels
            pair_accumulator = MetricAccumulator()
            pair_accumulator.add(labels, target)
            accumulators[variant].add_confusion(pair_accumulator.confusion)
            confusions[variant] = pair_accumulator.confusion.tolist()
            transition = _transition(predictions[VARIANTS[0]], labels)
            aggregate_transitions[variant] += transition
            if int(transition[1:, 0].sum()) != 0:
                raise AssertionError(f"{pair_id}/{variant}: changed-to-unchanged transition")
        if not np.array_equal(predictions[VARIANTS[2]] != 0, predictions[VARIANTS[3]] != 0):
            raise AssertionError(f"{pair_id}: A2/A3 binary support differs")
        if not np.array_equal(predictions[VARIANTS[2]] != 0, predictions[VARIANTS[4]] != 0):
            raise AssertionError(f"{pair_id}: A2/A4 binary support differs")
        promoted = (predictions[VARIANTS[2]] != 0) & (predictions[VARIANTS[0]] == 0)
        rescued_replaced = (
            (predictions[VARIANTS[3]] == int(Label.REPLACED))
            & (predictions[VARIANTS[2]] != int(Label.REPLACED))
            & (target == int(Label.REPLACED))
        )
        pair_reports.append(
            {
                "id": pair_id,
                "status": "success",
                "confusion": confusions,
                "feature_funnel": frozen["feature_funnel"],
                "promotion": frozen["promotion"],
                "direct_replacement_pair_count": frozen[
                    "direct_replacement_pair_count"
                ],
                "moved_reasoning_records": frozen["moved_reasoning_records"],
                "qualitative_scores": {
                    "guarded_new_changed_pixels": int(promoted.sum()),
                    "guarded_true_replaced_pixels": int(
                        np.logical_and(promoted, target == int(Label.REPLACED)).sum()
                    ),
                    "guarded_other_true_change_pixels": int(
                        np.logical_and(promoted, (target != 0) & (target != int(Label.REPLACED))).sum()
                    ),
                    "guarded_false_veto_pixels": int(
                        np.logical_and(promoted, target == 0).sum()
                    ),
                    "direct_replaced_rescue_pixels": int(rescued_replaced.sum()),
                },
                "geometry_artifacts": str(
                    Path(context["gate_records"][pair_id]["parent_artifacts"]).resolve()
                ),
                "artifacts": str((output / "pairs" / pair_id).resolve()),
            }
        )
        print(f"[evaluate {number}/{len(context['selection'])}] {pair_id}", flush=True)

    metrics = {name: item.compute() for name, item in accumulators.items()}
    tables = {name: _table3(value) for name, value in metrics.items()}
    expected_a0 = context["reports"]["best"]["variants"][
        context["a0_parent_variant"]
    ]["table3_iou_percent"]
    for section in ("binary", "multiclass"):
        for name, expected_value in expected_a0[section].items():
            if not np.isclose(
                tables[VARIANTS[0]][section][name], expected_value, atol=1e-10
            ):
                raise AssertionError(f"A0 differs from best parent: {section}/{name}")
    total_pixels = int(accumulators[VARIANTS[0]].confusion.sum())
    expected_pixels = len(context["selection"]) * 640 * 480
    if total_pixels != expected_pixels:
        raise AssertionError(f"expected {expected_pixels:,} pixels, got {total_pixels}")

    report = {
        "protocol": {
            "dataset": "ChangeSim",
            "experiment": config["experiment_id"],
            "pairs_selected": len(context["selection"]),
            "pairs_succeeded": len(context["selection"]),
            "evaluated_pixels": total_pixels,
            "seed": config["selection"]["seed"],
            "selection_role": config["protocol"]["selection_role"],
            "a0_parent_variant": context["a0_parent_variant"],
            "predictions_frozen_before_current_gt_evaluation": True,
            "ground_truth_used_in_inference": False,
            "ten_pairs_previously_used_for_development": (
                config["protocol"]["selection_role"] == "development_ablation"
            ),
            "sam3_proposals_reused": True,
            "sam3_feature_maps_reused": True,
            "promoted_tracks_share_hard_superset_batch_context": True,
            "promoted_tracking_caches_generated_this_run": len(missing),
            "promoted_tracking_caches_imported_this_run": len(imported),
            "promoted_tracking_caches_reused_locally_this_run": (
                len(plans) - len(missing) - len(imported)
            ),
            "promoted_tracking_caches_reused_this_run": len(plans) - len(missing),
            "sam2_directional_model_calls_this_run": int(
                sum(
                    bool(plan["hard_source_ids"]) + bool(plan["hard_target_ids"])
                    for plan in plans
                    if plan["id"] in set(missing)
                )
            ),
        },
        "feature_audit": audit,
        "color_audit": color_audit,
        "feature_funnel": _sum_nested_funnels(plans),
        "variants": {
            variant: {
                "description": (
                    f"Frozen {context['a0_parent_variant']} parent, replayed "
                    "byte-for-byte."
                    if variant == VARIANTS[0]
                    else VARIANT_DESCRIPTIONS[variant]
                ),
                "metrics": metrics[variant],
                "table3_iou_percent": tables[variant],
                "transition_from_a0": aggregate_transitions[variant].tolist(),
                "new_changed_pixels": int(aggregate_transitions[variant][0, 1:].sum()),
            }
            for variant in VARIANTS
        },
        "pairs": pair_reports,
        "failures": [],
        "elapsed_seconds": time.perf_counter() - experiment_started,
        "provenance": {
            "config_sha256": context["config_hash"],
            "implementation_sha256": implementation_hash,
            "execution_sha256": execution_hash,
            "parent_report_sha256": parent_hashes,
            "best_parent_freeze_sha256": _sha256_file(context["best_freeze_path"]),
            "a0_parent_variant": context["a0_parent_variant"],
            "prediction_freeze_sha256": _sha256_file(output / "predictions_frozen.json"),
            "ground_truth_used_in_inference": False,
            "tuned_on_test": False,
            "limitations": [
                "The frozen SAM3 features compare R0,1 with real I1, not a clean-render feature map.",
                "These pairs are retrospective development evidence for this post-hoc combination, not held out.",
                "Guarded tracks reuse masks produced in the shared hard-superset SAM2 state.",
                "Ordinary replacement conversion is exact among newly promoted objects; frozen parent object instances are unavailable for new-to-parent object-level overlap tests.",
                "Feature-veto and direct semantic rules are unpublished extensions to GOLDILOCS.",
            ],
        },
    }
    save_json(output / "report.json", report)
    if not args.skip_html:
        _build_html(output)
    print(json.dumps(tables, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
