#!/usr/bin/env python3
"""Run the isolated SAM3-proposal/SAM2 moved-association ablation.

The completed ``SAM3 masks + SAM2 tracking`` experiment remains immutable.
For each of its fixed ten pairs, this runner recovers the exact proposals that
failed the clean-render gate, propagates them in both temporal directions,
and stores the raw SAM2 masks.  All association variants then run from that
small cache on CPU, so changing a matching rule never requires another model
run.

Ground-truth files are deliberately opened only after every variant label map
has been saved and hashed in ``predictions_frozen.json``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

from ocmask.changesim import MetricAccumulator, normalize_target
from ocmask.config import load_config
from ocmask.stages.dinov2_motion import _finish_labels
from ocmask.stages.sam2_tracking_backend import Sam2MaskTracker
from ocmask.stages.sam3_moved_association import (
    AssociationVariant,
    MovedAssociationSettings,
    associate_moved_objects,
)
from ocmask.stages.sam3_pairwise import (
    load_cached_inputs,
    load_proposal_cache,
    proposals_to_objects,
)
from ocmask.io import save_image, save_json
from ocmask.types import Label, ObjectMask
from ocmask.visualization import colorize, instance_overlay, overlay


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs/stages/changesim-sam3-sam2-moved-association-no-splat-densegrid96.yaml"
VARIANT_DESCRIPTIONS = {
    AssociationVariant.RECIPROCAL_MUTUAL_OVERLAP.value: (
        "Accept only source/target pairs that are mutual best matches and pass "
        "both SAM2 propagation directions."
    ),
    AssociationVariant.HUNGARIAN_ONE_TO_ONE.value: (
        "Use a global one-to-one Hungarian assignment over reciprocal SAM2 scores."
    ),
    AssociationVariant.HUNGARIAN_MOTION_VERIFIED.value: (
        "Solve the reciprocal one-to-one assignment, then suppress assigned "
        "source/target masks with aligned IoU at or above 0.5 as unchanged."
    ),
}


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to the config's recommended_output directory.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check the parent, proposal IDs, and existing track caches without a model run.",
    )
    parser.add_argument(
        "--force-retrack",
        action="store_true",
        help="Replace only this experiment's track caches; parent artifacts remain read-only.",
    )
    parser.add_argument(
        "--skip-html", action="store_true", help="Write report.json but do not rebuild index.html."
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Keep track caches, frozen labels, and decisions; omit qualitative images.",
    )
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _object_mask_hash(objects: Iterable[ObjectMask]) -> str:
    digest = hashlib.sha256()
    for obj in objects:
        mask = np.ascontiguousarray(obj.mask, dtype=bool)
        digest.update(json.dumps(mask.shape).encode("utf-8"))
        digest.update(np.packbits(mask.reshape(-1)).tobytes())
        digest.update(str(float(obj.score)).encode("utf-8"))
        digest.update(json.dumps(obj.metadata, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_output_location(output: Path, immutable_roots: Sequence[Path]) -> None:
    """Prevent an experiment typo from modifying either immutable parent."""

    resolved = output.resolve()
    for root in immutable_roots:
        root = root.resolve()
        if resolved == root or _is_relative_to(resolved, root):
            raise ValueError(f"output must not be inside immutable parent {root}")


def _prepare_output(output: Path, config_sha256: str) -> None:
    """Resume only a directory already owned by this exact configuration."""

    marker = output / "experiment.json"
    if output.exists() and any(output.iterdir()):
        if not marker.is_file():
            raise RuntimeError(
                f"refusing non-empty unowned output directory: {output}"
            )
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("config_sha256") != config_sha256:
            raise RuntimeError(
                "output belongs to another configuration; choose a new directory"
            )
    output.mkdir(parents=True, exist_ok=True)


def _load_parent(parent: Path) -> tuple[dict, list[str], dict[str, dict]]:
    report = json.loads((parent / "report.json").read_text(encoding="utf-8"))
    selection = json.loads((parent / "selection.json").read_text(encoding="utf-8"))
    selected = list(selection["ids"])
    records = {str(record["id"]): record for record in report["pairs"]}
    if not selected or len(set(selected)) != len(selected):
        raise RuntimeError("the immutable parent must contain unique, non-empty pairs")
    if report.get("failures") or report["protocol"].get("pairs_succeeded") != len(selected):
        raise RuntimeError("the immutable parent is not a complete successful run")
    if set(selected) != set(records):
        raise RuntimeError("parent selection and report pair records disagree")
    for pair_id in selected:
        pair_root = Path(records[pair_id]["artifacts"]).resolve()
        if not _is_relative_to(pair_root, parent.resolve()):
            raise RuntimeError(f"{pair_id}: parent artifact escaped parent directory")
    return report, selected, records


def _manifest_targets_without_opening_gt(path: Path) -> dict[str, Path]:
    """Read target path strings only; unlike load_manifest this never opens GT."""

    base = path.resolve().parent
    targets: dict[str, Path] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "id" not in row or "target" not in row:
            raise ValueError(f"manifest line {line_number} lacks id or target")
        target = Path(row["target"])
        if not target.is_absolute():
            target = (base / target).resolve()
        targets[str(row["id"])] = target
    return targets


def _candidate_records(parent_pair: Path, stage: str) -> list[dict]:
    ledger = json.loads(
        (parent_pair / "tracking_attempts.json").read_text(encoding="utf-8")
    )
    try:
        records = ledger["stages"][stage]["attempts"]
    except KeyError as exc:
        raise RuntimeError(f"{parent_pair.name}: missing parent stage {stage}") from exc
    if not isinstance(records, list):
        raise RuntimeError(f"{parent_pair.name}: invalid attempt ledger for {stage}")
    return records


def recover_changed_candidates(
    parent_pair: Path, proposal_cache: Path
) -> tuple[list[ObjectMask], list[ObjectMask], dict[str, list[dict]]]:
    """Recover the exact changed-candidate order selected by the parent run.

    ``proposal_id`` is a one-based index into the immutable SAM3 cache.  Area
    checks ensure a stale or reordered cache is rejected before SAM2 is loaded.
    """

    stages = {
        "source_to_target": _candidate_records(parent_pair, "source_to_target"),
        "target_to_source": _candidate_records(parent_pair, "target_to_source"),
    }
    all_source = proposals_to_objects(load_proposal_cache(proposal_cache / "source.npz"))
    all_target = proposals_to_objects(load_proposal_cache(proposal_cache / "target.npz"))

    def select(name: str, objects: list[ObjectMask]) -> list[ObjectMask]:
        selected: list[ObjectMask] = []
        seen: set[int] = set()
        for expected_index, record in enumerate(stages[name]):
            proposal_id = int(record["proposal_id"])
            if proposal_id in seen:
                raise RuntimeError(f"{parent_pair.name}/{name}: duplicate proposal ID {proposal_id}")
            seen.add(proposal_id)
            if not 1 <= proposal_id <= len(objects):
                raise RuntimeError(f"{parent_pair.name}/{name}: proposal ID out of range")
            obj = objects[proposal_id - 1]
            actual_id = int(obj.metadata["automatic_proposal_id"])
            if actual_id != proposal_id:
                raise RuntimeError(f"{parent_pair.name}/{name}: proposal cache reordered")
            area = int(np.asarray(obj.mask, dtype=bool).sum())
            if area != int(record["source_area"]):
                raise RuntimeError(
                    f"{parent_pair.name}/{name}: proposal {proposal_id} area "
                    f"{area} != parent {record['source_area']}"
                )
            if int(record.get("source_index", expected_index)) != expected_index:
                raise RuntimeError(f"{parent_pair.name}/{name}: candidate order is not stable")
            selected.append(obj)
        return selected

    return (
        select("source_to_target", all_source),
        select("target_to_source", all_target),
        stages,
    )


def _tracking_implementation_identity(baseline_config: dict) -> dict:
    """Fingerprint model/code inputs without coupling cache validity to CPU rules."""

    sam2_spec = importlib.util.find_spec("sam2")
    if sam2_spec is None or not sam2_spec.submodule_search_locations:
        raise RuntimeError("installed SAM2 package could not be resolved")
    sam2_package = Path(next(iter(sam2_spec.submodule_search_locations))).resolve()
    checkpoint = Path(baseline_config["sam2"]["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = (REPOSITORY / checkpoint).resolve()
    model_cfg = Path(baseline_config["sam2"]["model_cfg"])
    cfg_candidates = [
        model_cfg,
        REPOSITORY / model_cfg,
        REPOSITORY / "src/mast3r" / model_cfg,
        sam2_package / model_cfg,
    ]
    resolved_cfg = next((item.resolve() for item in cfg_candidates if item.is_file()), None)
    files = {
        "sam2_tracking_backend": REPOSITORY / "src/ocmask/stages/sam2_tracking_backend.py",
        "sam2_adapter": REPOSITORY / "src/ocmask/adapters/sam2.py",
    }
    package_digest = hashlib.sha256()
    package_sources = sorted(sam2_package.rglob("*.py"))
    if not package_sources:
        raise RuntimeError(f"installed SAM2 package has no Python sources: {sam2_package}")
    for path in package_sources:
        package_digest.update(str(path.relative_to(sam2_package)).encode("utf-8"))
        package_digest.update(bytes.fromhex(_sha256_file(path)))

    # Importing torch reads build metadata but does not create a CUDA context;
    # cache-only reruns still perform no model work or GPU calls.
    import torch

    identity = {
        "backend": "sam2",
        "baseline_config_sha256": _sha256_json(baseline_config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "model_cfg": str(resolved_cfg) if resolved_cfg else baseline_config["sam2"]["model_cfg"],
        "model_cfg_sha256": _sha256_file(resolved_cfg) if resolved_cfg else None,
        "code_sha256": {name: _sha256_file(path) for name, path in files.items()},
        "installed_sam2": {
            "distribution_version": importlib.metadata.version("sam-2"),
            "package_root": str(sam2_package),
            "python_source_count": len(package_sources),
            "python_tree_sha256": package_digest.hexdigest(),
        },
        "torch": {
            "version": torch.__version__,
            "compiled_cuda_version": torch.version.cuda,
        },
    }
    identity["tracking_protocol_sha256"] = _sha256_json(identity)
    return identity


def _track_input_hash(
    inputs,
    source: Sequence[ObjectMask],
    target: Sequence[ObjectMask],
    stage_records: dict[str, list[dict]],
    tracking_protocol_sha256: str,
) -> str:
    payload = {
        "source_image": _sha256_array(inputs.source_render),
        "target_image": _sha256_array(inputs.target_image),
        "source_candidates": _object_mask_hash(source),
        "target_candidates": _object_mask_hash(target),
        "parent_stage_records": _sha256_json(stage_records),
        "tracking_protocol_sha256": tracking_protocol_sha256,
    }
    return _sha256_json(payload)


def _packed_masks(masks: Sequence[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if not masks:
        return np.empty((0, shape[0], (shape[1] + 7) // 8), dtype=np.uint8)
    stacked = np.stack([np.asarray(mask, dtype=bool) for mask in masks])
    if stacked.shape[1:] != shape:
        raise ValueError("tracking masks must share the target frame shape")
    return np.packbits(stacked, axis=2)


def save_tracking_cache(
    cache_dir: Path,
    forward_attempts,
    backward_attempts,
    *,
    shape: tuple[int, int],
    input_sha256: str,
    source_ids: Sequence[int],
    target_ids: Sequence[int],
) -> None:
    """Persist every raw track mask, including rejected non-empty attempts."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = cache_dir / "tracks.npz"
    temporary = cache_dir / "tracks.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            height=np.int32(shape[0]),
            width=np.int32(shape[1]),
            forward_masks_packed=_packed_masks([item.mask for item in forward_attempts], shape),
            backward_masks_packed=_packed_masks([item.mask for item in backward_attempts], shape),
            forward_accepted=np.asarray([item.accepted for item in forward_attempts], dtype=bool),
            backward_accepted=np.asarray([item.accepted for item in backward_attempts], dtype=bool),
            forward_logits=np.asarray([
                np.nan if item.object_score_logit is None else item.object_score_logit
                for item in forward_attempts
            ], dtype=np.float32),
            backward_logits=np.asarray([
                np.nan if item.object_score_logit is None else item.object_score_logit
                for item in backward_attempts
            ], dtype=np.float32),
        )
    temporary.replace(arrays_path)

    def rows(ids: Sequence[int], attempts) -> list[dict]:
        return [
            {
                "proposal_id": int(proposal_id),
                "accepted": bool(attempt.accepted),
                "object_score_logit": (
                    None if attempt.object_score_logit is None else float(attempt.object_score_logit)
                ),
                "rejection_reasons": list(attempt.rejection_reasons),
                "target_area": int(np.asarray(attempt.mask, dtype=bool).sum()),
                "mask_sha256": _sha256_array(np.asarray(attempt.mask, dtype=bool)),
            }
            for proposal_id, attempt in zip(ids, attempts, strict=True)
        ]

    metadata = {
        "schema_version": 1,
        "input_sha256": input_sha256,
        "arrays_sha256": _sha256_file(arrays_path),
        "ground_truth_used": False,
        "calls": ["source_to_target", "target_to_source"],
        "source_to_target": rows(source_ids, forward_attempts),
        "target_to_source": rows(target_ids, backward_attempts),
    }
    save_json(cache_dir / "metadata.json", metadata)


def load_tracking_cache(
    cache_dir: Path, *, input_sha256: str
) -> tuple[list[np.ndarray | None], list[np.ndarray | None], dict]:
    metadata_path = cache_dir / "metadata.json"
    arrays_path = cache_dir / "tracks.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError("tracking cache is incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise RuntimeError("unsupported tracking-cache schema")
    if metadata.get("input_sha256") != input_sha256:
        raise RuntimeError("tracking cache input hash mismatch")
    if metadata.get("arrays_sha256") != _sha256_file(arrays_path):
        raise RuntimeError("tracking cache array hash mismatch")
    with np.load(arrays_path) as cache:
        height, width = int(cache["height"]), int(cache["width"])

        def restore(prefix: str) -> list[np.ndarray | None]:
            packed = np.asarray(cache[f"{prefix}_masks_packed"], dtype=np.uint8)
            masks = np.unpackbits(packed, axis=2, count=width).astype(bool)
            masks = masks[:, :height, :width]
            accepted = np.asarray(cache[f"{prefix}_accepted"], dtype=bool)
            rows = metadata[
                "source_to_target" if prefix == "forward" else "target_to_source"
            ]
            if len(masks) != len(accepted) or len(masks) != len(rows):
                raise RuntimeError("tracking cache record counts disagree")
            output: list[np.ndarray | None] = []
            for mask, keep, row in zip(masks, accepted, rows, strict=True):
                if bool(keep) != bool(row["accepted"]):
                    raise RuntimeError("tracking cache acceptance metadata mismatch")
                if _sha256_array(mask) != row["mask_sha256"]:
                    raise RuntimeError("tracking cache mask hash mismatch")
                output.append(mask.copy() if keep else None)
            return output

        return restore("forward"), restore("backward"), metadata


def _validate_parent_scalar_replay(metadata: dict, stages: dict[str, list[dict]]) -> None:
    """Check that the two-call replay makes the parent's discrete decisions.

    Raw mask areas and bfloat16 object-presence logits may move by a few units
    across Torch compilation runs even when acceptance and the final raster do
    not change.  They are diagnostic values, not classification inputs, so we
    require equal proposal order, acceptance, and rejection reasons here.  A
    stronger pixel comparison against the parent's final ``labels.png`` runs
    before any association variant or ground-truth read and records all
    residual numerical boundary drift.
    """

    for stage in ("source_to_target", "target_to_source"):
        cached = metadata[stage]
        parent = stages[stage]
        if len(cached) != len(parent):
            raise RuntimeError(f"{stage}: parent and replay attempt counts differ")
        for current, previous in zip(cached, parent, strict=True):
            if int(current["proposal_id"]) != int(previous["proposal_id"]):
                raise RuntimeError(f"{stage}: proposal order changed")
            if bool(current["accepted"]) != bool(previous["tracker_accepted"]):
                raise RuntimeError(f"{stage}: tracker acceptance differs from parent")
            if list(current["rejection_reasons"]) != list(previous["tracker_rejection_reasons"]):
                raise RuntimeError(f"{stage}: rejection reason differs from parent")
            left, right = current["object_score_logit"], previous["object_score_logit"]
            if left is None or right is None:
                if left is not None or right is not None:
                    raise RuntimeError(f"{stage}: object score availability differs")
            elif not np.isfinite(float(left)) or not np.isfinite(float(right)):
                raise RuntimeError(f"{stage}: object score is not finite")


def _copy_object(obj: ObjectMask, label: Label, source: str) -> ObjectMask:
    return ObjectMask(
        mask=np.asarray(obj.mask, dtype=bool).copy(),
        score=float(obj.score),
        label=label,
        source=source,
        metadata=copy.deepcopy(obj.metadata),
    )


def compose_variant_labels(
    artifact: Path,
    source_candidates: Sequence[ObjectMask],
    target_candidates: Sequence[ObjectMask],
    association,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply the unchanged GOLDILOCS visibility/priority/replacement ending."""

    unmatched_source = association.diagnostics.unmatched_source_indices
    unmatched_target = association.diagnostics.unmatched_target_indices
    removed = [
        _copy_object(source_candidates[index], Label.REMOVED, "reciprocal_unmatched_source")
        for index in unmatched_source
    ]
    added = [
        _copy_object(target_candidates[index], Label.ADDED, "reciprocal_unmatched_target")
        for index in unmatched_target
    ]
    labels, filtered = _finish_labels(
        artifact,
        added,
        removed,
        list(association.moved_masks),
        return_filtered_objects=True,
    )
    counts = {name: len(items) for name, items in filtered.items()}
    return labels, counts


def replay_parent_labels_from_track_cache(
    artifact: Path,
    source_candidates: Sequence[ObjectMask],
    target_candidates: Sequence[ObjectMask],
    forward_tracks: Sequence[np.ndarray | None],
    backward_tracks: Sequence[np.ndarray | None],
) -> np.ndarray:
    """Reproduce the immutable parent's asymmetric moved-mask composition.

    The completed v4 parent writes a forward propagated raster for a
    successful source->target attempt.  In the reverse direction it writes
    the original target proposal when target->source succeeds.  This control
    intentionally mirrors that asymmetric composition; experiment variants
    instead use one matched target proposal per accepted association.
    """

    if len(source_candidates) != len(forward_tracks):
        raise ValueError("parent replay requires one forward track per source")
    if len(target_candidates) != len(backward_tracks):
        raise ValueError("parent replay requires one backward track per target")
    added: list[ObjectMask] = []
    removed: list[ObjectMask] = []
    moved: list[ObjectMask] = []
    for source, tracked in zip(source_candidates, forward_tracks, strict=True):
        if tracked is None:
            removed.append(_copy_object(source, Label.REMOVED, "parent_replay_source_failed"))
        else:
            moved.append(
                ObjectMask(
                    mask=np.asarray(tracked, dtype=bool).copy(),
                    score=float(source.score),
                    label=Label.MOVED,
                    source="parent_replay_source_to_target",
                    metadata=copy.deepcopy(source.metadata),
                )
            )
    for target, tracked in zip(target_candidates, backward_tracks, strict=True):
        if tracked is None:
            added.append(_copy_object(target, Label.ADDED, "parent_replay_target_failed"))
        else:
            moved.append(_copy_object(target, Label.MOVED, "parent_replay_target_to_source"))
    return _finish_labels(artifact, added, removed, moved)


def _centroid(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    return (int(xs.mean()), int(ys.mean())) if len(xs) else (0, 0)


def _blend_mask(image: np.ndarray, mask: np.ndarray, color: np.ndarray) -> None:
    image[mask] = 0.55 * image[mask] + 0.45 * color


def _association_image(
    source_image: np.ndarray,
    target_image: np.ndarray,
    sources: Sequence[ObjectMask],
    targets: Sequence[ObjectMask],
    association,
) -> np.ndarray:
    """Show moved pairs in color and static-suppressed pairs in amber."""

    left = np.asarray(source_image, dtype=np.float32).copy()
    right = np.asarray(target_image, dtype=np.float32).copy()
    palette = np.asarray(
        [(54, 211, 153), (96, 165, 250), (250, 204, 21), (244, 114, 182),
         (167, 139, 250), (251, 146, 60), (45, 212, 191), (248, 113, 113)],
        dtype=np.float32,
    )
    for number, match in enumerate(association.matches):
        color = palette[number % len(palette)]
        _blend_mask(left, np.asarray(sources[match.source_index].mask, bool), color)
        _blend_mask(right, np.asarray(targets[match.target_index].mask, bool), color)
    for source_index, target_index in association.diagnostics.suppressed_static_pairs:
        color = np.asarray((245, 158, 11), dtype=np.float32)
        _blend_mask(left, np.asarray(sources[source_index].mask, bool), color)
        _blend_mask(right, np.asarray(targets[target_index].mask, bool), color)
    combined = np.concatenate([left, right], axis=1).clip(0, 255).astype(np.uint8)
    canvas = Image.fromarray(combined)
    draw = ImageDraw.Draw(canvas)
    offset = left.shape[1]
    for number, match in enumerate(association.matches):
        color = tuple(int(value) for value in palette[number % len(palette)])
        sx, sy = _centroid(sources[match.source_index].mask)
        tx, ty = _centroid(targets[match.target_index].mask)
        draw.line((sx, sy, tx + offset, ty), fill=color, width=2)
        draw.text((sx, sy), f"S{match.source_proposal_id}", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        draw.text((tx + offset, ty), f"T{match.target_proposal_id}", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    # The motion-verified variant first assigns reciprocal pairs, then
    # suppresses same-location pairs as unchanged. Showing these in amber is
    # important: they are intentional decisions, not lost associations.
    for source_index, target_index in association.diagnostics.suppressed_static_pairs:
        sx, sy = _centroid(sources[source_index].mask)
        tx, ty = _centroid(targets[target_index].mask)
        draw.line((sx, sy, tx + offset, ty), fill=(245, 158, 11), width=2)
        draw.text((sx, sy), "static", fill=(255, 210, 90), stroke_width=2, stroke_fill=(0, 0, 0))
    draw.line((offset, 0, offset, combined.shape[0]), fill=(255, 255, 255), width=2)
    return np.asarray(canvas)


def _matrix_image(association) -> np.ndarray:
    """Render reciprocal scores; white boxes mark selected assignments."""

    scores = association.diagnostics.reciprocal_score
    eligible = association.diagnostics.assignment_eligible
    rows, columns = scores.shape
    cell = max(6, min(28, 560 // max(rows, columns, 1)))
    height, width = max(1, rows) * cell, max(1, columns) * cell
    image = np.full((height, width, 3), (15, 23, 42), dtype=np.uint8)
    for row in range(rows):
        for column in range(columns):
            score = float(np.clip(scores[row, column], 0.0, 1.0))
            color = np.asarray((25 + 35 * score, 45 + 130 * score, 80 + 175 * score))
            if not eligible[row, column]:
                color *= 0.35
            image[row * cell:(row + 1) * cell, column * cell:(column + 1) * cell] = color.astype(np.uint8)
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    for row, column in association.diagnostics.selected_pairs:
        draw.rectangle(
            (column * cell, row * cell, (column + 1) * cell - 1, (row + 1) * cell - 1),
            outline=(255, 255, 255), width=max(1, cell // 7),
        )
    for row, column in association.diagnostics.suppressed_static_pairs:
        draw.rectangle(
            (column * cell, row * cell, (column + 1) * cell - 1, (row + 1) * cell - 1),
            outline=(245, 158, 11), width=max(1, cell // 7),
        )
    return np.asarray(canvas)


def _error_image(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Green=correct change, red=FP, yellow=FN, purple=wrong change class."""

    output = np.full((*prediction.shape, 3), (20, 28, 40), dtype=np.uint8)
    pred_changed = prediction != int(Label.UNCHANGED)
    target_changed = target != int(Label.UNCHANGED)
    output[pred_changed & target_changed & (prediction == target)] = (45, 200, 110)
    output[pred_changed & ~target_changed] = (240, 70, 85)
    output[~pred_changed & target_changed] = (250, 195, 45)
    output[pred_changed & target_changed & (prediction != target)] = (180, 85, 220)
    return output


def _table3(metrics: dict) -> dict:
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


def _settings(config: dict) -> MovedAssociationSettings:
    return MovedAssociationSettings(**config["association"])


def _variant_names(config: dict) -> list[str]:
    names = [AssociationVariant(value).value for value in config["variants"]]
    if len(names) != 3 or len(set(names)) != 3:
        raise ValueError("this ablation requires each of the three variants exactly once")
    return names


def _build_html(output: Path) -> None:
    builder = REPOSITORY / "scripts/build_sam3_moved_association_report.py"
    subprocess.run(
        [
            sys.executable,
            str(builder),
            "--root",
            str(output),
            "--report",
            str(output / "report.json"),
            "--output",
            str(output / "index.html"),
            "--force",
        ],
        cwd=REPOSITORY,
        check=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    config = load_config(args.config)
    if config.get("tracking_backend") != "sam2":
        raise ValueError("this runner supports only the frozen SAM2 tracker")
    variants = _variant_names(config)
    settings = _settings(config)
    parent_root = (REPOSITORY / config["parent_evaluation"]).resolve()
    proposal_root = (REPOSITORY / config["proposal_cache_parent"]).resolve()
    output = (
        args.output.resolve()
        if args.output
        else (REPOSITORY / config["recommended_output"]).resolve()
    )
    _validate_output_location(output, (parent_root, proposal_root))
    config_sha256 = _sha256_json(config)
    _prepare_output(output, config_sha256)

    parent_report, selected_ids, parent_records = _load_parent(parent_root)
    parent_report_sha256 = _sha256_file(parent_root / "report.json")
    manifest_path = (REPOSITORY / config["manifest"]).resolve()
    # This index reads only JSON strings. Target image pixels remain unopened.
    target_paths = _manifest_targets_without_opening_gt(manifest_path)
    missing_targets = [pair_id for pair_id in selected_ids if pair_id not in target_paths]
    if missing_targets:
        raise RuntimeError(f"fixed pairs missing from manifest: {missing_targets}")

    first_artifact = Path(parent_records[selected_ids[0]]["parent_artifacts"])
    baseline_config = json.loads((first_artifact / "config.json").read_text(encoding="utf-8"))
    tracking_identity = _tracking_implementation_identity(baseline_config)
    implementation_sha256 = _sha256_json(
        {
            "runner": _sha256_file(Path(__file__)),
            "association": _sha256_file(REPOSITORY / "src/ocmask/stages/sam3_moved_association.py"),
        }
    )
    save_json(
        output / "experiment.json",
        {
            "experiment_id": config["experiment_id"],
            "config": str(args.config.resolve()),
            "config_sha256": config_sha256,
            "implementation_sha256": implementation_sha256,
            "parent_evaluation": str(parent_root),
            "parent_report_sha256": parent_report_sha256,
            "proposal_cache_parent": str(proposal_root),
            "selection": selected_ids,
            "tracking_identity": tracking_identity,
            "ground_truth_used_in_inference": False,
        },
    )
    save_json(output / "selection.json", {"ids": selected_ids})

    pair_inputs: dict[str, dict] = {}
    missing_cache_ids: list[str] = []
    for pair_id in selected_ids:
        record = parent_records[pair_id]
        parent_pair = Path(record["artifacts"])
        artifact = Path(record["parent_artifacts"])
        proposal_cache = proposal_root / "pairs" / pair_id / "proposal_cache"
        source, target, stages = recover_changed_candidates(parent_pair, proposal_cache)
        inputs = load_cached_inputs(artifact)
        input_hash = _track_input_hash(
            inputs,
            source,
            target,
            stages,
            tracking_identity["tracking_protocol_sha256"],
        )
        cache_dir = output / "pairs" / pair_id / "tracking_cache"
        cache_valid = False
        if not args.force_retrack:
            try:
                load_tracking_cache(cache_dir, input_sha256=input_hash)
                cache_valid = True
            except (FileNotFoundError, RuntimeError, KeyError, ValueError):
                cache_valid = False
        if not cache_valid:
            missing_cache_ids.append(pair_id)
        pair_inputs[pair_id] = {
            "record": record,
            "parent_pair": parent_pair,
            "artifact": artifact,
            "source": source,
            "target": target,
            "stages": stages,
            "inputs": inputs,
            "input_hash": input_hash,
            "cache_dir": cache_dir,
        }

    print(
        f"Validated {len(selected_ids)} fixed pairs; {len(missing_cache_ids)} require SAM2 tracking.",
        flush=True,
    )
    if args.validate_only:
        save_json(
            output / "input_validation.json",
            {
                "passed": True,
                "pairs": selected_ids,
                "tracking_caches_ready": len(missing_cache_ids) == 0,
                "missing_tracking_caches": missing_cache_ids,
                "ground_truth_opened": False,
            },
        )
        return 0

    if missing_cache_ids:
        tracker = Sam2MaskTracker(baseline_config)
        try:
            for number, pair_id in enumerate(missing_cache_ids, 1):
                data = pair_inputs[pair_id]
                started = time.perf_counter()
                # These are the only two model calls made by this experiment.
                forward = tracker.track(
                    [item.mask for item in data["source"]],
                    data["inputs"].source_render,
                    data["inputs"].target_image,
                )
                backward = tracker.track(
                    [item.mask for item in data["target"]],
                    data["inputs"].target_image,
                    data["inputs"].source_render,
                )
                if len(forward) != len(data["source"]) or len(backward) != len(data["target"]):
                    raise RuntimeError(f"{pair_id}: SAM2 returned an incomplete attempt list")
                save_tracking_cache(
                    data["cache_dir"],
                    forward,
                    backward,
                    shape=data["inputs"].target_image.shape[:2],
                    input_sha256=data["input_hash"],
                    source_ids=[int(item.metadata["automatic_proposal_id"]) for item in data["source"]],
                    target_ids=[int(item.metadata["automatic_proposal_id"]) for item in data["target"]],
                )
                _, _, metadata = load_tracking_cache(
                    data["cache_dir"], input_sha256=data["input_hash"]
                )
                if config["tracking_cache"].get("require_parent_scalar_replay", True):
                    _validate_parent_scalar_replay(metadata, data["stages"])
                print(
                    f"[tracking {number}/{len(missing_cache_ids)}] {pair_id}: "
                    f"{len(forward)} forward + {len(backward)} backward, "
                    f"{time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        finally:
            tracker.release()

    # Phase 1: construct every prediction using only RGB, proposals, tracks,
    # geometry support, and predeclared thresholds. No GT file is opened here.
    prediction_records: list[dict] = []
    parent_replay_records: list[dict] = []
    pair_report_records: list[dict] = []
    variant_pair_records: dict[str, dict[str, dict]] = {name: {} for name in variants}
    for number, pair_id in enumerate(selected_ids, 1):
        data = pair_inputs[pair_id]
        pair_dir = output / "pairs" / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        forward, backward, track_metadata = load_tracking_cache(
            data["cache_dir"], input_sha256=data["input_hash"]
        )
        if config["tracking_cache"].get("require_parent_scalar_replay", True):
            _validate_parent_scalar_replay(track_metadata, data["stages"])

        parent_replay = replay_parent_labels_from_track_cache(
            data["artifact"], data["source"], data["target"], forward, backward
        )
        immutable_parent_labels_path = data["parent_pair"] / "labels.png"
        immutable_parent_labels = np.asarray(
            Image.open(immutable_parent_labels_path), dtype=np.uint8
        )
        difference_pixels = int(np.count_nonzero(parent_replay != immutable_parent_labels))
        if difference_pixels:
            # SAM2 bfloat16 compilation can move a handful of boundary pixels
            # even when proposal order, acceptance, and rejection reasons are
            # identical.  Preserve the exact count instead of pretending the
            # historical model call is byte-reproducible.  This is computed
            # before GT is opened and therefore cannot tune the association.
            print(
                f"[parent replay] {pair_id}: {difference_pixels} numerical "
                "boundary pixels differ",
                flush=True,
            )
        parent_replay_records.append(
            {
                "pair_id": pair_id,
                "exact": difference_pixels == 0,
                "difference_pixels": difference_pixels,
                "difference_fraction": float(
                    difference_pixels / immutable_parent_labels.size
                ),
                "replay_array_sha256": _sha256_array(parent_replay),
                "immutable_labels_sha256": _sha256_file(immutable_parent_labels_path),
            }
        )

        if not args.cache_only:
            save_image(pair_dir / "input0.png", data["inputs"].source_render)
            save_image(pair_dir / "input1.png", data["inputs"].target_image)
            save_image(
                pair_dir / "source_candidates.png",
                instance_overlay(
                    data["inputs"].source_render,
                    data["source"],
                    instance_ids=[int(item.metadata["automatic_proposal_id"]) for item in data["source"]],
                ),
            )
            save_image(
                pair_dir / "target_candidates.png",
                instance_overlay(
                    data["inputs"].target_image,
                    data["target"],
                    instance_ids=[int(item.metadata["automatic_proposal_id"]) for item in data["target"]],
                ),
            )
        pair_variant_views: dict[str, dict] = {}
        for variant_name in variants:
            association = associate_moved_objects(
                data["source"],
                forward,
                data["target"],
                backward,
                variant=variant_name,
                settings=settings,
            )
            labels, final_counts = compose_variant_labels(
                data["artifact"], data["source"], data["target"], association
            )
            variant_dir = pair_dir / variant_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            labels_path = variant_dir / "labels.png"
            save_image(labels_path, labels)
            if not args.cache_only:
                native_target = np.asarray(
                    Image.fromarray(data["inputs"].target_image).resize(
                        labels.shape[::-1], Image.Resampling.LANCZOS
                    )
                )
                save_image(variant_dir / "prediction_overlay.png", overlay(native_target, labels))
                save_image(
                    variant_dir / "associations.png",
                    _association_image(
                        data["inputs"].source_render,
                        data["inputs"].target_image,
                        data["source"],
                        data["target"],
                        association,
                    ),
                )
                save_image(variant_dir / "association_matrix.png", _matrix_image(association))
            save_json(variant_dir / "association.json", association.summary())
            labels_hash = _sha256_file(labels_path)
            prediction_records.append(
                {
                    "pair_id": pair_id,
                    "variant": variant_name,
                    "path": str(labels_path.resolve()),
                    "sha256": labels_hash,
                }
            )
            match_rows = [
                {
                    "source_id": match.source_proposal_id,
                    "target_id": match.target_proposal_id,
                    "score": round(match.reciprocal_score, 4),
                    "decision": "moved",
                    "reason": (
                        "reciprocal + motion" if match.motion_verified else "reciprocal"
                    ),
                }
                for match in association.matches
            ]
            match_rows.extend(
                {
                    "source_id": int(data["source"][source_index].metadata["automatic_proposal_id"]),
                    "target_id": int(data["target"][target_index].metadata["automatic_proposal_id"]),
                    "score": round(
                        float(association.diagnostics.reciprocal_score[source_index, target_index]),
                        4,
                    ),
                    "decision": "unchanged",
                    "reason": "assigned but aligned IoU >= static cutoff",
                }
                for source_index, target_index in association.diagnostics.suppressed_static_pairs
            )
            pair_variant_views[variant_name] = {
                "status": "prediction_frozen",
                "artifacts": str(variant_dir.resolve()),
                "description": VARIANT_DESCRIPTIONS[variant_name],
                "diagnostics": {
                    "source_candidates": len(data["source"]),
                    "target_candidates": len(data["target"]),
                    "accepted_associations": len(association.matches),
                    "suppressed_static": len(association.diagnostics.suppressed_static_pairs),
                    "unmatched_source": len(association.diagnostics.unmatched_source_indices),
                    "unmatched_target": len(association.diagnostics.unmatched_target_indices),
                    **{f"final_{key}": value for key, value in final_counts.items()},
                },
                "associations": match_rows,
                "prediction_sha256": labels_hash,
            }
            variant_pair_records[variant_name][pair_id] = {
                "labels_path": labels_path,
                "labels_sha256": labels_hash,
                "view": pair_variant_views[variant_name],
            }
        pair_report_records.append(
            {
                "id": pair_id,
                "status": "success",
                "featured": pair_id == "Warehouse_6_Seq_0_2",
                "description": "Source/target changed candidates, reciprocal matches, prediction, and evaluation error.",
                "parent_artifacts": str(data["parent_pair"].resolve()),
                "baseline": {"artifacts": str(data["parent_pair"].resolve())},
                "artifacts": {
                    "input0": str((pair_dir / "input0.png").resolve()),
                    "input1": str((pair_dir / "input1.png").resolve()),
                    "source_candidates": str((pair_dir / "source_candidates.png").resolve()),
                    "target_candidates": str((pair_dir / "target_candidates.png").resolve()),
                },
                "variants": pair_variant_views,
            }
        )
        print(f"[prediction {number}/{len(selected_ids)}] {pair_id}", flush=True)

    freeze = {
        "schema_version": 1,
        "pair_count": len(selected_ids),
        "prediction_count": len(prediction_records),
        "expected_prediction_count": len(selected_ids) * len(variants),
        "ground_truth_opened_before_freeze": False,
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
        "predictions": prediction_records,
        "parent_replay_controls": parent_replay_records,
    }
    if freeze["prediction_count"] != freeze["expected_prediction_count"]:
        raise RuntimeError("not all predictions were frozen")
    save_json(output / "predictions_frozen.json", freeze)

    # Phase 2: evaluation begins only after the freeze marker exists.
    accumulators = {name: MetricAccumulator() for name in variants}
    pair_record_by_id = {record["id"]: record for record in pair_report_records}
    for pair_id in selected_ids:
        target = normalize_target(target_paths[pair_id])
        pair_dir = output / "pairs" / pair_id
        if not args.cache_only:
            save_image(pair_dir / "ground_truth.png", colorize(target))
            pair_record_by_id[pair_id]["artifacts"]["ground_truth"] = str(
                (pair_dir / "ground_truth.png").resolve()
            )
        for variant_name in variants:
            saved = variant_pair_records[variant_name][pair_id]
            if _sha256_file(saved["labels_path"]) != saved["labels_sha256"]:
                raise RuntimeError(f"{pair_id}/{variant_name}: prediction changed after freeze")
            prediction = np.asarray(Image.open(saved["labels_path"]), dtype=np.uint8)
            pair_accumulator = MetricAccumulator()
            pair_accumulator.add(prediction, target)
            accumulators[variant_name].add_confusion(pair_accumulator.confusion)
            view = saved["view"]
            view["status"] = "evaluated"
            view["confusion"] = pair_accumulator.confusion.tolist()
            if not args.cache_only:
                save_image(
                    Path(view["artifacts"]) / "error_map.png",
                    _error_image(prediction, target),
                )

    variant_reports = {}
    for variant_name in variants:
        metrics = accumulators[variant_name].compute()
        accounting_keys = (
            "source_candidates",
            "target_candidates",
            "accepted_associations",
            "suppressed_static",
            "unmatched_source",
            "unmatched_target",
        )
        accounting = {
            key: sum(
                int(variant_pair_records[variant_name][pair_id]["view"]["diagnostics"].get(key, 0))
                for pair_id in selected_ids
            )
            for key in accounting_keys
        }
        variant_reports[variant_name] = {
            "status": "complete",
            "description": VARIANT_DESCRIPTIONS[variant_name],
            "settings": asdict(settings),
            "object_accounting": accounting,
            "metrics": metrics,
            "table3_iou_percent": _table3(metrics),
        }

    report = {
        "protocol": {
            "dataset": "ChangeSim",
            "experiment": config["experiment_id"],
            "pairs_selected": len(selected_ids),
            "pairs_succeeded": len(selected_ids),
            "seed": parent_report["protocol"].get("seed"),
            "predictions_frozen_before_gt": True,
            "ground_truth_used_in_inference": False,
            "selection_role": config.get("inference", {}).get(
                "selection_role", "development ablation; not held-out"
            ),
            "protocol_change": (
                "Association variants emit the matched target SAM3 proposal as moved; "
                "the frozen parent composes forward tracked rasters and reverse proposals."
            ),
        },
        "baseline": {
            "id": parent_report["protocol"].get("experiment"),
            "status": "frozen",
            "description": "Completed SAM3 masks + SAM2 tracker parent experiment.",
            "metrics": parent_report["metrics"],
            "table3_iou_percent": parent_report["table3_iou_percent"],
        },
        "variants": variant_reports,
        "pairs": pair_report_records,
        "failures": [],
        "provenance": {
            "config_sha256": config_sha256,
            "implementation_sha256": implementation_sha256,
            "parent_report_sha256": parent_report_sha256,
            "tracking_protocol_sha256": tracking_identity["tracking_protocol_sha256"],
            "prediction_freeze_sha256": _sha256_file(output / "predictions_frozen.json"),
            "sam2_model_calls_per_uncached_pair": 2,
            "sam3_proposals_reused": True,
            "parent_outputs_modified": False,
            "track_cache_parent_replay_exact_for_all_pairs": all(
                record["exact"] for record in parent_replay_records
            ),
            "parent_replay_difference_pixels_total": sum(
                record["difference_pixels"] for record in parent_replay_records
            ),
            "parent_replay_pixels_total": sum(
                np.asarray(Image.open(Path(parent_records[record["pair_id"]]["artifacts"]) / "labels.png")).size
                for record in parent_replay_records
            ),
            "parent_replay_control_sha256": _sha256_json(parent_replay_records),
            "limitations": [
                "Reciprocal IDs and Hungarian assignment are unpublished experiments, not GOLDILOCS rules.",
                "The 0.5 static cutoff is an inferred no-3D control reused on different evidence: aligned independent proposals.",
                "A genuinely moved or replaced object with high aligned proposal IoU can be incorrectly suppressed as unchanged.",
                "Matched moved output uses the target proposal mask rather than the parent's forward-track-plus-reverse union.",
                "The repeated bfloat16 SAM2 calls preserve all discrete tracking decisions but are not byte-identical at every mask boundary; exact drift is recorded per pair.",
            ],
        },
    }
    save_json(output / "report.json", report)
    if not args.skip_html:
        _build_html(output)
    print(json.dumps({name: value["table3_iou_percent"] for name, value in variant_reports.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
