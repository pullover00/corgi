"""Offline pairwise GOLDILOCS replay with SAM3 proposals and/or SAM3.1 tracks."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from ..cache import load_reconstruction
from ..geometry import canonical_cloud, render_points
from ..io import save_image, save_json
from ..masks import (
    annotate_track_support,
    reject_inconsistent_tracks,
    same_place,
)
from ..pipeline import _resize_rgb
from ..types import Label, ObjectMask
from ..visualization import instance_overlay, overlay
from .dinov2_motion import _finish_labels
from .sam31_backend import Sam31MaskTracker, Sam31TrackAttempt
from .sam3_proposals import Sam3AutomaticMaskGenerator, Sam3Proposal


@dataclass(frozen=True)
class CachedPairInputs:
    """Raw images and geometry masks reused from one immutable baseline pair."""

    source_render: np.ndarray
    target_image: np.ndarray
    clean_render: np.ndarray
    cross_coverage: np.ndarray
    clean_coverage: np.ndarray
    input_provenance: dict


class MaskTracker(Protocol):
    """Small common interface shared by the SAM2 and SAM3.1 experiments."""

    def track(
        self,
        masks: list[np.ndarray],
        source_image: np.ndarray,
        target_image: np.ndarray,
    ) -> list[Sam31TrackAttempt]: ...

    def last_batch_plan(self) -> list[list[int]]: ...


def _sha256_file(path: Path) -> str:
    """Hash one immutable parent artifact without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    """Hash an array together with its shape and dtype."""

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(json.dumps(value.shape).encode("utf-8"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _image_record(path: Path, origin: str, image: np.ndarray) -> dict:
    """Create machine-readable provenance for one actual inference image."""

    return {
        "origin": origin,
        "path": str(path.resolve()),
        "file_sha256": _sha256_file(path),
        "array_sha256": _sha256_array(image),
        "shape": list(image.shape),
        "dtype": str(image.dtype),
    }


_DIRECTION_FILES = {
    "forward": {"source": "render_0_to_1.png", "clean": "render_clean_to_1.png"},
    "backward": {"source": "render_1_to_0.png", "clean": "render_clean_to_0.png"},
}


def _recompute_backward_coverage(
    reconstruction_path: Path, geometry_path: Path, geometry_config: dict, expected_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute (cross, clean) visibility coverage for the P1-through-C0 render.

    ``render_1_to_0.png`` and ``render_clean_to_0.png`` are cached pixels, but
    ``save_extended_geometry_artifacts`` discarded the coverage mask each
    ``render_points`` call also returns (it kept only the image). Only the
    forward direction's coverage was ever persisted to ``geometry.npz``. This
    mirrors the exact calls that produced the cached backward renders,
    recovering their coverage instead of reusing the forward one.
    """

    reconstruction = load_reconstruction(reconstruction_path)
    points0, points1 = reconstruction.points
    image0 = _resize_rgb(reconstruction.images[0], expected_shape)
    image1 = _resize_rgb(reconstruction.images[1], expected_shape)
    with np.load(geometry_path) as geometry:
        keep0 = np.asarray(geometry["keep0"], dtype=bool)
        keep1 = np.asarray(geometry["keep1"], dtype=bool)

    _, _, cross_coverage = render_points(
        points1,
        image1,
        reconstruction.intrinsics[0],
        reconstruction.world_to_camera[0],
        expected_shape,
        z_epsilon=geometry_config["z_buffer_epsilon"],
        splat_radius=geometry_config.get("splat_radius", 0),
        fill_holes=geometry_config.get("hole_fill_enabled", False),
        hole_fill_min_neighbors=geometry_config.get("hole_fill_min_neighbors", 5),
        hole_fill_max_relative_depth=geometry_config.get("hole_fill_max_relative_depth", 0.02),
    )
    star_points, star_colors = canonical_cloud(points0, image0, keep0, points1, image1, keep1)
    _, _, clean_coverage = render_points(
        star_points,
        star_colors,
        reconstruction.intrinsics[0],
        reconstruction.world_to_camera[0],
        expected_shape,
        z_epsilon=geometry_config["z_buffer_epsilon"],
    )
    return np.asarray(cross_coverage, dtype=bool), np.asarray(clean_coverage, dtype=bool)


def load_cached_inputs(
    artifact_dir: str | Path,
    direction: str = "forward",
    geometry_config: dict | None = None,
) -> CachedPairInputs:
    """Load raw reconstruction-grid inputs, never annotated debug images.

    Revision 3 accidentally passed
    ``sam_debug/02_source_to_clean_target.png`` to the tracker.  That file is
    an explanatory overlay containing colored masks and object numbers.  It
    changed most pixels and made clean-scene tracking meaningless.  The
    revision-4 contract below names each raw parent artifact explicitly and
    records hashes so a report can prove which pixels reached the model.

    ``direction="backward"`` reuses the same immutable parent reconstruction
    but compares the P1-through-C0 render against the real *source* (I0)
    image instead -- the mirror image of the forward direction. Its coverage
    mask is not cached (see ``_recompute_backward_coverage``), so
    ``geometry_config`` (the parent run's ``geometry:`` config block) is
    required to recompute it identically to how the cached render was made.
    """

    if direction not in _DIRECTION_FILES:
        raise ValueError(f"unknown direction {direction!r}")
    if direction == "backward" and geometry_config is None:
        raise ValueError("geometry_config is required to recompute backward coverage")

    artifact_dir = Path(artifact_dir)
    files = _DIRECTION_FILES[direction]
    source_path = artifact_dir / files["source"]
    clean_path = artifact_dir / files["clean"]
    reconstruction_path = artifact_dir / "reconstruction.npz"
    geometry_path = artifact_dir / "geometry.npz"

    source_render = np.asarray(Image.open(source_path).convert("RGB"))
    clean_render = np.asarray(Image.open(clean_path).convert("RGB"))

    if direction == "forward":
        with np.load(geometry_path) as geometry:
            cross_coverage = np.asarray(geometry["coverage01"], dtype=bool)
            clean_coverage = np.asarray(geometry["clean_coverage"], dtype=bool)
        expected_shape = cross_coverage.shape
    else:
        expected_shape = source_render.shape[:2]
        cross_coverage, clean_coverage = _recompute_backward_coverage(
            reconstruction_path, geometry_path, geometry_config, expected_shape
        )
    if clean_coverage.shape != expected_shape:
        raise ValueError("cross and clean coverage must have equal shapes")

    reconstruction = load_reconstruction(reconstruction_path)
    target_index = 1 if direction == "forward" else 0
    target_image = np.asarray(reconstruction.images[target_index], dtype=np.uint8)
    if target_image.shape[:2] != expected_shape:
        target_image = np.asarray(
            Image.fromarray(target_image).resize(
                expected_shape[::-1], Image.Resampling.LANCZOS
            )
        )

    images = {
        "source_render": source_render,
        "clean_render": clean_render,
        "target_image": target_image,
    }
    for name, image in images.items():
        if image.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8, got {image.dtype}")
        if image.shape != (*expected_shape, 3):
            raise ValueError(
                f"{name} must have shape {(*expected_shape, 3)}, got {image.shape}"
            )

    provenance = {
        "input_contract_revision": 4,
        "direction": direction,
        "debug_artifacts_used": False,
        "source_render": _image_record(
            source_path, files["source"], source_render
        ),
        "clean_render": _image_record(
            clean_path, files["clean"], clean_render
        ),
        "target_image": _image_record(
            reconstruction_path, f"reconstruction.npz:image{target_index}", target_image
        ),
        "geometry": {
            "path": str(geometry_path.resolve()),
            "file_sha256": _sha256_file(geometry_path),
            "cross_coverage_sha256": _sha256_array(cross_coverage),
            "clean_coverage_sha256": _sha256_array(clean_coverage),
            "shape": list(expected_shape),
            "recomputed": direction == "backward",
        },
    }
    return CachedPairInputs(
        source_render=source_render,
        target_image=target_image,
        clean_render=clean_render,
        cross_coverage=cross_coverage,
        clean_coverage=clean_coverage,
        input_provenance=provenance,
    )


def _debug_stages(artifact_dir: Path) -> dict[str, dict]:
    records = json.loads(
        (artifact_dir / "sam_debug/index.json").read_text(encoding="utf-8")
    )
    return {record["stage"]: record for record in records}


def load_selected_sam2_masks(
    artifact_dir: str | Path,
) -> tuple[list[ObjectMask], list[ObjectMask]]:
    """Recover exact post-visibility SAM2 masks from baseline tracking stages."""

    artifact_dir = Path(artifact_dir)
    debug = artifact_dir / "sam_debug"
    stages = _debug_stages(artifact_dir)

    def load(stage_name: str) -> list[ObjectMask]:
        masks = []
        for index, record in enumerate(stages[stage_name]["tracks"], start=1):
            mask = np.asarray(
                Image.open(debug / record["source_mask"]), dtype=np.uint8
            ) > 0
            masks.append(
                ObjectMask(
                    mask=mask,
                    score=float(record.get("source_proposal_score", 1.0)),
                    source="cached_sam2_proposal",
                    metadata={
                        "automatic_proposal_id": int(
                            record.get("source_proposal_id", index)
                        ),
                        "proposal_backend": "sam2",
                    },
                )
            )
        return masks

    return load("02_source_to_clean"), load("05_target_to_clean")


def proposals_to_objects(
    proposals: list[Sam3Proposal],
) -> list[ObjectMask]:
    """Convert SAM3 generator records into the pipeline's neutral mask type."""

    objects = []
    for index, proposal in enumerate(proposals, start=1):
        x0, y0, x1, y1 = proposal.crop_box_xyxy
        objects.append(
            ObjectMask(
                mask=proposal.mask,
                score=proposal.predicted_iou,
                source="sam3_automatic",
                metadata={
                    "automatic_proposal_id": index,
                    "proposal_backend": "sam3",
                    "stability_score": proposal.stability_score,
                    "point_coords": [list(proposal.point_xy)],
                    "crop_box_xywh": [x0, y0, x1 - x0, y1 - y0],
                },
            )
        )
    return objects


def save_proposal_cache(
    path: str | Path,
    proposals: list[Sam3Proposal],
    image_shape: tuple[int, int] | None = None,
) -> None:
    """Store masks compactly so tracking can resume without rerunning SAM3."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if proposals:
        masks = np.stack([proposal.mask for proposal in proposals])
        height, width = masks.shape[1:]
        packed = np.packbits(masks, axis=2)
    else:
        # Preserve the frame dimensions even when all SAM3 proposals are
        # rejected. This makes an empty cache a valid, resumable result.
        if image_shape is None:
            raise ValueError("image_shape is required for an empty proposal cache")
        height, width = image_shape
        packed = np.empty((0, 0, 0), dtype=np.uint8)
    np.savez_compressed(
        path,
        masks_packed=packed,
        height=np.int32(height),
        width=np.int32(width),
        predicted_iou=np.asarray(
            [proposal.predicted_iou for proposal in proposals], np.float32
        ),
        stability_score=np.asarray(
            [proposal.stability_score for proposal in proposals], np.float32
        ),
        points=np.asarray(
            [proposal.point_xy for proposal in proposals], np.float32
        ).reshape(-1, 2),
        crop_boxes=np.asarray(
            [proposal.crop_box_xyxy for proposal in proposals], np.int32
        ).reshape(-1, 4),
    )


def load_proposal_cache(path: str | Path) -> list[Sam3Proposal]:
    """Restore :func:`save_proposal_cache` without changing mask pixels."""

    with np.load(path) as cache:
        height = int(cache["height"])
        width = int(cache["width"])
        if len(cache["masks_packed"]):
            masks = np.unpackbits(
                cache["masks_packed"], axis=2, count=width
            )[:, :height, :width].astype(bool)
        else:
            masks = np.empty((0, height, width), dtype=bool)
        return [
            Sam3Proposal(
                mask=masks[index],
                predicted_iou=float(cache["predicted_iou"][index]),
                stability_score=float(cache["stability_score"][index]),
                point_xy=tuple(float(value) for value in cache["points"][index]),
                crop_box_xyxy=tuple(
                    int(value) for value in cache["crop_boxes"][index]
                ),
            )
            for index in range(len(masks))
        ]


def _attempt_objects(
    sources: list[ObjectMask],
    attempts: list[Sam31TrackAttempt],
    stage: str,
) -> tuple[list[ObjectMask | None], list[ObjectMask]]:
    """Convert raw SAM3.1 outputs and preserve rejected attempts for reports."""

    accepted: list[ObjectMask | None] = []
    raw: list[ObjectMask] = []
    for index, (source, attempt) in enumerate(zip(sources, attempts)):
        metadata = {
            "source_index": index,
            "source_metadata": copy.deepcopy(source.metadata),
            "sam_object_id": attempt.object_id,
            "tracker_batch_index": attempt.batch_index,
            "tracker_input_index": attempt.input_index,
            "object_score_logit": attempt.object_score_logit,
            "raw_target_area": int(attempt.mask.sum()),
            "tracker_accepted": bool(attempt.accepted),
            "rejection_reasons": list(attempt.rejection_reasons),
        }
        # Raw diagnostics and post-tracker gate candidates must not share a
        # mask or metadata dictionary.  Consistency/area-gate annotations are
        # deliberately written only to the candidate copy.
        raw_obj = ObjectMask(
            mask=np.asarray(attempt.mask, dtype=bool).copy(),
            score=source.score,
            source=f"sam31_{stage}_attempt",
            metadata=copy.deepcopy(metadata),
        )
        raw.append(raw_obj)
        if attempt.accepted:
            accepted.append(
                ObjectMask(
                    mask=np.asarray(attempt.mask, dtype=bool).copy(),
                    score=source.score,
                    source=f"sam31_{stage}_candidate",
                    metadata=copy.deepcopy(metadata),
                )
            )
        else:
            accepted.append(None)
    return accepted, raw


def _track(
    tracker: MaskTracker,
    sources: list[ObjectMask],
    source_image: np.ndarray,
    target_image: np.ndarray,
    stage: str,
) -> tuple[list[ObjectMask | None], list[ObjectMask]]:
    attempts = tracker.track(
        [source.mask for source in sources], source_image, target_image
    )
    if len(attempts) != len(sources):
        raise RuntimeError(f"{stage}: tracker returned a different result count")
    return _attempt_objects(sources, attempts, stage)


def _stage_records(
    sources: list[ObjectMask],
    raw: list[ObjectMask],
    tracker_candidates: list[ObjectMask | None],
    post_gate: list[ObjectMask | None],
) -> tuple[list[dict], dict]:
    """Describe each attempt and summarize the tracker-to-gate funnel."""

    if not (
        len(sources)
        == len(raw)
        == len(tracker_candidates)
        == len(post_gate)
    ):
        raise ValueError("tracking diagnostics require aligned object lists")
    records: list[dict] = []
    tracker_reasons: Counter[str] = Counter()
    gate_reasons: Counter[str] = Counter()
    for index, (source, raw_item, candidate, retained) in enumerate(
        zip(sources, raw, tracker_candidates, post_gate)
    ):
        raw_reasons = list(raw_item.metadata.get("rejection_reasons", []))
        candidate_reasons = (
            list(candidate.metadata.get("rejection_reasons", []))
            if candidate is not None
            else raw_reasons
        )
        for reason in raw_reasons:
            tracker_reasons[reason] += 1
        for reason in candidate_reasons:
            if reason not in raw_reasons:
                gate_reasons[reason] += 1
        record = {
            "source_index": index,
            "proposal_id": source.metadata.get("automatic_proposal_id"),
            "tracker_object_id": raw_item.metadata.get("sam_object_id"),
            "batch_index": raw_item.metadata.get("tracker_batch_index"),
            "tracker_accepted": candidate is not None,
            "post_consistency_gate_accepted": retained is not None,
            "source_area": int(np.asarray(source.mask, dtype=bool).sum()),
            "target_area": int(np.asarray(raw_item.mask, dtype=bool).sum()),
            "object_score_logit": raw_item.metadata.get("object_score_logit"),
            "tracker_rejection_reasons": raw_reasons,
            "consistency_gate_rejection_reasons": [
                reason
                for reason in candidate_reasons
                if reason not in raw_reasons
            ],
        }
        if candidate is not None:
            for key in (
                "source_target_iou",
                "source_target_area_ratio",
                "destination_support_fraction",
            ):
                if key in candidate.metadata:
                    record[key] = candidate.metadata[key]
        records.append(record)
    summary = {
        "attempted": len(sources),
        "tracker_accepted": sum(item is not None for item in tracker_candidates),
        "post_consistency_gate_accepted": sum(
            item is not None for item in post_gate
        ),
        "batch_plan": [
            [
                index
                for index, item in enumerate(raw)
                if item.metadata.get("tracker_batch_index") == batch_index
            ]
            for batch_index in sorted(
                {
                    int(item.metadata.get("tracker_batch_index", 0))
                    for item in raw
                }
            )
        ],
        "tracker_rejection_reasons": dict(sorted(tracker_reasons.items())),
        "consistency_gate_rejection_reasons": dict(sorted(gate_reasons.items())),
    }
    return records, summary


def _stage_overlay(
    output: Path,
    name: str,
    source_image: np.ndarray,
    target_image: np.ndarray,
    source_masks: list[ObjectMask],
    accepted: list[ObjectMask | None],
    attempts: list[ObjectMask],
) -> None:
    statuses = [item is not None for item in accepted]
    save_image(
        output / f"{name}_source.png",
        instance_overlay(source_image, source_masks, statuses=statuses),
    )
    save_image(
        output / f"{name}_target.png",
        instance_overlay(target_image, attempts, statuses=statuses),
    )


def run_cached_pair(
    artifact_dir: str | Path,
    output_dir: str | Path,
    tracker: MaskTracker,
    source_proposals: list[ObjectMask],
    target_proposals: list[ObjectMask],
    direction: str = "forward",
) -> tuple[np.ndarray, dict]:
    """Execute all four GOLDILOCS tracking stages on cached geometry.

    ``direction="backward"`` mirrors the whole stage across the P1-through-C0
    render vs the real source (I0) image; see ``load_cached_inputs``.
    """

    from ..masks import filter_visible

    artifact_dir = Path(artifact_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    inputs = load_cached_inputs(artifact_dir, direction=direction, geometry_config=config["geometry"])
    tracking = config["tracking"]
    area_bounds = tuple(tracking["track_area_ratio_bounds"])

    source_masks = filter_visible(
        source_proposals,
        inputs.cross_coverage,
        tracking["visibility_alpha"],
        tracking["minimum_mask_area"],
    )
    target_masks = filter_visible(
        target_proposals,
        inputs.cross_coverage,
        tracking["visibility_alpha"],
        tracking["minimum_mask_area"],
    )
    # Persist the exact unannotated pixels used by the model.  These images are
    # evidence, not inputs read back by later stages.
    save_image(output / "inference_source_render.png", inputs.source_render)
    save_image(output / "inference_clean_render.png", inputs.clean_render)
    save_image(output / "inference_target_image.png", inputs.target_image)
    save_image(
        output / "source_proposals.png",
        instance_overlay(inputs.source_render, source_proposals),
    )
    save_image(
        output / "target_proposals.png",
        instance_overlay(inputs.target_image, target_proposals),
    )

    source_to_clean_tracker, source_to_clean_raw = _track(
        tracker,
        source_masks,
        inputs.source_render,
        inputs.clean_render,
        "source_to_clean",
    )
    annotate_track_support(source_to_clean_raw, inputs.clean_coverage)
    annotate_track_support(source_to_clean_tracker, inputs.clean_coverage)
    source_to_clean = reject_inconsistent_tracks(
        source_masks,
        source_to_clean_tracker,
        tracking["minimum_track_iou"],
        area_ratio_bounds=area_bounds,
    )
    source_to_clean_records, source_to_clean_summary = _stage_records(
        source_masks,
        source_to_clean_raw,
        source_to_clean_tracker,
        source_to_clean,
    )
    _stage_overlay(
        output,
        "source_to_clean",
        inputs.source_render,
        inputs.clean_render,
        source_masks,
        source_to_clean,
        source_to_clean_raw,
    )
    source_changed = [
        source
        for source, tracked in zip(source_masks, source_to_clean)
        if tracked is None
    ]
    source_to_target_tracker, source_to_target_raw = _track(
        tracker,
        source_changed,
        inputs.source_render,
        inputs.target_image,
        "source_to_target",
    )
    source_to_target = list(source_to_target_tracker)
    source_to_target_records, source_to_target_summary = _stage_records(
        source_changed,
        source_to_target_raw,
        source_to_target_tracker,
        source_to_target,
    )
    _stage_overlay(
        output,
        "source_to_target",
        inputs.source_render,
        inputs.target_image,
        source_changed,
        source_to_target,
        source_to_target_raw,
    )

    same_place_cfg = config.get("same_place_pairing", {})
    same_place_min_iou = same_place_cfg.get("minimum_spatial_iou", 0.30)
    same_place_max_centroid = same_place_cfg.get(
        "maximum_normalized_centroid_distance", 0.10
    )

    removed: list[ObjectMask] = []
    moved: list[ObjectMask] = []
    # source_render and target_image share one pixel grid, so a re-found
    # source object is directly comparable to where it started: if it landed
    # in essentially the same place, the clean-gate rejection was wrong and
    # this was never a real move -- drop it back to unchanged instead of
    # assuming MOVED just because SAM2 tracked it somewhere.
    for source, tracked in zip(source_changed, source_to_target):
        if tracked is None:
            source.label = Label.REMOVED
            source.source = "sam31_source_failed"
            removed.append(source)
        elif same_place(
            source.mask, tracked.mask, same_place_min_iou, same_place_max_centroid
        ):
            continue
        else:
            tracked.label = Label.MOVED
            tracked.source = "sam31_source_to_target"
            moved.append(tracked)

    target_to_clean_tracker, target_to_clean_raw = _track(
        tracker,
        target_masks,
        inputs.target_image,
        inputs.clean_render,
        "target_to_clean",
    )
    annotate_track_support(target_to_clean_raw, inputs.clean_coverage)
    annotate_track_support(target_to_clean_tracker, inputs.clean_coverage)
    target_to_clean = reject_inconsistent_tracks(
        target_masks,
        target_to_clean_tracker,
        tracking["minimum_track_iou"],
        area_ratio_bounds=area_bounds,
    )
    target_to_clean_records, target_to_clean_summary = _stage_records(
        target_masks,
        target_to_clean_raw,
        target_to_clean_tracker,
        target_to_clean,
    )
    _stage_overlay(
        output,
        "target_to_clean",
        inputs.target_image,
        inputs.clean_render,
        target_masks,
        target_to_clean,
        target_to_clean_raw,
    )
    target_changed = [
        target
        for target, tracked in zip(target_masks, target_to_clean)
        if tracked is None
    ]
    target_to_source_tracker, target_to_source_raw = _track(
        tracker,
        target_changed,
        inputs.target_image,
        inputs.source_render,
        "target_to_source",
    )
    annotate_track_support(target_to_source_raw, inputs.cross_coverage)
    annotate_track_support(target_to_source_tracker, inputs.cross_coverage)
    target_to_source = list(target_to_source_tracker)
    target_to_source_records, target_to_source_summary = _stage_records(
        target_changed,
        target_to_source_raw,
        target_to_source_tracker,
        target_to_source,
    )
    _stage_overlay(
        output,
        "target_to_source",
        inputs.target_image,
        inputs.source_render,
        target_changed,
        target_to_source,
        target_to_source_raw,
    )

    added: list[ObjectMask] = []
    # target_image and source_render share one pixel grid, so the same
    # same-place check applies symmetrically here.
    for target, tracked in zip(target_changed, target_to_source):
        if tracked is None:
            target.label = Label.ADDED
            target.source = "sam31_target_failed"
            added.append(target)
        elif same_place(
            target.mask, tracked.mask, same_place_min_iou, same_place_max_centroid
        ):
            continue
        else:
            target.label = Label.MOVED
            target.source = "sam31_target_to_source"
            moved.append(target)

    candidate_object_counts = {
        "added": len(added),
        "removed": len(removed),
        "moved": len(moved),
    }
    labels, final_objects = _finish_labels(
        artifact_dir,
        added,
        removed,
        moved,
        return_filtered_objects=True,
    )
    save_image(output / "labels.png", labels)
    native_target = np.asarray(
        Image.fromarray(inputs.target_image).resize(
            labels.shape[::-1], Image.Resampling.LANCZOS
        )
    )
    save_image(output / "target.png", native_target)
    save_image(output / "overlay.png", overlay(native_target, labels))
    diagnostics = {
        "input_provenance": inputs.input_provenance,
        "proposal_counts": {
            "source_raw": len(source_proposals),
            "source_visible": len(source_masks),
            "target_raw": len(target_proposals),
            "target_visible": len(target_masks),
        },
        "track_counts": {
            "source_to_clean_tracker_accepted": source_to_clean_summary[
                "tracker_accepted"
            ],
            "source_to_clean": sum(item is not None for item in source_to_clean),
            "source_changed": len(source_changed),
            "source_to_target_tracker_accepted": source_to_target_summary[
                "tracker_accepted"
            ],
            "source_to_target": sum(item is not None for item in source_to_target),
            "target_to_clean_tracker_accepted": target_to_clean_summary[
                "tracker_accepted"
            ],
            "target_to_clean": sum(item is not None for item in target_to_clean),
            "target_changed": len(target_changed),
            "target_to_source_tracker_accepted": target_to_source_summary[
                "tracker_accepted"
            ],
            "target_to_source": sum(item is not None for item in target_to_source),
        },
        "candidate_object_counts": candidate_object_counts,
        "final_object_counts": {
            name: len(objects) for name, objects in final_objects.items()
        },
    }
    tracking_attempts = {
        "input_provenance": inputs.input_provenance,
        "stages": {
            "source_to_clean": {
                "funnel": source_to_clean_summary,
                "attempts": source_to_clean_records,
            },
            "source_to_target": {
                "funnel": source_to_target_summary,
                "attempts": source_to_target_records,
            },
            "target_to_clean": {
                "funnel": target_to_clean_summary,
                "attempts": target_to_clean_records,
            },
            "target_to_source": {
                "funnel": target_to_source_summary,
                "attempts": target_to_source_records,
            },
        },
        "candidate_object_counts": candidate_object_counts,
        "final_object_counts": diagnostics["final_object_counts"],
    }
    save_json(output / "tracking_attempts.json", tracking_attempts)
    save_json(output / "diagnostics.json", diagnostics)
    return labels, diagnostics
