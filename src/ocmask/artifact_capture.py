"""Optional, additive persistence of per-stage diagnostics for ablation studies.

Every function here is called from ``inference.run_pair`` only when its
caller passes ``save_stage_artifacts=True``. None of them compute anything
new or influence any inference decision or return value -- they only
serialize objects a stage already built (and, without this module, would
otherwise discard) to disk. With ``save_stage_artifacts=False`` (the
default, used by ``demo.py`` and every existing evaluation run), none of
these functions are called at all, so behavior is byte-for-byte unchanged.

Every stage folder follows the same shape: a JSON diagnostics file with IDs,
scores, thresholds, and decisions, plus an ``.npz`` for anything array-shaped
(masks, dense feature maps, similarity matrices). Visualizations are saved
where a stage already builds one cheaply, but this module never treats a PNG
as the source of truth -- the JSON/NPZ pair is.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .io import save_image
from .io import save_json as _save_json_raw
from .stages.sam3_pairwise import save_proposal_cache
from .types import ObjectMask
from .visualization import colorize


def _to_jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays so ``json.dumps`` accepts them.

    The stage dataclasses captured here were designed to be handled in
    memory, not serialized; several of their fields are numpy scalars
    (``np.float32``/``np.int64``/``np.bool_``) rather than plain Python
    types. Sanitizing here, once, is more robust than auditing every field
    of every dataclass this module touches.
    """

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def save_json(path: Path, value: Any) -> None:
    """``io.save_json``, but tolerant of numpy scalars/arrays anywhere inside."""

    _save_json_raw(path, _to_jsonable(value))


def _proposal_id(obj: ObjectMask) -> int:
    return int(obj.metadata["automatic_proposal_id"])


def _pack_masks(masks: Sequence[np.ndarray]) -> dict[str, Any]:
    """Pack a list of same-shape boolean masks for compact ``.npz`` storage."""

    if masks:
        stacked = np.stack([np.asarray(mask, dtype=bool) for mask in masks])
        height, width = stacked.shape[1:]
        packed = np.packbits(stacked, axis=2)
    else:
        packed = np.empty((0, 0, 0), dtype=np.uint8)
        height = width = 0
    return {"packed": packed, "height": np.int32(height), "width": np.int32(width)}


def save_stage2_proposals(
    stage_dir: Path,
    source_generated: Sequence[Any],
    target_generated: Sequence[Any],
    source_render_shape: tuple[int, int],
    target_image_shape: tuple[int, int],
) -> None:
    """Persist raw SAM3 automatic-proposal records (masks, scores, boxes, IDs).

    Uses the same on-disk format ``ocmask.weekend_cache`` already reads for a
    stage-2 cache hit, so a run made with this flag on doubles as a reusable
    proposal cache for future runs of the same pair.
    """

    stage_dir.mkdir(parents=True, exist_ok=True)
    save_proposal_cache(stage_dir / "source.npz", list(source_generated), image_shape=source_render_shape)
    save_proposal_cache(stage_dir / "target.npz", list(target_generated), image_shape=target_image_shape)


def save_visibility_and_gate(
    stage_dir: Path,
    source_raw: Sequence[ObjectMask],
    target_raw: Sequence[ObjectMask],
    source_visible: Sequence[ObjectMask],
    target_visible: Sequence[ObjectMask],
    source_changed_ids: set[int],
    target_changed_ids: set[int],
) -> None:
    """Record which raw proposals survived the cross-render visibility gate.

    This is what "which masks were removed by filtering" reduces to for this
    pipeline: :func:`ocmask.masks.filter_visible` drops proposals outright
    (rather than flagging them), so the removed set is the set difference
    between the raw and visible ID lists.
    """

    def _side(raw: Sequence[ObjectMask], visible: Sequence[ObjectMask], changed_ids: set[int]) -> dict[str, Any]:
        raw_ids = [_proposal_id(obj) for obj in raw]
        visible_ids = {_proposal_id(obj) for obj in visible}
        return {
            "raw_proposal_ids": raw_ids,
            "visible_proposal_ids": sorted(visible_ids),
            "removed_by_visibility_gate": sorted(set(raw_ids) - visible_ids),
            "changed_proposal_ids": sorted(changed_ids),
            "area_pixels_by_proposal_id": {
                str(_proposal_id(obj)): int(np.asarray(obj.mask, dtype=bool).sum()) for obj in raw
            },
            "predicted_iou_by_proposal_id": {
                str(_proposal_id(obj)): float(obj.score) for obj in raw
            },
        }

    save_json(
        stage_dir / "visibility.json",
        {
            "source": _side(source_raw, source_visible, source_changed_ids),
            "target": _side(target_raw, target_visible, target_changed_ids),
        },
    )


def save_dense_feature_map(path: Path, source_map: np.ndarray, target_map: np.ndarray) -> None:
    """Persist one stage's dense per-pixel embedding maps for both images."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        source=np.asarray(source_map, dtype=np.float16),
        target=np.asarray(target_map, dtype=np.float16),
    )


def save_stage4_appearance(
    stage_dir: Path,
    appearance: Any,
    source_visible: Sequence[ObjectMask],
    target_visible: Sequence[ObjectMask],
    source_changed_flags: np.ndarray,
    target_changed_flags: np.ndarray,
) -> None:
    """Persist SAM3 dense maps, pooled descriptors, calibration, and matches.

    Written in the exact ``sam3_features.npz`` / ``decisions.json`` shape
    ``inference._appearance_features_from_cache`` already knows how to read,
    so this doubles as a real stage-4 cache for a later rerun in addition to
    an ablation artifact. Descriptor vectors, validity flags, and the full
    cosine similarity matrix are added as extra keys that reader ignores.
    """

    from .stages.sam3_identity_location import cosine_similarity_matrix, pairwise_mask_iou

    stage_dir.mkdir(parents=True, exist_ok=True)
    similarity = cosine_similarity_matrix(appearance.source_features, appearance.target_features)
    spatial_iou = pairwise_mask_iou(source_visible, target_visible)
    np.savez_compressed(
        stage_dir / "sam3_features.npz",
        source=np.asarray(appearance.source_map, dtype=np.float16),
        target=np.asarray(appearance.target_map, dtype=np.float16),
        source_descriptor_vectors=np.asarray(appearance.source_features.vectors, dtype=np.float32),
        source_descriptor_valid=np.asarray(appearance.source_features.valid, dtype=bool),
        source_descriptor_effective_cells=np.asarray(appearance.source_features.effective_cells, dtype=np.float32),
        target_descriptor_vectors=np.asarray(appearance.target_features.vectors, dtype=np.float32),
        target_descriptor_valid=np.asarray(appearance.target_features.valid, dtype=bool),
        target_descriptor_effective_cells=np.asarray(appearance.target_features.effective_cells, dtype=np.float32),
        cosine_similarity_matrix=np.asarray(similarity, dtype=np.float32),
        spatial_iou_matrix=np.asarray(spatial_iou, dtype=np.float32),
    )
    save_json(
        stage_dir / "decisions.json",
        {
            "matches": appearance.match_records,
            "diagnostics": {
                "source_gate_changed_count": int(np.asarray(source_changed_flags, dtype=bool).sum()),
                "target_gate_changed_count": int(np.asarray(target_changed_flags, dtype=bool).sum()),
                "source_valid_descriptor_count": int(appearance.source_features.valid.sum()),
                "target_valid_descriptor_count": int(appearance.target_features.valid.sum()),
                "calibration": asdict(appearance.calibration),
                "identity_threshold": float(appearance.identity_threshold),
            },
            "source_proposal_ids": [_proposal_id(obj) for obj in source_visible],
            "target_proposal_ids": [_proposal_id(obj) for obj in target_visible],
        },
    )


def save_moved_candidate_tracking(
    stage_dir: Path,
    source_changed_objects: Sequence[ObjectMask],
    forward_attempts: Sequence[Any],
    target_changed_objects: Sequence[ObjectMask],
    reverse_attempts: Sequence[Any],
) -> None:
    """Persist forward (source->target) and reverse (target->source) SAM2 tracks.

    ``forward`` verifies whether a source-changed candidate has a same-object
    counterpart in the target image (moved-to evidence); ``reverse`` is the
    same check starting from target-changed candidates (moved-from evidence).
    Every attempt's accepted mask is kept, not just its accept/reject verdict.
    """

    stage_dir.mkdir(parents=True, exist_ok=True)

    def _side(objects: Sequence[ObjectMask], attempts: Sequence[Any]) -> tuple[list[dict], list[np.ndarray]]:
        records = []
        masks = []
        for obj, attempt in zip(objects, attempts, strict=True):
            records.append(
                {
                    "proposal_id": _proposal_id(obj),
                    "accepted": bool(attempt.accepted),
                    "object_score_logit": attempt.object_score_logit,
                    "rejection_reasons": list(attempt.rejection_reasons),
                }
            )
            masks.append(np.asarray(attempt.mask, dtype=bool))
        return records, masks

    forward_records, forward_masks = _side(source_changed_objects, forward_attempts)
    reverse_records, reverse_masks = _side(target_changed_objects, reverse_attempts)
    save_json(stage_dir / "attempts.json", {"forward": forward_records, "reverse": reverse_records})
    forward_packed = _pack_masks(forward_masks)
    reverse_packed = _pack_masks(reverse_masks)
    np.savez_compressed(
        stage_dir / "track_masks.npz",
        forward_masks_packed=forward_packed["packed"],
        forward_height=forward_packed["height"],
        forward_width=forward_packed["width"],
        reverse_masks_packed=reverse_packed["packed"],
        reverse_height=reverse_packed["height"],
        reverse_width=reverse_packed["width"],
    )


def save_evidence_fusion(stage_dir: Path, hybrid_diagnostics: Any, refined_labels: np.ndarray) -> None:
    """Persist stage 7's transition accounting and its post-fusion label map."""

    stage_dir.mkdir(parents=True, exist_ok=True)
    save_json(stage_dir / "diagnostics.json", hybrid_diagnostics.to_dict())
    save_image(stage_dir / "labels_after_fusion.png", refined_labels)
    save_image(stage_dir / "labels_after_fusion_color.png", colorize(refined_labels))


def save_feature_veto(
    stage_dir: Path,
    decision: Any,
    labels_before: np.ndarray,
    labels_after: np.ndarray,
) -> None:
    """Persist stage 8's per-pair veto evidence, funnel counts, and label maps."""

    stage_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        stage_dir / "diagnostics.json",
        {
            "pairs": [pair.to_dict() for pair in decision.pairs],
            "hard_source_ids": list(decision.hard_source_ids),
            "hard_target_ids": list(decision.hard_target_ids),
            "guarded_source_ids": list(decision.guarded_source_ids),
            "guarded_target_ids": list(decision.guarded_target_ids),
            "funnel": decision.funnel,
        },
    )
    save_image(stage_dir / "labels_before_veto.png", labels_before)
    save_image(stage_dir / "labels_after_veto.png", labels_after)
    save_image(stage_dir / "labels_after_veto_color.png", colorize(labels_after))


def save_real_source_sentinel(
    stage_dir: Path,
    real_source_proposals: Sequence[Any],
    real_source_map: np.ndarray,
    image_shape: tuple[int, int],
) -> None:
    """Persist stage 9's fresh SAM3 proposals/features over the real T0 image."""

    stage_dir.mkdir(parents=True, exist_ok=True)
    save_proposal_cache(
        stage_dir / "real_source_proposals.npz", list(real_source_proposals), image_shape=image_shape
    )
    np.savez_compressed(
        stage_dir / "real_source_map.npz", feature_map=np.asarray(real_source_map, dtype=np.float16)
    )


def save_association_resolution(stage_dir: Path, resolver_diagnostics: dict[str, Any], base_labels: np.ndarray) -> None:
    """Persist stage 10's association evidence: matches, presence, selections.

    ``resolver_diagnostics`` is the dict ``resolve_real_image_associations``
    returns; array-valued keys (the identity similarity matrices) are pulled
    into a companion ``.npz`` and the remainder is written as JSON.
    """

    stage_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = dict(resolver_diagnostics)
    array_keys = [key for key, value in diagnostics.items() if isinstance(value, np.ndarray)]
    arrays = {key: np.asarray(diagnostics.pop(key)) for key in array_keys}
    if arrays:
        np.savez_compressed(stage_dir / "similarity_matrices.npz", **arrays)
    save_json(stage_dir / "diagnostics.json", diagnostics)
    save_image(stage_dir / "labels_after_association.png", base_labels)
    save_image(stage_dir / "labels_after_association_color.png", colorize(base_labels))
