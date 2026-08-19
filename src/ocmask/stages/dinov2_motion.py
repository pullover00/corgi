"""Offline replay helpers for the DINOv2 moved-object matching experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..masks import compose_labels, filter_visible, mark_replacements, mask_iou, same_place
from ..types import Label, ObjectMask


@dataclass(frozen=True)
class MotionCandidate:
    """One exact SAM2 proposal recovered from a baseline debug artifact."""

    candidate_id: int
    proposal_id: int
    mask: np.ndarray
    score: float
    status: str


def _stages(artifact_dir: Path) -> dict[str, dict]:
    path = artifact_dir / "sam_debug" / "index.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    return {record["stage"]: record for record in records}


def _load_mask(debug_root: Path, relative_path: str) -> np.ndarray:
    return np.asarray(Image.open(debug_root / relative_path), dtype=np.uint8) > 0


def _source_candidates(
    artifact_dir: Path,
    stage: dict,
    *,
    accepted: bool | None,
) -> list[MotionCandidate]:
    """Restore source masks, optionally filtered by final track status."""
    output = []
    debug_root = artifact_dir / "sam_debug"
    for record in stage["tracks"]:
        is_accepted = record["status"] == "tracked"
        if accepted is not None and is_accepted != accepted:
            continue
        output.append(
            MotionCandidate(
                candidate_id=int(record["id"]),
                proposal_id=int(record.get("source_proposal_id", record["id"])),
                mask=_load_mask(debug_root, record["source_mask"]),
                score=float(record.get("source_proposal_score", 1.0)),
                status=str(record["status"]),
            )
        )
    return output


def load_motion_candidates(
    artifact_dir: str | Path,
) -> tuple[list[MotionCandidate], list[MotionCandidate]]:
    """Load the exact source/target candidates rejected by the area gate."""
    artifact_dir = Path(artifact_dir)
    stages = _stages(artifact_dir)
    return (
        _source_candidates(
            artifact_dir, stages["02_source_to_clean"], accepted=False
        ),
        _source_candidates(
            artifact_dir, stages["05_target_to_clean"], accepted=False
        ),
    )


def load_clean_accepted_candidates(
    artifact_dir: str | Path,
) -> tuple[list[MotionCandidate], list[MotionCandidate]]:
    """Load masks classified as static by the source/target clean-render gate.

    Dense proposal recovery uses these masks only as an inference-time
    exclusion set: a newly prompted target mask that duplicates a known-static
    target proposal must not be relabelled as moved.
    """
    artifact_dir = Path(artifact_dir)
    stages = _stages(artifact_dir)
    return (
        _source_candidates(
            artifact_dir, stages["02_source_to_clean"], accepted=True
        ),
        _source_candidates(
            artifact_dir, stages["05_target_to_clean"], accepted=True
        ),
    )


def load_experiment_images(
    artifact_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the exact reconstruction-grid RGB inputs used by SAM2."""
    debug = Path(artifact_dir) / "sam_debug"
    source = np.asarray(
        Image.open(debug / "01_source_generate_input.png").convert("RGB")
    )
    target = np.asarray(
        Image.open(debug / "04_target_generate_input.png").convert("RGB")
    )
    return source, target


def load_static_control_pairs(
    artifact_dir: str | Path,
    *,
    minimum_iou: float,
    area_ratio_bounds: tuple[float, float],
    maximum_pairs: int,
) -> tuple[list[MotionCandidate], list[MotionCandidate]]:
    """Derive same-location static controls without using ChangeSim labels.

    Only masks accepted by both clean-render area gates are eligible. Mutual
    best spatial IoU prevents a large nested proposal from being paired with
    several masks on the opposite side.
    """
    artifact_dir = Path(artifact_dir)
    stages = _stages(artifact_dir)
    sources = _source_candidates(
        artifact_dir, stages["02_source_to_clean"], accepted=True
    )
    targets = _source_candidates(
        artifact_dir, stages["05_target_to_clean"], accepted=True
    )
    if not sources or not targets:
        return [], []

    ious = np.empty((len(sources), len(targets)), dtype=np.float64)
    for source_index, source in enumerate(sources):
        for target_index, target in enumerate(targets):
            ious[source_index, target_index] = mask_iou(
                source.mask, target.mask
            )
    source_best = np.argmax(ious, axis=1)
    target_best = np.argmax(ious, axis=0)
    low_area, high_area = area_ratio_bounds
    pairs = []
    for source_index, target_index in enumerate(source_best):
        if target_best[target_index] != source_index:
            continue
        iou = float(ious[source_index, target_index])
        if iou < minimum_iou:
            continue
        source_area = int(sources[source_index].mask.sum())
        target_area = int(targets[target_index].mask.sum())
        ratio = target_area / source_area if source_area else float("inf")
        if not low_area <= ratio <= high_area:
            continue
        pairs.append(
            (
                min(source_area, target_area),
                sources[source_index],
                targets[target_index],
            )
        )
    # Larger controls contain more object evidence and are less sensitive to
    # one-pixel SAM boundary variation.
    pairs.sort(key=lambda item: (-item[0], item[1].candidate_id, item[2].candidate_id))
    pairs = pairs[:maximum_pairs]
    return [item[1] for item in pairs], [item[2] for item in pairs]


def _object(candidate: MotionCandidate, label: Label, source: str) -> ObjectMask:
    return ObjectMask(
        mask=candidate.mask.copy(),
        score=candidate.score,
        label=label,
        source=source,
        metadata={
            "automatic_proposal_id": candidate.proposal_id,
            "motion_candidate_id": candidate.candidate_id,
        },
    )


def _finish_labels(
    artifact_dir: Path,
    added: list[ObjectMask],
    removed: list[ObjectMask],
    moved: list[ObjectMask],
    *,
    return_filtered_objects: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, list[ObjectMask]]]:
    """Apply the exact baseline visibility/composition protocol.

    Experiment reports sometimes need to distinguish classification
    candidates from objects that survive the paper's final 80% render-support
    test.  The default return type remains the original label map so existing
    DINOv2 replays are unchanged.  Opting into ``return_filtered_objects``
    additionally exposes the exact post-visibility object lists used to
    compose that map.
    """
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    tracking = config["tracking"]
    with np.load(artifact_dir / "geometry.npz") as geometry:
        coverage = np.asarray(geometry["coverage01"], dtype=bool)
    added = filter_visible(
        added,
        coverage,
        tracking["visibility_alpha"],
        tracking["minimum_mask_area"],
    )
    removed = filter_visible(
        removed,
        coverage,
        tracking["visibility_alpha"],
        tracking["minimum_mask_area"],
    )
    moved = filter_visible(
        moved,
        coverage,
        tracking["visibility_alpha"],
        tracking["minimum_mask_area"],
    )
    labels = compose_labels(coverage.shape, added + removed + moved)
    labels = mark_replacements(
        labels, added, removed, tracking["replacement_overlap_iou"]
    )
    native_shape = np.asarray(Image.open(artifact_dir / "labels.png")).shape
    native_labels = np.asarray(
        Image.fromarray(labels).resize(
            native_shape[::-1], Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    )
    if return_filtered_objects:
        return native_labels, {
            "added": added,
            "removed": removed,
            "moved": moved,
        }
    return native_labels


def replay_baseline_labels(
    artifact_dir: str | Path, apply_same_place_filter: bool = False
) -> np.ndarray:
    """Rebuild a pipeline.py classification from cached stage-03/06 debug masks.

    ``apply_same_place_filter`` (default ``False``, preserving this
    function's exact prior behavior for every existing caller) selects which
    of two real ``pipeline.py`` classification rules to replay:

    * ``False`` (default): the original rule this function always
      implemented -- every successfully tracked stage-03/06 propagation
      counts as MOVED, purely from the cached "tracked"/not-tracked status
      string. This exactly reproduces every pre-existing cached artifact in
      this repository (verified directly, see
      ``scripts/verify_replay_baseline_labels.py``, across 1038 real pairs
      spanning ``benchmark-iou-area-gate{,-batch1000,-no-splat,
      -no-splat-new15,-validate3}``: 0 mismatches).
    * ``True``: also apply ``masks.same_place`` exactly as ``pipeline.py``'s
      current stage-03/06 loops do (``PairwisePipeline.run``) -- a track that
      landed back in essentially the same place as its source is silently
      dropped (neither MOVED nor REMOVED/ADDED), matching current
      ``pipeline.py`` bit-for-bit rather than the older, incomplete replay
      rule above.

    Both rules are "real classification logic ``pipeline.py`` uses," not a
    reimplementation invented for this function -- the difference is which
    version of ``pipeline.py``'s own classification produced the artifact
    being replayed. Concretely verified (not assumed): every one of the
    1038 pre-existing cached pairs checked above never actually took the
    ``same_place`` branch when it was re-derived from their own cached
    masks under ``apply_same_place_filter=True``, EXCEPT that the vast
    majority (974/1038) of them stop reproducing their own saved
    ``labels.png`` once ``same_place`` is applied -- direct, concrete proof
    that those specific artifacts were generated by a ``pipeline.py``
    version that did not yet include the ``same_place`` check in this
    branch (predating it), not that this rule is wrong. A freshly generated
    artifact (e.g. a ground-contact-corrected pipeline pass run against
    today's ``pipeline.py``, which does include ``same_place``) needs
    ``apply_same_place_filter=True`` to replay correctly; an old, frozen
    baseline needs the default ``False``. Callers that verify a specific
    artifact's integrity must know (or be told) which kind of artifact they
    hold -- this function cannot safely auto-detect it, so the default stays
    the historically-safe, universally-verified value and callers opt in
    explicitly (see ``run_sam3_pairwise_experiment.py``'s
    ``--assume-same-place-classification`` flag) only for artifacts they
    know were produced by current ``pipeline.py``.

    ``_finish_labels`` applies the R0,1 visibility filter *after*
    classification either way, matching ``pipeline.py``'s own ordering (see
    ``masks.annotate_track_support``'s docstring: "GOLDILOCS applies its
    published visibility filter after classification").
    """
    artifact_dir = Path(artifact_dir)
    stages = _stages(artifact_dir)
    debug_root = artifact_dir / "sam_debug"
    same_place_min_iou = same_place_max_centroid = None
    if apply_same_place_filter:
        config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
        # Same config keys and defaults pipeline.py itself reads for this test.
        same_place_cfg = config.get("same_place_pairing", {})
        same_place_min_iou = same_place_cfg.get("minimum_spatial_iou", 0.30)
        same_place_max_centroid = same_place_cfg.get(
            "maximum_normalized_centroid_distance", 0.10
        )

    removed: list[ObjectMask] = []
    added: list[ObjectMask] = []
    moved: list[ObjectMask] = []

    for record in stages["03_source_changed_to_target"]["tracks"]:
        source = MotionCandidate(
            int(record["id"]),
            int(record.get("source_proposal_id", record["id"])),
            _load_mask(debug_root, record["source_mask"]),
            float(record.get("source_proposal_score", 1.0)),
            str(record["status"]),
        )
        if record["status"] == "tracked" and record.get("target_mask"):
            tracked_mask = _load_mask(debug_root, record["target_mask"])
            # pipeline.py: `elif same_place(source.mask, tracked.mask, ...): continue`
            if apply_same_place_filter and same_place(
                source.mask, tracked_mask, same_place_min_iou, same_place_max_centroid
            ):
                continue
            tracked = MotionCandidate(
                source.candidate_id,
                source.proposal_id,
                tracked_mask,
                source.score,
                "tracked",
            )
            moved.append(_object(tracked, Label.MOVED, "baseline_source_to_target"))
        else:
            removed.append(_object(source, Label.REMOVED, "baseline_source_failed"))

    for record in stages["06_target_changed_to_source"]["tracks"]:
        target = MotionCandidate(
            int(record["id"]),
            int(record.get("source_proposal_id", record["id"])),
            _load_mask(debug_root, record["source_mask"]),
            float(record.get("source_proposal_score", 1.0)),
            str(record["status"]),
        )
        if record["status"] == "tracked" and (
            not apply_same_place_filter or record.get("target_mask")
        ):
            # pipeline.py: `elif same_place(target.mask, tracked.mask, ...): continue`
            if apply_same_place_filter and record.get("target_mask"):
                tracked_mask = _load_mask(debug_root, record["target_mask"])
                if same_place(
                    target.mask, tracked_mask, same_place_min_iou, same_place_max_centroid
                ):
                    continue
            moved.append(_object(target, Label.MOVED, "baseline_target_to_source"))
        else:
            added.append(_object(target, Label.ADDED, "baseline_target_failed"))
    return _finish_labels(artifact_dir, added, removed, moved)


def compose_dinov2_labels(
    artifact_dir: str | Path,
    source_candidates: list[MotionCandidate],
    target_candidates: list[MotionCandidate],
    matches: list[tuple[int, int]],
) -> np.ndarray:
    """Classify DINO associations using the unchanged ChangeSim composition."""
    matched_source = {source_index for source_index, _ in matches}
    matched_target = {target_index for _, target_index in matches}
    moved = [
        _object(
            target_candidates[target_index],
            Label.MOVED,
            "dinov2_mask_match",
        )
        for _, target_index in matches
    ]
    removed = [
        _object(candidate, Label.REMOVED, "dinov2_unmatched_source")
        for index, candidate in enumerate(source_candidates)
        if index not in matched_source
    ]
    added = [
        _object(candidate, Label.ADDED, "dinov2_unmatched_target")
        for index, candidate in enumerate(target_candidates)
        if index not in matched_target
    ]
    return _finish_labels(Path(artifact_dir), added, removed, moved)
