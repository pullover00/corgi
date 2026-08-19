"""Single-process orchestration of the full object-consistent-masks pipeline.

``run_pair`` is the one public entrypoint this module exists for: given two
RGB images (a "before"/source photo and an "after"/target photo of the same
place), it runs every stage of the method -- MASt3R+SAM2 reconstruction, SAM3
proposals, SAM2 re-tracking, SAM3 and DINOv2 dense appearance features,
moved-object tracking, evidence fusion, a feature-gated direct-replacement
pass, a fresh real-image sentinel pass, real-image association resolution,
and finally the object-consistent replacement refinement -- in one process
and returns the final per-pixel change map.

Every stage below reuses an already-existing, faithfully-extracted function
from ``ocmask.stages.*``; this module only sequences them and passes the
right array between them. See ``docs/rewrite_plan.md`` for the exact
correspondence between each stage here and the original per-stage research
script it was extracted from, including which computations turned out to be
dead code for this winning composition and were deliberately not ported.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .adapters.dinov2 import Dinov2FeatureExtractor
from .adapters.mast3r import Mast3rAdapter
from .adapters.sam2 import Sam2Adapter
from .cache import load_reconstruction
from .io import save_image, save_json
from .masks import filter_visible
from .pipeline import PairwisePipeline
from .stages.branch_b2 import ConsolidationSettings
from .stages.object_consistent_replacement import resolve_object_consistent_labels
from .stages.real_image_association_resolver import (
    resolve_real_image_associations,
    resolver_settings_from_config,
)
from .stages.sam2_tracking_backend import Sam2MaskTracker
from .stages.sam3_feature_veto import apply_feature_veto_direct_replacement
from .stages.sam3_guarded_hybrid import refine_with_motion_and_replacement_evidence
from .stages.sam3_identity_location import Sam3FeatureExtractor, compute_appearance_features
from .stages.sam3_pairwise import load_cached_inputs, proposals_to_objects, run_cached_pair
from .stages.sam3_proposals import Sam3AutomaticMaskGenerator
from .stages.slot_inconsistency import SlotSettings
from .types import ObjectMask
from .visualization import colorize, overlay


@dataclass
class InferenceResult:
    """Everything a caller (the demo script, the evaluation script) needs."""

    labels: np.ndarray
    """The headline result: the object-consistent full-mask prediction."""
    guarded_labels: np.ndarray
    """The same decision at a more conservative rasterization footprint."""
    base_labels: np.ndarray
    """Stages 1-10's prediction before stage 11's refinement (for comparison)."""
    target_image: np.ndarray
    """Native-resolution target ("after") image, for visualization."""
    artifacts_dir: Path
    timings: dict[str, float]
    diagnostics: dict[str, Any]


def _proposal_generator_kwargs(cfg: dict[str, Any], minimum_mask_area_key: str = "minimum_mask_area") -> dict[str, Any]:
    """Translate one config section into ``Sam3AutomaticMaskGenerator`` kwargs.

    The two proposal-generation config sections this pipeline uses
    (``sam3_proposals.proposals`` and ``obvious_object_sentinel.sam3.proposal_generation``)
    name their minimum-area key differently (``minimum_mask_area`` vs
    ``minimum_mask_area_pixels``); everything else lines up 1:1 with the
    generator's constructor.
    """

    return dict(
        points_per_side=int(cfg["points_per_side"]),
        points_per_batch=int(cfg["points_per_batch"]),
        pred_iou_threshold=float(cfg["pred_iou_threshold"]),
        stability_threshold=float(cfg["stability_threshold"]),
        stability_offset=float(cfg["stability_offset"]),
        crop_layers=int(cfg["crop_layers"]),
        crop_downscale_factor=int(cfg["crop_downscale_factor"]),
        box_nms_threshold=float(cfg["box_nms_threshold"]),
        crop_nms_threshold=float(cfg["crop_nms_threshold"]),
        minimum_mask_area=int(cfg[minimum_mask_area_key]),
        multimask_output=bool(cfg["multimask_output"]),
    )


def _accepted_tracks_by_proposal_id(
    objects: list[ObjectMask], attempts: list[Any]
) -> dict[int, np.ndarray]:
    """Map proposal ID -> accepted track mask, omitting rejected attempts entirely.

    This (not a dict with ``None`` values for rejections) is the contract
    ``sam3_guarded_hybrid.moved_verification_evidence`` expects -- it treats
    dict membership itself as "this proposal has an accepted track."
    """

    return {
        int(obj.metadata["automatic_proposal_id"]): np.asarray(attempt.mask, bool)
        for obj, attempt in zip(objects, attempts, strict=True)
        if attempt.accepted
    }


def run_pair(
    image0_path: str | Path,
    image1_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> InferenceResult:
    """Run the full method on one image pair and return the change prediction.

    ``image0_path`` is the "before"/source photo, ``image1_path`` the
    "after"/target photo of the same place; the returned labels are aligned
    to ``image1``'s pixel grid. ``config`` is the merged pipeline config
    (see ``ocmask.config.load_config('configs/pipeline.yaml')``).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    reconstruction_cfg = config["reconstruction"]

    # --- Stage 1: MASt3R + SAM2 reconstruction/rendering baseline ----------
    started = time.perf_counter()
    stage1 = PairwisePipeline(
        reconstruction_cfg, Mast3rAdapter(reconstruction_cfg), Sam2Adapter(reconstruction_cfg)
    ).run(image0_path, image1_path, output_dir, artifact_level="cache")
    artifact_dir = stage1.artifacts_dir
    inputs = load_cached_inputs(artifact_dir, direction="forward")
    reconstruction = load_reconstruction(artifact_dir / "reconstruction.npz")
    timings["01_reconstruction"] = time.perf_counter() - started

    # --- Stage 2: SAM3 automatic proposals over the aligned pair -----------
    # SAM3.1's own tracking output (also computed by the original research
    # script at this stage) is not consumed anywhere downstream -- every
    # later stage re-tracks these proposals with SAM2 instead (stage 3) --
    # so it is deliberately not run here.
    started = time.perf_counter()
    sam3_proposals_cfg = config["sam3_proposals"]
    generator = Sam3AutomaticMaskGenerator(
        sam3_proposals_cfg["sam3_image_checkpoint"],
        **_proposal_generator_kwargs(sam3_proposals_cfg["proposals"]),
    )
    source_proposals_raw = proposals_to_objects(generator.generate(inputs.source_render))
    target_proposals_raw = proposals_to_objects(generator.generate(inputs.target_image))
    generator.release()
    timings["02_sam3_proposals"] = time.perf_counter() - started

    # --- Stage 3: SAM2 re-tracking of stage 2's proposals -------------------
    # This is the raster every later stage refines.
    started = time.perf_counter()
    tracker = Sam2MaskTracker(reconstruction_cfg)
    stage3_dir = output_dir / "03_tracking"
    baseline_labels, stage3_diagnostics = run_cached_pair(
        artifact_dir, stage3_dir, tracker, source_proposals_raw, target_proposals_raw
    )
    tracking_cfg = reconstruction_cfg["tracking"]
    # Rebuilding this exact list (rather than threading it out of
    # run_cached_pair) is safe only because it uses the identical
    # visibility_alpha/minimum_mask_area stage 3 used internally -- the
    # changed-proposal-ID sets below are meaningless against a differently
    # filtered list.
    source_visible = filter_visible(
        source_proposals_raw, inputs.cross_coverage, tracking_cfg["visibility_alpha"], tracking_cfg["minimum_mask_area"]
    )
    target_visible = filter_visible(
        target_proposals_raw, inputs.cross_coverage, tracking_cfg["visibility_alpha"], tracking_cfg["minimum_mask_area"]
    )
    source_changed_ids = set(stage3_diagnostics["source_changed_proposal_ids"])
    target_changed_ids = set(stage3_diagnostics["target_changed_proposal_ids"])
    source_changed_flags = np.asarray(
        [int(obj.metadata["automatic_proposal_id"]) in source_changed_ids for obj in source_visible], dtype=bool
    )
    target_changed_flags = np.asarray(
        [int(obj.metadata["automatic_proposal_id"]) in target_changed_ids for obj in target_visible], dtype=bool
    )
    timings["03_tracking"] = time.perf_counter() - started

    # --- Stage 4: dense SAM3 appearance features + identity calibration ----
    started = time.perf_counter()
    sam3_features_cfg = config["sam3_features"]
    feature_extractor = Sam3FeatureExtractor(
        sam3_features_cfg["sam3"]["source"], sam3_features_cfg["sam3"]["checkpoint"]
    )
    appearance = compute_appearance_features(
        source_visible,
        target_visible,
        inputs.source_render,
        inputs.target_image,
        source_changed_flags,
        target_changed_flags,
        feature_extractor,
        sam3_features_cfg,
    )
    feature_extractor.release()
    timings["04_sam3_features"] = time.perf_counter() - started

    # --- Stage 5: dense DINOv2 appearance features --------------------------
    started = time.perf_counter()
    dinov2_extractor = Dinov2FeatureExtractor(config["dinov2_features"])
    source_dino_map = dinov2_extractor.feature_map(inputs.source_render)
    target_dino_map = dinov2_extractor.feature_map(inputs.target_image)
    dinov2_extractor.release()
    timings["05_dinov2_features"] = time.perf_counter() - started

    # --- Stage 6: forward/backward SAM2 tracking of changed candidates -----
    # The raw propagation only; no association/variant logic is computed
    # here (see docs/rewrite_plan.md for why that logic is dead for this
    # composition). Feeds both stage 7 (evidence fusion) and stage 11
    # (consolidation).
    started = time.perf_counter()
    source_changed_objects = [obj for obj, changed in zip(source_visible, source_changed_flags) if changed]
    target_changed_objects = [obj for obj, changed in zip(target_visible, target_changed_flags) if changed]
    forward_attempts = tracker.track(
        [obj.mask for obj in source_changed_objects], inputs.source_render, inputs.target_image
    )
    reverse_attempts = tracker.track(
        [obj.mask for obj in target_changed_objects], inputs.target_image, inputs.source_render
    )
    # Two different shapes of the same raw tracking result are needed
    # downstream: stage 7 wants "proposal ID -> accepted mask" (an entry's
    # mere presence in the dict means accepted; see
    # ``_accepted_tracks_by_proposal_id``'s docstring), while stage 11's
    # ``consolidate_hypotheses`` wants a full-length, order-aligned list
    # with an explicit ``None`` for every rejected attempt.
    forward_tracks_by_id = _accepted_tracks_by_proposal_id(source_changed_objects, forward_attempts)
    reverse_tracks_by_id = _accepted_tracks_by_proposal_id(target_changed_objects, reverse_attempts)
    forward_track_masks = [
        np.asarray(attempt.mask, bool) if attempt.accepted else None for attempt in forward_attempts
    ]
    forward_track_rows = [{"rejection_reasons": list(attempt.rejection_reasons)} for attempt in forward_attempts]
    timings["06_moved_candidate_tracking"] = time.perf_counter() - started

    # --- Stage 7: fuse same-place-replacement and moved-verification -------
    #     evidence onto stage 3's baseline.
    started = time.perf_counter()
    evidence_cfg = config["motion_and_replacement_evidence"]
    refined_labels, _hybrid_diagnostics = refine_with_motion_and_replacement_evidence(
        baseline_labels,
        appearance.match_records,
        [obj.mask for obj in source_proposals_raw],
        [obj.mask for obj in target_proposals_raw],
        forward_tracks_by_id,
        reverse_tracks_by_id,
        minimum_track_candidate_iou=float(evidence_cfg["moved_verification"]["minimum_track_candidate_iou"]),
    )
    timings["07_evidence_fusion"] = time.perf_counter() - started

    # --- Stage 8: feature-veto-gated direct replacement ---------------------
    started = time.perf_counter()
    direct_labels, _feature_veto_decision = apply_feature_veto_direct_replacement(
        source_visible,
        target_visible,
        appearance.source_features,
        appearance.target_features,
        ~source_changed_flags,
        ~target_changed_flags,
        refined_labels,
        inputs.cross_coverage,
        tracker,
        inputs.source_render,
        inputs.target_image,
        same_threshold=appearance.identity_threshold,
        config=config["feature_veto"],
    )
    timings["08_feature_veto_gate"] = time.perf_counter() - started

    # --- Stage 9: fresh SAM3 proposals + features over the *real*, ---------
    #     un-warped source image (a sentinel pass independent of MASt3R's
    #     rendering, used only by stage 10's absence verification).
    started = time.perf_counter()
    sentinel_cfg = config["obvious_object_sentinel"]
    sentinel_generator = Sam3AutomaticMaskGenerator(
        sentinel_cfg["sam3"]["checkpoint"],
        **_proposal_generator_kwargs(
            sentinel_cfg["sam3"]["proposal_generation"], minimum_mask_area_key="minimum_mask_area_pixels"
        ),
    )
    real_source_proposals, real_source_map = sentinel_generator.generate_with_feature_map(reconstruction.images[0])
    sentinel_generator.release()
    real_source_objects = proposals_to_objects(real_source_proposals)
    timings["09_real_source_sentinel"] = time.perf_counter() - started

    # --- Stage 10: real-image association resolver -------------------------
    started = time.perf_counter()
    resolver_settings = resolver_settings_from_config(config)
    base_labels, _resolver_diagnostics = resolve_real_image_associations(
        reconstruction,
        real_source_objects,
        target_visible,
        real_source_map,
        appearance.target_map,
        direct_labels,
        appearance.identity_threshold,
        tracker,
        reconstruction.images[0],
        reconstruction.images[1],
        settings=resolver_settings,
    )
    timings["10_association_resolution"] = time.perf_counter() - started
    tracker.release()

    # --- Stage 11: object-consistent replacement (the method's namesake) ---
    started = time.perf_counter()
    ocm_cfg = config["object_consistent_masks"]
    tracking_attempts = json.loads((stage3_dir / "tracking_attempts.json").read_text(encoding="utf-8"))
    guarded_labels, full_labels, stage11_diagnostics = resolve_object_consistent_labels(
        artifact_dir,
        base_labels,
        source_changed_objects,
        forward_track_masks,
        forward_track_rows,
        target_proposals_raw,
        tracking_attempts,
        appearance.source_map,
        appearance.target_map,
        source_dino_map,
        target_dino_map,
        inputs.source_render,
        inputs.target_image,
        consolidation=ConsolidationSettings(**ocm_cfg["consolidation"]),
        settings=SlotSettings(**ocm_cfg["slot_inconsistency"]),
        output_dir=output_dir / "11_object_consistent_masks",
    )
    timings["11_object_consistent_replacement"] = time.perf_counter() - started

    native_target = np.asarray(Image.open(stage3_dir / "target.png").convert("RGB"))
    save_image(output_dir / "labels.png", full_labels)
    save_image(output_dir / "labels_guarded.png", guarded_labels)
    save_image(output_dir / "labels_color.png", colorize(full_labels))
    save_image(output_dir / "overlay.png", overlay(native_target, full_labels))
    save_image(output_dir / "target.png", native_target)
    diagnostics = {"stage3": stage3_diagnostics, "stage11": stage11_diagnostics, "timings": timings}
    save_json(output_dir / "inference.json", diagnostics)

    return InferenceResult(
        labels=full_labels,
        guarded_labels=guarded_labels,
        base_labels=base_labels,
        target_image=native_target,
        artifacts_dir=output_dir,
        timings=timings,
        diagnostics=diagnostics,
    )
