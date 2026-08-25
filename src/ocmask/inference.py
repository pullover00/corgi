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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import artifact_capture
from .adapters.dinov2 import Dinov2FeatureExtractor
from .adapters.mast3r import Mast3rAdapter
from .adapters.sam2 import Sam2Adapter
from .cache import load_reconstruction
from .io import save_image, save_json
from .masks import filter_visible
from .numerics import (
    apply_post_reconstruction_numerics,
    capture_torch_numerical_state,
)
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
from .stages.sam3_identity_location import (
    AppearanceFeatures,
    Sam3FeatureExtractor,
    calibrate_identity_threshold,
    classify_identity_location,
    cosine_similarity_matrix,
    compute_appearance_features,
    mask_descriptors,
    pairwise_mask_iou,
)
from .stages.sam3_pairwise import (
    load_cached_inputs,
    load_proposal_cache,
    proposals_to_objects,
    run_cached_pair,
)
from .stages.sam3_proposals import Sam3AutomaticMaskGenerator
from .stages.slot_inconsistency import SlotSettings
from .types import ObjectMask
from .visualization import colorize, overlay
from .weekend_cache import ChangesimWeekendCache, StageCacheDirs, changed_proposal_ids_from_tracking_attempts


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


def _appearance_features_from_cache(
    stage4_dir: Path,
    source_visible: list[ObjectMask],
    target_visible: list[ObjectMask],
    source_changed_flags: np.ndarray,
    target_changed_flags: np.ndarray,
    sam3_features_cfg: dict[str, Any],
) -> tuple[AppearanceFeatures | None, dict[str, Any]]:
    """Reuse cached dense maps while recomputing every current CPU decision.

    ``decisions.json`` is validation/audit metadata, never an inference input:
    descriptors, pair calibration, and identity/location classifications are
    rebuilt from the dense tensors and the live proposal/clean-gate lists.
    Consequently a cache hit and a fresh run execute the same current CPU
    implementation. Returns ``(None, diagnostics)`` when the cached tensors
    cannot be trusted, allowing the caller to rerun the SAM3 backbone.
    """

    try:
        with np.load(stage4_dir / "sam3_features.npz") as cache:
            source_map = np.asarray(cache["source"])
            target_map = np.asarray(cache["target"])
        decisions = json.loads((stage4_dir / "decisions.json").read_text(encoding="utf-8"))
    except (OSError, EOFError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return None, {
            "dense_maps_reused": False,
            "decision_policy": "current_cpu_recomputed",
            "fallback_reason": f"cache_read_failed:{type(exc).__name__}",
        }

    if (
        source_map.ndim != 3
        or target_map.ndim != 3
        or source_map.shape[0] != target_map.shape[0]
    ):
        return None, {
            "dense_maps_reused": False,
            "decision_policy": "current_cpu_recomputed",
            "fallback_reason": "invalid_dense_feature_shapes",
            "source_shape": list(source_map.shape),
            "target_shape": list(target_map.shape),
        }

    diagnostics = decisions.get("diagnostics", {})
    # Cross-check against the live "visible, changed" accounting this cached
    # feature tensor was built from. A mismatch means the cached proposal
    # inventory is not the one being pooled now, even if its dimensions happen
    # to be compatible.
    if int(source_changed_flags.sum()) != diagnostics.get("source_gate_changed_count"):
        return None, {
            "dense_maps_reused": False,
            "decision_policy": "current_cpu_recomputed",
            "fallback_reason": "source_gate_count_mismatch",
        }
    if int(target_changed_flags.sum()) != diagnostics.get("target_gate_changed_count"):
        return None, {
            "dense_maps_reused": False,
            "decision_policy": "current_cpu_recomputed",
            "fallback_reason": "target_gate_count_mismatch",
        }

    minimum_cells = float(sam3_features_cfg["sam3"]["minimum_feature_cells"])
    try:
        source_features = mask_descriptors(
            source_map, source_visible, minimum_feature_cells=minimum_cells
        )
        target_features = mask_descriptors(
            target_map, target_visible, minimum_feature_cells=minimum_cells
        )
    except ValueError as exc:
        return None, {
            "dense_maps_reused": False,
            "decision_policy": "current_cpu_recomputed",
            "fallback_reason": f"descriptor_rebuild_failed:{exc}",
        }
    if int(source_features.valid.sum()) != diagnostics.get("source_valid_descriptor_count"):
        return None, {
            "dense_maps_reused": False,
            "decision_policy": "current_cpu_recomputed",
            "fallback_reason": "source_descriptor_count_mismatch",
        }
    if int(target_features.valid.sum()) != diagnostics.get("target_valid_descriptor_count"):
        return None, {
            "dense_maps_reused": False,
            "decision_policy": "current_cpu_recomputed",
            "fallback_reason": "target_descriptor_count_mismatch",
        }

    spatial_iou = pairwise_mask_iou(source_visible, target_visible)
    matching = sam3_features_cfg["matching"]
    calibration = calibrate_identity_threshold(
        cosine_similarity_matrix(source_features, target_features),
        spatial_iou,
        ~np.asarray(source_changed_flags, dtype=bool),
        ~np.asarray(target_changed_flags, dtype=bool),
        source_features.valid,
        target_features.valid,
        fallback_threshold=float(matching["fallback_minimum_cosine"]),
        control_iou=float(matching["calibration_control_iou"]),
        negative_iou=float(matching["calibration_negative_iou"]),
        maximum_negative_acceptance=float(
            matching["calibration_maximum_negative_acceptance"]
        ),
        minimum_positive_acceptance=float(
            matching["calibration_minimum_positive_acceptance"]
        ),
        minimum_positive_count=int(matching["calibration_minimum_positive_count"]),
        minimum_negative_count=int(matching["calibration_minimum_negative_count"]),
    )
    classification = classify_identity_location(
        source_visible,
        target_visible,
        source_features,
        target_features,
        source_changed_flags,
        target_changed_flags,
        calibration,
        sam3_features_cfg,
        spatial_iou=spatial_iou,
    )
    appearance = AppearanceFeatures(
        source_map=source_map,
        target_map=target_map,
        source_features=source_features,
        target_features=target_features,
        calibration=calibration,
        match_records=classification.match_records,
    )
    historical_calibration = diagnostics.get("calibration")
    historical_matches = decisions.get("matches")
    return appearance, {
        "dense_maps_reused": True,
        "decision_policy": "current_cpu_recomputed",
        "fallback_reason": None,
        "source_shape": list(source_map.shape),
        "target_shape": list(target_map.shape),
        "source_dtype": str(source_map.dtype),
        "target_dtype": str(target_map.dtype),
        "historical_decision_audit": {
            "calibration_equal": historical_calibration == asdict(calibration),
            "match_records_equal": historical_matches == classification.match_records,
        },
    }


def run_pair(
    image0_path: str | Path,
    image1_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    *,
    pair_id: str | None = None,
    cache: ChangesimWeekendCache | None = None,
    save_stage_artifacts: bool = False,
) -> InferenceResult:
    """Run the full method on one image pair and return the change prediction.

    ``image0_path`` is the "before"/source photo, ``image1_path`` the
    "after"/target photo of the same place; the returned labels are aligned
    to ``image1``'s pixel grid. ``config`` is the merged pipeline config
    (see ``ocmask.config.load_config('configs/pipeline.yaml')``).

    ``pair_id``/``cache`` are optional and only meaningful for ChangeSim
    pairs: if both are given, stages 1-4 look up a validated, pre-computed
    artifact directory via ``cache.lookup(pair_id, image0, image1)`` (see
    ``ocmask.weekend_cache``) and skip recomputation wherever one exists,
    falling back to full computation stage-by-stage otherwise. Omitting
    either (the default) computes every stage fresh, exactly as before this
    parameter existed.

    ``save_stage_artifacts``, when true, additionally writes every stage's
    intermediate evidence (proposals, descriptors, similarity matrices,
    accept/reject reasons, before/after label maps -- see
    ``ocmask.artifact_capture``) to ``output_dir`` for later ablation
    studies and failure analysis. This only adds file writes: it never
    changes what any stage computes, so ``labels``/``guarded_labels``/
    ``base_labels`` are identical whether or not it is set. Defaults to
    false, matching every caller's behavior before this parameter existed.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    numerical_states: dict[str, dict[str, Any]] = {}

    def record_numerical_state(stage: str) -> None:
        """Retain a compact audit trail for process-global torch switches."""

        numerical_states[stage] = capture_torch_numerical_state().to_dict()

    record_numerical_state("00_pair_start")
    reconstruction_cfg = config["reconstruction"]
    cache_requested = cache is not None and pair_id is not None
    cache_hit = (
        cache.lookup(pair_id, image0_path, image1_path)
        if cache_requested
        else StageCacheDirs()
    )
    cache_usage: dict[str, Any] = {
        "requested": cache_requested,
        "lookup_stop_reason": cache_hit.stop_reason,
        "stages": {
            f"stage{number}": {
                "offered": getattr(cache_hit, f"stage{number}_dir") is not None,
                "used": False,
                "path": (
                    str(getattr(cache_hit, f"stage{number}_dir").resolve())
                    if getattr(cache_hit, f"stage{number}_dir") is not None
                    else None
                ),
            }
            for number in range(1, 5)
        },
    }

    # --- Stage 1: MASt3R + SAM2 reconstruction/rendering baseline ----------
    started = time.perf_counter()
    if cache_hit.stage1_dir is not None:
        artifact_dir = cache_hit.stage1_dir
        cache_usage["stages"]["stage1"]["used"] = True
    else:
        stage1 = PairwisePipeline(
            reconstruction_cfg, Mast3rAdapter(reconstruction_cfg), Sam2Adapter(reconstruction_cfg)
        ).run(image0_path, image1_path, output_dir, artifact_level="cache")
        artifact_dir = stage1.artifacts_dir
    inputs = load_cached_inputs(artifact_dir, direction="forward")
    reconstruction = load_reconstruction(artifact_dir / "reconstruction.npz")
    timings["01_reconstruction"] = time.perf_counter() - started
    # Importing the MASt3R/CroCo stack in a fresh fused run changes CUDA matmul
    # from false/highest to true/high. A stage-1 cache hit skips that import.
    # Establish the same explicit boundary state on both paths so cache reuse
    # cannot change stages 2-11, while MASt3R itself still starts under the
    # clean pair-process policy above. PairwisePipeline also establishes this
    # boundary before its legacy SAM2 baseline, covering partial stage-1
    # retries whose reconstruction.npz already exists.
    apply_post_reconstruction_numerics()
    record_numerical_state("01_post_reconstruction_policy")

    # --- Stage 2: SAM3 automatic proposals over the aligned pair -----------
    # SAM3.1's own tracking output (also computed by the original research
    # script at this stage) is not consumed anywhere downstream -- every
    # later stage re-tracks these proposals with SAM2 instead (stage 3) --
    # so it is deliberately not run here.
    started = time.perf_counter()
    sam3_proposals_cfg = config["sam3_proposals"]
    if cache_hit.stage2_dir is not None:
        proposal_cache_dir = cache_hit.stage2_dir / "proposal_cache"
        source_proposals_raw = proposals_to_objects(load_proposal_cache(proposal_cache_dir / "source.npz"))
        target_proposals_raw = proposals_to_objects(load_proposal_cache(proposal_cache_dir / "target.npz"))
        cache_usage["stages"]["stage2"]["used"] = True
    else:
        generator = Sam3AutomaticMaskGenerator(
            sam3_proposals_cfg["sam3_image_checkpoint"],
            source=sam3_proposals_cfg["sam3_source"],
            **_proposal_generator_kwargs(sam3_proposals_cfg["proposals"]),
        )
        try:
            source_generated = generator.generate(inputs.source_render)
            target_generated = generator.generate(inputs.target_image)
            source_proposals_raw = proposals_to_objects(source_generated)
            target_proposals_raw = proposals_to_objects(target_generated)
        finally:
            generator.release()
        if save_stage_artifacts:
            artifact_capture.save_stage2_proposals(
                output_dir / "02_sam3_proposals",
                source_generated,
                target_generated,
                tuple(inputs.source_render.shape[:2]),
                tuple(inputs.target_image.shape[:2]),
            )
    timings["02_sam3_proposals"] = time.perf_counter() - started
    record_numerical_state("02_sam3_proposals_released")

    # --- Stage 3: SAM2 re-tracking of stage 2's proposals -------------------
    # Stage 3 gets its own tracker lifetime. In particular, do not keep its
    # multi-GB SAM2 models resident while stage 4 loads SAM3: a cache hit used
    # to avoid that overlap while a miss incurred it, making the two paths
    # behaviorally and operationally different.
    started = time.perf_counter()
    if cache_hit.stage3_dir is not None:
        stage3_dir = cache_hit.stage3_dir
        baseline_labels = np.asarray(Image.open(stage3_dir / "labels.png"))
        tracking_attempts_preview = json.loads(
            (stage3_dir / "tracking_attempts.json").read_text(encoding="utf-8")
        )
        stage3_diagnostics = {
            **json.loads(
                (stage3_dir / "diagnostics.json").read_text(encoding="utf-8")
            ),
            "source_changed_proposal_ids": changed_proposal_ids_from_tracking_attempts(
                tracking_attempts_preview, "source_to_clean"
            ),
            "target_changed_proposal_ids": changed_proposal_ids_from_tracking_attempts(
                tracking_attempts_preview, "target_to_clean"
            ),
            "cache_reused": True,
        }
        cache_usage["stages"]["stage3"]["used"] = True
    else:
        stage3_dir = output_dir / "03_tracking"
        stage3_tracker = Sam2MaskTracker(reconstruction_cfg)
        try:
            baseline_labels, stage3_diagnostics = run_cached_pair(
                artifact_dir,
                stage3_dir,
                stage3_tracker,
                source_proposals_raw,
                target_proposals_raw,
            )
            stage3_diagnostics["cache_reused"] = False
        finally:
            stage3_tracker.release()
    timings["03_tracking"] = time.perf_counter() - started
    record_numerical_state("03_sam2_tracker_released")

    tracking_cfg = reconstruction_cfg["tracking"]
    # Rebuild the exact proposal lists stage 3 tracked. The changed-ID sets
    # are meaningful only against these identical visibility settings.
    source_visible = filter_visible(
        source_proposals_raw,
        inputs.cross_coverage,
        tracking_cfg["visibility_alpha"],
        tracking_cfg["minimum_mask_area"],
    )
    target_visible = filter_visible(
        target_proposals_raw,
        inputs.cross_coverage,
        tracking_cfg["visibility_alpha"],
        tracking_cfg["minimum_mask_area"],
    )
    source_changed_ids = set(stage3_diagnostics["source_changed_proposal_ids"])
    target_changed_ids = set(stage3_diagnostics["target_changed_proposal_ids"])
    source_changed_flags = np.asarray(
        [
            int(obj.metadata["automatic_proposal_id"]) in source_changed_ids
            for obj in source_visible
        ],
        dtype=bool,
    )
    target_changed_flags = np.asarray(
        [
            int(obj.metadata["automatic_proposal_id"]) in target_changed_ids
            for obj in target_visible
        ],
        dtype=bool,
    )
    if save_stage_artifacts:
        artifact_capture.save_visibility_and_gate(
            output_dir / "02_sam3_proposals",
            source_proposals_raw,
            target_proposals_raw,
            source_visible,
            target_visible,
            source_changed_ids,
            target_changed_ids,
        )

    # --- Stage 4: dense SAM3 appearance features + identity calibration ----
    started = time.perf_counter()
    sam3_features_cfg = config["sam3_features"]
    appearance = None
    stage4_cache_diagnostics: dict[str, Any] = {
        "dense_maps_reused": False,
        "decision_policy": "current_cpu_recomputed",
        "fallback_reason": "stage4_not_offered",
    }
    if cache_hit.stage4_dir is not None:
        appearance, stage4_cache_diagnostics = _appearance_features_from_cache(
            cache_hit.stage4_dir,
            source_visible,
            target_visible,
            source_changed_flags,
            target_changed_flags,
            sam3_features_cfg,
        )
    if appearance is None:
        feature_extractor = Sam3FeatureExtractor(
            sam3_features_cfg["sam3"]["source"],
            sam3_features_cfg["sam3"]["checkpoint"],
        )
        try:
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
        finally:
            feature_extractor.release()
        stage4_cache_diagnostics["backbone_recomputed"] = True
        if save_stage_artifacts:
            artifact_capture.save_stage4_appearance(
                output_dir / "04_sam3_features",
                appearance,
                source_visible,
                target_visible,
                source_changed_flags,
                target_changed_flags,
            )
    else:
        cache_usage["stages"]["stage4"]["used"] = True
        stage4_cache_diagnostics["backbone_recomputed"] = False
    cache_usage["stages"]["stage4"].update(stage4_cache_diagnostics)
    timings["04_sam3_features"] = time.perf_counter() - started
    record_numerical_state("04_sam3_features_released")

    # --- Stage 5: dense DINOv2 appearance features --------------------------
    started = time.perf_counter()
    dinov2_extractor = Dinov2FeatureExtractor(config["dinov2_features"])
    try:
        source_dino_map = dinov2_extractor.feature_map(inputs.source_render)
        target_dino_map = dinov2_extractor.feature_map(inputs.target_image)
    finally:
        dinov2_extractor.release()
    if save_stage_artifacts:
        artifact_capture.save_dense_feature_map(
            output_dir / "05_dinov2_features" / "dinov2_features.npz",
            source_dino_map,
            target_dino_map,
        )
    timings["05_dinov2_features"] = time.perf_counter() - started
    record_numerical_state("05_dinov2_features_released")

    # --- Stages 6-8: moved tracking and guarded evidence -------------------
    # These stages intentionally share one SAM2 tracker, then release it
    # before stage 9 loads the independent SAM3 sentinel model.
    source_changed_objects = [
        obj for obj, changed in zip(source_visible, source_changed_flags) if changed
    ]
    target_changed_objects = [
        obj for obj, changed in zip(target_visible, target_changed_flags) if changed
    ]
    evidence_tracker = Sam2MaskTracker(reconstruction_cfg)
    try:
        started = time.perf_counter()
        forward_attempts = evidence_tracker.track(
            [obj.mask for obj in source_changed_objects],
            inputs.source_render,
            inputs.target_image,
        )
        reverse_attempts = evidence_tracker.track(
            [obj.mask for obj in target_changed_objects],
            inputs.target_image,
            inputs.source_render,
        )
        forward_tracks_by_id = _accepted_tracks_by_proposal_id(
            source_changed_objects, forward_attempts
        )
        reverse_tracks_by_id = _accepted_tracks_by_proposal_id(
            target_changed_objects, reverse_attempts
        )
        forward_track_masks = [
            np.asarray(attempt.mask, bool) if attempt.accepted else None
            for attempt in forward_attempts
        ]
        forward_track_rows = [
            {"rejection_reasons": list(attempt.rejection_reasons)}
            for attempt in forward_attempts
        ]
        timings["06_moved_candidate_tracking"] = time.perf_counter() - started
        if save_stage_artifacts:
            artifact_capture.save_moved_candidate_tracking(
                output_dir / "06_moved_candidate_tracking",
                source_changed_objects,
                forward_attempts,
                target_changed_objects,
                reverse_attempts,
            )

        started = time.perf_counter()
        evidence_cfg = config["motion_and_replacement_evidence"]
        refined_labels, hybrid_diagnostics = refine_with_motion_and_replacement_evidence(
            baseline_labels,
            appearance.match_records,
            [obj.mask for obj in source_proposals_raw],
            [obj.mask for obj in target_proposals_raw],
            forward_tracks_by_id,
            reverse_tracks_by_id,
            minimum_track_candidate_iou=float(
                evidence_cfg["moved_verification"]["minimum_track_candidate_iou"]
            ),
        )
        timings["07_evidence_fusion"] = time.perf_counter() - started
        if save_stage_artifacts:
            artifact_capture.save_evidence_fusion(
                output_dir / "07_evidence_fusion", hybrid_diagnostics, refined_labels
            )

        started = time.perf_counter()
        direct_labels, feature_veto_decision = apply_feature_veto_direct_replacement(
            source_visible,
            target_visible,
            appearance.source_features,
            appearance.target_features,
            ~source_changed_flags,
            ~target_changed_flags,
            refined_labels,
            inputs.cross_coverage,
            evidence_tracker,
            inputs.source_render,
            inputs.target_image,
            same_threshold=appearance.identity_threshold,
            config=config["feature_veto"],
        )
        timings["08_feature_veto_gate"] = time.perf_counter() - started
        if save_stage_artifacts:
            artifact_capture.save_feature_veto(
                output_dir / "08_feature_veto_gate",
                feature_veto_decision,
                refined_labels,
                direct_labels,
            )
    finally:
        evidence_tracker.release()
    record_numerical_state("08_sam2_tracker_released")

    # --- Stage 9: fresh SAM3 proposals/features over real source image ------
    started = time.perf_counter()
    sentinel_cfg = config["obvious_object_sentinel"]
    sentinel_generator = Sam3AutomaticMaskGenerator(
        sentinel_cfg["sam3"]["checkpoint"],
        source=sentinel_cfg["sam3"]["source"],
        **_proposal_generator_kwargs(
            sentinel_cfg["sam3"]["proposal_generation"],
            minimum_mask_area_key="minimum_mask_area_pixels",
        ),
    )
    try:
        real_source_proposals, real_source_map = (
            sentinel_generator.generate_with_feature_map(reconstruction.images[0])
        )
    finally:
        sentinel_generator.release()
    real_source_objects = proposals_to_objects(real_source_proposals)
    if save_stage_artifacts:
        artifact_capture.save_real_source_sentinel(
            output_dir / "09_real_source_sentinel",
            real_source_proposals,
            real_source_map,
            tuple(reconstruction.images[0].shape[:2]),
        )
    timings["09_real_source_sentinel"] = time.perf_counter() - started
    record_numerical_state("09_sam3_sentinel_released")

    # --- Stage 10: real-image association resolver -------------------------
    # A new tracker avoids retaining stages 6-8's SAM2 model across SAM3.
    started = time.perf_counter()
    resolver_tracker = Sam2MaskTracker(reconstruction_cfg)
    try:
        resolver_settings = resolver_settings_from_config(config)
        base_labels, resolver_diagnostics = resolve_real_image_associations(
            reconstruction,
            real_source_objects,
            target_visible,
            real_source_map,
            appearance.target_map,
            direct_labels,
            appearance.identity_threshold,
            resolver_tracker,
            reconstruction.images[0],
            reconstruction.images[1],
            settings=resolver_settings,
        )
    finally:
        resolver_tracker.release()
    timings["10_association_resolution"] = time.perf_counter() - started
    if save_stage_artifacts:
        artifact_capture.save_association_resolution(
            output_dir / "10_association_resolution", resolver_diagnostics, base_labels
        )
    record_numerical_state("10_sam2_tracker_released")

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
    save_image(output_dir / "labels_base.png", base_labels)
    save_image(output_dir / "labels_color.png", colorize(full_labels))
    save_image(output_dir / "overlay.png", overlay(native_target, full_labels))
    save_image(output_dir / "target.png", native_target)
    record_numerical_state("11_complete")
    diagnostics = {
        "cache": cache_usage,
        "stage3": stage3_diagnostics,
        "stage4_cache": stage4_cache_diagnostics,
        "stage11": stage11_diagnostics,
        "timings": timings,
        "numerical_states": numerical_states,
        "save_stage_artifacts": save_stage_artifacts,
    }
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
