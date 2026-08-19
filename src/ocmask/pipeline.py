from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .adapters.base import ReconstructionAdapter, SegmentationAdapter
from .cache import load_reconstruction, save_reconstruction
from .geometry import (
    canonical_cloud,
    confidence_gated_depth_filter,
    render_points,
    reverse_depth_filter,
)
from .geometry_artifacts import save_extended_geometry_artifacts
from .feature_consistency import (
    apply_aligned_sam2_feature_gate,
    apply_conditional_sam2_feature_gate,
)
from .io import load_rgb, pair_key, save_image, save_json
from .masks import (
    annotate_track_support,
    compose_labels,
    filter_visible,
    mark_replacements,
    reject_inconsistent_tracks,
    same_place,
)
from .ssim import colorize_ssim, compute_ssim_dissimilarity, heatmap_overlay
from .types import Label, ObjectMask, PairResult
from .visualization import colorize, instance_overlay, overlay


def _area_ratio_bounds(tracking_config: dict) -> tuple[float, float] | None:
    """Read the optional, unpublished target/source area-ratio ablation bound."""
    bounds = tracking_config.get("track_area_ratio_bounds")
    return tuple(bounds) if bounds is not None else None


def _apply_feature_consistency_gate(
    segmentation: SegmentationAdapter,
    source_masks: list[ObjectMask],
    tracks: list[ObjectMask | None],
    source_image: np.ndarray,
    target_image: np.ndarray,
    reliable_coverage: np.ndarray,
    tracking_config: dict,
) -> list[ObjectMask | None]:
    """Apply the optional SAM2 encoder-feature experiment to aligned tracks.

    This helper deliberately accepts the generic segmentation interface.  A
    future adapter can expose its own image features without coupling the
    pairwise pipeline to SAM2 imports.  Normal GOLDILOCS configurations take
    the early return and retain their previous behavior exactly.
    """
    feature = tracking_config.get("feature_consistency", {})
    if not feature.get("enabled", False) or not any(
        track is not None for track in tracks
    ):
        return tracks
    if feature.get("threshold_mode") != "pair_controls":
        raise ValueError(
            "feature_consistency.threshold_mode must be 'pair_controls'"
        )
    if feature.get("low_evidence_policy") != "keep_ambiguous":
        raise ValueError(
            "feature_consistency.low_evidence_policy must be 'keep_ambiguous'"
        )

    source_features = segmentation.image_feature_map(source_image)
    target_features = segmentation.image_feature_map(target_image)
    if source_features is None or target_features is None:
        raise RuntimeError(
            "The selected segmentation adapter does not expose image features"
        )

    mode = feature.get("mode", "dense_completeness")
    if mode == "conditional_prototype":
        return apply_conditional_sam2_feature_gate(
            source_masks,
            tracks,
            source_features,
            target_features,
            reliable_coverage,
            safe_iou=feature["safe_iou"],
            safe_area_ratio_bounds=tuple(feature["safe_area_ratio_bounds"]),
            maximum_negative_acceptance=feature[
                "maximum_negative_acceptance"
            ],
            minimum_positive_acceptance=feature[
                "minimum_positive_acceptance"
            ],
            minimum_calibration_cells=feature["minimum_calibration_cells"],
            minimum_reliable_cells=feature["minimum_reliable_cells"],
            minimum_reliable_source_fraction=feature[
                "minimum_reliable_source_fraction"
            ],
            minimum_reliable_target_fraction=feature[
                "minimum_reliable_target_fraction"
            ],
            low_evidence_policy=feature["low_evidence_policy"],
            negative_offsets=feature["negative_offsets"],
            reliable_cell_threshold=feature["reliable_cell_threshold"],
        )
    if mode != "dense_completeness":
        raise ValueError(
            "feature_consistency.mode must be 'dense_completeness' or "
            "'conditional_prototype'"
        )
    return apply_aligned_sam2_feature_gate(
        source_masks,
        tracks,
        source_features,
        target_features,
        reliable_coverage,
        maximum_negative_acceptance=feature[
            "maximum_negative_acceptance"
        ],
        minimum_positive_acceptance=feature["minimum_positive_acceptance"],
        minimum_calibration_cells=feature["minimum_calibration_cells"],
        minimum_reliable_cells=feature["minimum_reliable_cells"],
        boundary_ignore_cells=feature["boundary_ignore_cells"],
        correspondence_dilation_cells=feature[
            "correspondence_dilation_cells"
        ],
        minimum_reliable_source_fraction=feature[
            "minimum_reliable_source_fraction"
        ],
        minimum_explained_source_fraction=feature[
            "minimum_explained_source_fraction"
        ],
        minimum_target_purity_fraction=feature[
            "minimum_target_purity_fraction"
        ],
        low_evidence_policy=feature["low_evidence_policy"],
        negative_offsets=feature["negative_offsets"],
        reliable_cell_threshold=feature["reliable_cell_threshold"],
        mask_cell_threshold=feature["mask_cell_threshold"],
    )


def _reverse_depth_keep_mask(
    points: np.ndarray,
    point_confidence: np.ndarray,
    target_depth: np.ndarray,
    target_confidence: np.ndarray,
    target_intrinsics: np.ndarray,
    target_world_to_camera: np.ndarray,
    geo: dict,
) -> np.ndarray:
    """Dispatch to the confidence-gated filter only when explicitly enabled.

    Unpublished, opt-in reproduction ablation (see
    geometry.confidence_gated_depth_filter and
    outputs/sam-audit-paper-baseline/AUDIT.md). Default (absent config key)
    reproduces reverse_depth_filter's output exactly.
    """
    if not geo.get("confidence_gating_enabled", False):
        return reverse_depth_filter(
            points,
            target_depth,
            target_intrinsics,
            target_world_to_camera,
            geo["depth_epsilon"],
            geo["minimum_valid_depth"],
        )
    return confidence_gated_depth_filter(
        points,
        point_confidence,
        target_depth,
        target_confidence,
        target_intrinsics,
        target_world_to_camera,
        geo["depth_epsilon"],
        geo["minimum_valid_depth"],
        geo.get("opposing_confidence_threshold"),
        geo.get("own_confidence_threshold"),
    ).keep


def _resize_rgb(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize RGB appearance to the dense pointmap grid."""
    height, width = shape
    return np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS))


def _save_objects(path: Path, objects: list[ObjectMask]) -> None:
    """Store individual masks plus human- and machine-readable provenance."""
    path.mkdir(parents=True, exist_ok=True)
    records = []
    for index, obj in enumerate(objects):
        filename = f"{index:04d}.png"
        save_image(path / filename, obj.mask.astype(np.uint8) * 255)
        records.append(
            {
                "mask": filename,
                "label": obj.label.name.lower(),
                "score": obj.score,
                "source": obj.source,
                "metadata": obj.metadata,
            }
        )
    save_json(path / "objects.json", records)


def _save_sam_generation_debug(
    path: Path,
    name: str,
    image: np.ndarray,
    masks: list[ObjectMask],
) -> dict:
    """Persist one automatic-mask generation call and its numbered proposals."""
    path.mkdir(parents=True, exist_ok=True)
    save_image(path / f"{name}_input.png", image)
    save_image(path / f"{name}_proposals.png", instance_overlay(image, masks))
    return {
        "stage": name,
        "kind": "generate",
        "input": f"{name}_input.png",
        "output": f"{name}_proposals.png",
        "mask_count": len(masks),
        "masks": [
            {
                "id": index,
                "area": int(obj.mask.sum()),
                "score": float(obj.score),
                "stability_score": obj.metadata.get("stability_score"),
                "bbox_xywh": obj.metadata.get("bbox_xywh"),
                "point_coords": obj.metadata.get("point_coords"),
                "crop_box_xywh": obj.metadata.get("crop_box_xywh"),
                "valid_render_support_fraction": obj.metadata.get(
                    "valid_render_support_fraction"
                ),
                "selected_for_tracking": obj.metadata.get(
                    "selected_for_tracking", True
                ),
            }
            for index, obj in enumerate(masks, start=1)
        ],
    }


def _save_sam_tracking_debug(
    path: Path,
    name: str,
    source_image: np.ndarray,
    target_image: np.ndarray,
    source_masks: list[ObjectMask],
    tracked_masks: list[ObjectMask | None],
    attempted_masks: list[ObjectMask | None] | None = None,
) -> dict:
    """Persist accepted results and every raw SAM propagation attempt."""
    path.mkdir(parents=True, exist_ok=True)
    mask_path = path / f"{name}_masks"
    mask_path.mkdir(parents=True, exist_ok=True)
    if attempted_masks is None or len(attempted_masks) != len(tracked_masks):
        # Non-SAM adapters need not implement the optional diagnostic hook.
        attempted_masks = tracked_masks
    statuses = [tracked is not None for tracked in tracked_masks]
    target_objects = [
        attempt
        if attempt is not None
        else ObjectMask(np.zeros(target_image.shape[:2], bool))
        for attempt in attempted_masks
    ]
    save_image(
        path / f"{name}_source.png",
        instance_overlay(source_image, source_masks, statuses=statuses),
    )
    save_image(
        path / f"{name}_target.png",
        instance_overlay(target_image, target_objects, statuses=statuses),
    )
    failed_ids = [
        index
        for index, tracked in enumerate(tracked_masks, start=1)
        if tracked is None
    ]
    failed = [
        obj for obj, tracked in zip(source_masks, tracked_masks) if tracked is None
    ]
    save_image(
        path / f"{name}_failed.png",
        instance_overlay(
            source_image,
            failed,
            statuses=[False] * len(failed),
            instance_ids=failed_ids,
        ),
    )
    track_records = []
    for index, (source, tracked, attempt, success) in enumerate(
        zip(source_masks, tracked_masks, attempted_masks, statuses), start=1
    ):
        source_name = f"source_{index:04d}.png"
        save_image(mask_path / source_name, source.mask.astype(np.uint8) * 255)
        target_name = None
        target_area = 0
        target_nonblack_fraction = None
        target_mean_luminance = None
        if attempt is not None:
            target_name = f"target_{index:04d}.png"
            save_image(mask_path / target_name, attempt.mask.astype(np.uint8) * 255)
            target_area = int(attempt.mask.sum())
            pixels = np.asarray(target_image)[attempt.mask]
            if len(pixels):
                target_nonblack_fraction = float(np.any(pixels > 0, axis=1).mean())
                target_mean_luminance = float(
                    (
                        0.2126 * pixels[:, 0]
                        + 0.7152 * pixels[:, 1]
                        + 0.0722 * pixels[:, 2]
                    ).mean()
                )
        source_area = int(source.mask.sum())
        if attempt is not None:
            intersection = int(np.logical_and(source.mask, attempt.mask).sum())
            union = int(np.logical_or(source.mask, attempt.mask).sum())
            source_retention = intersection / source_area if source_area else 0.0
            target_precision = intersection / target_area if target_area else 0.0
            area_ratio = target_area / source_area if source_area else None
            measured_iou = intersection / union if union else 0.0
            attempt_metadata = attempt.metadata
        else:
            source_retention = None
            target_precision = None
            area_ratio = None
            measured_iou = None
            attempt_metadata = {}
        track_records.append(
            {
                "id": index,
                "status": (
                    "tracked"
                    if success
                    else "rejected"
                    if attempt is not None
                    else "failed"
                ),
                "source_mask": f"{name}_masks/{source_name}",
                "target_mask": (
                    f"{name}_masks/{target_name}" if target_name is not None else None
                ),
                "source_area": source_area,
                "source_proposal_id": source.metadata.get(
                    "automatic_proposal_id", index
                ),
                "target_area": target_area,
                "source_target_iou": measured_iou,
                "source_retention_fraction": source_retention,
                "target_precision_fraction": target_precision,
                "target_source_area_ratio": area_ratio,
                "target_nonblack_fraction": target_nonblack_fraction,
                "target_mean_luminance": target_mean_luminance,
                "object_score_logit": attempt_metadata.get("object_score_logit"),
                "source_proposal_score": attempt_metadata.get(
                    "source_proposal_score", source.score
                ),
                "destination_support_fraction": attempt_metadata.get(
                    "destination_support_fraction"
                ),
                # Dense encoder-feature diagnostics are present only in the
                # explicit SAM2 residual-gate ablation. Keeping every scalar in
                # the stage JSON makes threshold decisions auditable without
                # rerunning either model.
                "sam2_feature_gate_decision": attempt_metadata.get(
                    "sam2_dense_feature_gate_decision"
                ),
                "sam2_feature_threshold": attempt_metadata.get(
                    "sam2_dense_feature_threshold"
                ),
                "sam2_feature_calibration_valid": attempt_metadata.get(
                    "sam2_dense_feature_calibration_valid"
                ),
                "sam2_feature_positive_acceptance": attempt_metadata.get(
                    "sam2_dense_feature_positive_acceptance"
                ),
                "sam2_feature_negative_acceptance": attempt_metadata.get(
                    "sam2_dense_feature_negative_acceptance"
                ),
                "sam2_feature_source_explained_fraction": attempt_metadata.get(
                    "sam2_dense_feature_source_explained_fraction"
                ),
                "sam2_feature_target_purity_fraction": attempt_metadata.get(
                    "sam2_dense_feature_target_purity_fraction"
                ),
                "sam2_feature_source_reliable_fraction": attempt_metadata.get(
                    "sam2_dense_feature_source_reliable_fraction"
                ),
                "sam2_feature_prototype_cosine_similarity": attempt_metadata.get(
                    "sam2_feature_prototype_cosine_similarity"
                ),
                "sam2_conditional_feature_gate_decision": attempt_metadata.get(
                    "sam2_conditional_feature_gate_decision"
                ),
                "sam2_conditional_feature_threshold": attempt_metadata.get(
                    "sam2_conditional_feature_threshold"
                ),
                "sam2_conditional_feature_prototype_cosine": (
                    attempt_metadata.get(
                        "sam2_conditional_feature_prototype_cosine"
                    )
                ),
                "sam2_conditional_feature_iou": attempt_metadata.get(
                    "sam2_conditional_feature_iou"
                ),
                "sam2_conditional_feature_area_ratio": attempt_metadata.get(
                    "sam2_conditional_feature_area_ratio"
                ),
                "sam2_conditional_feature_source_reliable_fraction": (
                    attempt_metadata.get(
                        "sam2_conditional_feature_source_reliable_fraction"
                    )
                ),
                "sam2_conditional_feature_target_reliable_fraction": (
                    attempt_metadata.get(
                        "sam2_conditional_feature_target_reliable_fraction"
                    )
                ),
                "ambiguity_reasons": attempt_metadata.get(
                    "ambiguity_reasons", []
                ),
                "rejection_reasons": attempt_metadata.get("rejection_reasons", []),
            }
        )
    return {
        "stage": name,
        "kind": "track",
        "source": f"{name}_source.png",
        "target": f"{name}_target.png",
        "failed": f"{name}_failed.png",
        "input_count": len(source_masks),
        "tracked_count": sum(statuses),
        "failed_count": statuses.count(False),
        "tracks": track_records,
    }


class PairwisePipeline:
    """Paper-faithful pairwise inference orchestrator with stage caching."""
    def __init__(
        self,
        config: dict,
        reconstruction: ReconstructionAdapter,
        segmentation: SegmentationAdapter,
    ):
        self.config = config
        self.reconstruction_adapter = reconstruction
        self.segmentation_adapter = segmentation

    def run(
        self,
        image0_path: str | Path,
        image1_path: str | Path,
        output_root: str | Path,
        force_reconstruction: bool = False,
        artifact_level: str = "full",
        keep0_override: np.ndarray | None = None,
        keep1_override: np.ndarray | None = None,
        ground_contact_render_fill: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> PairResult:
        """Run one pair, optionally retaining lean reusable model caches.

        ``keep{0,1}_override`` let a caller substitute an already-computed
        ``keep0``/``keep1`` array (e.g. from
        ``ground_contact.exclude_from_keep``) for this run's own
        ``reverse_depth_filter`` output, so a ground-contact-corrected
        detection cloud actually feeds ``canonical_cloud`` and every SAM3/
        SAM2 stage downstream of it -- not just a side artifact nobody
        reads. Omitted (``None``, the default) reproduces today's behavior
        exactly. ``ground_contact_render_fill``, if given, is a
        ``(points, colors)`` pair of synthetic floor-surface points (see
        ``ground_contact.render_fill_for_mask``) appended *only* to the
        clean-plate renders saved by ``save_extended_geometry_artifacts``
        (``render_clean_to_0/1.png``) -- never to the detection cloud that
        feeds tracking, per ``ground_contact``'s module docstring on why
        those two paths must stay separate.
        """
        if artifact_level not in {"metrics", "minimal", "cache", "full"}:
            raise ValueError(
                "artifact_level must be 'metrics', 'minimal', 'cache', or 'full'"
            )
        save_debug = artifact_level == "full"
        # ``cache`` retains exactly the geometry products required by the A3
        # experiment chain, without the numerous PNG/PLY debug visualizations.
        save_cache = artifact_level in {"cache", "full"}
        save_files = artifact_level != "metrics"
        timings: dict[str, float] = {}
        size = (self.config["image"]["width"], self.config["image"]["height"])
        original0, original1 = load_rgb(image0_path, size), load_rgb(image1_path, size)
        key = pair_key(image0_path, image1_path, self.config)
        artifacts = Path(output_root) / key
        if save_files:
            artifacts.mkdir(parents=True, exist_ok=True)
            save_json(artifacts / "config.json", self.config)
            save_json(
                artifacts / "inputs.json",
                {"image0": str(Path(image0_path).resolve()), "image1": str(Path(image1_path).resolve())},
            )

        # MASt3R is the most expensive stage. Its cache key already incorporates
        # both input files and the complete reproduction configuration.
        reconstruction_path = artifacts / "reconstruction.npz"
        start = time.perf_counter()
        if reconstruction_path.exists() and not force_reconstruction:
            reconstruction = load_reconstruction(reconstruction_path)
            reconstruction_cache_hit = True
        else:
            reconstruction = self.reconstruction_adapter.reconstruct(original0, original1)
            if save_cache:
                save_reconstruction(reconstruction_path, reconstruction)
            reconstruction_cache_hit = False
        self.reconstruction_adapter.release()
        timings["reconstruction"] = time.perf_counter() - start

        start = time.perf_counter()
        # MASt3R may operate below the native input resolution. All geometric
        # processing stays on its calibrated grid; labels are resized only once
        # at the end with nearest-neighbor interpolation.
        points0, points1 = reconstruction.points
        height, width = points0.shape[:2]
        image0 = _resize_rgb(reconstruction.images[0], (height, width))
        image1 = _resize_rgb(reconstruction.images[1], (height, width))
        geo = self.config["geometry"]
        # A point is removed when it projects in front of the surface observed
        # at the other time. Such impossible front geometry is evidence of an
        # occluder or an object that changed.
        if keep0_override is not None:
            keep0 = np.asarray(keep0_override, dtype=bool)
        else:
            keep0 = _reverse_depth_keep_mask(
                points0,
                reconstruction.confidence[0],
                reconstruction.depths[1],
                reconstruction.confidence[1],
                reconstruction.intrinsics[1],
                reconstruction.world_to_camera[1],
                geo,
            )
        if keep1_override is not None:
            keep1 = np.asarray(keep1_override, dtype=bool)
        else:
            keep1 = _reverse_depth_keep_mask(
                points1,
                reconstruction.confidence[1],
                reconstruction.depths[0],
                reconstruction.confidence[0],
                reconstruction.intrinsics[0],
                reconstruction.world_to_camera[0],
                geo,
            )
        # The union of non-conflicting surfaces is the paper's canonical static
        # reconstruction P*. R0,1 contains T0 appearance viewed from camera 1;
        # clean1 contains only mutually consistent geometry in that same view.
        clean_points, clean_colors = canonical_cloud(
            points0, image0, keep0, points1, image1, keep1
        )
        render01, _, coverage01 = render_points(
            points0,
            image0,
            reconstruction.intrinsics[1],
            reconstruction.world_to_camera[1],
            (height, width),
            z_epsilon=geo["z_buffer_epsilon"],
            splat_radius=geo.get("splat_radius", 0),
            fill_holes=geo.get("hole_fill_enabled", False),
            hole_fill_min_neighbors=geo.get("hole_fill_min_neighbors", 5),
            hole_fill_max_relative_depth=geo.get(
                "hole_fill_max_relative_depth", 0.02
            ),
        )
        clean1, _, clean_coverage = render_points(
            clean_points,
            clean_colors,
            reconstruction.intrinsics[1],
            reconstruction.world_to_camera[1],
            (height, width),
            z_epsilon=geo["z_buffer_epsilon"],
        )
        # Feature controls and the paper's final visibility checks both need
        # pixels supported by the source cross-render and the clean render.
        reliable_coverage = coverage01 & clean_coverage
        timings["geometry_and_rendering"] = time.perf_counter() - start
        if save_cache:
            np.savez_compressed(
                artifacts / "geometry.npz",
                keep0=keep0,
                keep1=keep1,
                coverage01=coverage01,
                clean_coverage=clean_coverage,
            )
        # Persist both temporal pointmaps, each cleaned component, and all four
        # cross/canonical renders requested by the paper walkthrough.
        if save_debug:
            save_extended_geometry_artifacts(
                artifacts,
                reconstruction,
                keep0,
                keep1,
                geo,
                extra_render_points=(
                    ground_contact_render_fill[0]
                    if ground_contact_render_fill is not None
                    else None
                ),
                extra_render_colors=(
                    ground_contact_render_fill[1]
                    if ground_contact_render_fill is not None
                    else None
                ),
            )
        elif save_cache:
            # Downstream SAM3/SAM2 experiments consume only these two renders.
            # Saving them directly avoids point-cloud dumps and walkthrough art.
            save_image(artifacts / "render_0_to_1.png", render01)
            save_image(artifacts / "render_clean_to_1.png", clean1)

        start = time.perf_counter()
        sam_debug_path = artifacts / "sam_debug"
        sam_debug_records: list[dict] = []
        # Source-side pass: masks that disappear in the clean reconstruction
        # are candidates for removal or movement.
        source_proposals = self.segmentation_adapter.generate(render01)
        for proposal_id, obj in enumerate(source_proposals, start=1):
            obj.metadata["automatic_proposal_id"] = proposal_id
        # Appendix A.6 removes these masks after classification. Applying the
        # same alpha rule before independent SAM2 propagation avoids VOS work
        # for proposals that can never appear in the final prediction.
        source_masks = filter_visible(
            source_proposals,
            coverage01,
            self.config["tracking"]["visibility_alpha"],
            self.config["tracking"]["minimum_mask_area"],
        )
        selected_source_ids = {id(obj) for obj in source_masks}
        for obj in source_proposals:
            obj.metadata["selected_for_tracking"] = id(obj) in selected_source_ids
        if save_debug:
            sam_debug_records.append(
                _save_sam_generation_debug(
                    sam_debug_path,
                    "01_source_generate",
                    render01,
                    source_proposals,
                )
            )
        source_to_clean = self.segmentation_adapter.track(source_masks, render01, clean1)
        source_to_clean_attempts = self.segmentation_adapter.last_track_attempts()
        if len(source_to_clean_attempts) != len(source_to_clean):
            source_to_clean_attempts = source_to_clean
        # Clean-render support is useful diagnostic context, but an unsupported
        # destination is unknown evidence rather than proof of change.
        annotate_track_support(source_to_clean_attempts, clean_coverage)
        source_to_clean = reject_inconsistent_tracks(
            source_masks,
            source_to_clean,
            self.config["tracking"]["minimum_track_iou"],
            area_ratio_bounds=_area_ratio_bounds(self.config["tracking"]),
        )
        # Encoder features verify that an accepted aligned track explains the
        # whole source object rather than only a rasterized residual. This
        # unpublished experiment is intentionally absent from motion-search
        # stages, where a spatial displacement is the desired outcome.
        source_to_clean = _apply_feature_consistency_gate(
            self.segmentation_adapter,
            source_masks,
            source_to_clean,
            render01,
            clean1,
            reliable_coverage,
            self.config["tracking"],
        )
        if save_debug:
            sam_debug_records.append(
                _save_sam_tracking_debug(
                    sam_debug_path,
                    "02_source_to_clean",
                    render01,
                    clean1,
                    source_masks,
                    source_to_clean,
                    source_to_clean_attempts,
                )
            )
        source_changed = [mask for mask, tracked in zip(source_masks, source_to_clean) if tracked is None]
        source_to_target = self.segmentation_adapter.track(source_changed, render01, image1)
        source_to_target_attempts = self.segmentation_adapter.last_track_attempts()
        if len(source_to_target_attempts) != len(source_to_target):
            source_to_target_attempts = source_to_target
        # Do not apply aligned IoU here: this call deliberately searches I1 for
        # an object that may have moved to a different pixel location.
        if save_debug:
            sam_debug_records.append(
                _save_sam_tracking_debug(
                    sam_debug_path,
                    "03_source_changed_to_target",
                    render01,
                    image1,
                    source_changed,
                    source_to_target,
                    source_to_target_attempts,
                )
            )
        same_place_cfg = self.config.get("same_place_pairing", {})
        same_place_min_iou = same_place_cfg.get("minimum_spatial_iou", 0.30)
        same_place_max_centroid = same_place_cfg.get(
            "maximum_normalized_centroid_distance", 0.10
        )

        moved_from_source: list[ObjectMask] = []
        removed: list[ObjectMask] = []
        # A changed source mask found again in I1 moved; otherwise it was removed.
        # render01 and image1 share one pixel grid, so `source` and `tracked`
        # are directly comparable: if SAM2 found the object again in
        # essentially the same place, the clean-gate rejection was wrong and
        # nothing actually moved -- drop it rather than assume MOVED.
        for source, tracked in zip(source_changed, source_to_target):
            if tracked is None:
                source.label = Label.REMOVED
                removed.append(source)
            elif same_place(
                source.mask, tracked.mask, same_place_min_iou, same_place_max_centroid
            ):
                continue
            else:
                tracked.label = Label.MOVED
                tracked.source = "source_changed_to_target"
                moved_from_source.append(tracked)

        # Symmetric target-side pass: masks absent from P* are additions unless
        # they can be tracked back to the T0 cross-view rendering.
        target_proposals = self.segmentation_adapter.generate(image1)
        for proposal_id, obj in enumerate(target_proposals, start=1):
            obj.metadata["automatic_proposal_id"] = proposal_id
        target_masks = filter_visible(
            target_proposals,
            coverage01,
            self.config["tracking"]["visibility_alpha"],
            self.config["tracking"]["minimum_mask_area"],
        )
        selected_target_ids = {id(obj) for obj in target_masks}
        for obj in target_proposals:
            obj.metadata["selected_for_tracking"] = id(obj) in selected_target_ids
        if save_debug:
            sam_debug_records.append(
                _save_sam_generation_debug(
                    sam_debug_path,
                    "04_target_generate",
                    image1,
                    target_proposals,
                )
            )
        target_to_clean = self.segmentation_adapter.track(target_masks, image1, clean1)
        target_to_clean_attempts = self.segmentation_adapter.last_track_attempts()
        if len(target_to_clean_attempts) != len(target_to_clean):
            target_to_clean_attempts = target_to_clean
        annotate_track_support(target_to_clean_attempts, clean_coverage)
        target_to_clean = reject_inconsistent_tracks(
            target_masks,
            target_to_clean,
            self.config["tracking"]["minimum_track_iou"],
            area_ratio_bounds=_area_ratio_bounds(self.config["tracking"]),
        )
        target_to_clean = _apply_feature_consistency_gate(
            self.segmentation_adapter,
            target_masks,
            target_to_clean,
            image1,
            clean1,
            reliable_coverage,
            self.config["tracking"],
        )
        if save_debug:
            sam_debug_records.append(
                _save_sam_tracking_debug(
                    sam_debug_path,
                    "05_target_to_clean",
                    image1,
                    clean1,
                    target_masks,
                    target_to_clean,
                    target_to_clean_attempts,
                )
            )
        target_changed = [mask for mask, tracked in zip(target_masks, target_to_clean) if tracked is None]
        target_to_source = self.segmentation_adapter.track(target_changed, image1, render01)
        target_to_source_attempts = self.segmentation_adapter.last_track_attempts()
        if len(target_to_source_attempts) != len(target_to_source):
            target_to_source_attempts = target_to_source
        annotate_track_support(target_to_source_attempts, coverage01)
        # As above, a spatially displaced target is the expected evidence for
        # movement, so source/target IoU must not gate this propagation.
        if save_debug:
            sam_debug_records.append(
                _save_sam_tracking_debug(
                    sam_debug_path,
                    "06_target_changed_to_source",
                    image1,
                    render01,
                    target_changed,
                    target_to_source,
                    target_to_source_attempts,
                )
            )
        moved: list[ObjectMask] = []
        added: list[ObjectMask] = []
        # image1 and render01 share one pixel grid, so the same same-place
        # check applies symmetrically here.
        for target, tracked in zip(target_changed, target_to_source):
            if tracked is None:
                target.label = Label.ADDED
                added.append(target)
            elif same_place(
                target.mask, tracked.mask, same_place_min_iou, same_place_max_centroid
            ):
                continue
            else:
                target.label = Label.MOVED
                target.source = "target_changed_to_source"
                moved.append(target)
        # Both branches end in I1 coordinates: the source branch contributes
        # its propagated target mask, while the target branch contributes the
        # original I1 proposal. Keep their union as in the paper's symmetric
        # moved sets. Overlap composition later makes duplicate pixels benign.
        moved.extend(moved_from_source)
        timings["segmentation_and_tracking"] = time.perf_counter() - start
        if save_debug:
            save_json(sam_debug_path / "index.json", sam_debug_records)

        tracking = self.config["tracking"]
        # Visibility filtering prevents holes caused by parallax, occlusion, or
        # limited camera overlap from being reported as semantic changes.
        removed = filter_visible(removed, coverage01, tracking["visibility_alpha"], tracking["minimum_mask_area"])
        added = filter_visible(added, coverage01, tracking["visibility_alpha"], tracking["minimum_mask_area"])
        moved = filter_visible(moved, coverage01, tracking["visibility_alpha"], tracking["minimum_mask_area"])

        start = time.perf_counter()
        ssim_cfg = self.config["ssim"]
        warped: list[ObjectMask] = []
        if ssim_cfg.get("enabled", True):
            # SSIM compares viewpoint-aligned T0 appearance against I1. It
            # returns a per-pixel map whose values are averaged per object.
            dissimilarity = compute_ssim_dissimilarity(render01, image1, ssim_cfg)
            changed_union = np.zeros((height, width), dtype=bool)
            for obj in added + moved:
                changed_union |= obj.mask
            # Warp detection applies only to target objects not already
            # explained by a rigid addition or movement and supported by both
            # rendered views.
            static_masks = [
                obj for obj in target_masks
                if not np.logical_and(obj.mask, changed_union).any()
                and obj.mask.sum() >= tracking["minimum_mask_area"]
                and np.logical_and(obj.mask, reliable_coverage).sum()
                / obj.mask.sum()
                >= tracking["visibility_alpha"]
            ]
            scores = np.array(
                [float(dissimilarity[obj.mask].mean()) for obj in static_masks]
            )
            if len(scores):
                # The paper's statistical rule is mean plus one standard
                # deviation across eligible object scores.
                threshold = float(
                    scores.mean()
                    + ssim_cfg["threshold_stddevs"] * scores.std()
                )
                for obj, score in zip(static_masks, scores):
                    if score > threshold:
                        obj.label = Label.WARPED
                        obj.source = "ssim"
                        obj.metadata["ssim_dissimilarity"] = score
                        warped.append(obj)
            else:
                threshold = float("nan")
        else:
            # ChangeSim has no warped class. Avoid computing an unreliable SSIM
            # map and guarantee that this branch contributes no predictions.
            dissimilarity = np.zeros((height, width), dtype=np.float32)
            threshold = float("nan")
        timings["ssim"] = time.perf_counter() - start

        objects = added + removed + moved + warped
        # Rasterize with the fixed paper priority, then adapt overlapping
        # addition/removal regions to ChangeSim's extra "replaced" category.
        labels = compose_labels((height, width), objects)
        labels = mark_replacements(labels, added, removed, tracking["replacement_overlap_iou"])
        binary = (labels != Label.UNCHANGED).astype(np.uint8)
        labels_native = np.asarray(
            Image.fromarray(labels).resize(size, Image.Resampling.NEAREST), dtype=np.uint8
        )
        binary_native = (labels_native != Label.UNCHANGED).astype(np.uint8)
        if save_files:
            save_image(artifacts / "labels.png", labels_native)
        if save_debug:
            save_image(artifacts / "binary.png", binary_native * 255)
            save_image(artifacts / "labels_color.png", colorize(labels_native))
            save_image(artifacts / "overlay.png", overlay(original1, labels_native))
            # Preserve exact floating-point values and a readable heatmap.
            np.save(artifacts / "ssim_dissimilarity.npy", dissimilarity)
            ssim_heatmap = colorize_ssim(dissimilarity)
            save_image(artifacts / "ssim_heatmap.png", ssim_heatmap)
            save_image(
                artifacts / "ssim_heatmap_overlay.png",
                heatmap_overlay(image1, ssim_heatmap),
            )
            _save_objects(artifacts / "objects", objects)
        if save_files:
            save_json(
                artifacts / "metadata.json",
                {
                    "cache": {"reconstruction_hit": reconstruction_cache_hit},
                    "match_count": reconstruction.match_count,
                    "ssim_threshold": None if np.isnan(threshold) else threshold,
                    "object_counts": {
                        label.name.lower(): sum(obj.label == label for obj in objects)
                        for label in Label
                    },
                    "timings_seconds": timings,
                },
            )
        return PairResult(labels_native, binary_native, objects, artifacts, timings)
