"""SAM3 feature identity + aligned-location change-classification experiment.

This module is deliberately isolated from the production pairwise pipeline.
It consumes frozen SAM3 proposals and the frozen clean-gate decisions, then
tests whether dense SAM3 image features can replace the brittle rule
"tracking succeeded, therefore moved" with explicit object association.

The inference rules never consult ChangeSim ground truth:

* same identity + same aligned position -> unchanged;
* same identity + different position/orientation -> moved;
* low identity + strong same-position overlap -> replaced;
* remaining source-only changed proposals -> removed;
* remaining target-only changed proposals -> added.
"""

from __future__ import annotations

import gc
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from ..masks import compose_labels, filter_visible, mark_replacements
from ..numerics import (
    capture_torch_numerical_state,
    enable_sam3_numerics,
    exit_sam3_numerical_scope,
)
from ..types import Label, ObjectMask


@dataclass(frozen=True)
class FeatureDescriptorBatch:
    """Mask-pooled descriptors and their amount of spatial evidence."""

    vectors: np.ndarray
    valid: np.ndarray
    effective_cells: np.ndarray


@dataclass(frozen=True)
class SimilarityCalibration:
    """Pair-internal, GT-free identity threshold diagnostics."""

    threshold: float
    valid: bool
    positive_count: int
    negative_count: int
    positive_acceptance: float | None
    negative_acceptance: float | None
    inferred_threshold: float | None
    invalid_reason: str | None


@dataclass(frozen=True)
class IdentityMatch:
    """One reciprocal source/target identity association."""

    source_index: int
    target_index: int
    cosine: float
    score: float
    source_margin: float
    target_margin: float


@dataclass
class IdentityLocationResult:
    """Object lists and audit records produced before final rasterization."""

    added: list[ObjectMask]
    removed: list[ObjectMask]
    moved: list[ObjectMask]
    replaced: list[ObjectMask]
    match_records: list[dict[str, Any]]
    diagnostics: dict[str, Any]


class Sam3FeatureExtractor:
    """Expose SAM3's full-image embedding without changing proposal code.

    The automatic proposal model already computes this backbone tensor.  The
    proposal caches do not retain it, so the experiment recomputes exactly one
    feature map per source/target image and stores it on disk.  No per-mask
    model forwards are required.
    """

    def __init__(self, source: str | Path, checkpoint: str | Path):
        self.source = Path(source).resolve()
        self.checkpoint = str(Path(checkpoint).resolve())
        self._model = None
        self._predictor = None
        self._processor = None
        self._numerical_state = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not (self.source / "sam3").is_dir():
            raise FileNotFoundError(f"SAM3 source is unavailable: {self.source}")
        import torch

        state = capture_torch_numerical_state(torch)
        self._numerical_state = state
        source_entry = str(self.source)
        sys.path.insert(0, source_entry)
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            import sam3

            loaded_source = Path(sam3.__file__).resolve().parent.parent
            if loaded_source != self.source:
                raise RuntimeError(
                    "SAM3 was imported from an unexpected checkout: "
                    f"configured={self.source}, loaded={loaded_source}. "
                    "Run full-pipeline inference in a clean pair worker."
                )
            enable_sam3_numerics(torch)
            self._model = build_sam3_image_model(
                checkpoint_path=self.checkpoint,
                load_from_HF=False,
                device="cuda",
                eval_mode=True,
                enable_segmentation=False,
                enable_inst_interactivity=True,
                compile=False,
            )
            self._predictor = self._model.inst_interactive_predictor
            self._processor = Sam3Processor(self._model)
        except BaseException as load_error:
            predictor = self._predictor
            self._processor = None
            self._predictor = None
            self._model = None
            self._numerical_state = None
            try:
                exit_sam3_numerical_scope(predictor, state, torch)
            except BaseException as cleanup_error:
                # Keep the model/import failure as the primary exception while
                # retaining evidence that its cleanup also encountered a fault.
                raise load_error from cleanup_error
            raise
        finally:
            if sys.path and sys.path[0] == source_entry:
                sys.path.pop(0)
            else:
                # Defensive fallback for import hooks that edit sys.path.
                try:
                    sys.path.remove(source_entry)
                except ValueError:
                    pass

    def feature_map(self, image: np.ndarray) -> np.ndarray:
        """Return a channel-normalized CxHf xWf SAM3 image embedding."""

        import torch
        import torch.nn.functional as functional

        self.load()
        assert self._processor is not None and self._predictor is not None
        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("SAM3 feature input must have shape H x W x 3")
        with torch.inference_mode():
            state = self._processor.set_image(Image.fromarray(rgb.copy()))
            backbone = state["backbone_out"]["sam2_backbone_out"]
            _, vision_features, _, _ = (
                self._predictor.model._prepare_backbone_features(backbone)
            )
            # Reproduce the predictor's shape conversion but intentionally do
            # not add ``no_mem_embed``: that constant is useful to the mask
            # decoder but would dominate cosine appearance descriptors.
            features = [
                feature.permute(1, 2, 0).reshape(1, -1, *feature_size)
                for feature, feature_size in zip(
                    vision_features[::-1],
                    self._predictor._bb_feat_sizes[::-1],
                )
            ][::-1]
            embedding = functional.normalize(features[-1][0].float(), dim=0)
            output = embedding.cpu().numpy().astype(np.float16, copy=True)
        self._predictor.reset_predictor()
        return output

    def release(self) -> None:
        if self._model is None and self._numerical_state is None:
            return
        import torch

        predictor = self._predictor
        state = self._numerical_state
        # Detach owned resources before running fallible cleanup. A retry after
        # an exception is therefore a no-op instead of closing/restoring twice.
        self._processor = None
        self._predictor = None
        self._model = None
        self._numerical_state = None

        cleanup_error: BaseException | None = None
        try:
            exit_sam3_numerical_scope(predictor, state, torch)
        except BaseException as exc:
            cleanup_error = exc
        try:
            gc.collect()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
        if cleanup_error is not None:
            raise cleanup_error


def mask_descriptors(
    feature_map: np.ndarray,
    objects: Sequence[ObjectMask],
    *,
    minimum_feature_cells: float,
) -> FeatureDescriptorBatch:
    """Mean-pool dense features inside every soft-resized object mask."""

    features = np.asarray(feature_map, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError("feature_map must have shape C x H x W")
    channels, grid_height, grid_width = features.shape
    flat = features.reshape(channels, -1)
    flat /= np.maximum(np.linalg.norm(flat, axis=0, keepdims=True), 1e-12)
    if not objects:
        return FeatureDescriptorBatch(
            vectors=np.empty((0, channels), np.float32),
            valid=np.empty(0, bool),
            effective_cells=np.empty(0, np.float32),
        )

    weights = []
    for obj in objects:
        mask = np.asarray(obj.mask, dtype=np.float32)
        if mask.ndim != 2:
            raise ValueError("object masks must be two-dimensional")
        resized = np.asarray(
            Image.fromarray(mask, mode="F").resize(
                (grid_width, grid_height), Image.Resampling.BOX
            ),
            dtype=np.float32,
        )
        weights.append(np.clip(resized, 0.0, 1.0).reshape(-1))
    weight_matrix = np.stack(weights)
    evidence = weight_matrix.sum(axis=1)
    vectors = weight_matrix @ flat.T
    vectors /= np.maximum(evidence[:, None], 1e-12)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    valid = (evidence >= float(minimum_feature_cells)) & np.all(
        np.isfinite(vectors), axis=1
    )
    vectors[~valid] = 0.0
    return FeatureDescriptorBatch(
        vectors=vectors.astype(np.float32),
        valid=valid.astype(bool),
        effective_cells=evidence.astype(np.float32),
    )


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pairwise_mask_iou(
    source: Sequence[ObjectMask], target: Sequence[ObjectMask]
) -> np.ndarray:
    """Compute exact mask IoUs while skipping non-overlapping bounding boxes."""

    output = np.zeros((len(source), len(target)), dtype=np.float32)
    if not source or not target:
        return output
    source_masks = [np.asarray(obj.mask, dtype=bool) for obj in source]
    target_masks = [np.asarray(obj.mask, dtype=bool) for obj in target]
    shape = source_masks[0].shape
    if any(mask.shape != shape for mask in source_masks + target_masks):
        raise ValueError("all association masks must share one aligned grid")
    source_boxes = [_bbox(mask) for mask in source_masks]
    target_boxes = [_bbox(mask) for mask in target_masks]
    source_areas = np.asarray([mask.sum() for mask in source_masks], np.int64)
    target_areas = np.asarray([mask.sum() for mask in target_masks], np.int64)
    for source_index, (source_mask, source_box) in enumerate(
        zip(source_masks, source_boxes)
    ):
        sx0, sy0, sx1, sy1 = source_box
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        for target_index, (target_mask, target_box) in enumerate(
            zip(target_masks, target_boxes)
        ):
            tx0, ty0, tx1, ty1 = target_box
            x0, y0 = max(sx0, tx0), max(sy0, ty0)
            x1, y1 = min(sx1, tx1), min(sy1, ty1)
            if x1 <= x0 or y1 <= y0:
                continue
            intersection = int(
                np.logical_and(
                    source_mask[y0:y1, x0:x1], target_mask[y0:y1, x0:x1]
                ).sum()
            )
            union = int(source_areas[source_index] + target_areas[target_index] - intersection)
            if union:
                output[source_index, target_index] = intersection / union
    return output


def _mask_geometry(mask: np.ndarray) -> dict[str, float | None]:
    """Return aligned centroid and an optional 180-degree mask orientation."""

    binary = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(binary)
    height, width = binary.shape
    if not len(xs):
        return {
            "area": 0.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "orientation_degrees": None,
            "anisotropy": 0.0,
        }
    centered = np.stack((xs - xs.mean(), ys - ys.mean()), axis=1)
    covariance = centered.T @ centered / max(len(centered), 1)
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    largest, smallest = float(values.max()), float(values.min())
    anisotropy = (largest - smallest) / max(largest + smallest, 1e-12)
    angle = math.degrees(math.atan2(float(major[1]), float(major[0]))) % 180.0
    return {
        "area": float(len(xs)),
        "centroid_x": float(xs.mean() / max(width - 1, 1)),
        "centroid_y": float(ys.mean() / max(height - 1, 1)),
        "orientation_degrees": angle,
        "anisotropy": anisotropy,
    }


def _orientation_difference(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_angle = first["orientation_degrees"]
    second_angle = second["orientation_degrees"]
    if first_angle is None or second_angle is None:
        return 0.0
    difference = abs(float(first_angle) - float(second_angle)) % 180.0
    return min(difference, 180.0 - difference)


def cosine_similarity_matrix(
    source: FeatureDescriptorBatch, target: FeatureDescriptorBatch
) -> np.ndarray:
    if source.vectors.shape[1:] != target.vectors.shape[1:]:
        raise ValueError("source and target descriptor dimensions differ")
    if not len(source.vectors) or not len(target.vectors):
        return np.empty((len(source.vectors), len(target.vectors)), np.float32)
    return np.clip(source.vectors @ target.vectors.T, -1.0, 1.0).astype(np.float32)


def calibrate_identity_threshold(
    similarity: np.ndarray,
    spatial_iou: np.ndarray,
    source_static: np.ndarray,
    target_static: np.ndarray,
    source_valid: np.ndarray,
    target_valid: np.ndarray,
    *,
    fallback_threshold: float,
    control_iou: float,
    negative_iou: float,
    maximum_negative_acceptance: float,
    minimum_positive_acceptance: float,
    minimum_positive_count: int,
    minimum_negative_count: int,
) -> SimilarityCalibration:
    """Infer a conservative identity threshold from hard-static controls."""

    similarity = np.asarray(similarity, np.float32)
    spatial_iou = np.asarray(spatial_iou, np.float32)
    source_indices = np.flatnonzero(np.asarray(source_static, bool) & source_valid)
    target_indices = np.flatnonzero(np.asarray(target_static, bool) & target_valid)
    positives: list[tuple[int, int]] = []
    if len(source_indices) and len(target_indices):
        sub_iou = spatial_iou[np.ix_(source_indices, target_indices)]
        source_best = np.argmax(sub_iou, axis=1)
        target_best = np.argmax(sub_iou, axis=0)
        for local_source, local_target in enumerate(source_best):
            if target_best[local_target] != local_source:
                continue
            if sub_iou[local_source, local_target] < control_iou:
                continue
            positives.append(
                (int(source_indices[local_source]), int(target_indices[local_target]))
            )
    positive_scores = np.asarray(
        [similarity[source, target] for source, target in positives], np.float32
    )
    positive_set = set(positives)
    negative_scores = np.asarray(
        [
            similarity[source, target]
            for source, _ in positives
            for _, target in positives
            if (source, target) not in positive_set
            and spatial_iou[source, target] <= negative_iou
        ],
        np.float32,
    )
    reasons = []
    if len(positive_scores) < minimum_positive_count:
        reasons.append("insufficient_positive_controls")
    if len(negative_scores) < minimum_negative_count:
        reasons.append("insufficient_negative_controls")
    if reasons:
        return SimilarityCalibration(
            threshold=float(fallback_threshold),
            valid=False,
            positive_count=len(positive_scores),
            negative_count=len(negative_scores),
            positive_acceptance=None,
            negative_acceptance=None,
            inferred_threshold=None,
            invalid_reason=";".join(reasons),
        )
    inferred = float(
        np.quantile(
            negative_scores,
            1.0 - float(maximum_negative_acceptance),
            method="higher",
        )
    )
    threshold = max(float(fallback_threshold), inferred)
    positive_acceptance = float(np.mean(positive_scores >= threshold))
    negative_acceptance = float(np.mean(negative_scores >= threshold))
    if positive_acceptance < minimum_positive_acceptance:
        return SimilarityCalibration(
            threshold=float(fallback_threshold),
            valid=False,
            positive_count=len(positive_scores),
            negative_count=len(negative_scores),
            positive_acceptance=positive_acceptance,
            negative_acceptance=negative_acceptance,
            inferred_threshold=inferred,
            invalid_reason="positive_acceptance_below_minimum",
        )
    return SimilarityCalibration(
        threshold=threshold,
        valid=True,
        positive_count=len(positive_scores),
        negative_count=len(negative_scores),
        positive_acceptance=positive_acceptance,
        negative_acceptance=negative_acceptance,
        inferred_threshold=inferred,
        invalid_reason=None,
    )


def _second_best(values: np.ndarray, best: int) -> float:
    if len(values) <= 1:
        return -1.0
    return float(np.max(np.delete(values, best)))


def associate_identities(
    source: FeatureDescriptorBatch,
    target: FeatureDescriptorBatch,
    spatial_iou: np.ndarray,
    source_areas: np.ndarray,
    target_areas: np.ndarray,
    *,
    minimum_cosine: float,
    minimum_margin: float,
    area_ratio_bounds: tuple[float, float],
    same_location_bonus: float,
    require_mutual_nearest: bool,
) -> tuple[list[IdentityMatch], np.ndarray, np.ndarray]:
    """Associate identities globally, with a small aligned-location tie-break."""

    similarity = cosine_similarity_matrix(source, target)
    low_area, high_area = map(float, area_ratio_bounds)
    ratios = target_areas[None, :] / np.maximum(source_areas[:, None], 1.0)
    feasible = (
        source.valid[:, None]
        & target.valid[None, :]
        & (similarity >= float(minimum_cosine))
        & (ratios >= low_area)
        & (ratios <= high_area)
    )
    score = similarity + float(same_location_bonus) * spatial_iou
    masked = np.where(feasible, score, -np.inf)
    if not feasible.size or not np.any(feasible):
        return [], similarity, feasible
    row_best = np.argmax(masked, axis=1)
    column_best = np.argmax(masked, axis=0)
    accepted_edges = np.zeros_like(feasible)
    for source_index, target_index in zip(*np.nonzero(feasible)):
        if require_mutual_nearest and not (
            row_best[source_index] == target_index
            and column_best[target_index] == source_index
        ):
            continue
        source_margin = score[source_index, target_index] - _second_best(
            masked[source_index], target_index
        )
        target_margin = score[source_index, target_index] - _second_best(
            masked[:, target_index], source_index
        )
        if source_margin < minimum_margin or target_margin < minimum_margin:
            continue
        accepted_edges[source_index, target_index] = True

    matches: list[IdentityMatch] = []
    if np.any(accepted_edges):
        rows, columns = linear_sum_assignment(
            np.where(accepted_edges, -score, 1e6)
        )
        for source_index, target_index in zip(rows, columns):
            if not accepted_edges[source_index, target_index]:
                continue
            matches.append(
                IdentityMatch(
                    source_index=int(source_index),
                    target_index=int(target_index),
                    cosine=float(similarity[source_index, target_index]),
                    score=float(score[source_index, target_index]),
                    source_margin=float(
                        score[source_index, target_index]
                        - _second_best(masked[source_index], target_index)
                    ),
                    target_margin=float(
                        score[source_index, target_index]
                        - _second_best(masked[:, target_index], source_index)
                    ),
                )
            )
    return matches, similarity, accepted_edges


def _is_duplicate(index: int, consumed: set[int], within_iou: np.ndarray, threshold: float) -> bool:
    return bool(consumed) and bool(
        np.any(within_iou[index, np.asarray(sorted(consumed), dtype=int)] >= threshold)
    )


def _within_iou(objects: Sequence[ObjectMask]) -> np.ndarray:
    matrix = pairwise_mask_iou(objects, objects)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def classify_identity_location(
    source_objects: Sequence[ObjectMask],
    target_objects: Sequence[ObjectMask],
    source_features: FeatureDescriptorBatch,
    target_features: FeatureDescriptorBatch,
    source_changed: np.ndarray,
    target_changed: np.ndarray,
    calibration: SimilarityCalibration,
    config: dict[str, Any],
    spatial_iou: np.ndarray | None = None,
) -> IdentityLocationResult:
    """Apply the preregistered identity/location/replacement decision tree."""

    source_objects = list(source_objects)
    target_objects = list(target_objects)
    source_changed = np.asarray(source_changed, dtype=bool)
    target_changed = np.asarray(target_changed, dtype=bool)
    if source_changed.shape != (len(source_objects),):
        raise ValueError("source_changed length mismatch")
    if target_changed.shape != (len(target_objects),):
        raise ValueError("target_changed length mismatch")
    if spatial_iou is None:
        spatial_iou = pairwise_mask_iou(source_objects, target_objects)
    else:
        spatial_iou = np.asarray(spatial_iou, dtype=np.float32)
        if spatial_iou.shape != (len(source_objects), len(target_objects)):
            raise ValueError("spatial_iou shape mismatch")
    source_geometry = [_mask_geometry(obj.mask) for obj in source_objects]
    target_geometry = [_mask_geometry(obj.mask) for obj in target_objects]
    source_areas = np.asarray([item["area"] for item in source_geometry], np.float32)
    target_areas = np.asarray([item["area"] for item in target_geometry], np.float32)
    matching = config["matching"]
    classification = config["classification"]
    matches, similarity, feasible = associate_identities(
        source_features,
        target_features,
        spatial_iou,
        source_areas,
        target_areas,
        minimum_cosine=calibration.threshold,
        minimum_margin=float(matching["minimum_bidirectional_margin"]),
        area_ratio_bounds=tuple(matching["area_ratio_bounds"]),
        same_location_bonus=float(matching["same_location_bonus"]),
        require_mutual_nearest=bool(matching["require_mutual_nearest"]),
    )

    moved: list[ObjectMask] = []
    records: list[dict[str, Any]] = []
    consumed_source: set[int] = set()
    consumed_target: set[int] = set()
    unchanged_matches = 0
    moved_matches = 0
    rotation_matches = 0
    unchanged_low_area, unchanged_high_area = map(
        float, classification["unchanged_area_ratio_bounds"]
    )
    for match in matches:
        source_index, target_index = match.source_index, match.target_index
        consumed_source.add(source_index)
        consumed_target.add(target_index)
        source_geo = source_geometry[source_index]
        target_geo = target_geometry[target_index]
        ratio = float(target_geo["area"] / max(float(source_geo["area"]), 1.0))
        angle_difference = _orientation_difference(source_geo, target_geo)
        orientation_reliable = (
            float(source_geo["anisotropy"])
            >= float(classification["orientation_anisotropy"])
            and float(target_geo["anisotropy"])
            >= float(classification["orientation_anisotropy"])
        )
        rotated = orientation_reliable and angle_difference >= float(
            classification["orientation_change_degrees"]
        )
        same_location = (
            spatial_iou[source_index, target_index]
            >= float(classification["unchanged_mask_iou"])
            and unchanged_low_area <= ratio <= unchanged_high_area
            and not rotated
        )
        label = Label.UNCHANGED if same_location else Label.MOVED
        if label == Label.MOVED:
            moved.append(
                ObjectMask(
                    mask=np.asarray(target_objects[target_index].mask, bool).copy(),
                    score=float(target_objects[target_index].score),
                    label=Label.MOVED,
                    source="sam3_identity_displaced",
                    metadata={
                        "source_proposal_id": source_objects[source_index].metadata.get(
                            "automatic_proposal_id"
                        ),
                        "target_proposal_id": target_objects[target_index].metadata.get(
                            "automatic_proposal_id"
                        ),
                        "identity_cosine": match.cosine,
                    },
                )
            )
            moved_matches += 1
            rotation_matches += int(rotated)
        else:
            unchanged_matches += 1
        records.append(
            {
                "decision": label.name.lower(),
                "source_index": source_index,
                "target_index": target_index,
                "source_proposal_id": source_objects[source_index].metadata.get(
                    "automatic_proposal_id"
                ),
                "target_proposal_id": target_objects[target_index].metadata.get(
                    "automatic_proposal_id"
                ),
                "cosine": match.cosine,
                "association_score": match.score,
                "spatial_iou": float(spatial_iou[source_index, target_index]),
                "area_ratio": ratio,
                "orientation_difference_degrees": angle_difference,
                "orientation_reliable": orientation_reliable,
                "rotation_triggered": rotated,
            }
        )

    source_within = _within_iou(source_objects)
    target_within = _within_iou(target_objects)
    duplicate_threshold = float(classification["duplicate_suppression_iou"])
    duplicate_source = {
        index
        for index in range(len(source_objects))
        if index not in consumed_source
        and _is_duplicate(index, consumed_source, source_within, duplicate_threshold)
    }
    duplicate_target = {
        index
        for index in range(len(target_objects))
        if index not in consumed_target
        and _is_duplicate(index, consumed_target, target_within, duplicate_threshold)
    }

    # Direct replacement is deliberately evaluated before added/removed.  It
    # can therefore recover an old/new object pair even when the clean-render
    # gate incorrectly called one or both proposals static.
    replacement_limit = calibration.threshold - float(
        classification["replacement_identity_margin"]
    )
    remaining_source = [
        index
        for index in range(len(source_objects))
        if index not in consumed_source and index not in duplicate_source
    ]
    remaining_target = [
        index
        for index in range(len(target_objects))
        if index not in consumed_target and index not in duplicate_target
    ]
    replacement_edges: list[tuple[float, int, int]] = []
    for source_index in remaining_source:
        for target_index in remaining_target:
            if not source_features.valid[source_index] or not target_features.valid[target_index]:
                continue
            iou = float(spatial_iou[source_index, target_index])
            if iou < float(classification["replacement_minimum_iou"]):
                continue
            if float(similarity[source_index, target_index]) > replacement_limit:
                continue
            source_geo = source_geometry[source_index]
            target_geo = target_geometry[target_index]
            distance = math.hypot(
                float(source_geo["centroid_x"]) - float(target_geo["centroid_x"]),
                float(source_geo["centroid_y"]) - float(target_geo["centroid_y"]),
            )
            if distance > float(classification["replacement_maximum_centroid_distance"]):
                continue
            ratio = float(target_geo["area"] / max(float(source_geo["area"]), 1.0))
            low, high = map(float, matching["area_ratio_bounds"])
            if not low <= ratio <= high:
                continue
            replacement_edges.append((iou, source_index, target_index))
    replacement_edges.sort(reverse=True)
    replaced: list[ObjectMask] = []
    replacement_source: set[int] = set()
    replacement_target: set[int] = set()
    for iou, source_index, target_index in replacement_edges:
        if source_index in replacement_source or target_index in replacement_target:
            continue
        mask = np.logical_and(
            source_objects[source_index].mask, target_objects[target_index].mask
        )
        if not np.any(mask):
            continue
        replacement_source.add(source_index)
        replacement_target.add(target_index)
        consumed_source.add(source_index)
        consumed_target.add(target_index)
        replaced.append(
            ObjectMask(
                mask=mask,
                score=min(
                    float(source_objects[source_index].score),
                    float(target_objects[target_index].score),
                ),
                label=Label.REPLACED,
                source="sam3_same_place_different_identity",
                metadata={
                    "source_proposal_id": source_objects[source_index].metadata.get(
                        "automatic_proposal_id"
                    ),
                    "target_proposal_id": target_objects[target_index].metadata.get(
                        "automatic_proposal_id"
                    ),
                    "identity_cosine": float(similarity[source_index, target_index]),
                    "spatial_iou": iou,
                },
            )
        )
        records.append(
            {
                "decision": "replaced",
                "source_index": source_index,
                "target_index": target_index,
                "source_proposal_id": source_objects[source_index].metadata.get(
                    "automatic_proposal_id"
                ),
                "target_proposal_id": target_objects[target_index].metadata.get(
                    "automatic_proposal_id"
                ),
                "cosine": float(similarity[source_index, target_index]),
                "spatial_iou": iou,
                "replacement_identity_limit": replacement_limit,
            }
        )

    # Suppress proposal duplicates of every consumed identity/replacement pair
    # once more before turning unmatched gate failures into semantic changes.
    duplicate_source |= {
        index
        for index in range(len(source_objects))
        if index not in consumed_source
        and _is_duplicate(index, consumed_source, source_within, duplicate_threshold)
    }
    duplicate_target |= {
        index
        for index in range(len(target_objects))
        if index not in consumed_target
        and _is_duplicate(index, consumed_target, target_within, duplicate_threshold)
    }
    removed = [
        ObjectMask(
            mask=np.asarray(source_objects[index].mask, bool).copy(),
            score=float(source_objects[index].score),
            label=Label.REMOVED,
            source="sam3_identity_unmatched_source",
            metadata=dict(source_objects[index].metadata),
        )
        for index in range(len(source_objects))
        if source_changed[index]
        and index not in consumed_source
        and index not in duplicate_source
    ]
    added = [
        ObjectMask(
            mask=np.asarray(target_objects[index].mask, bool).copy(),
            score=float(target_objects[index].score),
            label=Label.ADDED,
            source="sam3_identity_unmatched_target",
            metadata=dict(target_objects[index].metadata),
        )
        for index in range(len(target_objects))
        if target_changed[index]
        and index not in consumed_target
        and index not in duplicate_target
    ]
    diagnostics = {
        "calibration": asdict(calibration),
        "identity_match_count": len(matches),
        "unchanged_identity_match_count": unchanged_matches,
        "moved_identity_match_count": moved_matches,
        "orientation_triggered_moved_count": rotation_matches,
        "direct_replacement_count": len(replaced),
        "removed_unmatched_count": len(removed),
        "added_unmatched_count": len(added),
        "source_duplicate_suppressed_count": len(duplicate_source),
        "target_duplicate_suppressed_count": len(duplicate_target),
        "source_valid_descriptor_count": int(source_features.valid.sum()),
        "target_valid_descriptor_count": int(target_features.valid.sum()),
        "feature_feasible_edge_count": int(feasible.sum()),
        "source_gate_changed_count": int(source_changed.sum()),
        "target_gate_changed_count": int(target_changed.sum()),
    }
    return IdentityLocationResult(
        added=added,
        removed=removed,
        moved=moved,
        replaced=replaced,
        match_records=records,
        diagnostics=diagnostics,
    )


def compose_identity_labels(
    artifact_dir: str | Path,
    result: IdentityLocationResult,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply the frozen visibility/priority protocol plus direct replacement."""

    artifact = Path(artifact_dir)
    classification = config["classification"]
    with np.load(artifact / "geometry.npz") as geometry:
        coverage = np.asarray(geometry["coverage01"], dtype=bool)
    alpha = float(classification["visibility_alpha"])
    minimum_area = int(classification["minimum_mask_area"])
    added = filter_visible(result.added, coverage, alpha, minimum_area)
    removed = filter_visible(result.removed, coverage, alpha, minimum_area)
    moved = filter_visible(result.moved, coverage, alpha, minimum_area)
    replaced = filter_visible(result.replaced, coverage, alpha, minimum_area)
    labels = compose_labels(coverage.shape, added + removed + moved)
    labels = mark_replacements(
        labels,
        added,
        removed,
        float(classification["replacement_overlap_iou"]),
    )
    for replacement in replaced:
        labels[np.asarray(replacement.mask, bool)] = int(Label.REPLACED)
    native_shape = np.asarray(Image.open(artifact / "labels.png"), dtype=np.uint8).shape
    native = np.asarray(
        Image.fromarray(labels).resize(native_shape[::-1], Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    return native, {
        "added": len(added),
        "removed": len(removed),
        "moved": len(moved),
        "direct_replaced": len(replaced),
    }


@dataclass(frozen=True)
class AppearanceFeatures:
    """Everything the rest of the pipeline needs from the dense-feature stage.

    ``source_map``/``target_map`` are the raw dense SAM3 embeddings (kept so
    later stages -- the feature-veto gate, the association resolver -- can
    pool descriptors for their own object lists without a second backbone
    pass). ``source_features``/``target_features`` are already-pooled
    per-proposal descriptors for the same visible object lists used here,
    reused as-is by the feature-veto gate rather than recomputed.
    ``identity_threshold`` is ``calibration.threshold`` (it already falls
    back to the configured default when calibration is invalid; callers
    never need to branch on ``calibration.valid`` themselves).
    ``match_records`` is the per-pair identity/location decision trail
    (unchanged/moved/replaced) that the motion-and-replacement evidence
    stage pairs against proposal masks; nothing here rasterizes its own
    label map (see ``classify_identity_location``'s docstring for why that
    raster itself is not part of the winning composition).
    """

    source_map: np.ndarray
    target_map: np.ndarray
    source_features: FeatureDescriptorBatch
    target_features: FeatureDescriptorBatch
    calibration: SimilarityCalibration
    match_records: list[dict[str, Any]]

    @property
    def identity_threshold(self) -> float:
        return float(self.calibration.threshold)


def compute_appearance_features(
    source_objects: Sequence[ObjectMask],
    target_objects: Sequence[ObjectMask],
    source_render: np.ndarray,
    target_image: np.ndarray,
    source_changed: np.ndarray,
    target_changed: np.ndarray,
    extractor: Sam3FeatureExtractor,
    config: dict[str, Any],
) -> AppearanceFeatures:
    """Dense SAM3 appearance features plus a pair-internal identity threshold.

    Faithful extraction of ``scripts/run_sam3_identity_location_experiment.py``'s
    per-pair computation (see that script around its ``_pair_inputs``/main-loop
    feature and classification steps). ``source_objects``/``target_objects``
    must be the same *visible* (cross-render-covered, area-filtered) proposal
    lists stage 3 tracked; ``source_changed``/``target_changed`` are boolean
    arrays aligned with them, True where stage 3's clean-render gate rejected
    that proposal (i.e. membership in its ``source_changed_proposal_ids`` /
    ``target_changed_proposal_ids`` diagnostics) -- calibration treats the
    complement (statically accepted proposals) as its identity controls.

    Only ``(source_map, target_map, calibration.threshold)`` were originally
    scoped as downstream-relevant; reading the motion-and-replacement
    evidence stage's actual data dependency (its ``decisions.json`` input is
    this function's ``match_records``) showed ``classify_identity_location``
    is load-bearing too, not dead code -- so it is computed here as well.
    ``compose_identity_labels`` (this module's own standalone label raster)
    remains genuinely unused downstream and is not called.
    """

    source_map = extractor.feature_map(source_render)
    target_map = extractor.feature_map(target_image)
    sam3_cfg = config["sam3"]
    minimum_cells = float(sam3_cfg["minimum_feature_cells"])
    source_features = mask_descriptors(source_map, source_objects, minimum_feature_cells=minimum_cells)
    target_features = mask_descriptors(target_map, target_objects, minimum_feature_cells=minimum_cells)
    spatial_iou = pairwise_mask_iou(source_objects, target_objects)
    matching = config["matching"]
    calibration = calibrate_identity_threshold(
        cosine_similarity_matrix(source_features, target_features),
        spatial_iou,
        ~np.asarray(source_changed, dtype=bool),
        ~np.asarray(target_changed, dtype=bool),
        source_features.valid,
        target_features.valid,
        fallback_threshold=float(matching["fallback_minimum_cosine"]),
        control_iou=float(matching["calibration_control_iou"]),
        negative_iou=float(matching["calibration_negative_iou"]),
        maximum_negative_acceptance=float(matching["calibration_maximum_negative_acceptance"]),
        minimum_positive_acceptance=float(matching["calibration_minimum_positive_acceptance"]),
        minimum_positive_count=int(matching["calibration_minimum_positive_count"]),
        minimum_negative_count=int(matching["calibration_minimum_negative_count"]),
    )
    classification = classify_identity_location(
        source_objects,
        target_objects,
        source_features,
        target_features,
        source_changed,
        target_changed,
        calibration,
        config,
        spatial_iou=spatial_iou,
    )
    return AppearanceFeatures(
        source_map=source_map,
        target_map=target_map,
        source_features=source_features,
        target_features=target_features,
        calibration=calibration,
        match_records=classification.match_records,
    )
