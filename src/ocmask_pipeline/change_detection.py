"""Object-level change detection over three images on one aligned pixel grid.

The three frames are:

``render_t0``
    The old scene rendered into the T1 camera (see ``reconstruction.py``).
``clean_render``
    The canonical point-cloud render in the same camera, with mutual
    depth-conflict filtering applied so add/remove transients are pruned.
``image_t1``
    The real current image, and the pixel grid the output is aligned to.

Each frame gets an independent SAM3 automatic-mask inventory with pooled
SAM3 + DINOv2 descriptors per object. A correspondence between two frames is
accepted as an identity only when both descriptors agree (or a SAM2-tracked
mask and both descriptors agree). The clean render is a third-view bridge
when the render/photo appearance gap makes a direct comparison fail.

Objects that end up unmatched are, by default, revisited once more before
being called removed/added: ``recover_unmatched_via_tracking`` reuses the
SAM2 track already computed for every object (regardless of whether it later
got a partner) and checks it against the opposite frame's own dense
SAM3/DINOv2 feature maps at the same identity thresholds used everywhere
else here. This recovers objects that SAM3's automatic mask generator
proposed in only one of the two frames, which is not evidence of a real
change.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import binary_dilation, label as connected_components
from scipy.optimize import linear_sum_assignment

from .color_residual import color_replacement_candidate_mask
from .io import save_image, save_json
from .stages.sam2_tracking_backend import Sam2MaskTracker
from .stages.sam3_identity_location import (
    FeatureDescriptorBatch,
    cosine_similarity_matrix,
    mask_descriptors,
    pairwise_mask_iou,
)
from .stages.sam3_proposals import Sam3AutomaticMaskGenerator, Sam3Proposal
from .adapters.dinov2 import Dinov2FeatureExtractor
from .types import Label, ObjectMask
from .visualization import colorize, instance_overlay, overlay


@dataclass(frozen=True)
class ThreeImageSettings:
    """Thresholds for proposal cleanup, association, and classification."""

    minimum_mask_area: int = 16
    maximum_mask_area_fraction: float = 0.25
    duplicate_iou: float = 0.80
    duplicate_containment: float = 0.65
    duplicate_minimum_area_ratio: float = 0.0
    part_maximum_area_ratio: float = 0.40
    part_bbox_margin_pixels: int = 3
    minimum_sam_feature_cells: float = 1.0
    minimum_dino_feature_cells: float = 1.0
    minimum_sam_cosine: float = 0.65
    minimum_dino_cosine: float = 0.60
    minimum_identity_margin: float = 0.02
    area_ratio_low: float = 0.25
    area_ratio_high: float = 4.0
    minimum_track_iou: float = 0.20
    same_location_iou: float = 0.45
    tracking_batch_size: int = 16
    recover_unmatched_via_tracking: bool = True
    # Geometric identity test (point-cloud centroid + overlap), an
    # alternative to 2D-appearance cosine similarity that does not need to
    # bridge the render/photo appearance gap -- see change_detection's
    # module docstring and docs/goldilocs_analysis.html Section 4. Both
    # fractions are relative to reconstruction.ReferenceScene's own
    # scene_scale (a robust radius estimate), not an absolute distance, so
    # one setting works across PASLCD scenes at different arbitrary
    # reconstruction scales. Disabled (all pairs pass) when no 3D position
    # data is supplied to run_object_state_resolution, so this is fully
    # backward compatible with callers that only have 2D renders.
    enable_geometric_identity: bool = True
    geometric_centroid_fraction: float = 0.15
    geometric_overlap_fraction: float = 0.15
    minimum_geometric_overlap: float = 0.20
    # GOLDILOCS's visibility filter (Appendix A.6 / masks.filter_visible):
    # drops a removed/added/moved decision whose mask is mostly supported by
    # unrendered (occluded/out-of-view/parallax-gap) pixels rather than real
    # geometry, using render_t0's own per-pixel coverage -- replaces this
    # pipeline's removed border-touching-proposal filter, which (verified on
    # PASLCD) destroyed real edge-adjacent objects far more than it removed
    # background; coverage is the actual underlying signal border-touching
    # was a poor proxy for. 0.8 is ChangeSim's / every GOLDILOCS config's own
    # published default, not a PASLCD-tuned value. Disabled (no decisions
    # filtered) when no render_t0_coverage is supplied to
    # run_object_state_resolution, same fallback convention as the
    # geometric-identity settings above.
    enable_visibility_filter: bool = True
    minimum_render_support_fraction: float = 0.8
    # Our extension: a continuous, confidence-weighted visibility score
    # (see _confidence_weighted_visible_fraction) in place of the binary
    # coverage-fraction test above, splatting VGGT-Omega's own per-point
    # multi-view depth confidence into the render_t0 view instead of just a
    # rendered/not-rendered bit. Neither GOLDILOCS nor SceneDiff (arXiv
    # 2512.16908) have an equivalent -- both reconstruct from a single
    # best-matching pair rather than jointly fusing many reference views,
    # so neither has a genuine multi-view confidence signal to use here.
    # Requires render_t0_confidence; falls back to the binary test above
    # when not supplied, same convention as every other setting here.
    use_confidence_weighted_visibility: bool = False
    # PASLCD-specific precision safeguard (roadmap item 6): GOLDILOCS's own
    # dominant precision mechanism is a majority vote across several *query*
    # photos of the same change (paper Table 11: 59-80% -> 95-98% precision
    # on/off) -- structurally unavailable here, since PASLCD gives exactly
    # one query image per test case. This is the reference-side analogue:
    # render each of the (up to 24) reference images' own point cloud into
    # the query view independently, never merged with each other
    # (reconstruction.localize_and_render_query's render_t0_corroboration),
    # and require a removed/moved-source object's own geometry to be
    # corroborated by more than a couple of them -- catching the case where
    # the aggregate reconstruction's confident-looking geometry there
    # actually traces back to one near-degenerate stereo pair, not real
    # multi-view agreement. Only ever applied where a t0-side mask exists
    # (removed, and moved's source location); added objects have no
    # reference-side geometry to corroborate by construction, so this can
    # never affect them. Disabled (no decisions filtered) when no
    # render_t0_corroboration is supplied, same fallback convention as the
    # other reference-signal settings above.
    enable_reference_corroboration: bool = False
    minimum_corroborating_views: int = 2
    # Same-footprint object-replacement detection (roadmap item 5's
    # replacement -- dropped the ported SSIM/Warped idea for a mechanism
    # grounded in an actual verified PASLCD failure: a red block swapped for
    # a blue one, absorbed into a single large tabletop SAM3 proposal in
    # both frames, so identity matching between two proposals never gets a
    # chance to run on it at all -- color-aware or not. Runs as a final,
    # separate pass over render_t0/image_t1 directly (see
    # change_detection.find_color_replacement_regions and
    # color_residual.py), restricted to pixels no earlier stage already
    # explained, so it can only add new REPLACED detections, never touch or
    # override an existing decision. Disabled (no color-residual pass run)
    # when render_t0_coverage is not supplied, same fallback convention as
    # every other reference-signal setting above.
    enable_color_replacement_detection: bool = False
    color_replacement_minimum_colorfulness: float = 40.0
    color_replacement_residual_percentile: float = 90.0
    minimum_color_replacement_area: int = 40
    # Checked only when render_t0_confidence is supplied (see
    # _confidence_weighted_visible_fraction's [0, 1] normalization) --
    # suppresses candidates sitting on geometry the reconstruction itself
    # was never confident about, e.g. a thin draped object at an oblique
    # angle, one observed false positive (yellow caution tape) during
    # validation.
    color_replacement_minimum_confidence: float = 0.3
    # Tried and reverted: a minimum-bounding-box-width filter, meant to
    # reject thin slivers found on a scene with cluttered thin structures
    # (a pole amid ropes/netting, where the z-buffer interleaves several
    # distinct real 3D locations -- each individually confident and well-
    # corroborated, so the gate above doesn't catch them -- at adjacent
    # pixels along a thin silhouette). It did remove that false positive,
    # but also rejected a genuine thin true positive elsewhere (a real
    # GT-supported 4x26px detection); at 19-query scale the two effects
    # canceled out (1 improved, 1 regressed, net ~0%). Two other candidate
    # discriminators (neighborhood color variance, within-component color
    # variance) were also tested and neither separated real thin detections
    # from the pole artifacts either -- real and fake overlap in all three
    # signals tried. Not shipped; a real fix would need a signal this
    # session didn't find, not a retuned threshold on any of these three.

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ThreeImageSettings":
        values = config.get("three_image_comparison", {})
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError("Unknown three_image_comparison settings: " + ", ".join(unknown))
        return cls(**values)


@dataclass(frozen=True)
class FrameInventory:
    """Object masks, two independent descriptor batches, and (optionally)
    the frame's own per-pixel 3D world positions, for one frame."""

    objects: tuple[ObjectMask, ...]
    sam: FeatureDescriptorBatch
    dino: FeatureDescriptorBatch
    world_positions: np.ndarray | None = None  # (H, W, 3), NaN where unobserved


@dataclass(frozen=True)
class TrackingEvidence:
    """Bidirectional SAM2 tracks for the three pairwise frame combinations."""

    t0_to_t1: tuple[np.ndarray | None, ...]
    t1_to_t0: tuple[np.ndarray | None, ...]
    t0_to_clean: tuple[np.ndarray | None, ...]
    clean_to_t0: tuple[np.ndarray | None, ...]
    t1_to_clean: tuple[np.ndarray | None, ...]
    clean_to_t1: tuple[np.ndarray | None, ...]


@dataclass(frozen=True)
class ThreeImageResult:
    labels: np.ndarray
    objects: tuple[ObjectMask, ...]
    decisions: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]
    artifacts_dir: Path
    timings: dict[str, float]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    components, count = connected_components(np.asarray(mask, dtype=bool))
    if count <= 1:
        return np.asarray(mask, dtype=bool).copy()
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == int(np.argmax(sizes))


def _border_sides(mask: np.ndarray) -> int:
    binary = np.asarray(mask, dtype=bool)
    return sum(bool(side.any()) for side in (binary[0], binary[-1], binary[:, 0], binary[:, -1]))


def select_object_proposals(
    proposals: Sequence[Sam3Proposal], settings: ThreeImageSettings
) -> list[ObjectMask]:
    """Turn overlapping automatic proposals into a compact object inventory.

    This removes empty/tiny masks, oversized (whole-frame-spanning)
    proposals, disconnected residuals, and near-duplicate/nested proposals.
    It cannot infer semantic foreground classes; that would require labelled
    fine-tuning data or a text-conditioned detector.
    """
    candidates = []
    for index, proposal in enumerate(proposals, start=1):
        x0, y0, x1, y1 = proposal.crop_box_xyxy
        candidates.append(
            ObjectMask(
                mask=np.asarray(proposal.mask, dtype=bool),
                score=float(proposal.predicted_iou),
                source="sam3_automatic",
                metadata={
                    "automatic_proposal_id": index,
                    "proposal_backend": "sam3",
                    "stability_score": float(proposal.stability_score),
                    "point_coords": [list(proposal.point_xy)],
                    "crop_box_xywh": [x0, y0, x1 - x0, y1 - y0],
                },
            )
        )
    if not candidates:
        return []

    height, width = np.asarray(candidates[0].mask).shape
    image_area = height * width
    filtered: list[ObjectMask] = []
    for item in candidates:
        mask = _largest_component(item.mask)
        if mask.shape != (height, width):
            raise ValueError("all proposals in an inventory must share one shape")
        area = int(mask.sum())
        fraction = area / max(image_area, 1)
        # border_sides is kept as diagnostic metadata only -- a prior version
        # rejected any mask touching any frame edge here, which (verified on
        # PASLCD) was destroying real edge-adjacent objects (a table/wall/
        # ceiling fixture reaching the frame border) far more than it removed
        # actual unbounded background. Removed; area/duplicate filtering below
        # is what should suppress genuine background regions.
        sides = _border_sides(mask)
        if area < settings.minimum_mask_area:
            continue
        if fraction > settings.maximum_mask_area_fraction:
            continue
        filtered.append(
            ObjectMask(
                mask=mask,
                score=item.score,
                source=item.source,
                metadata={**item.metadata, "area": area, "area_fraction": fraction, "border_sides": sides},
            )
        )

    def rank(item: ObjectMask) -> tuple[int, float, float, int]:
        # Every candidate already cleared SAM3's quality/stability gates. Prefer
        # the complete enclosing mask so a head, button, lid, or face does not
        # become another object beside the whole instance.
        return (
            int(np.asarray(item.mask, bool).sum()),
            float(item.score),
            float(item.metadata.get("stability_score", 0.0)),
            -int(item.metadata.get("automatic_proposal_id", 0)),
        )

    retained: list[ObjectMask] = []
    for candidate in sorted(filtered, key=rank, reverse=True):
        candidate_mask = np.asarray(candidate.mask, dtype=bool)
        candidate_area = int(candidate_mask.sum())
        duplicate = False
        for existing in retained:
            existing_mask = np.asarray(existing.mask, dtype=bool)
            existing_area = int(existing_mask.sum())
            intersection = int(np.logical_and(candidate_mask, existing_mask).sum())
            union = candidate_area + existing_area - intersection
            iou = intersection / max(union, 1)
            containment = intersection / max(min(candidate_area, existing_area), 1)
            area_ratio = min(candidate_area, existing_area) / max(candidate_area, existing_area, 1)
            if iou >= settings.duplicate_iou or (
                containment >= settings.duplicate_containment and area_ratio >= settings.duplicate_minimum_area_ratio
            ):
                duplicate = True
                break
        if not duplicate:
            retained.append(candidate)

    retained.sort(key=lambda item: int(item.metadata.get("automatic_proposal_id", 0)))
    for proposal_id, item in enumerate(retained, start=1):
        item.metadata["object_id"] = proposal_id
    return retained


def _track_iou(tracks: Sequence[np.ndarray | None], targets: Sequence[ObjectMask]) -> np.ndarray:
    output = np.zeros((len(tracks), len(targets)), dtype=np.float32)
    target_masks = [np.asarray(item.mask, dtype=bool) for item in targets]
    target_areas = np.asarray([mask.sum() for mask in target_masks], dtype=np.int64)
    for row, track in enumerate(tracks):
        if track is None:
            continue
        source = np.asarray(track, dtype=bool)
        source_area = int(source.sum())
        for column, target in enumerate(target_masks):
            intersection = int(np.logical_and(source, target).sum())
            union = source_area + int(target_areas[column]) - intersection
            if union:
                output[row, column] = intersection / union
    return output


def _bidirectional_track_score(
    forward: Sequence[np.ndarray | None],
    backward: Sequence[np.ndarray | None],
    source: Sequence[ObjectMask],
    target: Sequence[ObjectMask],
) -> tuple[np.ndarray, np.ndarray]:
    forward_iou = _track_iou(forward, target)
    backward_iou = _track_iou(backward, source).T
    return np.minimum(forward_iou, backward_iou), np.maximum(forward_iou, backward_iou)


def _feature_matrices(source: FrameInventory, target: FrameInventory) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sam = cosine_similarity_matrix(source.sam, target.sam)
    dino = cosine_similarity_matrix(source.dino, target.dino)
    valid = (
        source.sam.valid[:, None]
        & target.sam.valid[None, :]
        & source.dino.valid[:, None]
        & target.dino.valid[None, :]
    )
    return sam, dino, valid


def _areas(objects: Sequence[ObjectMask]) -> np.ndarray:
    return np.asarray([np.asarray(item.mask, bool).sum() for item in objects], np.float32)


def _object_point_clouds(objects: Sequence[ObjectMask], world_positions: np.ndarray | None) -> list[np.ndarray]:
    """Each object's own 3D points: its 2D mask applied to the shared
    per-frame position buffer, dropping unobserved (NaN) pixels. Empty
    arrays for objects with no observed 3D support at all (fully inside a
    render hole, e.g.) -- callers treat those as geometrically unresolvable,
    not as a match or a rejection."""
    if world_positions is None:
        return [np.empty((0, 3), dtype=np.float32) for _ in objects]
    clouds = []
    for item in objects:
        points = world_positions[np.asarray(item.mask, dtype=bool)]
        valid = np.isfinite(points).all(axis=1)
        clouds.append(points[valid])
    return clouds


def _point_set_overlap(a: np.ndarray, b: np.ndarray, radius: float) -> float:
    """Symmetric, conservative point-set overlap: the smaller of (fraction
    of a within radius of some point in b) and (fraction of b within radius
    of some point in a). A coarse Chamfer/IoU stand-in, cheap via KD-tree --
    exact point correspondence isn't the point, "occupies roughly the same
    3D region" is."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    from scipy.spatial import cKDTree

    dist_a, _ = cKDTree(b).query(a, k=1)
    dist_b, _ = cKDTree(a).query(b, k=1)
    return float(min((dist_a <= radius).mean(), (dist_b <= radius).mean()))


def _point_cloud_scale(points: np.ndarray) -> float:
    """Characteristic spatial extent of one point cloud -- the 90th-
    percentile radius around its own centroid, same computation as
    reconstruction._scene_scale but applied per-object rather than to the
    whole scene. Used to size a containment radius relative to *this
    particular object's* own size: a single large parent (a table, a wall)
    can easily span more than a small fixed fraction of the whole scene, so
    sizing the radius off the whole-scene scale wrongly fails genuine parts
    near a big parent's edges (see _suppress_feature_matched_parts)."""
    if points.size == 0:
        return 1.0
    centroid = points.mean(axis=0)
    radii = np.linalg.norm(points - centroid, axis=1)
    scale = float(np.percentile(radii, 90))
    return scale if scale > 1e-9 else 1.0


def _directional_containment(points: np.ndarray, reference: np.ndarray, radius: float) -> float:
    """Fraction of ``points`` within ``radius`` of the point cloud
    ``reference`` -- directional, unlike ``_point_set_overlap``'s symmetric
    measure. Used to test whether a small object's own points actually sit
    near a much larger reference object's surface; the reverse fraction
    would be near-meaningless here since most of a large object's points
    are always far from any one small part of it."""
    if len(points) == 0 or len(reference) == 0:
        return 0.0
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(reference).query(points, k=1)
    return float((distances <= radius).mean())


def _geometric_matrix(
    source_clouds: list[np.ndarray],
    target_clouds: list[np.ndarray],
    scene_scale: float,
    settings: "ThreeImageSettings",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Geometric-identity test: two proposals are the same physical object
    if their 3D point clouds share a centroid and actually overlap in space
    -- unlike 2D appearance cosine similarity, this needs no render/photo
    appearance bridge, since both point sets already live in one world
    frame. Thresholds are fractions of ``scene_scale`` (see
    reconstruction._scene_scale), not an absolute distance, so they hold
    across PASLCD scenes at different arbitrary reconstruction scales.

    Returns (geometric_ok, geometric_score, resolvable): ``resolvable[i,j]``
    is False when either object has no observed 3D points at all (a render
    hole) -- those pairs fall back to whatever other evidence is available,
    they are not treated as a geometric rejection.
    """
    n0, n1 = len(source_clouds), len(target_clouds)
    ok = np.zeros((n0, n1), dtype=bool)
    score = np.zeros((n0, n1), dtype=np.float32)
    resolvable = np.zeros((n0, n1), dtype=bool)
    centroid_radius = settings.geometric_centroid_fraction * scene_scale
    overlap_radius = settings.geometric_overlap_fraction * scene_scale

    source_centroids = [cloud.mean(axis=0) if len(cloud) else None for cloud in source_clouds]
    target_centroids = [cloud.mean(axis=0) if len(cloud) else None for cloud in target_clouds]

    for i, (sp, sc) in enumerate(zip(source_clouds, source_centroids)):
        if sc is None:
            continue
        for j, (tp, tc) in enumerate(zip(target_clouds, target_centroids)):
            if tc is None:
                continue
            resolvable[i, j] = True
            if float(np.linalg.norm(sc - tc)) > centroid_radius:
                continue
            overlap = _point_set_overlap(sp, tp, overlap_radius)
            score[i, j] = overlap
            ok[i, j] = overlap >= settings.minimum_geometric_overlap
    return ok, score, resolvable


def _reciprocal_with_margin(scores: np.ndarray, feasible: np.ndarray, margin: float) -> np.ndarray:
    accepted = np.zeros_like(feasible, dtype=bool)
    if not feasible.size or not feasible.any():
        return accepted
    masked = np.where(feasible, scores, -np.inf)
    row_best = np.argmax(masked, axis=1)
    column_best = np.argmax(masked, axis=0)
    for row, column in zip(*np.nonzero(feasible)):
        if row_best[row] != column or column_best[column] != row:
            continue
        row_values = np.delete(masked[row], column)
        column_values = np.delete(masked[:, column], row)
        row_second = float(row_values.max(initial=-1.0))
        column_second = float(column_values.max(initial=-1.0))
        if scores[row, column] - max(row_second, column_second) >= margin:
            accepted[row, column] = True
    return accepted


def _assign(scores: np.ndarray, feasible: np.ndarray) -> list[tuple[int, int]]:
    if not feasible.size or not feasible.any():
        return []
    rows, columns = linear_sum_assignment(np.where(feasible, -scores, 1e6))
    return [(int(row), int(column)) for row, column in zip(rows, columns) if feasible[row, column]]


def _identity_candidates(
    source: FrameInventory,
    target: FrameInventory,
    bidirectional_track: np.ndarray,
    any_track: np.ndarray,
    settings: ThreeImageSettings,
    scene_scale: float | None = None,
) -> tuple[list[tuple[int, int]], dict[str, np.ndarray]]:
    sam, dino, valid = _feature_matrices(source, target)
    source_area = _areas(source.objects)
    target_area = _areas(target.objects)
    ratio = target_area[None, :] / np.maximum(source_area[:, None], 1.0)
    identity = (
        valid
        & (sam >= settings.minimum_sam_cosine)
        & (dino >= settings.minimum_dino_cosine)
        & (ratio >= settings.area_ratio_low)
        & (ratio <= settings.area_ratio_high)
    )
    feature_score = np.minimum(sam, dino)
    reciprocal = _reciprocal_with_margin(feature_score, identity, settings.minimum_identity_margin)
    tracked = bidirectional_track >= settings.minimum_track_iou

    # Pass 1: identical to the pre-geometric gate/score. One-direction
    # tracks are useful for ranking, but cannot independently establish
    # identity. This pass is frozen exactly as it always was, so it
    # reproduces every match the old code already got right, whatever
    # happens in pass 2 below.
    old_feasible = identity & (tracked | reciprocal)
    score = feature_score + 0.20 * bidirectional_track + 0.05 * any_track
    matched = _assign(score, old_feasible)

    diagnostics = {
        "sam": sam,
        "dino": dino,
        "valid": valid,
        "identity": identity,
        "reciprocal": reciprocal,
        "tracked": tracked,
        "score": score,
    }

    # Geometric identity (3D point-cloud centroid + overlap): a fourth
    # signal, alongside identity/tracked/reciprocal, for objects with
    # observed 3D support -- needs no render/photo appearance bridge, unlike
    # sam/dino cosine similarity, which our own pixel-loss analysis found to
    # be the dominant failure mode on PASLCD (see docs/goldilocs_analysis.html).
    #
    # Two passes, not one shared assignment: _assign solves one global
    # optimum (Hungarian) over every object in the frame at once, so simply
    # adding geometric-only-feasible pairs into that SAME problem -- even
    # via a safe union with old_feasible, even with no geometric term in the
    # score -- can still reroute the global optimum and silently swap out a
    # match the old gate already had right elsewhere in the frame. Real
    # end-to-end validation (Cantina_Instance_1, n=6) hit this twice: a
    # mandatory-AND geometric gate dropped mean F1 12%/precision 20%, and a
    # follow-up union-gate variant (sharing pass 1's assignment problem)
    # still left one query worse than before geometric identity existed at
    # all, with an unchanged score -- proving it was the shared assignment
    # problem itself, not the gate logic or the score weighting. Running
    # geometric-assisted matching as its own separate assignment, restricted
    # to objects pass 1 left unmatched on both sides, makes pass 1 provably
    # unaffected: geometric can only rescue objects that were already going
    # to be reported as removed+added, never touch one pass 1 resolved.
    if not (
        settings.enable_geometric_identity
        and scene_scale is not None
        and source.world_positions is not None
        and target.world_positions is not None
    ):
        diagnostics.update(
            geometric=np.ones_like(identity, dtype=bool),
            geometric_score=np.zeros_like(sam),
            geometric_resolvable=np.zeros_like(identity, dtype=bool),
        )
        return matched, diagnostics

    source_clouds = _object_point_clouds(source.objects, source.world_positions)
    target_clouds = _object_point_clouds(target.objects, target.world_positions)
    geometric, geometric_score, resolvable = _geometric_matrix(source_clouds, target_clouds, scene_scale, settings)
    diagnostics.update(geometric=geometric, geometric_score=geometric_score, geometric_resolvable=resolvable)

    matched_rows = {row for row, _ in matched}
    matched_columns = {column for _, column in matched}
    leftover_rows = [row for row in range(len(source.objects)) if row not in matched_rows]
    leftover_columns = [column for column in range(len(target.objects)) if column not in matched_columns]
    if leftover_rows and leftover_columns:
        rows = np.asarray(leftover_rows)
        columns = np.asarray(leftover_columns)
        new_feasible = geometric & (identity | tracked | reciprocal)
        restricted_feasible = np.where(
            resolvable[np.ix_(rows, columns)],
            old_feasible[np.ix_(rows, columns)] | new_feasible[np.ix_(rows, columns)],
            old_feasible[np.ix_(rows, columns)],
        )
        restricted_score = score[np.ix_(rows, columns)]
        for local_row, local_column in _assign(restricted_score, restricted_feasible):
            matched.append((int(rows[local_row]), int(columns[local_column])))

    return matched, diagnostics


def _endpoint_map(bidirectional_track: np.ndarray, any_track: np.ndarray, minimum_iou: float) -> dict[int, int]:
    feasible = bidirectional_track >= minimum_iou
    score = bidirectional_track + 0.10 * any_track
    return {source: target for source, target in _assign(score, feasible)}


def _identity_at(
    sam: np.ndarray, dino: np.ndarray, valid: np.ndarray, row: int, column: int, settings: ThreeImageSettings
) -> bool:
    return bool(
        valid[row, column]
        and sam[row, column] >= settings.minimum_sam_cosine
        and dino[row, column] >= settings.minimum_dino_cosine
    )


def _visible_fraction(mask: np.ndarray, coverage: np.ndarray) -> float:
    """Fraction of ``mask`` lying on genuinely rendered (non-hole) pixels --
    direct port of GOLDILOCS's masks.visible_fraction (also SceneDiff's own
    binary visibility mask, arXiv 2512.16908: both reconstruct from a single
    best-matching pair and only ever have a binary rendered/not-rendered
    signal to filter on)."""
    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    return float(np.logical_and(mask, coverage).sum() / area) if area else 0.0


def _confidence_weighted_visible_fraction(mask: np.ndarray, confidence: np.ndarray) -> float:
    """Continuous generalization of ``_visible_fraction``: the mean of
    ``confidence`` over ``mask`` (NaN/uncovered pixels contribute 0, exactly
    reducing to ``_visible_fraction`` if confidence were a pure 0/1 signal).
    ``confidence`` is expected pre-normalized to roughly [0, 1] -- see
    reconstruction.reconstruct_and_render's confidence_scale.

    Unlike GOLDILOCS/SceneDiff's binary coverage, this can distinguish a
    mask barely-covered by low-confidence points from one solidly covered
    by points the multi-view reconstruction agrees on -- a distinction only
    possible because our reconstruction backbone jointly fuses many
    reference views (and so produces a genuine per-pixel multi-view
    confidence) rather than reconstructing from one best-matching pair."""
    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    if not area:
        return 0.0
    values = np.where(np.isnan(confidence), 0.0, confidence)
    return float(values[mask].mean())


def _corroboration_score(mask: np.ndarray, corroboration: np.ndarray) -> float:
    """Median per-pixel cross-reference-view corroboration count within
    ``mask`` (see reconstruction.ReconstructionResult.render_t0_corroboration):
    how many of the reference scene's own individually-rendered images agree
    with the aggregate render_t0 there. Median, not mean, so a handful of
    well- or poorly-corroborated edge pixels cannot dominate the verdict for
    an object whose interior is solidly one or the other."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0.0
    return float(np.median(corroboration[mask]))


def resolve_three_image_changes(
    t0: FrameInventory,
    clean: FrameInventory,
    t1: FrameInventory,
    tracks: TrackingEvidence,
    settings: ThreeImageSettings,
    scene_scale: float | None = None,
    render_t0_coverage: np.ndarray | None = None,
    render_t0_confidence: np.ndarray | None = None,
    render_t0_corroboration: np.ndarray | None = None,
) -> tuple[np.ndarray, list[ObjectMask], list[dict[str, Any]], dict[str, Any]]:
    """Resolve unchanged/moved/removed/added object states.

    ``scene_scale`` (see reconstruction._scene_scale) is required for the
    geometric-identity test to activate; omit it (or leave the three
    inventories' ``world_positions`` as None) to fall back to the original
    2D-appearance-only gate everywhere, unchanged.

    ``render_t0_coverage`` (render_t0's own per-pixel boolean coverage, same
    grid as the masks below) activates GOLDILOCS's (and SceneDiff's) binary
    visibility filter: a removed/added/moved decision mostly supported by
    unrendered pixels is dropped rather than reported as a change, since a
    render hole makes it genuinely unknown -- not evidence either way --
    whether something was there before. ``render_t0_confidence`` (render_t0's
    per-pixel multi-view depth confidence, normalized to ~[0, 1]) instead
    activates a continuous confidence-weighted version of the same filter
    when ``settings.use_confidence_weighted_visibility`` is set. Omit both
    to keep every decision, unfiltered.

    ``render_t0_corroboration`` (render_t0's per-pixel cross-reference-view
    agreement count) activates a second, independent filter when
    ``settings.enable_reference_corroboration`` is set: a removed/moved
    decision whose own t0-side geometry is corroborated by fewer than
    ``settings.minimum_corroborating_views`` of the reference scene's own
    individually-rendered images is dropped, the PASLCD-specific analogue of
    GOLDILOCS's cross-query-view majority vote (see module docstring).
    """

    shape_candidates = [item.mask.shape for frame in (t0, clean, t1) for item in frame.objects]
    if not shape_candidates:
        raise ValueError("at least one object proposal is required")
    shape = shape_candidates[0]
    if any(candidate != shape for candidate in shape_candidates):
        raise ValueError("all three object inventories must use one aligned grid")

    direct_bi, direct_any = _bidirectional_track_score(tracks.t0_to_t1, tracks.t1_to_t0, t0.objects, t1.objects)
    t0_clean_bi, t0_clean_any = _bidirectional_track_score(
        tracks.t0_to_clean, tracks.clean_to_t0, t0.objects, clean.objects
    )
    t1_clean_bi, t1_clean_any = _bidirectional_track_score(
        tracks.t1_to_clean, tracks.clean_to_t1, t1.objects, clean.objects
    )
    direct_pairs, direct_features = _identity_candidates(t0, t1, direct_bi, direct_any, settings, scene_scale)
    spatial = pairwise_mask_iou(t0.objects, t1.objects)
    # The clean point-cloud view can be incomplete, so requiring both track
    # directions here would discard the very bridge that is meant to recover
    # a render/photo domain gap. A one-way endpoint track is only a location
    # proposal; SAM3 and DINO still have to agree below before identity passes.
    t0_clean_map = _endpoint_map(t0_clean_any, t0_clean_any, settings.minimum_track_iou)
    t1_clean_map = _endpoint_map(t1_clean_any, t1_clean_any, settings.minimum_track_iou)
    clean_sam = cosine_similarity_matrix(clean.sam, clean.sam)
    clean_dino = cosine_similarity_matrix(clean.dino, clean.dino)
    clean_valid = clean.sam.valid[:, None] & clean.sam.valid[None, :] & clean.dino.valid[:, None] & clean.dino.valid[None, :]

    consumed_t0: set[int] = set()
    consumed_t1: set[int] = set()
    decisions: list[dict[str, Any]] = []
    output_objects: list[ObjectMask] = []

    def record_pair(source: int, target: int, decision: Label, evidence: str) -> None:
        consumed_t0.add(source)
        consumed_t1.add(target)
        clean_source = t0_clean_map.get(source)
        clean_target = t1_clean_map.get(target)
        clean_confirmed = clean_source is not None and clean_target is not None
        mask = np.logical_or(t0.objects[source].mask, t1.objects[target].mask)
        decisions.append(
            {
                "decision": decision.name.lower(),
                "t0_object_id": source + 1,
                "t1_object_id": target + 1,
                "clean_t0_object_id": None if clean_source is None else clean_source + 1,
                "clean_t1_object_id": None if clean_target is None else clean_target + 1,
                "clean_location_confirmed": clean_confirmed,
                "evidence": evidence,
                "spatial_iou": float(spatial[source, target]),
                "track_iou": float(direct_bi[source, target]),
                "sam_cosine": float(direct_features["sam"][source, target]),
                "dino_cosine": float(direct_features["dino"][source, target]),
            }
        )
        if decision != Label.UNCHANGED:
            output_objects.append(
                ObjectMask(
                    mask=mask,
                    score=float(min(t0.objects[source].score, t1.objects[target].score)),
                    label=decision,
                    source=f"three_image_{evidence}",
                    metadata={"t0_object_id": source + 1, "t1_object_id": target + 1},
                )
            )

    for source, target in direct_pairs:
        decision = Label.UNCHANGED if spatial[source, target] >= settings.same_location_iou else Label.MOVED
        record_pair(source, target, decision, "direct_identity")

    # A reliable track can survive a render/photo feature-domain gap. Validate
    # such a pair by comparing its two endpoints inside the common clean-render
    # feature domain.
    bridge_edges = np.zeros_like(direct_bi, dtype=bool)
    bridge_scores = np.zeros_like(direct_bi, dtype=np.float32)
    for source, target in zip(*np.nonzero(direct_any >= settings.minimum_track_iou)):
        if source in consumed_t0 or target in consumed_t1:
            continue
        clean_source = t0_clean_map.get(int(source))
        clean_target = t1_clean_map.get(int(target))
        if clean_source is None or clean_target is None:
            continue
        if not _identity_at(clean_sam, clean_dino, clean_valid, clean_source, clean_target, settings):
            continue
        bridge_edges[source, target] = True
        bridge_scores[source, target] = (
            min(clean_sam[clean_source, clean_target], clean_dino[clean_source, clean_target]) + direct_bi[source, target]
        )
    for source, target in _assign(bridge_scores, bridge_edges):
        decision = Label.UNCHANGED if spatial[source, target] >= settings.same_location_iou else Label.MOVED
        record_pair(source, target, decision, "clean_bridge_identity")

    for source, item in enumerate(t0.objects):
        if source in consumed_t0:
            continue
        output_objects.append(
            ObjectMask(mask=np.asarray(item.mask, bool).copy(), score=item.score, label=Label.REMOVED,
                       source="three_image_unmatched_t0", metadata={"t0_object_id": source + 1})
        )
        decisions.append({"decision": "removed", "t0_object_id": source + 1, "t1_object_id": None})
    for target, item in enumerate(t1.objects):
        if target in consumed_t1:
            continue
        output_objects.append(
            ObjectMask(mask=np.asarray(item.mask, bool).copy(), score=item.score, label=Label.ADDED,
                       source="three_image_unmatched_t1", metadata={"t1_object_id": target + 1})
        )
        decisions.append({"decision": "added", "t0_object_id": None, "t1_object_id": target + 1})

    use_confidence = settings.use_confidence_weighted_visibility and render_t0_confidence is not None
    visibility_filter_rejected = 0
    if settings.enable_visibility_filter and (render_t0_coverage is not None or use_confidence):
        decision_by_ids = {(row.get("t0_object_id"), row.get("t1_object_id")): row for row in decisions}
        retained_objects = []
        for item in output_objects:
            if use_confidence:
                support = _confidence_weighted_visible_fraction(item.mask, render_t0_confidence)
            else:
                support = _visible_fraction(item.mask, render_t0_coverage)
            item.metadata["render_support_fraction"] = support
            if support >= settings.minimum_render_support_fraction:
                retained_objects.append(item)
                continue
            visibility_filter_rejected += 1
            # Also demote the matching decisions row so
            # recover_unmatched_via_tracking (which reads decisions, not
            # output_objects, to find removed/added candidates) does not
            # try to recover an id this filter just rejected -- a render
            # hole makes the region's evidence unknown, not proof of
            # tracking failure, so it should stay dropped, not be retried.
            ids = (item.metadata.get("t0_object_id"), item.metadata.get("t1_object_id"))
            row = decision_by_ids.get(ids)
            if row is not None:
                row["decision"] = "visibility_filtered"
                row["render_support_fraction"] = support
        output_objects = retained_objects

    corroboration_rejected = 0
    if settings.enable_reference_corroboration and render_t0_corroboration is not None:
        decision_by_ids = {(row.get("t0_object_id"), row.get("t1_object_id")): row for row in decisions}
        retained_objects = []
        for item in output_objects:
            t0_object_id = item.metadata.get("t0_object_id")
            if t0_object_id is None:
                # Added has no t0-side geometry to corroborate by construction.
                retained_objects.append(item)
                continue
            source_mask = t0.objects[t0_object_id - 1].mask
            corroboration = _corroboration_score(source_mask, render_t0_corroboration)
            item.metadata["reference_corroboration"] = corroboration
            if corroboration >= settings.minimum_corroborating_views:
                retained_objects.append(item)
                continue
            corroboration_rejected += 1
            ids = (item.metadata.get("t0_object_id"), item.metadata.get("t1_object_id"))
            row = decision_by_ids.get(ids)
            if row is not None:
                row["decision"] = "low_corroboration"
                row["reference_corroboration"] = corroboration
        output_objects = retained_objects

    labels = np.zeros(shape, dtype=np.uint8)
    # Added is weakest where proposal masks overlap; same-identity motion is
    # the most specific object-level explanation.
    priority = (Label.ADDED, Label.REMOVED, Label.MOVED)
    for label_value in priority:
        for item in output_objects:
            if item.label == label_value:
                labels[np.asarray(item.mask, bool)] = int(label_value)
    # moved/removed/added counted from output_objects (post-visibility-filter)
    # rather than decisions, so this reflects what was actually rasterized;
    # unchanged is never filtered (see resolve_three_image_changes's
    # docstring), so it still comes from decisions directly.
    counts = {"unchanged": sum(row["decision"] == "unchanged" for row in decisions)}
    for label_value in (Label.MOVED, Label.REMOVED, Label.ADDED):
        counts[label_value.name.lower()] = sum(1 for item in output_objects if item.label == label_value)
    diagnostics = {
        "object_counts": {"render_t0": len(t0.objects), "clean_render": len(clean.objects), "image_t1": len(t1.objects)},
        "decision_counts": counts,
        "visibility_filter_rejected": visibility_filter_rejected,
        "corroboration_rejected": corroboration_rejected,
        "changed_pixel_fraction": float(np.mean(labels != int(Label.UNCHANGED))),
        "association_evidence": {
            "direct_bidirectional_track_iou": direct_bi.tolist(),
            "direct_any_direction_track_iou": direct_any.tolist(),
            "direct_sam_cosine": direct_features["sam"].tolist(),
            "direct_dino_cosine": direct_features["dino"].tolist(),
            "direct_spatial_iou": spatial.tolist(),
            "direct_geometric_score": direct_features["geometric_score"].tolist(),
            "direct_geometric_resolvable": direct_features["geometric_resolvable"].tolist(),
            "scene_scale": scene_scale,
            "t0_to_clean_bidirectional_track_iou": t0_clean_bi.tolist(),
            "t0_to_clean_any_direction_track_iou": t0_clean_any.tolist(),
            "t1_to_clean_bidirectional_track_iou": t1_clean_bi.tolist(),
            "t1_to_clean_any_direction_track_iou": t1_clean_any.tolist(),
        },
    }
    return labels, output_objects, decisions, diagnostics


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    union = int(np.logical_or(first, second).sum())
    if not union:
        return 0.0
    return int(np.logical_and(first, second).sum()) / union


def recover_unmatched_via_tracking(
    labels: np.ndarray,
    output_objects: list[ObjectMask],
    decisions: list[dict[str, Any]],
    tracks: TrackingEvidence,
    t0: FrameInventory,
    t1: FrameInventory,
    sam_t0_map: np.ndarray,
    dino_t0_map: np.ndarray,
    sam_t1_map: np.ndarray,
    dino_t1_map: np.ndarray,
    settings: ThreeImageSettings,
) -> tuple[np.ndarray, list[ObjectMask], list[dict[str, Any]], dict[str, Any]]:
    """Recover objects SAM3 only proposed in one frame, using the track SAM2
    already computed for every object instead of discarding it once an
    unmatched object is about to be declared removed/added.

    ``resolve_three_image_changes`` only ever pairs objects that both have an
    independent SAM3 proposal in their own frame; ``tracks.t0_to_t1[i]`` (a
    real mask propagated into image_t1 by Sam2MaskTracker) is computed for
    every t0 object regardless, but is silently dropped for any object that
    ends up unmatched. This revisits exactly those "removed"/"added" verdicts:
    if the existing track for an unmatched object is accepted, and pooling
    the target frame's own dense SAM3/DINOv2 feature maps under that tracked
    mask still agrees with the source object's descriptor at the same
    thresholds used everywhere else in this module, the proposal-generation
    miss is not evidence of a real change.
    """
    removed_t0_ids = {row["t0_object_id"] - 1 for row in decisions if row["decision"] == "removed"}
    added_t1_ids = {row["t1_object_id"] - 1 for row in decisions if row["decision"] == "added"}
    object_index_by_removed_t0 = {
        item.metadata["t0_object_id"] - 1: index for index, item in enumerate(output_objects) if item.label == Label.REMOVED
    }
    object_index_by_added_t1 = {
        item.metadata["t1_object_id"] - 1: index for index, item in enumerate(output_objects) if item.label == Label.ADDED
    }

    def area_ratio_ok(source_area: float, candidate_area: float) -> bool:
        ratio = candidate_area / max(source_area, 1.0)
        return settings.area_ratio_low <= ratio <= settings.area_ratio_high

    def try_recover(
        source_id: int,
        candidate_mask: np.ndarray | None,
        source_inventory: FrameInventory,
        candidate_map_sam: np.ndarray,
        candidate_map_dino: np.ndarray,
    ) -> tuple[float, float, float] | None:
        if candidate_mask is None:
            return None
        if not source_inventory.sam.valid[source_id] or not source_inventory.dino.valid[source_id]:
            return None
        candidate = ObjectMask(mask=candidate_mask)
        cand_sam = mask_descriptors(candidate_map_sam, [candidate], minimum_feature_cells=settings.minimum_sam_feature_cells)
        cand_dino = mask_descriptors(candidate_map_dino, [candidate], minimum_feature_cells=settings.minimum_dino_feature_cells)
        if not (cand_sam.valid[0] and cand_dino.valid[0]):
            return None
        sam_cos = float(source_inventory.sam.vectors[source_id] @ cand_sam.vectors[0])
        dino_cos = float(source_inventory.dino.vectors[source_id] @ cand_dino.vectors[0])
        if sam_cos < settings.minimum_sam_cosine or dino_cos < settings.minimum_dino_cosine:
            return None
        source_area = float(np.asarray(source_inventory.objects[source_id].mask, bool).sum())
        if not area_ratio_ok(source_area, float(np.asarray(candidate_mask, bool).sum())):
            return None
        return sam_cos, dino_cos, float(candidate_mask.sum())

    recovered_decisions: list[dict[str, Any]] = []
    dropped_object_indices: set[int] = set()
    recovered_t0_ids: set[int] = set()
    recovered_t1_ids: set[int] = set()

    for t0_id in sorted(removed_t0_ids):
        result = try_recover(t0_id, tracks.t0_to_t1[t0_id], t0, sam_t1_map, dino_t1_map)
        if result is None:
            continue
        sam_cos, dino_cos, _ = result
        tracked_mask = np.asarray(tracks.t0_to_t1[t0_id], dtype=bool)
        source_mask = np.asarray(t0.objects[t0_id].mask, dtype=bool)
        absorbed_t1 = None
        for t1_id in sorted(added_t1_ids - recovered_t1_ids):
            if _mask_iou(np.asarray(t1.objects[t1_id].mask, bool), tracked_mask) >= settings.same_location_iou:
                absorbed_t1 = t1_id
                break
        recovered_t0_ids.add(t0_id)
        dropped_object_indices.add(object_index_by_removed_t0[t0_id])
        if absorbed_t1 is not None:
            recovered_t1_ids.add(absorbed_t1)
            dropped_object_indices.add(object_index_by_added_t1[absorbed_t1])
            evidence_mask = np.logical_or(source_mask, np.asarray(t1.objects[absorbed_t1].mask, bool))
        else:
            evidence_mask = np.logical_or(source_mask, tracked_mask)
        location_iou = _mask_iou(source_mask, tracked_mask)
        new_label = Label.UNCHANGED if location_iou >= settings.same_location_iou else Label.MOVED
        recovered_decisions.append(
            {
                "decision": new_label.name.lower(),
                "t0_object_id": t0_id + 1,
                "t1_object_id": (absorbed_t1 + 1) if absorbed_t1 is not None else None,
                "evidence": "tracking_recovery_t0_to_t1",
                "track_iou": location_iou,
                "sam_cosine": sam_cos,
                "dino_cosine": dino_cos,
            }
        )
        if new_label != Label.UNCHANGED:
            output_objects.append(
                ObjectMask(
                    mask=evidence_mask,
                    score=float(t0.objects[t0_id].score),
                    label=new_label,
                    source="three_image_tracking_recovery",
                    metadata={"t0_object_id": t0_id + 1, "t1_object_id": (absorbed_t1 + 1) if absorbed_t1 is not None else None},
                )
            )

    for t1_id in sorted(added_t1_ids - recovered_t1_ids):
        result = try_recover(t1_id, tracks.t1_to_t0[t1_id], t1, sam_t0_map, dino_t0_map)
        if result is None:
            continue
        sam_cos, dino_cos, _ = result
        tracked_mask = np.asarray(tracks.t1_to_t0[t1_id], dtype=bool)
        target_mask = np.asarray(t1.objects[t1_id].mask, dtype=bool)
        absorbed_t0 = None
        for t0_id in sorted(removed_t0_ids - recovered_t0_ids):
            if _mask_iou(np.asarray(t0.objects[t0_id].mask, bool), tracked_mask) >= settings.same_location_iou:
                absorbed_t0 = t0_id
                break
        recovered_t1_ids.add(t1_id)
        dropped_object_indices.add(object_index_by_added_t1[t1_id])
        if absorbed_t0 is not None:
            recovered_t0_ids.add(absorbed_t0)
            dropped_object_indices.add(object_index_by_removed_t0[absorbed_t0])
            evidence_mask = np.logical_or(target_mask, np.asarray(t0.objects[absorbed_t0].mask, bool))
        else:
            evidence_mask = np.logical_or(target_mask, tracked_mask)
        location_iou = _mask_iou(target_mask, tracked_mask)
        new_label = Label.UNCHANGED if location_iou >= settings.same_location_iou else Label.MOVED
        recovered_decisions.append(
            {
                "decision": new_label.name.lower(),
                "t0_object_id": (absorbed_t0 + 1) if absorbed_t0 is not None else None,
                "t1_object_id": t1_id + 1,
                "evidence": "tracking_recovery_t1_to_t0",
                "track_iou": location_iou,
                "sam_cosine": sam_cos,
                "dino_cosine": dino_cos,
            }
        )
        if new_label != Label.UNCHANGED:
            output_objects.append(
                ObjectMask(
                    mask=evidence_mask,
                    score=float(t1.objects[t1_id].score),
                    label=new_label,
                    source="three_image_tracking_recovery",
                    metadata={"t0_object_id": (absorbed_t0 + 1) if absorbed_t0 is not None else None, "t1_object_id": t1_id + 1},
                )
            )

    kept_decisions = [
        row
        for row in decisions
        if not (
            (row["decision"] == "removed" and row["t0_object_id"] - 1 in recovered_t0_ids)
            or (row["decision"] == "added" and row["t1_object_id"] - 1 in recovered_t1_ids)
        )
    ]
    final_decisions = kept_decisions + recovered_decisions
    final_objects = [item for index, item in enumerate(output_objects) if index not in dropped_object_indices]

    new_labels = np.zeros(labels.shape, dtype=np.uint8)
    priority = (Label.ADDED, Label.REMOVED, Label.MOVED)
    for label_value in priority:
        for item in final_objects:
            if item.label == label_value:
                new_labels[np.asarray(item.mask, bool)] = int(label_value)

    counts = {
        name: sum(row["decision"] == name for row in final_decisions)
        for name in ("unchanged", "moved", "removed", "added")
    }
    recovery_diagnostics = {
        "decision_counts": counts,
        "changed_pixel_fraction": float(np.mean(new_labels != int(Label.UNCHANGED))),
        "tracking_recoveries": len(recovered_decisions),
        "tracking_recovered_t0_ids": sorted(i + 1 for i in recovered_t0_ids),
        "tracking_recovered_t1_ids": sorted(i + 1 for i in recovered_t1_ids),
    }
    return new_labels, final_objects, final_decisions, recovery_diagnostics


def _track_batches(
    tracker: Sam2MaskTracker, objects: Sequence[ObjectMask], source_image: np.ndarray, target_image: np.ndarray, batch_size: int
) -> tuple[np.ndarray | None, ...]:
    results: list[np.ndarray | None] = []
    for start in range(0, len(objects), batch_size):
        batch = objects[start : start + batch_size]
        attempts = tracker.track([np.asarray(item.mask, bool) for item in batch], source_image, target_image)
        results.extend(np.asarray(attempt.mask, bool).copy() if attempt.accepted else None for attempt in attempts)
    return tuple(results)


def _build_inventory(
    objects: Sequence[ObjectMask],
    sam_map: np.ndarray,
    dino_map: np.ndarray,
    settings: ThreeImageSettings,
    world_positions: np.ndarray | None = None,
) -> FrameInventory:
    return FrameInventory(
        objects=tuple(objects),
        sam=mask_descriptors(sam_map, objects, minimum_feature_cells=settings.minimum_sam_feature_cells),
        dino=mask_descriptors(dino_map, objects, minimum_feature_cells=settings.minimum_dino_feature_cells),
        world_positions=world_positions,
    )


def _suppress_feature_matched_parts(
    inventory: FrameInventory, settings: ThreeImageSettings, scene_scale: float | None = None
) -> FrameInventory:
    """Drop a small disjoint part inside a larger same-identity object box."""

    if len(inventory.objects) < 2:
        return inventory
    sam = cosine_similarity_matrix(inventory.sam, inventory.sam)
    dino = cosine_similarity_matrix(inventory.dino, inventory.dino)
    areas = _areas(inventory.objects)
    order = sorted(range(len(inventory.objects)), key=lambda index: -areas[index])

    # 2D bbox-containment and mask adjacency are both coincidental at times:
    # a genuinely separate object at a different depth can still project
    # inside the parent's 2D bbox and touch its silhouette (parallax /
    # occlusion), and appearance similarity alone can't tell a real part
    # from an unrelated same-class object nearby. Where 3D data is
    # available, additionally require most of the *candidate part's* own
    # points to actually sit near the parent's point cloud before
    # suppressing it -- directional containment, not the symmetric overlap
    # _identity_candidates uses for cross-frame matching, since the parent
    # is always much larger here (part_maximum_area_ratio below already
    # guarantees that), and a symmetric test would almost always read as
    # "no overlap" simply because most of a large object's points are far
    # from any one small part. Falls back to the original 2D-only decision
    # wherever 3D data isn't resolvable for a given pair, same convention
    # as _identity_candidates.
    #
    # The containment radius is sized off the PARENT's own extent
    # (_point_cloud_scale), not the whole scene's scale: real end-to-end
    # validation (19 queries) showed that reusing scene_scale regressed
    # mean F1 4% (13/19 queries worse) because a single large parent object
    # -- a table, a wall -- routinely spans more than a small fixed
    # fraction of the *scene's* extent, so genuine parts near its edges
    # failed containment against a scene-sized radius and wrongly stopped
    # being suppressed, letting SAM3's own over-segmentation back in.
    geometry_active = settings.enable_geometric_identity and scene_scale is not None and inventory.world_positions is not None
    clouds = _object_point_clouds(inventory.objects, inventory.world_positions) if geometry_active else None

    retained: list[int] = []
    margin = settings.part_bbox_margin_pixels
    for index in order:
        mask = np.asarray(inventory.objects[index].mask, dtype=bool)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        center_x, center_y = float(xs.mean()), float(ys.mean())
        is_part = False
        for parent in retained:
            ratio = float(areas[index] / max(areas[parent], 1.0))
            if ratio > settings.part_maximum_area_ratio:
                continue
            parent_mask = np.asarray(inventory.objects[parent].mask, dtype=bool)
            parent_y, parent_x = np.nonzero(parent_mask)
            if not len(parent_x):
                continue
            inside_box = (
                parent_x.min() - margin <= center_x <= parent_x.max() + margin
                and parent_y.min() - margin <= center_y <= parent_y.max() + margin
            )
            features_match = (
                inventory.sam.valid[index]
                and inventory.sam.valid[parent]
                and inventory.dino.valid[index]
                and inventory.dino.valid[parent]
                and sam[index, parent] >= settings.minimum_sam_cosine
                and dino[index, parent] >= settings.minimum_dino_cosine
            )
            adjacent = bool(np.logical_and(mask, binary_dilation(parent_mask, iterations=max(margin, 1))).any())
            geometric_confirms = True
            if geometry_active:
                part_points, parent_points = clouds[index], clouds[parent]
                if len(part_points) and len(parent_points):
                    containment_radius = settings.geometric_overlap_fraction * _point_cloud_scale(parent_points)
                    containment = _directional_containment(part_points, parent_points, containment_radius)
                    geometric_confirms = containment >= settings.minimum_geometric_overlap
            if inside_box and (features_match or adjacent) and geometric_confirms:
                is_part = True
                break
        if not is_part:
            retained.append(index)
    retained.sort()

    def subset(batch: FeatureDescriptorBatch) -> FeatureDescriptorBatch:
        indices = np.asarray(retained, dtype=int)
        return FeatureDescriptorBatch(
            vectors=batch.vectors[indices], valid=batch.valid[indices], effective_cells=batch.effective_cells[indices]
        )

    return FrameInventory(
        objects=tuple(inventory.objects[index] for index in retained), sam=subset(inventory.sam), dino=subset(inventory.dino),
        world_positions=inventory.world_positions,
    )


def _proposal_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["sam3_proposals"]["proposals"]
    return {
        "points_per_side": int(cfg["points_per_side"]),
        "points_per_batch": int(cfg["points_per_batch"]),
        "pred_iou_threshold": float(cfg["pred_iou_threshold"]),
        "stability_threshold": float(cfg["stability_threshold"]),
        "stability_offset": float(cfg["stability_offset"]),
        "crop_layers": int(cfg["crop_layers"]),
        "crop_downscale_factor": int(cfg["crop_downscale_factor"]),
        "box_nms_threshold": float(cfg["box_nms_threshold"]),
        "crop_nms_threshold": float(cfg["crop_nms_threshold"]),
        "minimum_mask_area": int(cfg["minimum_mask_area"]),
        "multimask_output": bool(cfg["multimask_output"]),
    }


def find_color_replacement_regions(
    render_t0: np.ndarray,
    image_t1: np.ndarray,
    render_t0_coverage: np.ndarray | None,
    already_changed: np.ndarray,
    settings: ThreeImageSettings,
    render_t0_confidence: np.ndarray | None = None,
) -> list[ObjectMask]:
    """Final, separate pass finding same-footprint object replacements a
    single-proposal-level SAM3+DINO decision pipeline structurally cannot:
    the replaced object is frequently absorbed into a much larger
    surrounding surface's proposal (e.g. a whole tabletop) in both frames, so
    no per-object comparison -- appearance, color, or otherwise -- ever runs
    on it. Operates directly on render_t0/image_t1, restricted to pixels
    ``already_changed`` does not already cover, so it can only add new
    REPLACED detections, never perturb or override an existing one. See
    color_residual.py's module docstring for the mechanism and its validated
    example, and this setting's own comment in ThreeImageSettings.
    """
    if not settings.enable_color_replacement_detection or render_t0_coverage is None:
        return []
    candidate = color_replacement_candidate_mask(
        render_t0,
        image_t1,
        render_t0_coverage,
        settings.color_replacement_minimum_colorfulness,
        settings.color_replacement_residual_percentile,
    )
    novel = candidate & ~np.asarray(already_changed, dtype=bool)
    if not novel.any():
        return []
    components, count = connected_components(novel)
    sizes = np.bincount(components.ravel())
    objects = []
    for component_id in range(1, count + 1):
        if sizes[component_id] < settings.minimum_color_replacement_area:
            continue
        component_mask = components == component_id
        if render_t0_confidence is not None:
            confidence = _confidence_weighted_visible_fraction(component_mask, render_t0_confidence)
            if confidence < settings.color_replacement_minimum_confidence:
                continue
        objects.append(
            ObjectMask(mask=component_mask, score=1.0, label=Label.REPLACED, source="color_residual")
        )
    return objects


def run_object_state_resolution(
    render_t0: np.ndarray,
    clean_render: np.ndarray,
    image_t1: np.ndarray,
    output_dir: str | Path,
    config: dict[str, Any],
    generator: "Sam3AutomaticMaskGenerator | None" = None,
    dino_extractor: "Dinov2FeatureExtractor | None" = None,
    tracker: "Sam2MaskTracker | None" = None,
    render_t0_positions: np.ndarray | None = None,
    clean_render_positions: np.ndarray | None = None,
    image_t1_positions: np.ndarray | None = None,
    scene_scale: float | None = None,
    render_t0_coverage: np.ndarray | None = None,
    render_t0_confidence: np.ndarray | None = None,
    render_t0_corroboration: np.ndarray | None = None,
) -> ThreeImageResult:
    """Run SAM3 + DINOv2 + SAM2-tracking inference over three aligned images
    and write a T1-aligned change-detection result.

    ``render_t0``/``clean_render``/``image_t1`` must already be produced by
    ``reconstruction.py`` (optionally refined by ``refine.py``) and share one
    pixel grid.

    ``generator``/``dino_extractor``/``tracker``, if supplied, are used
    as-is and left alive for the caller to reuse across multiple calls
    (e.g. every query image in one PASLCD scene instance) instead of paying
    each model's load/compile cost -- observed at ~150s combined -- on every
    single call. If omitted (the default, used by every existing caller),
    behavior is unchanged: each is built fresh here and released before
    returning.

    ``render_t0_positions``/``clean_render_positions``/``image_t1_positions``
    (each an (H, W, 3) per-pixel world position, NaN where unobserved -- see
    ``reconstruction.ReconstructionResult``) and ``scene_scale`` activate
    geometric reasoning in two places: the identity test in
    ``resolve_three_image_changes`` for the direct (render_t0-vs-image_t1)
    association pass, and ``_suppress_feature_matched_parts``'s same-frame
    part/parent containment check for each of the three inventories below.
    Omit all four (the default) to keep the original 2D-appearance-only
    behavior unchanged in both places -- e.g. for SceneDiff callers that
    don't have 3D position data to offer.

    ``render_t0_coverage`` (render_t0's own per-pixel boolean coverage --
    see ``reconstruction.ReconstructionResult``) activates the binary
    visibility filter in ``resolve_three_image_changes``; ``render_t0_confidence``
    (render_t0's per-pixel multi-view depth confidence) instead activates
    its continuous, confidence-weighted form when
    ``settings.use_confidence_weighted_visibility`` is set. Both are
    independent of the geometric settings above, and independently
    omittable.

    ``render_t0_corroboration`` (render_t0's per-pixel cross-reference-view
    agreement count) activates the reference-corroboration filter when
    ``settings.enable_reference_corroboration`` is set -- see
    ``resolve_three_image_changes``'s docstring.

    ``render_t0_coverage`` also activates the color-replacement pass (see
    ``find_color_replacement_regions``) when
    ``settings.enable_color_replacement_detection`` is set -- a final,
    separate step that can only add new REPLACED detections in pixels no
    earlier stage already explained, never touch an existing decision.
    ``render_t0_confidence``, if also supplied, additionally suppresses
    candidates sitting on geometry the reconstruction itself was not
    confident about.
    """

    images = tuple(np.asarray(image, dtype=np.uint8) for image in (render_t0, clean_render, image_t1))
    if any(image.ndim != 3 or image.shape[2] != 3 for image in images):
        raise ValueError("three-image inputs must be H x W x 3 RGB arrays")
    if len({image.shape for image in images}) != 1:
        raise ValueError("three-image inputs must already share one aligned pixel grid")
    render_t0, clean_render, image_t1 = images
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = ThreeImageSettings.from_config(config)
    timings: dict[str, float] = {}

    started = time.perf_counter()
    sam_cfg = config["sam3_proposals"]
    owns_generator = generator is None
    if generator is None:
        generator = Sam3AutomaticMaskGenerator(sam_cfg["sam3_image_checkpoint"], source=sam_cfg["sam3_source"], **_proposal_kwargs(config))
    try:
        raw_t0, sam_t0 = generator.generate_with_feature_map(render_t0)
        raw_clean, sam_clean = generator.generate_with_feature_map(clean_render)
        raw_t1, sam_t1 = generator.generate_with_feature_map(image_t1)
    finally:
        if owns_generator:
            generator.release()
    objects_t0 = select_object_proposals(raw_t0, settings)
    objects_clean = select_object_proposals(raw_clean, settings)
    objects_t1 = select_object_proposals(raw_t1, settings)
    timings["01_sam3_inventory_and_features"] = time.perf_counter() - started

    started = time.perf_counter()
    owns_dino = dino_extractor is None
    if dino_extractor is None:
        dino_extractor = Dinov2FeatureExtractor(config["dinov2_features"])
    try:
        dino_t0 = dino_extractor.feature_map(render_t0)
        dino_clean = dino_extractor.feature_map(clean_render)
        dino_t1 = dino_extractor.feature_map(image_t1)
    finally:
        if owns_dino:
            dino_extractor.release()
    inventory_t0 = _suppress_feature_matched_parts(
        _build_inventory(objects_t0, sam_t0, dino_t0, settings, render_t0_positions), settings, scene_scale
    )
    inventory_clean = _suppress_feature_matched_parts(
        _build_inventory(objects_clean, sam_clean, dino_clean, settings, clean_render_positions), settings, scene_scale
    )
    inventory_t1 = _suppress_feature_matched_parts(
        _build_inventory(objects_t1, sam_t1, dino_t1, settings, image_t1_positions), settings, scene_scale
    )
    objects_t0 = list(inventory_t0.objects)
    objects_clean = list(inventory_clean.objects)
    objects_t1 = list(inventory_t1.objects)
    timings["02_dinov2_and_pooling"] = time.perf_counter() - started

    started = time.perf_counter()
    owns_tracker = tracker is None
    if tracker is None:
        tracker = Sam2MaskTracker(config["sam2_tracking"])
    try:
        track = lambda objects, source, target: _track_batches(tracker, objects, source, target, settings.tracking_batch_size)  # noqa: E731
        tracking = TrackingEvidence(
            t0_to_t1=track(objects_t0, render_t0, image_t1),
            t1_to_t0=track(objects_t1, image_t1, render_t0),
            t0_to_clean=track(objects_t0, render_t0, clean_render),
            clean_to_t0=track(objects_clean, clean_render, render_t0),
            t1_to_clean=track(objects_t1, image_t1, clean_render),
            clean_to_t1=track(objects_clean, clean_render, image_t1),
        )
    finally:
        if owns_tracker:
            tracker.release()
    timings["03_bidirectional_tracking"] = time.perf_counter() - started

    started = time.perf_counter()
    labels, objects, decisions, diagnostics = resolve_three_image_changes(
        inventory_t0, inventory_clean, inventory_t1, tracking, settings, scene_scale,
        render_t0_coverage, render_t0_confidence, render_t0_corroboration,
    )
    timings["04_object_state_resolution"] = time.perf_counter() - started

    if settings.recover_unmatched_via_tracking:
        started = time.perf_counter()
        labels, objects, decisions, recovery_diagnostics = recover_unmatched_via_tracking(
            labels, objects, decisions, tracking, inventory_t0, inventory_t1, sam_t0, dino_t0, sam_t1, dino_t1, settings
        )
        timings["05_tracking_recovery"] = time.perf_counter() - started
        diagnostics = {
            **diagnostics,
            "decision_counts": recovery_diagnostics["decision_counts"],
            "changed_pixel_fraction": recovery_diagnostics["changed_pixel_fraction"],
            "tracking_recovery": recovery_diagnostics,
        }

    started = time.perf_counter()
    already_changed = labels != int(Label.UNCHANGED)
    replaced_objects = find_color_replacement_regions(
        render_t0, image_t1, render_t0_coverage, already_changed, settings, render_t0_confidence
    )
    timings["06_color_replacement_detection"] = time.perf_counter() - started
    if replaced_objects:
        objects = list(objects) + replaced_objects
        for item in replaced_objects:
            labels[item.mask] = int(Label.REPLACED)
            decisions = list(decisions) + [
                {
                    "decision": "replaced",
                    "t0_object_id": None,
                    "t1_object_id": None,
                    "evidence": "color_residual",
                    "area": int(item.mask.sum()),
                }
            ]
        diagnostics = {
            **diagnostics,
            "decision_counts": {**diagnostics["decision_counts"], "replaced": len(replaced_objects)},
            "changed_pixel_fraction": float(np.mean(labels != int(Label.UNCHANGED))),
        }

    save_image(output_dir / "render_t0.png", render_t0)
    save_image(output_dir / "clean_render.png", clean_render)
    save_image(output_dir / "target.png", image_t1)
    save_image(output_dir / "labels.png", labels)
    save_image(output_dir / "labels_color.png", colorize(labels))
    save_image(output_dir / "overlay.png", overlay(image_t1, labels))
    save_image(output_dir / "objects_t0.png", instance_overlay(render_t0, objects_t0))
    save_image(output_dir / "objects_clean.png", instance_overlay(clean_render, objects_clean))
    save_image(output_dir / "objects_t1.png", instance_overlay(image_t1, objects_t1))

    # Pre-select_object_proposals raw SAM3 automatic-mask inventory, for
    # inspecting what select_object_proposals's dedup/filtering removed.
    def _raw_overlay(image: np.ndarray, raw: Sequence[Sam3Proposal]) -> np.ndarray:
        return instance_overlay(image, [ObjectMask(mask=np.asarray(p.mask, dtype=bool)) for p in raw])

    save_image(output_dir / "objects_t0_raw.png", _raw_overlay(render_t0, raw_t0))
    save_image(output_dir / "objects_clean_raw.png", _raw_overlay(clean_render, raw_clean))
    save_image(output_dir / "objects_t1_raw.png", _raw_overlay(image_t1, raw_t1))
    diagnostics = {**diagnostics, "settings": asdict(settings), "timings": timings, "decisions": decisions}
    save_json(output_dir / "inference.json", diagnostics)
    return ThreeImageResult(
        labels=labels, objects=tuple(objects), decisions=tuple(decisions), diagnostics=diagnostics,
        artifacts_dir=output_dir, timings=timings,
    )
