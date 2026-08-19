from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .cache import load_reconstruction
from .changesim import load_manifest, normalize_target
from .geometry import render_points
from .types import Label, Reconstruction

# GOLDILOCS SAM2 proposal/tracking audit (outputs/sam-audit-paper-baseline/AUDIT.md)
# found this contamination by hand, on one pair, and never persisted the
# measurement as code. This module turns it into a reusable, scriptable one.
# Ground truth is read only to report statistics; it never influences keep0/
# keep1 or any prediction.


def _resize_rgb(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Match pipeline.py's own resize exactly, so indexing stays aligned."""
    height, width = shape
    return np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS))


def measure_pair_provenance(
    reconstruction: Reconstruction,
    keep0: np.ndarray,
    keep1: np.ndarray,
    geometry_config: dict,
    target: np.ndarray | None = None,
) -> dict:
    """Measure what fraction of the canonical render's covered pixels are
    won by T1 (P1) rather than T0 (P0) geometry -- the direct, quantified
    version of the audit's "R*,1 retains T1 surfaces" finding.

    ``render_points`` already keeps only the nearest surviving point at each
    destination pixel and accepts arbitrary per-point colors; passing colors
    that simply encode "which reconstruction this point came from" turns
    that existing, completely unmodified renderer into a provenance
    measurement -- no changes to render_points or canonical_cloud needed.
    """
    points0, points1 = reconstruction.points
    height, width = points0.shape[:2]
    image0 = _resize_rgb(reconstruction.images[0], (height, width))
    image1 = _resize_rgb(reconstruction.images[1], (height, width))

    p0_points, p1_points = points0[keep0], points1[keep1]
    p0_labels = np.zeros((len(p0_points), 3), dtype=np.uint8)
    p1_labels = np.full((len(p1_points), 3), 255, dtype=np.uint8)
    label_image, _, covered = render_points(
        np.concatenate([p0_points, p1_points]),
        np.concatenate([p0_labels, p1_labels]),
        reconstruction.intrinsics[1],
        reconstruction.world_to_camera[1],
        (height, width),
        z_epsilon=geometry_config["z_buffer_epsilon"],
    )
    # A second render with the real appearance colors lets the "is this
    # pixel identical to I1" corroborating check use exact RGB values.
    clean1, _, _ = render_points(
        np.concatenate([p0_points, p1_points]),
        np.concatenate([image0[keep0], image1[keep1]]),
        reconstruction.intrinsics[1],
        reconstruction.world_to_camera[1],
        (height, width),
        z_epsilon=geometry_config["z_buffer_epsilon"],
    )

    from_p1 = covered & (label_image[..., 0] == 255)
    result: dict = {
        "covered_pixels": int(covered.sum()),
        "p1_share_overall": float(from_p1.sum() / covered.sum()) if covered.any() else None,
    }
    if target is not None:
        for label in (Label.ADDED, Label.REMOVED, Label.MOVED, Label.REPLACED):
            region = covered & (target == int(label))
            if not region.any():
                continue
            name = label.name.lower()
            result[f"pixels_{name}"] = int(region.sum())
            result[f"p1_share_within_{name}"] = float(from_p1[region].sum() / region.sum())
            result[f"p1_identical_to_i1_within_{name}"] = float(
                np.mean(np.all(clean1[region] == image1[region], axis=-1))
            )
    return result


def _weighted_mean(values: list[tuple[float, int]]) -> float | None:
    total_weight = sum(weight for _, weight in values)
    if not total_weight:
        return None
    return sum(value * weight for value, weight in values) / total_weight


def _summarize(per_pair: list[dict]) -> dict:
    """Pixel-weighted aggregate of every per-pair statistic across pairs."""
    summary: dict = {"pairs_measured": len(per_pair)}
    summary["p1_share_overall"] = _weighted_mean(
        [(p["p1_share_overall"], p["covered_pixels"]) for p in per_pair if p["p1_share_overall"] is not None]
    )
    for label in (Label.ADDED, Label.REMOVED, Label.MOVED, Label.REPLACED):
        name = label.name.lower()
        pixel_key = f"pixels_{name}"
        contributing = [p for p in per_pair if pixel_key in p]
        if not contributing:
            continue
        summary[f"pairs_with_{name}_support"] = len(contributing)
        summary[f"p1_share_within_{name}"] = _weighted_mean(
            [(p[f"p1_share_within_{name}"], p[pixel_key]) for p in contributing]
        )
        summary[f"p1_identical_to_i1_within_{name}"] = _weighted_mean(
            [(p[f"p1_identical_to_i1_within_{name}"], p[pixel_key]) for p in contributing]
        )
    return summary


def measure_evaluation_provenance(evaluation_dir: str | Path, manifest_path: str | Path) -> dict:
    """Measure provenance for every full-artifact pair in an evaluation run.

    Requires ``--artifact-level full`` (needs cached reconstruction.npz and
    geometry.npz); pairs saved at a lighter artifact level are skipped.
    """
    evaluation_dir = Path(evaluation_dir)
    manifest = {pair.pair_id: pair for pair in load_manifest(manifest_path)}
    evaluation = json.loads((evaluation_dir / "report.json").read_text(encoding="utf-8"))

    per_pair = []
    skipped = []
    for pair_result in evaluation["pairs"]:
        pair_id = pair_result["id"]
        if pair_id not in manifest or "artifacts" not in pair_result:
            skipped.append(pair_id)
            continue
        artifact_dir = Path(pair_result["artifacts"])
        reconstruction_path = artifact_dir / "reconstruction.npz"
        geometry_path = artifact_dir / "geometry.npz"
        config_path = artifact_dir / "config.json"
        if not (reconstruction_path.exists() and geometry_path.exists() and config_path.exists()):
            skipped.append(pair_id)
            continue

        reconstruction = load_reconstruction(reconstruction_path)
        with np.load(geometry_path) as geometry:
            keep0, keep1 = geometry["keep0"], geometry["keep1"]
        config = json.loads(config_path.read_text(encoding="utf-8"))

        pair = manifest[pair_id]
        target = normalize_target(pair.target)
        height, width = reconstruction.points[0].shape[:2]
        if target.shape != (height, width):
            target = np.asarray(
                Image.fromarray(target).resize((width, height), Image.Resampling.NEAREST)
            )

        stats = measure_pair_provenance(reconstruction, keep0, keep1, config["geometry"], target)
        stats["id"] = pair_id
        per_pair.append(stats)

    return {"pairs": per_pair, "skipped": skipped, "summary": _summarize(per_pair)}
