from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def compute_ssim_dissimilarity(
    reference: np.ndarray, target: np.ndarray, config: dict
) -> np.ndarray:
    """Return a scalar per-pixel ``1 - SSIM`` map.

    scikit-image returns one SSIM map per RGB channel. Averaging the channels
    preserves the object-level mean used by the classifier while yielding a
    proper two-dimensional heatmap for inspection.
    """
    _, similarity = structural_similarity(
        reference,
        target,
        channel_axis=config["channel_axis"],
        data_range=255,
        win_size=config["win_size"],
        gaussian_weights=config["gaussian_weights"],
        sigma=config["sigma"],
        full=True,
    )
    dissimilarity = 1.0 - similarity
    if dissimilarity.ndim == 3:
        dissimilarity = dissimilarity.mean(axis=2)
    return dissimilarity.astype(np.float32)


def colorize_ssim(dissimilarity: np.ndarray, maximum: float = 1.0) -> np.ndarray:
    """Map SSIM dissimilarity to blue→cyan→yellow→red using a fixed scale."""
    value = np.clip(np.asarray(dissimilarity, np.float32) / maximum, 0, 1)
    output = np.zeros((*value.shape, 3), dtype=np.uint8)
    output[..., 0] = np.clip(1.5 - np.abs(4 * value - 3), 0, 1) * 255
    output[..., 1] = np.clip(1.5 - np.abs(4 * value - 2), 0, 1) * 255
    output[..., 2] = np.clip(1.5 - np.abs(4 * value - 1), 0, 1) * 255
    return output


def heatmap_overlay(
    target: np.ndarray, heatmap: np.ndarray, opacity: float = 0.55
) -> np.ndarray:
    """Blend a false-color heatmap over the target image."""
    return np.clip(
        (1 - opacity) * np.asarray(target, np.float32)
        + opacity * np.asarray(heatmap, np.float32),
        0,
        255,
    ).astype(np.uint8)

