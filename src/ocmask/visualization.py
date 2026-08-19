from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .types import Label, ObjectMask


COLORS = {
    Label.UNCHANGED: (0, 0, 0),
    Label.ADDED: (0, 200, 70),
    Label.REMOVED: (230, 40, 40),
    Label.MOVED: (40, 120, 240),
    Label.WARPED: (230, 180, 20),
    Label.REPLACED: (180, 60, 220),
}


def colorize(labels: np.ndarray) -> np.ndarray:
    """Convert integer labels into the fixed reproduction color palette."""
    output = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label, color in COLORS.items():
        output[labels == label] = color
    return output


def overlay(image: np.ndarray, labels: np.ndarray, opacity: float = 0.55) -> np.ndarray:
    """Blend changed pixels over the target image while leaving static pixels intact."""
    colors = colorize(labels)
    changed = labels != Label.UNCHANGED
    result = np.asarray(image, dtype=np.float32).copy()
    result[changed] = (1 - opacity) * result[changed] + opacity * colors[changed]
    return np.clip(result, 0, 255).astype(np.uint8)


def instance_overlay(
    image: np.ndarray,
    objects: list[ObjectMask],
    statuses: list[bool] | None = None,
    opacity: float = 0.45,
    instance_ids: list[int] | None = None,
) -> np.ndarray:
    """Draw numbered SAM instances over an image for pipeline debugging.

    When ``statuses`` is supplied, successful tracks are green and failed
    source masks are red. Otherwise, a deterministic high-contrast palette
    distinguishes automatic mask proposals. Instance numbers are stable within
    a generate/track call and allow source and target views to be compared.
    """
    base = np.asarray(image, dtype=np.uint8)
    result = base.astype(np.float32).copy()
    palette = np.array(
        [
            (255, 99, 71),
            (64, 224, 208),
            (255, 215, 0),
            (138, 43, 226),
            (50, 205, 50),
            (30, 144, 255),
            (255, 105, 180),
            (255, 140, 0),
        ],
        dtype=np.float32,
    )
    for index, obj in enumerate(objects):
        mask = np.asarray(obj.mask, dtype=bool)
        if mask.shape != base.shape[:2]:
            raise ValueError("SAM debug mask and image dimensions must match")
        if statuses is None:
            color = palette[index % len(palette)]
        else:
            color = np.array(
                (35, 220, 90) if statuses[index] else (245, 55, 55),
                dtype=np.float32,
            )
        result[mask] = (1 - opacity) * result[mask] + opacity * color

        # A bright one-pixel contour makes nested and overlapping masks visible.
        interior = (
            mask
            & np.roll(mask, 1, axis=0)
            & np.roll(mask, -1, axis=0)
            & np.roll(mask, 1, axis=1)
            & np.roll(mask, -1, axis=1)
        )
        boundary = mask & ~interior
        result[boundary] = color

    if instance_ids is None:
        instance_ids = list(range(1, len(objects) + 1))
    if len(instance_ids) != len(objects):
        raise ValueError("instance_ids and objects must have equal length")

    canvas = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    for instance_id, obj in zip(instance_ids, objects):
        ys, xs = np.nonzero(obj.mask)
        if not len(xs):
            continue
        x, y = int(np.median(xs)), int(np.median(ys))
        text = str(instance_id)
        box = draw.textbbox((x, y), text, anchor="mm")
        draw.rectangle(
            (box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1),
            fill=(0, 0, 0),
        )
        draw.text((x, y), text, fill=(255, 255, 255), anchor="mm")
    return np.asarray(canvas)
