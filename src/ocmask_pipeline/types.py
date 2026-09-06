from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


class Label(IntEnum):
    """Canonical per-pixel/per-object change labels."""
    UNCHANGED = 0
    ADDED = 1
    REMOVED = 2
    MOVED = 3
    WARPED = 4
    # Same-footprint object replacement (e.g. a red block swapped for a blue
    # one) -- found by a dense color-residual pass, not by identity matching
    # between two proposals, since the old/new object is often never
    # segmented as its own SAM3 proposal in the first place (it gets
    # absorbed into a larger surrounding surface's mask). See
    # change_detection.find_color_replacement_regions.
    REPLACED = 5


@dataclass
class ObjectMask:
    """One object proposal with its final class and diagnostic provenance."""
    mask: np.ndarray
    score: float = 1.0
    label: Label = Label.UNCHANGED
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackAttempt:
    """One target-frame mask-tracking result corresponding to one input mask."""
    object_id: int
    mask: np.ndarray
    object_score_logit: float | None
    accepted: bool
    rejection_reasons: tuple[str, ...]
    batch_index: int = 0
    input_index: int = 0
