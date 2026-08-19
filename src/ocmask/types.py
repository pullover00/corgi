from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np


class Label(IntEnum):
    """Canonical prediction IDs; REPLACED is a ChangeSim protocol extension."""
    UNCHANGED = 0
    ADDED = 1
    REMOVED = 2
    MOVED = 3
    WARPED = 4
    REPLACED = 5


@dataclass
class Reconstruction:
    """Dense stereo outputs expressed in a shared world coordinate system."""
    images: tuple[np.ndarray, np.ndarray]
    points: tuple[np.ndarray, np.ndarray]
    depths: tuple[np.ndarray, np.ndarray]
    intrinsics: tuple[np.ndarray, np.ndarray]
    world_to_camera: tuple[np.ndarray, np.ndarray]
    confidence: tuple[np.ndarray, np.ndarray]
    match_count: int = 0


@dataclass
class ObjectMask:
    """One object proposal with its final class and diagnostic provenance."""
    mask: np.ndarray
    score: float = 1.0
    label: Label = Label.UNCHANGED
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PairResult:
    """Public result returned by pairwise and ablation pipelines."""
    labels: np.ndarray
    binary: np.ndarray
    objects: list[ObjectMask]
    artifacts_dir: Path
    timings: dict[str, float]
