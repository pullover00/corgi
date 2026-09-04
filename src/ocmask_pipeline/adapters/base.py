from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import ObjectMask


class SegmentationAdapter(ABC):
    @abstractmethod
    def generate(self, image: np.ndarray) -> list[ObjectMask]:
        raise NotImplementedError

    @abstractmethod
    def track(
        self, masks: list[ObjectMask], source_image: np.ndarray, target_image: np.ndarray
    ) -> list[ObjectMask | None]:
        """Return one target mask (or None) per input mask."""
        raise NotImplementedError

    def last_track_attempts(self) -> list[ObjectMask | None]:
        """Return raw attempts from the most recent tracking call.

        Production adapters may override this diagnostic hook to retain masks
        and confidence values that were rejected by their own acceptance rule.
        The pipeline never uses these attempts for classification.
        """
        return []

    def image_feature_map(self, image: np.ndarray) -> np.ndarray | None:
        """Return an optional dense image-encoder feature map.

        The pairwise pipeline uses this capability only for explicitly enabled
        research ablations.  Keeping it optional means alternative mask
        adapters do not need to expose internal features merely to satisfy the
        normal GOLDILOCS segmentation/tracking interface.
        """
        return None

    def release(self) -> None:
        """Release heavyweight model state once this adapter is done."""
        return None
