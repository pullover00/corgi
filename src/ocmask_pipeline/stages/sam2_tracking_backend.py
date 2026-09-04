"""Mask-prompted tracking through SAM2: given a mask in one image, find its
corresponding region (if any) in another image of the same scene."""

from __future__ import annotations

import numpy as np

from ..adapters.sam2 import Sam2Adapter
from ..types import ObjectMask, TrackAttempt


class Sam2MaskTracker:
    """Track externally supplied masks with the SAM2 video-propagation backend."""

    def __init__(self, baseline_config: dict) -> None:
        self._adapter = Sam2Adapter(baseline_config)
        self._last_batch_plan: list[list[int]] = []

    def track(
        self,
        masks: list[np.ndarray],
        source_image: np.ndarray,
        target_image: np.ndarray,
    ) -> list[TrackAttempt]:
        """Return SAM2 propagation attempts, one per input mask."""

        if not masks:
            self._last_batch_plan = []
            return []
        sources = [
            ObjectMask(
                mask=np.asarray(mask, dtype=bool),
                source="external_cached_proposal",
            )
            for mask in masks
        ]
        accepted = self._adapter.track(sources, source_image, target_image)
        raw = self._adapter.last_track_attempts()
        if len(accepted) != len(sources) or len(raw) != len(sources):
            raise RuntimeError("SAM2 returned an incomplete tracking result")

        self._last_batch_plan = [list(range(len(sources)))]
        attempts = []
        for index, (result, attempt) in enumerate(zip(accepted, raw), start=1):
            if attempt is None:
                mask = np.zeros(target_image.shape[:2], dtype=bool)
                score = None
                reasons = ("missing_tracking_attempt",)
            else:
                mask = np.asarray(attempt.mask, dtype=bool)
                score = attempt.metadata.get("object_score_logit")
                reasons = tuple(attempt.metadata.get("rejection_reasons", ()))
            attempts.append(
                TrackAttempt(
                    object_id=index,
                    mask=mask,
                    object_score_logit=(None if score is None else float(score)),
                    accepted=result is not None,
                    rejection_reasons=reasons,
                    batch_index=0,
                    input_index=index - 1,
                )
            )
        return attempts

    def last_batch_plan(self) -> list[list[int]]:
        """Return the single SAM2 multi-object state used by the last call."""
        return [list(indices) for indices in self._last_batch_plan]

    def release(self) -> None:
        """Drop SAM2 image/video models and release cached CUDA allocations."""
        self._adapter.release()
