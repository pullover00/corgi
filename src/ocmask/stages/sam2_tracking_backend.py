"""Experiment-only exact-mask tracking through the existing SAM2 adapter.

This wrapper gives the cached-proposal runner the same neutral result contract
as :class:`Sam31MaskTracker`.  It lets the proposal model and tracking model be
varied independently without changing GOLDILOCS classification code.
"""

from __future__ import annotations

import numpy as np

from ..adapters.sam2 import Sam2Adapter
from ..types import ObjectMask
from .sam31_backend import Sam31TrackAttempt


class Sam2MaskTracker:
    """Track externally supplied masks with the paper-faithful SAM2 backend."""

    def __init__(self, baseline_config: dict) -> None:
        self._adapter = Sam2Adapter(baseline_config)
        self._last_batch_plan: list[list[int]] = []

    def track(
        self,
        masks: list[np.ndarray],
        source_image: np.ndarray,
        target_image: np.ndarray,
    ) -> list[Sam31TrackAttempt]:
        """Return SAM2 propagation attempts in the shared experiment format."""

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
                Sam31TrackAttempt(
                    object_id=index,
                    mask=mask,
                    object_score_logit=(
                        None if score is None else float(score)
                    ),
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
