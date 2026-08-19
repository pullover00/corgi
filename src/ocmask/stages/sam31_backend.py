"""Thin, experiment-only access to SAM 3.1 mask-prompted tracking.

GOLDILOCS starts every track from an already known object mask.  The SAM3.1
multiplex tracker has a batched ``add_new_masks`` API, so this adapter creates
one tracker state per bounded batch and seeds every mask into that state in one
call.  Keeping the masks in one state is important: creating one singleton
state per object and later aggregating the states changes object ordering and
can associate a propagated mask with the wrong object ID.

This adapter deliberately lives under ``experiments``.  It does not alter the
paper-faithful SAM2 adapter or claim that the internal SAM 3.1 call is a stable
upstream API.
"""

from __future__ import annotations

import gc
import io
import logging
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Sam31TrackAttempt:
    """One target-frame result corresponding to one input mask."""

    # Object IDs are local to a bounded tracker batch.  They are exposed to
    # make ID/order invariance testable without relying on SAM3.1 internals.
    object_id: int
    mask: np.ndarray
    object_score_logit: float | None
    accepted: bool
    rejection_reasons: tuple[str, ...]
    batch_index: int = 0
    input_index: int = 0


def plan_mask_batches(
    masks: list[np.ndarray],
    maximum_batch_size: int,
    mode: str = "overlap_safe",
) -> list[list[int]]:
    """Plan deterministic tracker batches while preserving every prompt.

    SAM3.1's multiplex ``add_new_masks`` implementation subtracts all other
    masks in the same state from each prompt.  Automatic-mask proposals are
    frequently nested, so naive consecutive batches can erase a prompt before
    tracking starts.  ``overlap_safe`` uses deterministic first-fit graph
    coloring: a mask enters the first non-full batch whose union does not
    overlap it.  The packed-bit representation keeps this CPU planning step
    inexpensive even for hundreds of dense proposals.

    The returned indices retain their original order within each batch.  The
    caller restores outputs to global input order after inference.
    """

    if maximum_batch_size < 1:
        raise ValueError("maximum_batch_size must be positive")
    if mode not in {"overlap_safe", "sequential", "singleton"}:
        raise ValueError(
            "batching mode must be 'overlap_safe', 'sequential', or 'singleton'"
        )
    if not masks:
        return []
    normalized = [np.asarray(mask, dtype=bool) for mask in masks]
    shape = normalized[0].shape
    if any(mask.shape != shape for mask in normalized):
        raise ValueError("all masks must have the same shape")
    if mode == "singleton":
        return [[index] for index in range(len(normalized))]
    if mode == "sequential":
        return [
            list(range(start, min(start + maximum_batch_size, len(normalized))))
            for start in range(0, len(normalized), maximum_batch_size)
        ]

    packed = [np.packbits(mask.reshape(-1)) for mask in normalized]
    batches: list[list[int]] = []
    batch_unions: list[np.ndarray] = []
    for index, candidate in enumerate(packed):
        compatible = [
            batch_index
            for batch_index, (batch, occupied) in enumerate(
                zip(batches, batch_unions)
            )
            if len(batch) < maximum_batch_size
            and not np.bitwise_and(candidate, occupied).any()
        ]
        if not compatible:
            batches.append([index])
            batch_unions.append(candidate.copy())
            continue
        # Fill the most occupied compatible state first.  This minimizes calls;
        # the earliest batch wins ties, keeping the plan deterministic.
        destination = max(compatible, key=lambda item: len(batches[item]))
        batches[destination].append(index)
        np.bitwise_or(
            batch_unions[destination],
            candidate,
            out=batch_unions[destination],
        )
    return batches


class Sam31MaskTracker:
    """Load SAM3.1 once and track exact mask prompts through two RGB frames."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        batch_size: int = 8,
        minimum_mask_area: int = 100,
        minimum_object_score: float = 0.0,
        use_fa3: bool = False,
        use_rope_real: bool = False,
        batching_mode: str = "overlap_safe",
        disable_output_non_overlap: bool = True,
        model_max_num_objects: int | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if model_max_num_objects is not None and model_max_num_objects < batch_size:
            raise ValueError(
                "model_max_num_objects cannot be smaller than batch_size"
            )
        self.checkpoint = str(Path(checkpoint).resolve())
        self.batch_size = int(batch_size)
        # Normally the model capacity equals the largest tracker state.  Tests
        # may deliberately schedule singleton states while keeping the same
        # eight-object model as the multiplex candidate; otherwise the
        # "batch-invariance" comparison changes two variables at once.
        self.model_max_num_objects = int(
            model_max_num_objects
            if model_max_num_objects is not None
            else batch_size
        )
        self.minimum_mask_area = int(minimum_mask_area)
        self.minimum_object_score = float(minimum_object_score)
        self.use_fa3 = bool(use_fa3)
        self.use_rope_real = bool(use_rope_real)
        self.batching_mode = str(batching_mode)
        self.disable_output_non_overlap = bool(disable_output_non_overlap)
        self._predictor = None
        self._last_batch_plan: list[list[int]] = []

    def load(self) -> None:
        """Build the pinned SAM3.1 multiplex predictor lazily."""

        if self._predictor is not None:
            return
        from sam3.model_builder import build_sam3_predictor

        # The current upstream builder first probes a tracker-only model with
        # the full prefixed checkpoint and prints thousands of expected
        # non-matching detector keys before correctly loading the assembled
        # detector/tracker model. Keep that diagnostic noise out of benchmark
        # logs; load errors still propagate normally.
        with redirect_stdout(io.StringIO()):
            self._predictor = build_sam3_predictor(
                checkpoint_path=self.checkpoint,
                version="sam3.1",
                compile=False,
                warm_up=False,
                max_num_objects=self.model_max_num_objects,
                multiplex_count=16,
                use_fa3=self.use_fa3,
                use_rope_real=self.use_rope_real,
                async_loading_frames=False,
            )
        if self.disable_output_non_overlap:
            # The non-Hydra builder currently leaves this demo default enabled,
            # although the upstream multiplex wrapper documents it as false.
            # Overlap-safe packing prevents prompt conflicts; disabling this
            # second output-level constraint prevents unrelated masks from
            # competing again after propagation.
            self._predictor.model.tracker.model.non_overlap_masks_for_output = False

    def release(self) -> None:
        """Release model and cached CUDA allocations between experiment stages."""

        if self._predictor is None:
            return
        import torch

        self._predictor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _track_batch(
        self,
        masks: list[np.ndarray],
        source_image: np.ndarray,
        target_image: np.ndarray,
    ) -> list[Sam31TrackAttempt]:
        """Track at most ``batch_size`` masks in one two-frame SAM3.1 state."""

        import torch

        assert self._predictor is not None
        model = self._predictor.model
        frames = [
            Image.fromarray(np.asarray(source_image, dtype=np.uint8), mode="RGB"),
            Image.fromarray(np.asarray(target_image, dtype=np.uint8), mode="RGB"),
        ]
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            # Upstream emits one INFO line per object plus a tqdm bar for every
            # two-frame call. Capture that internal chatter so a ten-pair run
            # remains readable; exceptions still propagate to the runner.
            previous_logging_level = logging.root.manager.disable
            logging.disable(logging.INFO)
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    state = model.init_state(
                        resource_path=frames,
                        offload_video_to_cpu=True,
                        async_loading_frames=False,
                    )
                    # Populate frame-zero features before creating the child
                    # tracker state.  The child receives this cache by
                    # reference, so frame-one features can be added later.
                    model._prepare_backbone_feats(state, frame_idx=0, reverse=False)
                    tracker_state = model._init_new_sam2_state(state)
                    object_ids = list(range(1, len(masks) + 1))
                    source_masks = torch.stack(
                        [
                            torch.as_tensor(mask, dtype=torch.float32)
                            for mask in masks
                        ],
                        dim=0,
                    )

                    # This is the key corrected operation: all exact masks are
                    # allocated and conditioned together in one multiplex
                    # state.  There is no dummy point and no later state merge.
                    model.tracker.add_new_masks(
                        tracker_state,
                        frame_idx=0,
                        obj_ids=object_ids,
                        masks=source_masks,
                    )
                    model.tracker.propagate_in_video_preflight(
                        tracker_state, run_mem_encoder=True
                    )

                    # Add the target-frame backbone features to the shared
                    # cache, then propagate only frame one.  SAM3.1 returns
                    # video-resolution mask logits and per-object presence
                    # logits in the same object-ID order as the joint state.
                    model._prepare_backbone_feats(state, frame_idx=1, reverse=False)
                    target_result = next(
                        model.tracker.propagate_in_video(
                            tracker_state,
                            start_frame_idx=1,
                            max_frame_num_to_track=0,
                            reverse=False,
                            tqdm_disable=True,
                            run_mem_encoder=True,
                        )
                    )
                    (
                        target_frame_idx,
                        target_ids,
                        _target_low_res,
                        target_video_res,
                        target_scores,
                    ) = target_result
            finally:
                logging.disable(previous_logging_level)

        if int(target_frame_idx) != 1:
            raise RuntimeError(
                f"SAM3.1 returned frame {target_frame_idx}, expected target frame 1"
            )
        returned_ids = [int(obj_id) for obj_id in target_ids]
        expected_ids = list(range(1, len(masks) + 1))
        if len(set(returned_ids)) != len(returned_ids):
            raise RuntimeError(f"SAM3.1 returned duplicate object IDs: {returned_ids}")
        if returned_ids != expected_ids:
            raise RuntimeError(
                "SAM3.1 object-ID/order mismatch: "
                f"expected {expected_ids}, received {returned_ids}"
            )
        if len(target_video_res) != len(returned_ids):
            raise RuntimeError("SAM3.1 returned a different number of masks and IDs")
        if len(target_scores) != len(returned_ids):
            raise RuntimeError("SAM3.1 returned a different number of scores and IDs")

        height, width = target_image.shape[:2]
        output_masks: dict[int, np.ndarray] = {}
        output_scores: dict[int, float] = {}
        for index, obj_id in enumerate(returned_ids):
            # SAM mask logits use zero as their foreground threshold.  Casting
            # the logits directly to bool would incorrectly mark every
            # non-zero negative background logit as foreground.
            output_masks[obj_id] = np.asarray(
                (target_video_res[index].squeeze() > 0.0).detach().cpu(),
                dtype=bool,
            )
            output_scores[obj_id] = float(
                target_scores[index].squeeze().detach().float().cpu()
            )

        attempts: list[Sam31TrackAttempt] = []
        for obj_id in range(1, len(masks) + 1):
            mask = output_masks.get(
                obj_id, np.zeros((height, width), dtype=bool)
            )
            score = output_scores.get(obj_id)
            reasons: list[str] = []
            if obj_id not in output_masks:
                reasons.append("missing_target_output")
            if int(mask.sum()) < self.minimum_mask_area:
                reasons.append("below_minimum_mask_area")
            if score is None:
                reasons.append("missing_object_score")
            elif score <= self.minimum_object_score:
                reasons.append("object_absent")
            attempts.append(
                Sam31TrackAttempt(
                    object_id=obj_id,
                    mask=mask,
                    object_score_logit=score,
                    accepted=not reasons,
                    rejection_reasons=tuple(reasons),
                )
            )

        # Explicitly sever the many CUDA tensors held by one inference state.
        state.clear()
        del state
        gc.collect()
        return attempts

    def track(
        self,
        masks: list[np.ndarray],
        source_image: np.ndarray,
        target_image: np.ndarray,
    ) -> list[Sam31TrackAttempt]:
        """Track masks in bounded batches while preserving input order."""

        if not masks:
            return []
        self.load()
        source = np.asarray(source_image, dtype=np.uint8)
        target = np.asarray(target_image, dtype=np.uint8)
        if source.shape != target.shape:
            raise ValueError("source and target images must have equal shapes")
        shape = source.shape[:2]
        normalized_masks = [np.asarray(mask, dtype=bool) for mask in masks]
        if any(mask.shape != shape for mask in normalized_masks):
            raise ValueError("every mask must match the source image shape")

        batches = plan_mask_batches(
            normalized_masks, self.batch_size, mode=self.batching_mode
        )
        self._last_batch_plan = [list(indices) for indices in batches]
        ordered: list[Sam31TrackAttempt | None] = [None] * len(normalized_masks)
        for batch_index, indices in enumerate(batches):
            batch_attempts = self._track_batch(
                [normalized_masks[index] for index in indices],
                source,
                target,
            )
            if len(batch_attempts) != len(indices):
                raise RuntimeError("SAM3.1 returned an incomplete multiplex batch")
            for input_index, attempt in zip(indices, batch_attempts):
                ordered[input_index] = replace(
                    attempt,
                    batch_index=batch_index,
                    input_index=input_index,
                )
        if any(attempt is None for attempt in ordered):
            raise RuntimeError("SAM3.1 did not return every planned input mask")
        return [attempt for attempt in ordered if attempt is not None]

    def last_batch_plan(self) -> list[list[int]]:
        """Return the most recent deterministic global-index batch plan."""

        return [list(indices) for indices in self._last_batch_plan]
