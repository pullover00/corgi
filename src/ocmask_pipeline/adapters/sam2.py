from __future__ import annotations

import hashlib
import tempfile
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

from .base import SegmentationAdapter
from ..types import ObjectMask


def _rgb_feature_key(image: np.ndarray) -> str:
    """Hash the exact in-memory RGB array used to compute an image embedding."""
    array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(memoryview(array))
    return digest.hexdigest()


def _generate_with_feature_capture(generator, image: np.ndarray):
    """Run SAM2 AMG and retain its full-image encoder output at zero extra cost.

    SAM2's automatic mask generator resets its predictor after each full image
    and crop.  Its first crop is the complete image, so the public
    ``get_image_embedding`` result must be cloned immediately after
    ``set_image``.  The wrapper also checks shape and pixels rather than
    relying on call order, making the assumption explicit for the pinned SAM2
    implementation.

    Returns:
        ``(records, feature_map)`` where ``feature_map`` is a CPU float32
        ``C x Hf x Wf`` NumPy array.
    """
    # Pillow-backed arrays can be C-contiguous but read-only. SAM2 converts the
    # array to a tensor and warns in that case, so make writability explicit.
    expected = np.array(image, dtype=np.uint8, copy=True, order="C")
    predictor = generator.predictor
    original_set_image = predictor.set_image
    captured: dict[str, np.ndarray] = {}

    def set_image_and_capture(candidate) -> None:
        original_set_image(candidate)
        array = np.asarray(candidate)
        is_full_image = (
            "feature_map" not in captured
            and array.shape == expected.shape
            and np.array_equal(array, expected)
        )
        if is_full_image:
            embedding = predictor.get_image_embedding()
            if embedding.ndim != 4 or embedding.shape[0] != 1:
                raise RuntimeError(
                    "Expected SAM2 image embedding with shape 1 x C x H x W"
                )
            # BF16 tensors cannot be converted directly to NumPy.  Pooling is
            # deliberately performed in FP32 for stable cosine comparisons.
            captured["feature_map"] = (
                embedding[0].detach().float().cpu().numpy().copy()
            )

    # The override is scoped to one generate call and restored even if SAM2
    # fails, so normal generator behavior is unchanged outside this helper.
    predictor.set_image = set_image_and_capture
    try:
        records = generator.generate(expected)
    finally:
        predictor.set_image = original_set_image

    if "feature_map" not in captured:
        raise RuntimeError(
            "SAM2 automatic mask generation did not expose a full-image embedding"
        )
    return records, captured["feature_map"]


def _save_video_frame(image: np.ndarray, path: Path) -> None:
    """Write one SAM2 video frame without Pillow's default chroma subsampling.

    The upstream directory loader expects JPEG frames.  Pillow otherwise uses
    4:2:0 chroma subsampling, which needlessly changes colored boundaries
    between automatic-mask generation and video tracking.
    """
    Image.fromarray(np.asarray(image, np.uint8)).save(
        path,
        format="JPEG",
        quality=100,
        subsampling=0,
    )


def _frame_object_score(
    state: dict, object_id: int, frame_idx: int
) -> float | None:
    """Read SAM2.1's raw object-presence logit for one object and frame."""
    obj_idx = state["obj_id_to_idx"][int(object_id)]
    outputs = state["output_dict_per_obj"][obj_idx]
    frame_output = outputs["non_cond_frame_outputs"].get(frame_idx)
    if frame_output is None:
        frame_output = outputs["cond_frame_outputs"].get(frame_idx)
    if frame_output is None:
        return None
    score = frame_output.get("object_score_logits")
    if score is None:
        return None
    return float(score.detach().float().cpu().reshape(-1)[0])


class Sam2Adapter(SegmentationAdapter):
    """SAM2 automatic masks plus two-frame video propagation."""

    def __init__(self, config: dict):
        self.config = config
        self._image_model = None
        self._generator = None
        self._video_predictor = None
        self._last_track_attempts: list[ObjectMask | None] = []
        # Feature maps live on the CPU and are bounded by an LRU cache.  Three
        # distinct images are used by one pair (R01, I1, and the clean render);
        # four entries retain those without growing across a full benchmark.
        self._feature_maps: OrderedDict[str, np.ndarray] = OrderedDict()

    def _inference_context(self):
        """Use BF16 autocast when mixed-precision CUDA inference is enabled."""
        import torch

        use_mixed_precision = self.config["sam2"].get(
            "mixed_precision", self.config.get("mixed_precision", False)
        )
        device = str(self.config.get("device", "cuda"))
        if use_mixed_precision and device.startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.bfloat16)

        return nullcontext()

    def _load(self):
        """Lazily construct SAM2's image generator and video propagator."""
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2, build_sam2_video_predictor
        except ImportError as exc:
            raise RuntimeError(
                "SAM2 is not installed. Follow README.md and install requirements-models.txt."
            ) from exc
        cfg = self.config["sam2"]

        if self._image_model is None:
            # Compiling the encoder makes the first call slower, but avoids
            # repeatedly executing the image backbone in eager mode afterward.
            image_overrides = []
            if cfg.get("compile_image_encoder", False):
                image_overrides.append("++model.compile_image_encoder=True")

            self._image_model = build_sam2(
                cfg["model_cfg"],
                cfg["checkpoint"],
                device=self.config.get("device", "cuda"),
                hydra_overrides_extra=image_overrides,
            )
            self._generator = SAM2AutomaticMaskGenerator(
                self._image_model,
                points_per_side=cfg["points_per_side"],
                pred_iou_thresh=cfg["pred_iou_thresh"],
                stability_score_thresh=cfg["stability_score_thresh"],
                crop_n_layers=cfg["crop_n_layers"],
                crop_n_points_downscale_factor=cfg["crop_n_points_downscale_factor"],
                multimask_output=cfg["multimask_output"],
                use_m2m=cfg["use_m2m"],
            )

        if self._video_predictor is None:
            # SAM2's VOS predictor compiles its image encoder and optimized
            # propagation path when the corresponding config flag is enabled.
            self._video_predictor = build_sam2_video_predictor(
                cfg["model_cfg"],
                cfg["checkpoint"],
                device=self.config.get("device", "cuda"),
                vos_optimized=cfg.get("vos_optimized", False),
            )

    def generate(self, image: np.ndarray) -> list[ObjectMask]:
        """Generate class-agnostic object proposals using paper parameters."""
        import torch

        self._load()
        rgb = np.array(image, dtype=np.uint8, copy=True, order="C")
        feature_config = self.config["tracking"].get("feature_consistency", {})
        # Inference mode removes autograd bookkeeping; BF16 autocast reduces
        # memory traffic and accelerates tensor-core operations on the RTX 4090.
        with torch.inference_mode(), self._inference_context():
            if feature_config.get("enabled", False):
                # Capture the global 64x64 image embedding during AMG's
                # existing full-image encoder call. Crop embeddings are not
                # mixed into this common coordinate system.
                records, feature_map = _generate_with_feature_capture(
                    self._generator, rgb
                )
                self._remember_feature_map(rgb, feature_map)
            else:
                records = self._generator.generate(rgb)
        minimum = self.config["tracking"]["minimum_mask_area"]
        return [
            ObjectMask(
                mask=np.asarray(record["segmentation"], bool),
                score=float(record.get("predicted_iou", 1.0)),
                source="sam2_automatic",
                metadata={
                    "area": int(record.get("area", np.asarray(record["segmentation"]).sum())),
                    "stability_score": float(record.get("stability_score", 0.0)),
                    # Preserve upstream proposal provenance so crop/prompt
                    # duplication can be diagnosed without rerunning SAM2.
                    "bbox_xywh": record.get("bbox"),
                    "point_coords": record.get("point_coords"),
                    "crop_box_xywh": record.get("crop_box"),
                },
            )
            for record in records
            if int(record.get("area", 0)) >= minimum
        ]

    def track(
        self, masks: list[ObjectMask], source_image: np.ndarray, target_image: np.ndarray
    ) -> list[ObjectMask | None]:
        """Propagate all masks through a temporary two-frame SAM2 video."""
        if not masks:
            self._last_track_attempts = []
            return []

        import torch

        self._load()
        with tempfile.TemporaryDirectory(prefix="ocmask-sam2-") as temporary:
            _save_video_frame(source_image, Path(temporary) / "00000.jpg")
            _save_video_frame(target_image, Path(temporary) / "00001.jpg")
            # SAM3's ambient autocast context is closed before SAM2 is loaded
            # (see ocmask.numerics and the SAM3 adapters). With that global
            # leak removed, use SAM2's original scoped BF16 policy; this is the
            # numerical path used by the frozen Goldilocs tracking artifacts.
            with torch.inference_mode(), self._inference_context():
                state = self._video_predictor.init_state(video_path=temporary)
                self._video_predictor.reset_state(state)
                # Seed every object in frame zero before a single propagation
                # pass. SAM2 reserves zero-like values, so IDs start at one.
                for index, obj in enumerate(masks, start=1):
                    self._video_predictor.add_new_mask(
                        inference_state=state, frame_idx=0, obj_id=index, mask=obj.mask
                    )
                target_logits = {}
                target_object_scores = {}
                for frame_idx, object_ids, logits in (
                    self._video_predictor.propagate_in_video(state)
                ):
                    if frame_idx == 1:
                        for object_id, logit in zip(object_ids, logits):
                            # NumPy cannot consume BF16 directly, so explicitly
                            # convert propagated logits back to FP32 on the CPU.
                            target_logits[int(object_id)] = (
                                logit.detach().float().cpu().numpy().squeeze()
                            )
                            # SAM2.1 predicts whether an object is present in
                            # each frame. The public propagation iterator omits
                            # this value, but retains it in the per-object state.
                            score = _frame_object_score(
                                state, int(object_id), frame_idx
                            )
                            if score is not None:
                                target_object_scores[int(object_id)] = score
                        break
        results: list[ObjectMask | None] = []
        attempts: list[ObjectMask | None] = []
        minimum = self.config["tracking"]["minimum_mask_area"]
        minimum_score = self.config["tracking"].get("minimum_track_score", 0.0)
        for index, source in enumerate(masks, start=1):
            logit = target_logits.get(index)
            object_score = target_object_scores.get(index, float("-inf"))
            # SAM2's documented mask-logit decision boundary is zero. A tiny
            # region or a negative object-presence logit is treated as tracking
            # failure rather than an object match.
            mask = (
                np.asarray(logit > 0, dtype=bool)
                if logit is not None
                else np.zeros(target_image.shape[:2], dtype=bool)
            )
            raw_area = int(mask.sum())
            rejection_reasons = []
            if logit is None:
                rejection_reasons.append("missing_target_logits")
            if raw_area < minimum:
                rejection_reasons.append("below_minimum_mask_area")
            # SAM2 itself treats logits <= 0 as object absence. A separately
            # configured positive minimum may impose a stricter inferred rule.
            if object_score <= 0:
                rejection_reasons.append("object_absent")
            elif object_score < minimum_score:
                rejection_reasons.append("below_minimum_track_score")
            attempt = ObjectMask(
                mask=mask,
                # This is the source proposal's predicted-IoU, not a target
                # tracking-confidence score. Keep the name explicit in metadata.
                score=source.score,
                source="sam2_track_attempt",
                metadata={
                    "source_index": index - 1,
                    "source_proposal_score": source.score,
                    "object_score_logit": (
                        object_score if np.isfinite(object_score) else None
                    ),
                    "raw_target_area": raw_area,
                    "rejection_reasons": rejection_reasons,
                },
            )
            attempts.append(attempt)
            if rejection_reasons:
                results.append(None)
            else:
                attempt.source = "sam2_track"
                results.append(
                    attempt
                )
        self._last_track_attempts = attempts
        return results

    def last_track_attempts(self) -> list[ObjectMask | None]:
        """Expose raw attempts for debug output without changing inference."""
        return list(self._last_track_attempts)

    def _remember_feature_map(
        self, image: np.ndarray, feature_map: np.ndarray
    ) -> np.ndarray:
        """Insert one CPU feature map into the bounded content-addressed cache."""
        key = _rgb_feature_key(image)
        value = np.asarray(feature_map, dtype=np.float32)
        self._feature_maps[key] = value
        self._feature_maps.move_to_end(key)
        maximum = int(
            self.config["tracking"]
            .get("feature_consistency", {})
            .get("feature_cache_entries", 4)
        )
        if maximum < 1:
            raise ValueError("feature_cache_entries must be at least one")
        while len(self._feature_maps) > maximum:
            self._feature_maps.popitem(last=False)
        return value

    def image_feature_map(self, image: np.ndarray) -> np.ndarray:
        """Return SAM2's raw image-encoder embedding for an exact RGB array.

        Proposal images normally hit the cache populated in :meth:`generate`.
        The clean canonical render has no automatic-mask call, so it incurs one
        explicit image-encoder pass.  Video-predictor features are intentionally
        not reused: that model reads temporary JPEG frames and its
        memory-conditioned state would make track verification circular.
        """
        import torch

        self._load()
        rgb = np.array(image, dtype=np.uint8, copy=True, order="C")
        key = _rgb_feature_key(rgb)
        cached = self._feature_maps.get(key)
        if cached is not None:
            self._feature_maps.move_to_end(key)
            return cached

        predictor = self._generator.predictor
        with torch.inference_mode(), self._inference_context():
            predictor.set_image(rgb)
            try:
                embedding = predictor.get_image_embedding()
                if embedding.ndim != 4 or embedding.shape[0] != 1:
                    raise RuntimeError(
                        "Expected SAM2 image embedding with shape 1 x C x H x W"
                    )
                feature_map = (
                    embedding[0].detach().float().cpu().numpy().copy()
                )
            finally:
                predictor.reset_predictor()
        return self._remember_feature_map(rgb, feature_map)

    def release(self) -> None:
        """Drop SAM2's image/video models and release cached CUDA allocations.

        Every caller (``PairwisePipeline`` for stage 1's clean-plate render,
        ``Sam2MaskTracker`` for stages 3/6/8/10) constructs a fresh
        ``Sam2Adapter`` per pair. Without an explicit release, PyTorch/CUDA
        can hold each instance's multi-GB image encoder and video predictor
        alive past the point Python's own refcounting would otherwise
        reclaim them -- torch.compile's cache (``compile_image_encoder`` and
        ``vos_optimized`` are both on by default) in particular keeps
        strong references to compiled graphs keyed by the specific model
        instance, so a long run that builds a new instance every pair leaks
        GPU memory pair over pair until it OOMs.
        """
        import gc

        import torch

        self._generator = None
        self._video_predictor = None
        self._image_model = None
        self._feature_maps.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
