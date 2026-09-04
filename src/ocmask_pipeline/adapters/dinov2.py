"""Frozen DINOv2 appearance encoder for the isolated motion experiment."""

from __future__ import annotations

import gc
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class DescriptorBatch:
    """One descriptor and evidence diagnostics per requested object mask."""

    vectors: np.ndarray
    valid: np.ndarray
    effective_patches: np.ndarray
    previews: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class FullImageFeatures:
    """Normalized DINOv2 patch tokens for one deterministically resized RGB.

    ``feature_map`` is stored in channel-first order (C x Hpatch x Wpatch).
    The shapes make the image-to-patch coordinate transform explicit in every
    experiment artifact instead of silently relying on a library transform.
    """

    feature_map: np.ndarray
    original_shape: tuple[int, int]
    resized_shape: tuple[int, int]
    patch_size: int


def checkpoint_sha256(path: str | Path) -> str:
    """Hash a checkpoint without loading its large tensor payload."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_masked_object_crop(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    size: int,
    context_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a translation-independent, background-neutral object crop.

    Returns a normalized CHW tensor, a patch-grid occupancy map, and a uint8
    preview. The crop keeps its aspect ratio and is centered on ImageNet-mean
    padding. Pixels outside the object mask are replaced with the same mean so
    source-render holes and target-image background cannot dominate identity.
    """
    image = np.asarray(image, dtype=np.uint8)
    mask = np.asarray(mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape H x W x 3")
    if mask.shape != image.shape[:2]:
        raise ValueError("mask and image dimensions must match")
    if size <= 0 or size % 14:
        raise ValueError("DINO crop size must be a positive multiple of 14")
    if context_fraction < 0:
        raise ValueError("context_fraction must be nonnegative")

    fill_rgb = np.round(IMAGENET_MEAN * 255).astype(np.uint8)
    if not np.any(mask):
        preview = np.broadcast_to(fill_rgb, (size, size, 3)).copy()
        normalized = (
            preview.astype(np.float32) / 255.0 - IMAGENET_MEAN
        ) / IMAGENET_STD
        return normalized.transpose(2, 0, 1), np.zeros((size // 14, size // 14), np.float32), preview

    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    span = max(x1 - x0, y1 - y0)
    margin = int(round(context_fraction * span))
    x0, x1 = max(0, x0 - margin), min(image.shape[1], x1 + margin)
    y0, y1 = max(0, y0 - margin), min(image.shape[0], y1 + margin)

    crop_rgb = image[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    crop_rgb[~crop_mask] = fill_rgb
    crop_height, crop_width = crop_mask.shape
    scale = min(size / crop_width, size / crop_height)
    resized_width = max(1, min(size, int(round(crop_width * scale))))
    resized_height = max(1, min(size, int(round(crop_height * scale))))
    offset_x = (size - resized_width) // 2
    offset_y = (size - resized_height) // 2

    preview = np.broadcast_to(fill_rgb, (size, size, 3)).copy()
    resized_rgb = np.asarray(
        Image.fromarray(crop_rgb).resize(
            (resized_width, resized_height), Image.Resampling.BICUBIC
        )
    )
    preview[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized_rgb

    soft_mask = np.zeros((size, size), dtype=np.float32)
    resized_mask = np.asarray(
        Image.fromarray(crop_mask.astype(np.float32), mode="F").resize(
            (resized_width, resized_height), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    soft_mask[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = np.clip(resized_mask, 0, 1)
    patch_size = size // 14
    occupancy = np.asarray(
        Image.fromarray(soft_mask, mode="F").resize(
            (patch_size, patch_size), Image.Resampling.BOX
        ),
        dtype=np.float32,
    )
    normalized = (
        preview.astype(np.float32) / 255.0 - IMAGENET_MEAN
    ) / IMAGENET_STD
    return normalized.transpose(2, 0, 1), occupancy, preview


class Dinov2FeatureExtractor:
    """Expose DINOv2's full-image dense patch embedding as a drop-in
    replacement for ``sam3_identity_location.Sam3FeatureExtractor``.

    Reuses :meth:`DinoV2AppearanceAdapter.full_image_features` so the
    identity/location experiment's calibration, pairing, and classification
    code (all backbone-agnostic once given a dense C x H x W array) runs
    unmodified against a different feature backbone. Only the image-to-grid
    resolution differs from SAM3 (a coarser 14px-patch grid vs. SAM3's native
    backbone stride); this is a known, unavoidable resolution asymmetry
    between the two backbones, not a tuning knob.
    """

    def __init__(self, config: dict):
        self._adapter = DinoV2AppearanceAdapter(config)
        cfg = config["dinov2"]
        self._width = int(cfg["full_image_width"])
        self._height = int(cfg["full_image_height"])
        self._patch_size = int(cfg.get("patch_size", 14))

    def load(self) -> None:
        self._adapter._load()

    def feature_map(self, image: np.ndarray) -> np.ndarray:
        """Return a channel-normalized C x Hf x Wf DINOv2 patch embedding."""

        output = self._adapter.full_image_features(
            image,
            width=self._width,
            height=self._height,
            patch_size=self._patch_size,
        )
        return output.feature_map.astype(np.float16, copy=True)

    def release(self) -> None:
        self._adapter.release()


class DinoV2AppearanceAdapter:
    """Load a pinned official DINOv2 model and pool mask-aware patch tokens."""

    def __init__(self, config: dict):
        self.config = config
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch

        cfg = self.config["dinov2"]
        source = Path(cfg["source"]).resolve()
        checkpoint = Path(cfg["checkpoint"]).resolve()
        if not (source / "dinov2").is_dir():
            raise FileNotFoundError(f"DINOv2 source is missing: {source}")
        if not checkpoint.exists():
            raise FileNotFoundError(f"DINOv2 checkpoint is missing: {checkpoint}")
        expected_hash = cfg.get("checkpoint_sha256")
        if expected_hash:
            actual_hash = checkpoint_sha256(checkpoint)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"DINOv2 checkpoint SHA256 mismatch: {actual_hash}"
                )

        sys.path.insert(0, str(source))
        try:
            from dinov2.hub.backbones import dinov2_vitb14_reg
        finally:
            # Imported modules retain their package paths; removing this entry
            # prevents the experimental dependency from shadowing other code.
            sys.path.pop(0)
        model = dinov2_vitb14_reg(pretrained=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval().to(self.config.get("device", "cuda"))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model

    def encode(
        self,
        image: np.ndarray,
        masks: list[np.ndarray],
    ) -> DescriptorBatch:
        """Encode masks in deterministic batches and return CPU descriptors."""
        import torch
        import torch.nn.functional as functional

        self._load()
        cfg = self.config["dinov2"]
        size = int(cfg["crop_size"])
        prepared = [
            prepare_masked_object_crop(
                image,
                mask,
                size=size,
                context_fraction=float(cfg["context_fraction"]),
            )
            for mask in masks
        ]
        if not prepared:
            dimension = int(cfg.get("descriptor_dimension", 768))
            return DescriptorBatch(
                vectors=np.empty((0, dimension), np.float32),
                valid=np.empty(0, bool),
                effective_patches=np.empty(0, np.float32),
                previews=(),
            )

        vectors = []
        valid = []
        effective = []
        device = self.config.get("device", "cuda")
        mixed = bool(cfg.get("mixed_precision", True)) and str(device).startswith("cuda")
        batch_size = int(cfg["batch_size"])
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            pixels = torch.from_numpy(
                np.stack([item[0] for item in batch])
            ).to(device)
            occupancies = torch.from_numpy(
                np.stack([item[1] for item in batch])
            ).to(device)
            with torch.inference_mode(), (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if mixed
                else __import__("contextlib").nullcontext()
            ):
                output = self._model.forward_features(pixels)
                tokens = output["x_norm_patchtokens"].float()
                tokens = functional.normalize(tokens, dim=-1)
                weights = occupancies.flatten(1).float()
                occupancy_sum = weights.sum(dim=1)
                pooled = (tokens * weights[:, :, None]).sum(dim=1)
                pooled = functional.normalize(pooled, dim=-1)
            vectors.append(pooled.cpu().numpy())
            effective.append(occupancy_sum.cpu().numpy())
            valid.append(
                (
                    occupancy_sum
                    >= float(cfg["minimum_effective_patches"])
                ).cpu().numpy()
            )
        return DescriptorBatch(
            vectors=np.concatenate(vectors).astype(np.float32),
            valid=np.concatenate(valid).astype(bool),
            effective_patches=np.concatenate(effective).astype(np.float32),
            previews=tuple(item[2] for item in prepared),
        )

    def full_image_features(
        self,
        image: np.ndarray,
        *,
        width: int,
        height: int,
        patch_size: int = 14,
    ) -> FullImageFeatures:
        """Extract one dense, normalized patch map from a complete image.

        This differs deliberately from :meth:`encode`: no mask crop or object
        pooling is performed.  All object masks therefore share the same
        source/target coordinate system, which is required for dense visual
        prompting and lets one backbone pass serve every proposal in a pair.
        """
        import torch
        import torch.nn.functional as functional

        self._load()
        image = np.asarray(image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape H x W x 3")
        if width <= 0 or height <= 0:
            raise ValueError("full-image dimensions must be positive")
        if width % patch_size or height % patch_size:
            raise ValueError("full-image dimensions must be divisible by patch_size")

        resized = np.asarray(
            Image.fromarray(image).resize(
                (int(width), int(height)), Image.Resampling.BICUBIC
            ),
            dtype=np.uint8,
        )
        normalized = (
            resized.astype(np.float32) / 255.0 - IMAGENET_MEAN
        ) / IMAGENET_STD
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)[None]).to(
            self.config.get("device", "cuda")
        )
        device = str(self.config.get("device", "cuda"))
        mixed = bool(self.config["dinov2"].get("mixed_precision", True)) and (
            device.startswith("cuda")
        )
        with torch.inference_mode(), (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if mixed
            else __import__("contextlib").nullcontext()
        ):
            tokens = self._model.forward_features(tensor)[
                "x_norm_patchtokens"
            ].float()[0]
            tokens = functional.normalize(tokens, dim=-1)

        grid_height = height // patch_size
        grid_width = width // patch_size
        if tokens.shape[0] != grid_height * grid_width:
            raise RuntimeError(
                "DINOv2 patch-token count does not match configured image grid"
            )
        feature_map = (
            tokens.reshape(grid_height, grid_width, -1)
            .permute(2, 0, 1)
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
        )
        return FullImageFeatures(
            feature_map=feature_map,
            original_shape=tuple(map(int, image.shape[:2])),
            resized_shape=(int(height), int(width)),
            patch_size=int(patch_size),
        )

    def patch_feature_map(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return one normalized Cx16x16 patch map for visualization.

        This diagnostic follows the exact same crop and model path as
        :meth:`encode`; it is intentionally separate so normal experiment
        batches do not retain hundreds of dense feature tensors.
        """
        import torch
        import torch.nn.functional as functional

        self._load()
        cfg = self.config["dinov2"]
        pixels, occupancy, preview = prepare_masked_object_crop(
            image,
            mask,
            size=int(cfg["crop_size"]),
            context_fraction=float(cfg["context_fraction"]),
        )
        device = self.config.get("device", "cuda")
        mixed = bool(cfg.get("mixed_precision", True)) and str(device).startswith("cuda")
        tensor = torch.from_numpy(pixels[None]).to(device)
        with torch.inference_mode(), (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if mixed
            else __import__("contextlib").nullcontext()
        ):
            tokens = self._model.forward_features(tensor)[
                "x_norm_patchtokens"
            ].float()
            tokens = functional.normalize(tokens, dim=-1)[0]
        side = int(cfg["crop_size"]) // 14
        feature_map = (
            tokens.reshape(side, side, -1)
            .permute(2, 0, 1)
            .cpu()
            .numpy()
            .copy()
        )
        return feature_map, occupancy, preview

    def release(self) -> None:
        """Release the frozen backbone after the offline experiment."""
        if self._model is None:
            return
        self._model = None
        # Model parameters may participate in Python reference cycles created
        # by framework hooks. Collect them before asking CUDA to return its
        # cached blocks so SAM2 can start with the lowest possible peak VRAM.
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
