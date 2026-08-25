"""Automatic class-agnostic proposals from SAM3's interactive image tracker.

SAM3 does not currently ship an ``AutomaticMaskGenerator`` class in its
official source package.  For a controlled comparison, this module applies
the same point-grid, crop, quality, stability, edge, and NMS procedure used by
SAM2's automatic mask generator to SAM3's SAM1-compatible image predictor.
"""

from __future__ import annotations

import gc
import math
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

from ..numerics import (
    capture_torch_numerical_state,
    enable_sam3_numerics,
    exit_sam3_numerical_scope,
)


@dataclass(frozen=True)
class Sam3Proposal:
    """One automatic mask and the diagnostics used to retain it."""

    mask: np.ndarray
    predicted_iou: float
    stability_score: float
    point_xy: tuple[float, float]
    crop_box_xyxy: tuple[int, int, int, int]


def _point_grid(points_per_side: int) -> np.ndarray:
    offset = 1.0 / (2.0 * points_per_side)
    axis = np.linspace(offset, 1.0 - offset, points_per_side)
    xx, yy = np.meshgrid(axis, axis)
    return np.stack((xx, yy), axis=-1).reshape(-1, 2)


def _crop_boxes(
    image_shape: tuple[int, int],
    crop_layers: int,
    overlap_ratio: float = 512.0 / 1500.0,
) -> tuple[list[tuple[int, int, int, int]], list[int]]:
    """Reproduce SAM's overlapping crop pyramid in XYXY coordinates."""

    height, width = image_shape
    short_side = min(height, width)
    boxes = [(0, 0, width, height)]
    layers = [0]

    def crop_length(length: int, count: int, overlap: int) -> int:
        return int(math.ceil((overlap * (count - 1) + length) / count))

    for layer in range(crop_layers):
        count = 2 ** (layer + 1)
        overlap = int(overlap_ratio * short_side * (2.0 / count))
        crop_width = crop_length(width, count, overlap)
        crop_height = crop_length(height, count, overlap)
        x_starts = [int((crop_width - overlap) * index) for index in range(count)]
        y_starts = [int((crop_height - overlap) * index) for index in range(count)]
        for x0, y0 in product(x_starts, y_starts):
            boxes.append(
                (
                    x0,
                    y0,
                    min(x0 + crop_width, width),
                    min(y0 + crop_height, height),
                )
            )
            layers.append(layer + 1)
    return boxes, layers


def _mask_boxes(masks):
    """Return XYXY boxes for a ``N x H x W`` boolean tensor."""

    import torch

    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)
    count, height, width = masks.shape
    ys = torch.arange(height, device=masks.device)
    xs = torch.arange(width, device=masks.device)
    row_present = masks.any(dim=2)
    col_present = masks.any(dim=1)
    top = torch.where(row_present, ys, height).amin(dim=1)
    bottom = torch.where(row_present, ys, -1).amax(dim=1)
    left = torch.where(col_present, xs, width).amin(dim=1)
    right = torch.where(col_present, xs, -1).amax(dim=1)
    boxes = torch.stack((left, top, right, bottom), dim=1)
    empty = ~masks.reshape(count, -1).any(dim=1)
    boxes[empty] = 0
    return boxes


def _stability_score(logits, threshold: float, offset: float):
    """IoU between masks at stricter and looser logit thresholds."""

    intersection = (logits > threshold + offset).sum(dim=(-2, -1))
    union = (logits > threshold - offset).sum(dim=(-2, -1))
    return intersection.float() / union.clamp_min(1).float()


def _near_crop_edge(
    boxes,
    crop_box: tuple[int, int, int, int],
    image_box: tuple[int, int, int, int],
    tolerance: float = 20.0,
):
    """Identify masks truncated by an internal crop edge."""

    import torch

    x0, y0, _, _ = crop_box
    full_boxes = boxes + torch.tensor(
        [x0, y0, x0, y0], device=boxes.device
    )
    crop = torch.tensor(crop_box, dtype=torch.float32, device=boxes.device)
    image = torch.tensor(image_box, dtype=torch.float32, device=boxes.device)
    near_crop = torch.isclose(full_boxes.float(), crop[None], atol=tolerance, rtol=0)
    near_image = torch.isclose(full_boxes.float(), image[None], atol=tolerance, rtol=0)
    return (near_crop & ~near_image).any(dim=1)


def _encode_rle(mask) -> dict:
    """Encode one torch boolean mask as an uncompressed column-major RLE."""

    import torch

    height, width = mask.shape
    flat = mask.transpose(0, 1).reshape(-1)
    changes = torch.nonzero(flat[1:] != flat[:-1], as_tuple=False).flatten() + 1
    boundaries = torch.cat(
        (
            torch.tensor([0], device=flat.device),
            changes,
            torch.tensor([height * width], device=flat.device),
        )
    )
    lengths = (boundaries[1:] - boundaries[:-1]).cpu().tolist()
    counts = [] if not bool(flat[0]) else [0]
    counts.extend(int(value) for value in lengths)
    return {"size": (height, width), "counts": counts}


def _decode_rle(rle: dict) -> np.ndarray:
    height, width = rle["size"]
    flat = np.empty(height * width, dtype=bool)
    index = 0
    foreground = False
    for count in rle["counts"]:
        flat[index : index + count] = foreground
        index += count
        foreground = not foreground
    return flat.reshape(width, height).T


class Sam3AutomaticMaskGenerator:
    """SAM2-style automatic mask generation using the SAM3 image checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        source: str | Path | None = None,
        points_per_side: int = 64,
        points_per_batch: int = 32,
        pred_iou_threshold: float = 0.8,
        stability_threshold: float = 0.8,
        stability_offset: float = 1.0,
        crop_layers: int = 1,
        crop_downscale_factor: int = 2,
        box_nms_threshold: float = 0.7,
        crop_nms_threshold: float = 0.7,
        minimum_mask_area: int = 32,
        multimask_output: bool = True,
    ) -> None:
        self.checkpoint = str(Path(checkpoint).resolve())
        self.source = Path(source).resolve() if source is not None else None
        self.points_per_side = int(points_per_side)
        self.points_per_batch = int(points_per_batch)
        self.pred_iou_threshold = float(pred_iou_threshold)
        self.stability_threshold = float(stability_threshold)
        self.stability_offset = float(stability_offset)
        self.crop_layers = int(crop_layers)
        self.crop_downscale_factor = int(crop_downscale_factor)
        self.box_nms_threshold = float(box_nms_threshold)
        self.crop_nms_threshold = float(crop_nms_threshold)
        self.minimum_mask_area = int(minimum_mask_area)
        self.multimask_output = bool(multimask_output)
        self._model = None
        self._predictor = None
        self._processor = None
        self._numerical_state = None
        # Optional experiment-only capture of the full-frame image embedding.
        # It lets a caller generate proposals and appearance features from one
        # backbone pass while leaving the normal ``generate`` API untouched.
        self._capture_full_image_feature = False
        self._captured_full_image_feature: np.ndarray | None = None

    def load(self) -> None:
        """Build SAM3 with only the interactive image path enabled."""

        if self._predictor is not None:
            return
        import torch

        state = capture_torch_numerical_state(torch)
        self._numerical_state = state
        inserted = False
        try:
            if self.source is not None:
                if not (self.source / "sam3").is_dir():
                    raise FileNotFoundError(
                        f"Configured SAM3 source is unavailable: {self.source}"
                    )
                sys.path.insert(0, str(self.source))
                inserted = True
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            if self.source is not None:
                import sam3

                loaded_source = Path(sam3.__file__).resolve().parent.parent
                if loaded_source != self.source:
                    raise RuntimeError(
                        "SAM3 was imported from an unexpected checkout: "
                        f"configured={self.source}, loaded={loaded_source}. "
                        "Run full-pipeline inference in a clean pair worker."
                    )
            # model_builder enables TF32 only on its first import. Reapply the
            # same policy for every separately scoped SAM3 model lifetime.
            enable_sam3_numerics(torch)
            self._model = build_sam3_image_model(
                checkpoint_path=self.checkpoint,
                load_from_HF=False,
                device="cuda",
                eval_mode=True,
                enable_segmentation=False,
                enable_inst_interactivity=True,
                compile=False,
            )
            self._predictor = self._model.inst_interactive_predictor
            self._processor = Sam3Processor(self._model)
        except BaseException as load_error:
            predictor = self._predictor
            self._processor = None
            self._predictor = None
            self._model = None
            self._numerical_state = None
            try:
                exit_sam3_numerical_scope(predictor, state, torch)
            except BaseException as cleanup_error:
                # Keep the model/import failure as the primary exception while
                # retaining evidence that its cleanup also encountered a fault.
                raise load_error from cleanup_error
            raise
        finally:
            if inserted:
                source_entry = str(self.source)
                if sys.path and sys.path[0] == source_entry:
                    sys.path.pop(0)
                else:
                    # Defensive fallback for import hooks that edit sys.path.
                    try:
                        sys.path.remove(source_entry)
                    except ValueError:
                        pass

    def release(self) -> None:
        """Release the image model before loading the SAM3.1 video tracker."""

        if self._model is None and self._numerical_state is None:
            return
        import torch

        predictor = self._predictor
        state = self._numerical_state
        # Detach owned resources before running fallible cleanup. A retry after
        # an exception is therefore a no-op instead of closing/restoring twice.
        self._processor = None
        self._predictor = None
        self._model = None
        self._numerical_state = None

        cleanup_error: BaseException | None = None
        # Upstream SAM3 manually enters this context in its predictor
        # constructor. Close it before dropping the owning model reference.
        try:
            exit_sam3_numerical_scope(predictor, state, torch)
        except BaseException as exc:
            cleanup_error = exc
        try:
            gc.collect()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
        if cleanup_error is not None:
            raise cleanup_error

    @staticmethod
    def _filter_records(records: list[dict], keep) -> list[dict]:
        indices = keep.detach().cpu().tolist()
        return [records[int(index)] for index in indices]

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: tuple[int, int, int, int],
        layer: int,
    ) -> list[dict]:
        import torch
        from torchvision.ops import nms

        assert self._predictor is not None
        image_height, image_width = image.shape[:2]
        x0, y0, x1, y1 = crop_box
        crop = image[y0:y1, x0:x1]
        crop_height, crop_width = crop.shape[:2]
        # SAM3's image model owns the shared vision backbone.  Its interactive
        # tracker intentionally has no private backbone, so populate the
        # predictor features from the image-model state once per crop.
        state = self._processor.set_image(Image.fromarray(np.array(crop, copy=True)))
        backbone_out = state["backbone_out"]["sam2_backbone_out"]
        _, vision_features, _, _ = (
            self._predictor.model._prepare_backbone_features(backbone_out)
        )
        if (
            self._capture_full_image_feature
            and layer == 0
            and crop_box == (0, 0, image_width, image_height)
        ):
            # Match Sam3FeatureExtractor: capture the lowest-resolution image
            # embedding before the mask decoder's no-memory constant is added.
            import torch.nn.functional as functional

            feature_size = self._predictor._bb_feat_sizes[-1]
            embedding = vision_features[-1].permute(1, 2, 0).reshape(
                1, -1, *feature_size
            )[0]
            self._captured_full_image_feature = (
                functional.normalize(embedding.float(), dim=0)
                .cpu()
                .numpy()
                .astype(np.float16, copy=True)
            )
        vision_features[-1] = (
            vision_features[-1] + self._predictor.model.no_mem_embed
        )
        features = [
            feature.permute(1, 2, 0).view(1, -1, *feature_size)
            for feature, feature_size in zip(
                vision_features[::-1],
                self._predictor._bb_feat_sizes[::-1],
            )
        ][::-1]
        self._predictor._features = {
            "image_embed": features[-1],
            "high_res_feats": features[:-1],
        }
        self._predictor._is_image_set = True
        self._predictor._orig_hw = [(crop_height, crop_width)]

        side = max(
            1,
            int(self.points_per_side / (self.crop_downscale_factor**layer)),
        )
        normalized_grid = _point_grid(side)
        points = normalized_grid * np.array([crop_width, crop_height])[None]
        records: list[dict] = []
        for start in range(0, len(points), self.points_per_batch):
            pixel_points = points[start : start + self.points_per_batch]
            points_tensor = torch.as_tensor(
                pixel_points,
                dtype=torch.float32,
                device=self._predictor.device,
            )
            transformed = self._predictor._transforms.transform_coords(
                points_tensor,
                normalize=True,
                orig_hw=(crop_height, crop_width),
            )
            labels = torch.ones(
                (len(points_tensor), 1),
                dtype=torch.int32,
                device=points_tensor.device,
            )
            logits, iou_scores, _ = self._predictor._predict(
                transformed[:, None],
                labels,
                multimask_output=self.multimask_output,
                return_logits=True,
            )
            logits = logits.flatten(0, 1)
            iou_scores = iou_scores.flatten()
            repeated_points = points_tensor.repeat_interleave(
                logits.shape[0] // len(points_tensor), dim=0
            )

            keep = iou_scores > self.pred_iou_threshold
            logits = logits[keep]
            iou_scores = iou_scores[keep]
            repeated_points = repeated_points[keep]
            stability = _stability_score(logits, 0.0, self.stability_offset)
            keep = stability >= self.stability_threshold
            logits = logits[keep]
            iou_scores = iou_scores[keep]
            repeated_points = repeated_points[keep]
            stability = stability[keep]

            masks = logits > 0.0
            areas = masks.sum(dim=(-2, -1))
            keep = areas >= self.minimum_mask_area
            masks = masks[keep]
            iou_scores = iou_scores[keep]
            repeated_points = repeated_points[keep]
            stability = stability[keep]
            boxes = _mask_boxes(masks)
            keep = ~_near_crop_edge(
                boxes,
                crop_box,
                (0, 0, image_width, image_height),
            )
            masks = masks[keep]
            iou_scores = iou_scores[keep]
            repeated_points = repeated_points[keep]
            stability = stability[keep]
            boxes = boxes[keep]

            for mask, box, score, stable, point in zip(
                masks, boxes, iou_scores, stability, repeated_points
            ):
                # Padding before RLE keeps dense GPU masks crop-sized.
                full_mask = torch.zeros(
                    (image_height, image_width),
                    dtype=torch.bool,
                    device=mask.device,
                )
                full_mask[y0:y1, x0:x1] = mask
                full_box = box + torch.tensor(
                    [x0, y0, x0, y0], device=box.device
                )
                records.append(
                    {
                        "rle": _encode_rle(full_mask),
                        "box": full_box.detach().float().cpu(),
                        "predicted_iou": float(score.detach().float().cpu()),
                        "stability_score": float(stable.detach().float().cpu()),
                        "point_xy": (
                            float(point[0].detach().float().cpu()) + x0,
                            float(point[1].detach().float().cpu()) + y0,
                        ),
                        "crop_box": crop_box,
                    }
                )
            del logits, masks

        self._predictor.reset_predictor()
        if not records:
            return []
        boxes = torch.stack([record["box"] for record in records])
        scores = torch.tensor(
            [record["predicted_iou"] for record in records],
            dtype=torch.float32,
        )
        return self._filter_records(
            records, nms(boxes, scores, self.box_nms_threshold)
        )

    def generate(self, image: np.ndarray) -> list[Sam3Proposal]:
        """Generate deterministic class-agnostic masks for one RGB image."""

        import torch
        from torchvision.ops import nms

        self.load()
        rgb = np.asarray(image, dtype=np.uint8)
        boxes, layers = _crop_boxes(rgb.shape[:2], self.crop_layers)
        records: list[dict] = []
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            for crop_box, layer in zip(boxes, layers):
                records.extend(self._process_crop(rgb, crop_box, layer))
        if len(boxes) > 1 and records:
            full_boxes = torch.stack([record["box"] for record in records])
            crop_scores = torch.tensor(
                [
                    1.0
                    / float(
                        (record["crop_box"][2] - record["crop_box"][0])
                        * (record["crop_box"][3] - record["crop_box"][1])
                    )
                    for record in records
                ],
                dtype=torch.float32,
            )
            records = self._filter_records(
                records, nms(full_boxes, crop_scores, self.crop_nms_threshold)
            )

        return [
            Sam3Proposal(
                mask=_decode_rle(record["rle"]),
                predicted_iou=record["predicted_iou"],
                stability_score=record["stability_score"],
                point_xy=record["point_xy"],
                crop_box_xyxy=record["crop_box"],
            )
            for record in records
        ]

    def generate_with_feature_map(
        self, image: np.ndarray
    ) -> tuple[list[Sam3Proposal], np.ndarray]:
        """Return point-grid proposals and their shared real-image embedding.

        The proposal path is byte-identical to :meth:`generate`; this method
        only retains the full-frame backbone tensor that path already computes.
        """

        self._capture_full_image_feature = True
        self._captured_full_image_feature = None
        try:
            proposals = self.generate(image)
            feature = self._captured_full_image_feature
            if feature is None:
                raise RuntimeError("SAM3 full-image feature capture did not run")
            return proposals, feature.copy()
        finally:
            self._capture_full_image_feature = False
            self._captured_full_image_feature = None
