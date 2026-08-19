from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

from .adapters.base import SegmentationAdapter
from .io import load_rgb, pair_key, save_image, save_json
from .masks import compose_labels, mark_replacements, mask_iou
from .types import Label, ObjectMask, PairResult
from .visualization import colorize, overlay


class No3DPipeline:
    """Direct 2D counterpart used only for the paper's no-3D trend ablation.

    The paper does not specify this control in enough detail for exact parity.
    This implementation removes all reconstruction/viewpoint alignment while
    retaining the same segmentation, tracking, SSIM, and label composition.
    """

    def __init__(self, config: dict, segmentation: SegmentationAdapter):
        self.config = config
        self.segmentation = segmentation

    def run(self, image0_path, image1_path, output_root, **_) -> PairResult:
        """Run the inferred direct-2D ablation and write compatible artifacts."""
        started = time.perf_counter()
        size = (self.config["image"]["width"], self.config["image"]["height"])
        image0, image1 = load_rgb(image0_path, size), load_rgb(image1_path, size)
        key = "no3d-" + pair_key(image0_path, image1_path, self.config)
        artifacts = Path(output_root) / key
        artifacts.mkdir(parents=True, exist_ok=True)
        source = self.segmentation.generate(image0)
        source_tracks = self.segmentation.track(source, image0, image1)
        removed, moved = [], []
        movement_iou = self.config["tracking"]["no3d_movement_iou"]
        # Without geometry, mask displacement in image coordinates is the only
        # available movement cue. Failed forward tracks are treated as removals.
        for obj, tracked in zip(source, source_tracks):
            if tracked is None:
                obj.label = Label.REMOVED
                obj.source = "no3d_source_track_failure"
                removed.append(obj)
            elif mask_iou(obj.mask, tracked.mask) < movement_iou:
                tracked.label = Label.MOVED
                tracked.source = "no3d_spatial_track"
                moved.append(tracked)

        target = self.segmentation.generate(image1)
        # The reverse pass isolates additions that have no T0 correspondence.
        target_tracks = self.segmentation.track(target, image1, image0)
        added = []
        for obj, tracked in zip(target, target_tracks):
            if tracked is None:
                obj.label = Label.ADDED
                obj.source = "no3d_target_track_failure"
                added.append(obj)

        cfg = self.config["ssim"]
        _, similarity = structural_similarity(
            image0,
            image1,
            channel_axis=cfg["channel_axis"],
            data_range=255,
            win_size=cfg["win_size"],
            gaussian_weights=cfg["gaussian_weights"],
            sigma=cfg["sigma"],
            full=True,
        )
        dissimilarity = 1 - similarity
        changed = np.zeros(image1.shape[:2], bool)
        for obj in added + moved:
            changed |= obj.mask
        static = [obj for obj in target if not np.logical_and(obj.mask, changed).any()]
        scores = np.array([dissimilarity[obj.mask].mean() for obj in static])
        warped = []
        if len(scores):
            threshold = scores.mean() + cfg["threshold_stddevs"] * scores.std()
            for obj, score in zip(static, scores):
                if score > threshold:
                    obj.label = Label.WARPED
                    obj.source = "no3d_ssim"
                    warped.append(obj)
        objects = added + removed + moved + warped
        labels = compose_labels(image1.shape[:2], objects)
        labels = mark_replacements(
            labels, added, removed, self.config["tracking"]["replacement_overlap_iou"]
        )
        binary = (labels != Label.UNCHANGED).astype(np.uint8)
        save_image(artifacts / "labels.png", labels)
        save_image(artifacts / "binary.png", binary * 255)
        save_image(artifacts / "labels_color.png", colorize(labels))
        save_image(artifacts / "overlay.png", overlay(image1, labels))
        save_image(artifacts / "ssim_dissimilarity.png", dissimilarity)
        timings = {"no3d_total": time.perf_counter() - started}
        save_json(artifacts / "inputs.json", {"image0": str(Path(image0_path).resolve()), "image1": str(Path(image1_path).resolve())})
        save_json(artifacts / "metadata.json", {"ablation": "no-3d", "timings_seconds": timings})
        return PairResult(labels, binary, objects, artifacts, timings)
