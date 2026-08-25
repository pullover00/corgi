from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .base import ReconstructionAdapter
from ..model_paths import configure_mast3r_paths
from ..types import Reconstruction


class Mast3rAdapter(ReconstructionAdapter):
    """Thin adapter around the upstream MASt3R/DUSt3R public APIs."""

    def __init__(self, config: dict):
        self.config = config
        self._model = None

    @staticmethod
    def _desc_conf_output_key(config: dict) -> str:
        """Translate the paper's confidence name to MASt3R's result key.

        GOLDILOCS reports ``desc_conf='3d'`` as a method-level setting. The
        pinned MASt3R implementation stores that confidence tensor in its
        ``desc_conf`` result entry; it has no result entry named ``3d``.
        """
        configured = config.get("desc_conf_output_key")
        if configured:
            return str(configured)
        value = config.get("desc_conf", "desc_conf")
        return "desc_conf" if value == "3d" else str(value)

    @staticmethod
    def _reshape_dense_outputs(points, depths, confidences):
        """Restore MASt3R's flattened dense outputs to image-shaped arrays."""
        shaped_points = []
        shaped_depths = []
        shaped_confidences = []
        for pointmap, depthmap, confidence in zip(points, depths, confidences):
            confidence_np = confidence.detach().cpu().numpy().astype(np.float32)
            height, width = confidence_np.shape
            pointmap_np = (
                pointmap.detach().cpu().numpy().astype(np.float32).reshape(height, width, 3)
            )
            depthmap_np = (
                depthmap.detach().cpu().numpy().astype(np.float32).reshape(height, width)
            )
            shaped_points.append(pointmap_np)
            shaped_depths.append(depthmap_np)
            shaped_confidences.append(confidence_np)
        return tuple(shaped_points), tuple(shaped_depths), tuple(shaped_confidences)

    def _imports(self):
        """Delay heavyweight imports so geometry tests do not require CUDA."""
        # Upstream MASt3R has no Python packaging metadata. Register its pinned
        # source checkout before importing MASt3R and its DUSt3R submodule.
        configure_mast3r_paths()
        try:
            import torch
            from dust3r.utils.image import load_images
            from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
            from mast3r.image_pairs import make_pairs
            from mast3r.model import AsymmetricMASt3R
        except ImportError as exc:
            raise RuntimeError(
                "MASt3R is not installed. Follow README.md and install requirements-models.txt."
            ) from exc
        return torch, sparse_global_alignment, make_pairs, load_images, AsymmetricMASt3R

    def reconstruct(self, image0: np.ndarray, image1: np.ndarray) -> Reconstruction:
        """Reconstruct both images in one optimized world coordinate frame."""
        torch, sparse_global_alignment, make_pairs, load_images, AsymmetricMASt3R = self._imports()
        device = self.config.get("device", "cuda")
        cfg = self.config["mast3r"]
        if self._model is None:
            # ``from_pretrained`` accepts either the configured local checkpoint
            # or an upstream model identifier. Reproduction configs use local
            # files so checkpoint hashes can be recorded.
            self._model = AsymmetricMASt3R.from_pretrained(cfg["checkpoint"]).to(device).eval()

        with tempfile.TemporaryDirectory(prefix="ocmask-mast3r-") as temporary:
            paths = [Path(temporary) / "0.png", Path(temporary) / "1.png"]
            Image.fromarray(image0).save(paths[0])
            Image.fromarray(image1).save(paths[1])
            # MASt3R owns its inference resolution and preserves scale metadata
            # needed to produce correctly calibrated dense outputs.
            images = load_images([str(path) for path in paths], size=512, verbose=False)
            pairs = make_pairs(
                images, scene_graph=cfg["scenegraph_type"], prefilter=None, symmetrize=True
            )
            cache_path = str(Path(temporary) / "cache")
            Path(cache_path).mkdir()
            # Force CUDA autocast off around sparse_global_alignment. MASt3R's
            # reciprocal-NN implementation allocates FP32 distance buffers and
            # assigns descriptor-derived values into them; under BF16 autocast
            # that assignment becomes an index_put_ between a bf16 source and
            # fp32 destination, which PyTorch refuses.
            #
            # This isn't about anything *this* function does: SAM3's own
            # predictor classes (sam3_multiplex_base.py and friends, in the
            # installed sam3 package) call `torch.autocast(...).__enter__()`
            # at construction time with no matching __exit__ -- "keep using
            # for the entire model process" per their own comment -- which
            # leaves BF16 autocast enabled process-wide, ambiently, for every
            # later CUDA op for the rest of the run once any SAM3 predictor
            # has been built (e.g. by an earlier pair's proposal/sentinel
            # stage). A plain unwrapped call silently inherits that leaked
            # state; only an explicit enabled=False forces it off here
            # regardless of what a prior stage or pair left behind.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
                scene = sparse_global_alignment(
                    [str(path) for path in paths],
                    pairs,
                    cache_path,
                    self._model,
                    lr1=cfg["lr1"],
                    niter1=cfg["niter1"],
                    lr2=cfg["lr2"],
                    niter2=cfg["niter2"],
                    device=device,
                    opt_depth="depth" in cfg["optim_level"],
                    shared_intrinsics=cfg["shared_intrinsics"],
                    matching_conf_thr=cfg["matching_conf_thr"],
                    desc_conf=self._desc_conf_output_key(cfg),
                    subsample=cfg["subsample"],
                )
            # Deliberately disable upstream depth cleaning. The stride here must
            # equal sparse_global_alignment's stride: optimized depthmaps contain
            # one anchor per stride-sized block. MASt3R still returns a point for
            # every image pixel by applying dense canonical depth offsets.
            points_t, depths_t, confidences_t = scene.get_dense_pts3d(
                clean_depth=False, subsample=cfg["subsample"]
            )
            poses_t = scene.get_im_poses()
            intrinsics_t = scene.intrinsics
            scene_images = scene.imgs

            points, depths, confidence = self._reshape_dense_outputs(
                points_t, depths_t, confidences_t
            )
            cam_to_world = tuple(value.detach().cpu().numpy().astype(np.float32) for value in poses_t)
            world_to_camera = tuple(np.linalg.inv(value) for value in cam_to_world)
            images_np = tuple(
                np.clip(np.asarray(value) * 255, 0, 255).astype(np.uint8) for value in scene_images
            )
            intrinsics = tuple(value.detach().cpu().numpy().astype(np.float32) for value in intrinsics_t)
            match_count = int(sum(np.count_nonzero(value > 0) for value in confidence))
        return Reconstruction(
            images=images_np,
            points=points,
            depths=depths,
            intrinsics=intrinsics,
            world_to_camera=world_to_camera,
            confidence=confidence,
            match_count=match_count,
        )

    def release(self) -> None:
        """Drop MASt3R before SAM2 loads on the 16 GiB target GPU."""
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
