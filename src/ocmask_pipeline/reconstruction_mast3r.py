"""MASt3R-based reference-scene reconstruction, for an ablation against the
VGGT-Omega path in reconstruction.py.

Unlike VGGT-Omega's single feed-forward pass over all frames, MASt3R
reconstructs a scene by pairwise-matching images (``make_pairs``) and then
jointly refining poses/points across the whole set with iterative
optimization (``sparse_global_alignment``). This is the same upstream API
the original (pre-cleanup) change_detect repo's two-image
``Mast3rAdapter.reconstruct`` used -- generalized here from exactly 2 images
to an arbitrary reference set, with the same call parameters. Cost scales
with the number of pairs the chosen scene graph produces, which for
``scenegraph_type="complete"`` is quadratic in the image count -- see
``build_paslcd_reference_scenes_mast3r.py`` for how that's kept tractable
for PASLCD's ~24-image reference sets.

Produces the same reconstruction.ReferenceScene shape as
reconstruct_reference_scene, so both methods' outputs can be compared/loaded
uniformly for the ablation.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .reconstruction import ReferenceScene


def _configure_mast3r_paths(mast3r_root: str | Path) -> None:
    """Expose upstream MASt3R and its bundled DUSt3R submodule for import.

    MASt3R ships with no Python packaging metadata; its own demos rely on
    being run from the repo root, which implicitly puts these directories on
    sys.path. Mirrors change_detect/src/ocmask/model_paths.configure_mast3r_paths.
    """
    checkout = Path(mast3r_root)
    package = checkout / "mast3r"
    dust3r_package = checkout / "dust3r" / "dust3r"
    if not package.is_dir() or not dust3r_package.is_dir():
        raise RuntimeError(f"MASt3R sources are incomplete at {checkout}")
    for source_root in (checkout, checkout / "dust3r"):
        value = str(source_root)
        if value not in sys.path:
            sys.path.insert(0, value)


def reconstruct_reference_scene_mast3r(
    image_paths: list[str | Path], config: dict[str, Any]
) -> ReferenceScene:
    """Reconstruct a reference scene from its own image set with MASt3R's
    sparse global alignment, in isolation (no query photo), producing the
    same ReferenceScene shape reconstruction.reconstruct_reference_scene
    does for VGGT-Omega."""

    cfg = config["reconstruction_mast3r"]
    _configure_mast3r_paths(cfg["mast3r_root"])

    import torch
    from dust3r.utils.image import load_images
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.image_pairs import make_pairs
    from mast3r.model import AsymmetricMASt3R

    device = cfg.get("device", "cuda")
    model = AsymmetricMASt3R.from_pretrained(cfg["checkpoint"]).to(device).eval()

    image_paths = [str(p) for p in image_paths]
    with tempfile.TemporaryDirectory(prefix="ocmask-pipeline-mast3r-") as temporary:
        images = load_images(image_paths, size=512, verbose=False)
        pairs = make_pairs(images, scene_graph=cfg["scenegraph_type"], prefilter=None, symmetrize=True)
        cache_path = str(Path(temporary) / "cache")
        Path(cache_path).mkdir()

        # See Mast3rAdapter.reconstruct in the original change_detect repo:
        # SAM3's predictor classes leave BF16 autocast enabled process-wide
        # once constructed; sparse_global_alignment's reciprocal-NN code
        # cannot run under that. Not applicable to this process today (no
        # SAM3 in this env), kept anyway since it costs nothing and this
        # function may end up called from a shared process later.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
            scene = sparse_global_alignment(
                image_paths,
                pairs,
                cache_path,
                model,
                lr1=cfg["lr1"],
                niter1=cfg["niter1"],
                lr2=cfg["lr2"],
                niter2=cfg["niter2"],
                device=device,
                opt_depth="depth" in cfg["optim_level"],
                shared_intrinsics=cfg["shared_intrinsics"],
                matching_conf_thr=cfg["matching_conf_thr"],
                desc_conf=cfg["desc_conf_output_key"],
                subsample=cfg["subsample"],
            )

        points_t, depths_t, confidences_t = scene.get_dense_pts3d(clean_depth=False, subsample=cfg["subsample"])
        poses_t = scene.get_im_poses()
        intrinsics_t = scene.intrinsics
        scene_images = scene.imgs

    num_frames = len(image_paths)
    height, width = confidences_t[0].detach().cpu().numpy().shape
    world_points = np.zeros((num_frames, height, width, 3), dtype=np.float32)
    depth_conf = np.zeros((num_frames, height, width), dtype=np.float32)
    colors = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    extrinsic = np.zeros((num_frames, 3, 4), dtype=np.float32)
    intrinsic = np.zeros((num_frames, 3, 3), dtype=np.float32)

    for i in range(num_frames):
        world_points[i] = points_t[i].detach().cpu().numpy().astype(np.float32).reshape(height, width, 3)
        depth_conf[i] = confidences_t[i].detach().cpu().numpy().astype(np.float32).reshape(height, width)
        colors[i] = np.clip(np.asarray(scene_images[i]) * 255, 0, 255).astype(np.uint8)
        cam_to_world = poses_t[i].detach().cpu().numpy().astype(np.float32)
        extrinsic[i] = np.linalg.inv(cam_to_world)[:3, :]
        intrinsic[i] = intrinsics_t[i].detach().cpu().numpy().astype(np.float32)

    del model
    torch.cuda.empty_cache()

    return ReferenceScene(
        image_paths=[Path(p) for p in image_paths],
        world_points=world_points,
        colors=colors,
        depth_conf=depth_conf,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
    )
