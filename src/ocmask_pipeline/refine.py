"""Refine render_t0/clean_render with DI2FIX's DifixPipeline before the
change-detection stage.

Difix is a single-step, reference-conditioned diffusion model trained to
remove rendering artifacts from underconstrained regions of a 3D
reconstruction. Both rendered images are passed as ``image``; the real
target photo (``image_t1``) is passed as ``ref_image`` so the model has a
ground-truth view of the actual current scene to condition on.

Caveat (see docs/METHODS.md): this can hallucinate plausible-looking content
into genuine gaps in ``clean_render`` left by the depth-conflict filter.
That has not caused a bad change decision in testing so far, because
``clean_render``'s objects only ever serve as an identity-confirmation
bridge in change_detection.py -- never an independent source of an
added/removed verdict -- but it has not been stress-tested broadly.

Must run in a conda env with DI2FIX's ``src/`` on the path and a
diffusers/torch pin matching pipeline_difix.py (see README.md) -- this is
deliberately a separate environment from both vggt-omega and the
SAM3/DINOv2/SAM2 stage.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_refiner(config: dict[str, Any]):
    """Load DI2FIX's DifixPipeline onto the GPU. Split out of refine_renders
    so a resident worker (scripts/refine_worker.py) can load it ONCE: the
    load is ~2.5 min of a ~2.7 min refine stage (measured 2026-09-09), the
    single denoising step on two images is seconds."""
    refine_cfg = config["refine"]
    di2fix_src = Path(refine_cfg["di2fix_root"]) / "src"
    if str(di2fix_src) not in sys.path:
        sys.path.insert(0, str(di2fix_src))

    import torch
    from pipeline_difix import DifixPipeline

    # trust_remote_code is required by diffusers' loader for this repo (it
    # ships custom unet/vae code) even though we import DifixPipeline from
    # DI2FIX's own local src/, not the remote copy -- diffusers checks the
    # repo metadata before it knows that.
    pipe = DifixPipeline.from_pretrained(refine_cfg["model"], torch_dtype=torch.float16, trust_remote_code=True)
    pipe.to("cuda")
    return pipe


def refine_with(pipe, render_t0: np.ndarray, clean_render: np.ndarray, image_t1: np.ndarray,
                config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """refine_renders with an already-loaded pipeline."""
    refine_cfg = config["refine"]
    ref_image = Image.fromarray(np.asarray(image_t1, dtype=np.uint8))
    height, width = ref_image.size[1], ref_image.size[0]

    def fix(image: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        output = pipe(
            refine_cfg["prompt"],
            image=pil_image,
            ref_image=ref_image,
            height=height,
            width=width,
            num_inference_steps=int(refine_cfg["num_inference_steps"]),
            timesteps=[int(refine_cfg["timestep"])],
            guidance_scale=float(refine_cfg["guidance_scale"]),
        ).images[0]
        return np.asarray(output, dtype=np.uint8)

    return fix(render_t0), fix(clean_render)


def refine_renders(render_t0: np.ndarray, clean_render: np.ndarray, image_t1: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return (fixed_render_t0, fixed_clean_render); image_t1 is untouched
    and used only as the reference image. Loads the model per call -- see
    load_refiner for the resident alternative."""
    return refine_with(load_refiner(config), render_t0, clean_render, image_t1, config)
