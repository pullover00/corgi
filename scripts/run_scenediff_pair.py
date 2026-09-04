#!/usr/bin/env python3
"""Run the full pipeline on one SceneDiff benchmark pair: extract frames,
reconstruct, optionally refine, and detect changes.

Handles two SceneDiff-specific data quirks discovered during development
(see docs/METHODS.md's "Data preparation" note):

1. ``video1.mp4``/``video2.mp4`` have some objects repainted with flat
   synthetic colors (a review-tool visualization artifact). The real footage
   is ``original_video{1,2}.*`` -- always used here.
2. Those original files carry sensor-orientation metadata that OpenCV does
   not apply by default; ``CAP_PROP_ORIENTATION_AUTO`` is required or frames
   come out sideways.

Run stage 1 (this script) in the vggt-omega conda env; it shells out to the
other two envs for stages 2/3 via ``conda run``, same as run_pipeline.sh.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DIFIX_ENV = os.environ.get("DIFIX3D_CONDA_ENV", "difix3d")
DETECT_ENV = os.environ.get("DETECTION_CONDA_ENV", "goldilocs")


def resolve_original_video(pair_dir: Path, video_number: int) -> Path:
    prefix = f"original_video{video_number}"
    for name in sorted(pair_dir.iterdir()):
        if name.stem == prefix and name.suffix.lower() in (".mp4", ".mov"):
            return name
    raise FileNotFoundError(f"no {prefix}.(mp4|mov|MOV) found in {pair_dir}")


def representative_frame_index(objects: list[dict], video_key: str, frame_key: str) -> int:
    counts = Counter(int(obj[frame_key]) for obj in objects if obj.get(video_key) and int(obj.get(frame_key, -1)) >= 0)
    return counts.most_common(1)[0][0] if counts else 0


def sample_frame_indices(frame_count: int, num_samples: int) -> list[int]:
    if num_samples >= frame_count:
        return list(range(frame_count))
    step = (frame_count - 1) / (num_samples - 1)
    indices = sorted({round(i * step) for i in range(num_samples)})
    if 0 not in indices:
        indices = [0] + indices
    return sorted(set(indices))


def extract_frames(video_path: Path, indices: list[int], out_dir: Path, prefix: str) -> list[Path]:
    import cv2
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"could not read frame {index} from {video_path}")
            out_path = out_dir / f"{prefix}_{index:05d}.png"
            Image.fromarray(frame_bgr[:, :, ::-1]).save(out_path)
            paths.append(out_path)
    finally:
        capture.release()
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-root", type=Path, required=True, help="path to the scenediff_benchmark data root")
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--frames-per-video", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO / "configs/pipeline.yaml")
    parser.add_argument("--skip-refine", action="store_true")
    args = parser.parse_args()

    import json
    import pickle

    pair_dir = args.benchmark_root / "data" / args.pair_id
    segments = pickle.loads((pair_dir / "segments.pkl").read_bytes())
    objects = segments["objects"]
    t0_reference_frame = representative_frame_index(objects, "in_video1", "video1_frame_idx")
    t1_reference_frame = representative_frame_index(objects, "in_video2", "video2_frame_idx")

    import cv2

    def frame_count(video_path: Path) -> int:
        capture = cv2.VideoCapture(str(video_path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        return count

    video1 = resolve_original_video(pair_dir, 1)
    video2 = resolve_original_video(pair_dir, 2)
    t0_indices = sample_frame_indices(frame_count(video1), args.frames_per_video)
    t1_indices = sample_frame_indices(frame_count(video2), args.frames_per_video)
    if t0_reference_frame not in t0_indices:
        t0_indices = sorted(t0_indices + [t0_reference_frame])
    if t1_reference_frame not in t1_indices:
        t1_indices = sorted(t1_indices + [t1_reference_frame])

    frames_dir = args.output_dir / "frames"
    t0_frames = extract_frames(video1, t0_indices, frames_dir, "t0")
    t1_frames = extract_frames(video2, t1_indices, frames_dir, "t1")
    t0_reference_index = t0_indices.index(t0_reference_frame)
    t1_reference_index = t1_indices.index(t1_reference_frame)

    from ocmask_pipeline.config import load_config
    from ocmask_pipeline.reconstruction import reconstruct_and_render
    from PIL import Image

    config = load_config(args.config)
    result = reconstruct_and_render(t0_frames, t1_frames, t0_reference_index, t1_reference_index, config)

    recon_dir = args.output_dir / "reconstruction"
    recon_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.render_t0).save(recon_dir / "render_t0.png")
    Image.fromarray(result.clean_render).save(recon_dir / "clean_render.png")
    Image.fromarray(result.image_t1).save(recon_dir / "image_t1.png")
    print(f"wrote reconstruction to {recon_dir}")

    render_t0, clean_render, image_t1 = recon_dir / "render_t0.png", recon_dir / "clean_render.png", recon_dir / "image_t1.png"

    if not args.skip_refine:
        refined_dir = args.output_dir / "refined"
        subprocess.run(
            [
                "conda", "run", "-n", DIFIX_ENV, "python", str(REPO / "scripts/refine.py"),
                "--render-t0", str(render_t0), "--clean-render", str(clean_render), "--image-t1", str(image_t1),
                "--config", str(args.config), "--output-dir", str(refined_dir),
            ],
            check=True,
        )
        render_t0, clean_render = refined_dir / "render_t0.png", refined_dir / "clean_render.png"

    result_dir = args.output_dir / "result"
    subprocess.run(
        [
            "conda", "run", "-n", DETECT_ENV, "python", str(REPO / "scripts/detect.py"),
            "--render-t0", str(render_t0), "--clean-render", str(clean_render), "--image-t1", str(image_t1),
            "--config", str(args.config), "--output-dir", str(result_dir),
        ],
        check=True,
    )
    print(f"done: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
