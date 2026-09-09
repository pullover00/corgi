#!/usr/bin/env python3
"""Reconstruct all selected SceneDiff pairs, refine each with DI2FIX (a real
pipeline stage, not optional -- was silently missing from every SceneDiff run
until 2026-09-08), then run detect_batch.py ONCE across all of them instead
of once per pair -- avoids paying the ~150s SAM3/DINOv2/SAM2 cold-start
model-load cost separately per pair. Run in the vggt-omega env; shells out to
DIFIX3D_CONDA_ENV for refine and DETECTION_CONDA_ENV for the batched detect
stage, same convention as run_paslcd_pair.py.

Methodology (changed 2026-09-08): reconstruct the t0 ("before") scene from
its full multi-frame set via reconstruct_reference_scene, then localize and
render a SINGLE t1 query frame against it via localize_and_render_query --
same reference-set-vs-single-query pattern as PASLCD, instead of jointly
reconstructing t0 and t1 from multiple frames each. This removes a confound
in cross-dataset comparisons (PASLCD's 24-view t0 vs SceneDiff's previous
~10-frame-both-sides scheme were not doing the same thing) and matches how
change is actually queried in the target use case: one new observation
compared against an existing reconstruction, not two observation sets.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DETECT_ENV = os.environ.get("DETECTION_CONDA_ENV", "goldilocs")
DIFIX_ENV = os.environ.get("DIFIX3D_CONDA_ENV", "difix3d")


def representative_frame_index(objects, video_key, frame_key):
    from collections import Counter
    counts = Counter(int(obj[frame_key]) for obj in objects if obj.get(video_key) and int(obj.get(frame_key, -1)) >= 0)
    return counts.most_common(1)[0][0] if counts else 0


def sample_frame_indices(frame_count, num_samples):
    if num_samples >= frame_count:
        return list(range(frame_count))
    step = (frame_count - 1) / (num_samples - 1)
    indices = sorted({round(i * step) for i in range(num_samples)})
    if 0 not in indices:
        indices = [0] + indices
    return sorted(set(indices))


def resolve_original_video(pair_dir: Path, video_number: int) -> Path:
    prefix = f"original_video{video_number}"
    for name in sorted(pair_dir.iterdir()):
        if name.stem == prefix and name.suffix.lower() in (".mp4", ".mov"):
            return name
    raise FileNotFoundError(f"no {prefix}.(mp4|mov|MOV) found in {pair_dir}")


def extract_frames(video_path, indices, out_dir, prefix):
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--pair-ids-file", type=Path, required=True)
    parser.add_argument("--frames-per-video", type=int, default=10)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-refine", action="store_true",
                         help="explicit opt-out only -- refine is a real pipeline stage, on by default")
    args = parser.parse_args()

    from ocmask_pipeline.config import load_config
    from ocmask_pipeline.reconstruction import reconstruct_reference_scene, localize_and_render_query
    from PIL import Image
    import cv2

    def frame_count(video_path):
        capture = cv2.VideoCapture(str(video_path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        return count

    args.output_root.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    pair_ids = [line.strip() for line in args.pair_ids_file.read_text().splitlines() if line.strip()]

    manifest = []
    query_meta = {}
    for pair_id in pair_ids:
        pair_dir = args.benchmark_root / "data" / pair_id
        out_dir = args.output_root / pair_id
        try:
            segments = pickle.loads((pair_dir / "segments.pkl").read_bytes())
            objects = segments["objects"]
            t0_reference_frame = representative_frame_index(objects, "in_video1", "video1_frame_idx")
            t1_reference_frame = representative_frame_index(objects, "in_video2", "video2_frame_idx")

            video1 = resolve_original_video(pair_dir, 1)
            video2 = resolve_original_video(pair_dir, 2)
            t0_indices = sample_frame_indices(frame_count(video1), args.frames_per_video)
            if t0_reference_frame not in t0_indices:
                t0_indices = sorted(t0_indices + [t0_reference_frame])

            frames_dir = out_dir / "frames"
            t0_frames = extract_frames(video1, t0_indices, frames_dir, "t0")
            t1_frames = extract_frames(video2, [t1_reference_frame], frames_dir, "t1")
            query_frame = t1_frames[0]

            # Reference-set-vs-single-query: reconstruct t0 from its full
            # frame set alone, then localize+render only the single t1 query
            # frame against it (same pattern as PASLCD's run_paslcd_pair.py).
            reference_scene = reconstruct_reference_scene(t0_frames, config)
            result = localize_and_render_query(t0_frames, query_frame, reference_scene, 0, config)
            print(f"  [{pair_id}] alignment residual: {result.alignment_residual}", flush=True)

            recon_dir = out_dir / "reconstruction"
            recon_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(result.render_t0).save(recon_dir / "render_t0.png")
            Image.fromarray(result.clean_render).save(recon_dir / "clean_render.png")
            Image.fromarray(result.image_t1).save(recon_dir / "image_t1.png")

            render_t0_path, clean_render_path = recon_dir / "render_t0.png", recon_dir / "clean_render.png"
            if not args.skip_refine and config.get("refine", {}).get("enabled", False):
                refined_dir = out_dir / "refined"
                subprocess.run(
                    ["conda", "run", "-n", DIFIX_ENV, "python", str(REPO / "scripts/refine.py"),
                     "--render-t0", str(render_t0_path), "--clean-render", str(clean_render_path),
                     "--image-t1", str(recon_dir / "image_t1.png"), "--config", str(args.config),
                     "--output-dir", str(refined_dir)],
                    check=True,
                )
                render_t0_path, clean_render_path = refined_dir / "render_t0.png", refined_dir / "clean_render.png"

            manifest.append({
                "render_t0": str(render_t0_path),
                "clean_render": str(clean_render_path),
                "image_t1": str(recon_dir / "image_t1.png"),
                "output_dir": str(out_dir / "detect"),
            })
            query_meta[pair_id] = {"t1_frame_idx": t1_reference_frame, "output_dir": str(out_dir / "detect")}
            print(f"[{pair_id}] reconstruction OK, t1_frame_idx={t1_reference_frame}", flush=True)
        except Exception as e:
            print(f"[{pair_id}] RECONSTRUCTION FAILED: {e}", flush=True)

    manifest_path = args.output_root / "detect_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    meta_path = args.output_root / "query_meta.json"
    meta_path.write_text(json.dumps(query_meta, indent=2))
    print(f"wrote {manifest_path} ({len(manifest)} entries), {meta_path}")

    print(f"=== running detect_batch.py once on {len(manifest)} pairs ===", flush=True)
    subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", DETECT_ENV, "python", str(REPO / "scripts/detect_batch.py"),
         "--manifest", str(manifest_path), "--config", str(args.config)],
        check=True,
    )
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
