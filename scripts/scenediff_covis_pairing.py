#!/usr/bin/env python3
"""Official SceneDiff co-visibility view pairing, reused verbatim.

Runs SceneDiff's own pairing on each sequence pair and records which T1
(after-sequence) frames it selects for each T0 (before-sequence) frame:

  * co-visibility = SceneDiff's ``calculate_mask_percentage`` evaluated in
    BOTH directions and averaged (utils.py:761, as in
    ``SceneDiff._compute_similarity_matrix``);
  * a partner is suitable when that value exceeds
    ``processing.visible_percentage`` (0.5 in their shipped config); if a frame
    has no partner above it, the single highest-co-visibility partner is taken
    (``SceneDiff._compute_similarity_weights``).

Both routines are called UNBOUND from the official classes rather than
reimplemented. Their modules import faiss / open3d / torch_scatter at module
level for unrelated functionality that the pairing never touches, so those are
stubbed before import; the script then asserts that the functions it actually
calls were sourced from the official repository.

Frame cadence follows the official ``prepare_scene_data``: SceneDiff re-encodes
``original_video{1,2}`` to 30 fps with ffmpeg and takes every 30th frame, i.e.
one frame per second, indexed in the 30 fps annotation space. We reproduce that
by enumerating annotation indices 0, 30, 60, ... and mapping each to a real
frame with ``annotation_to_original_index`` -- arithmetically the same, without
re-encoding 710 videos.

Geometry comes from SceneDiff's own Pi3 model, as in their pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
BENCH = REPO / "data/scenediff_benchmark"
SCENE_DIFF = Path("/home/tessa/scene_diff")


def _install_stubs() -> None:
    """Stub modules the official files import but the pairing path never uses."""
    for name in ("faiss", "open3d", "torch_scatter"):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        if name == "torch_scatter":
            def _unavailable(*a, **k):
                raise RuntimeError("torch_scatter stub: not used by the pairing path")
            module.scatter_mean = _unavailable
        sys.modules[name] = module


def load_official():
    """Import the official pairing pieces and verify their provenance."""
    import inspect

    _install_stubs()
    sys.path.insert(0, str(SCENE_DIFF))
    sys.path.insert(0, str(SCENE_DIFF / "submodules/Pi3"))
    from utils import calculate_mask_percentage, get_img_coor, load_images_as_tensor_from_list
    from modules.geometry_model import GeometryModel
    from modules.scenediff import SceneDiff

    for fn in (calculate_mask_percentage, get_img_coor, load_images_as_tensor_from_list,
               SceneDiff._compute_similarity_matrix, SceneDiff._compute_similarity_weights):
        src = inspect.getsourcefile(fn)
        assert src and str(SCENE_DIFF) in src, f"{fn.__name__} is not the official implementation ({src})"
    return {"calculate_mask_percentage": calculate_mask_percentage, "get_img_coor": get_img_coor,
            "load_images": load_images_as_tensor_from_list, "GeometryModel": GeometryModel,
            "SceneDiff": SceneDiff}


def official_commit() -> str:
    import subprocess
    return subprocess.run(["git", "-C", str(SCENE_DIFF), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def annotation_frame_plan(video, resample_rate: int) -> list[dict]:
    """Annotation-space indices 0, 30, 60, ... mapped to real original frames."""
    from run_scenediff_batch import annotation_to_original_index, video_meta

    n_true, fps = video_meta(video)
    n_annotation = int(round(n_true * 30.0 / fps)) if fps > 0 else n_true
    plan, seen = [], set()
    for annotation_idx in range(0, max(n_annotation, 1), resample_rate):
        mapped = annotation_to_original_index(annotation_idx, video)
        if mapped["original_idx"] in seen:
            continue            # a short 10 fps tail can map two annotation indices onto one frame
        seen.add(mapped["original_idx"])
        plan.append({"annotation_idx": annotation_idx, "original_idx": mapped["original_idx"],
                     "fps": mapped["original_fps"], "ratio": mapped["ratio"],
                     "original_n_true": n_true, "annotation_n": n_annotation})
    return plan


def materialize(video, plan, out_dir: Path, prefix: str) -> tuple[list[Path], list[dict]]:
    """Write the planned frames with the robust reader; log decode method."""
    from PIL import Image
    from run_scenediff_batch import read_frame_robust

    out_dir.mkdir(parents=True, exist_ok=True)
    paths, log = [], []
    for entry in plan:
        path = out_dir / f"{prefix}_{entry['annotation_idx']:06d}.jpg"
        if not path.exists():
            frame, how = read_frame_robust(video, entry["original_idx"])
            if frame is None:
                raise RuntimeError(f"could not read frame {entry['original_idx']} from {video}")
            Image.fromarray(frame[:, :, ::-1]).save(path, quality=95)
        else:
            how = "cached"
        log.append({**entry, "path": str(path), "decode": how})
        paths.append(path)
    return paths, log


def pair_covisibility(official, config, pair: str, resample_rate: int, cache_root: Path) -> dict:
    import torch
    from run_scenediff_batch import resolve_original_video

    pair_dir = BENCH / "data" / pair
    video1, video2 = resolve_original_video(pair_dir, 1), resolve_original_video(pair_dir, 2)
    plan1, plan2 = annotation_frame_plan(video1, resample_rate), annotation_frame_plan(video2, resample_rate)
    frames_dir = cache_root / pair
    paths1, log1 = materialize(video1, plan1, frames_dir, "v1")
    paths2, log2 = materialize(video2, plan2, frames_dir, "v2")

    file_list = [str(p) for p in paths1 + paths2]
    images = official["load_images"](file_list)[None].cuda()
    H, W = images.shape[-2:]

    geometry = official["_geometry"]
    res = geometry.estimate_depth_and_poses(images)
    intrinsic = geometry.compute_intrinsics(res, file_list, H, W)
    depth_map, _point_map, poses, _voxel = geometry.normalize_scene_scale(
        res["point_map"][0].cpu().numpy(), res["depth_map"], res["poses"])

    grid_intrinsic = torch.tensor([2.0 / W, 0, -1, 0, 2.0 / H, -1]).reshape(2, 3).cuda()
    img_coors = official["get_img_coor"](H, W).cuda()

    shim = types.SimpleNamespace(config=config, device="cuda")
    similarity = official["SceneDiff"]._compute_similarity_matrix(
        shim, poses, depth_map, intrinsic, grid_intrinsic, img_coors, H, W, len(paths1), len(paths2))
    weights = official["SceneDiff"]._compute_similarity_weights(shim, similarity)

    n1 = len(paths1)
    selections = []
    for i in range(n1):
        row_w = weights[i, n1:]
        row_s = similarity[i, n1:]
        chosen = (row_w > 0).nonzero().flatten().tolist()
        above = [j for j in chosen if float(row_s[j]) > config["processing"]["visible_percentage"]]
        for j in chosen:
            selections.append({
                "t0_annotation_idx": plan1[i]["annotation_idx"], "t0_original_idx": plan1[i]["original_idx"],
                "t1_annotation_idx": plan2[j]["annotation_idx"], "t1_original_idx": plan2[j]["original_idx"],
                "covisibility": round(float(row_s[j]), 6),
                "above_threshold": bool(float(row_s[j]) > config["processing"]["visible_percentage"]),
                "selection_rule": "above_threshold" if above else "argmax_fallback",
            })
    return {"pair": pair, "video1": str(video1), "video2": str(video2),
            "t0_frames": log1, "t1_frames": log2, "selections": selections,
            "n_t0_frames": len(paths1), "n_t1_frames": len(paths2),
            "similarity_max": round(float(similarity[:n1, n1:].max()), 6),
            "unique_t1": sorted({s["t1_annotation_idx"] for s in selections})}


def main() -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair-ids-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, default=REPO / "results/scenediff_covis/frames")
    ap.add_argument("--resample-rate", type=int, default=None, help="default: the official config's value")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    official = load_official()
    config = yaml.safe_load((SCENE_DIFF / "configs/scenediff_config.yml").read_text())
    resample = args.resample_rate or int(config["dataset"]["resample_rate"])
    official["_geometry"] = official["GeometryModel"](config, "cuda")
    official["_geometry"]._load_pi3_model()

    pairs = [l.strip() for l in args.pair_ids_file.read_text().splitlines() if l.strip()]
    if args.limit:
        pairs = pairs[: args.limit]
    out = json.loads(args.out.read_text()) if args.out.exists() else {}
    out.setdefault("_meta", {})
    out["_meta"] = {"scenediff_commit": official_commit(), "resample_rate": resample,
                    "visible_percentage": config["processing"]["visible_percentage"],
                    "pi3_model": config["models"]["pi3"]["name"]}

    failures = out.get("_failures", {})
    for i, pair in enumerate(pairs, 1):
        if pair in out:
            continue
        try:
            out[pair] = pair_covisibility(official, config, pair, resample, args.cache_root)
            r = out[pair]
            print(f"[{i}/{len(pairs)}] {pair[:42]:<42} T0={r['n_t0_frames']:>3} T1={r['n_t1_frames']:>3} "
                  f"sel={len(r['selections']):>4} uniqueT1={len(r['unique_t1']):>3} maxcov={r['similarity_max']:.3f}",
                  flush=True)
        except Exception as e:  # noqa: BLE001 -- one bad pair must not sink the sweep
            failures[pair] = f"{type(e).__name__}: {e}"
            print(f"[{i}/{len(pairs)}] {pair[:42]:<42} FAILED {failures[pair][:90]}", flush=True)
        if i % 10 == 0:
            out["_failures"] = failures
            args.out.write_text(json.dumps(out, indent=1))
    out["_failures"] = failures
    args.out.write_text(json.dumps(out, indent=1))
    done = [k for k in out if not k.startswith("_")]
    uniq = sum(len(out[k]["unique_t1"]) for k in done)
    print(f"\nwrote {args.out}: {len(done)} pairs, {len(failures)} failures, "
          f"{uniq} unique T1 queries total ({uniq / max(len(done), 1):.1f} per pair)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
