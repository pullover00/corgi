#!/usr/bin/env python3
"""Run the full pipeline on an ad-hoc capture: one reference image set plus
one or more unposed query photos, with no ground truth.

This is the same reference-set-vs-single-query protocol as
run_paslcd_pair.py -- reconstruct the reference scene once in isolation
(reconstruct_reference_scene), then localize and render each query against
it (localize_and_render_query), refine, and detect -- but for a directory of
images the user captured themselves rather than a benchmark instance. Since
there are no GT masks it reports the pipeline's own decisions
(inference.json) instead of IoU/F1, and writes a side-by-side panel per
query so the intermediate stages can be inspected by eye.

Default layout (data/demo1/):
  target/**/*.jpg    reference ("before") photos, recursively
  inference/*.png    query ("after") photos, one run per file

Unlike the benchmark runners this one applies EXIF orientation when copying
inputs: the demo1 reference photos are phone captures carrying orientation
tag 6, and VGGT-Omega's loader does not transpose them, so without this the
reference set is reconstructed sideways relative to the query.

Run in the vggt-omega conda env; shells out to difix3d/goldilocs for the
refine and detect stages, exactly as run_paslcd_pair.py does.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DIFIX_ENV = os.environ.get("DIFIX3D_CONDA_ENV", "difix3d")
DETECT_ENV = os.environ.get("DETECTION_CONDA_ENV", "goldilocs")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def even_subsample(items: list[Path], n: int) -> list[Path]:
    """Uniform-stride subsample preserving first/last, as in run_paslcd_pair."""
    if n >= len(items) or n <= 1:
        return items if n >= len(items) else [items[0]]
    step = (len(items) - 1) / (n - 1)
    return [items[i] for i in sorted({round(i * step) for i in range(n)})]


def stage_inputs(images: list[Path], out_dir: Path) -> list[Path]:
    """Copy inputs to a working dir with EXIF orientation applied. VGGT-Omega
    reads pixels straight off disk, so an un-transposed phone photo enters
    the reconstruction rotated 90 degrees from the query."""
    from PIL import Image, ImageOps

    out_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for i, p in enumerate(images):
        dst = out_dir / f"{i:03d}_{p.stem}.png"
        if not dst.exists():
            ImageOps.exif_transpose(Image.open(p)).convert("RGB").save(dst)
        staged.append(dst)
    return staged


def save_panel(paths: dict[str, Path], detect_dir: Path, panel_path: Path) -> None:
    """render_t0 | clean_render | image_t1 | labels overlay, one row."""
    import numpy as np
    from PIL import Image

    tiles, labels = [], ["render_t0 (refined)", "clean_render (refined)", "image_t1 (query)", "detected changes"]
    for key in ("render_t0", "clean_render", "image_t1"):
        tiles.append(Image.open(paths[key]).convert("RGB"))
    query = tiles[2].copy()
    label_png = detect_dir / "labels.png"
    if label_png.exists():
        lab = np.asarray(Image.open(label_png).convert("L").resize(query.size, Image.NEAREST))
        ov = np.array(query).copy()
        ov[lab != 0] = (0.45 * ov[lab != 0] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
        query = Image.fromarray(ov)
    tiles.append(query)

    w, h = 480, int(480 * tiles[0].height / tiles[0].width)
    panel = Image.new("RGB", (w * len(tiles), h + 18), (16, 16, 16))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(panel)
    for i, (t, name) in enumerate(zip(tiles, labels)):
        panel.paste(t.resize((w, h)), (i * w, 18))
        draw.text((i * w + 6, 4), name, fill=(255, 255, 0))
    panel.save(panel_path)


def reference_fingerprint(sources: list[Path], config_path: Path) -> dict:
    """What the cached reference reconstruction depends on: the source images
    (identity, size, mtime) and the reconstruction config section. If any of
    it changes the cache is stale -- the reference set was edited three times
    on 2026-09-09 alone, and a silently stale reconstruction would put every
    detected change in the wrong place."""
    from ocmask_pipeline.config import load_config

    return {
        "sources": [{"path": str(p.resolve()), "size": p.stat().st_size, "mtime": p.stat().st_mtime}
                    for p in sources],
        "reconstruction": load_config(config_path).get("reconstruction"),
    }


def cached_reference_scene(sources: list[Path], cache: Path, config_path: Path, config,
                           rebuild: bool = False):
    """Reference scene from ``cache`` (.npz + .json sidecar) when its
    fingerprint matches, else rebuilt and saved. Returns (scene, staged_paths).

    The staged copies live in a STABLE directory beside the cache, not in the
    per-run inputs dir: localize_and_render_query raises unless the query-time
    t0 paths equal the cached scene's image_paths exactly, so per-run staging
    would make the cache unusable by construction. Only the isolated
    reference pass is saved -- localization still runs VGGT-Omega on
    reference + query jointly for every query, by design.
    """
    from ocmask_pipeline.reconstruction import ReferenceScene, reconstruct_reference_scene

    staged = stage_inputs(sources, cache.parent / f"{cache.stem}_inputs")
    fp = reference_fingerprint(sources, config_path)
    side = cache.with_suffix(".json")
    if not rebuild and cache.exists() and side.exists():
        try:
            if json.loads(side.read_text()) == fp:
                scene = ReferenceScene.load(cache)
                if scene.image_paths == staged:
                    print(f"reference scene: loaded from cache {cache}", flush=True)
                    return scene, staged
                print("reference scene: cache paths differ from staged inputs, rebuilding", flush=True)
            else:
                print("reference scene: sources or reconstruction config changed, rebuilding", flush=True)
        except Exception as e:  # a corrupt cache must never block a run
            print(f"reference scene: cache unreadable ({e}), rebuilding", flush=True)
    print("reference scene: reconstructing (once, in isolation from queries)...", flush=True)
    scene = reconstruct_reference_scene(staged, config)
    cache.parent.mkdir(parents=True, exist_ok=True)
    scene.save(cache)
    side.write_text(json.dumps(fp, indent=2))
    print(f"reference scene: saved to {cache}", flush=True)
    return scene, staged


def refine_via_worker(worker_dir: Path, paths: dict, config_path: Path, refined_dir: Path,
                      timeout_s: float = 600.0) -> bool:
    """Hand the refine job to a resident refine_worker.py if one is alive.
    Returns True on success; False (with a printed reason) when there is no
    live worker, it failed, or it timed out -- the caller then falls back to
    the per-run subprocess, so a missing worker never changes the result,
    only the time."""
    hb = worker_dir / "heartbeat"
    if not hb.exists() or time.time() - hb.stat().st_mtime > 10.0:
        print("refine: no live worker, using the per-run subprocess", flush=True)
        return False
    rid = f"{int(time.time() * 1000)}_{os.getpid()}"
    (worker_dir / "requests").mkdir(parents=True, exist_ok=True)
    (worker_dir / "results").mkdir(parents=True, exist_ok=True)
    (worker_dir / "requests" / f"{rid}.json").write_text(json.dumps({
        "render_t0": str(paths["render_t0"]), "clean_render": str(paths["clean_render"]),
        "image_t1": str(paths["image_t1"]), "config": str(config_path), "output_dir": str(refined_dir)}))
    done, err = worker_dir / "results" / f"{rid}.done", worker_dir / "results" / f"{rid}.err"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if done.exists():
            print(f"refine: worker finished in {json.loads(done.read_text())['seconds']:.1f}s", flush=True)
            return True
        if err.exists():
            print("refine: worker FAILED, falling back to the subprocess:\n" + err.read_text(), flush=True)
            return False
        if time.time() - hb.stat().st_mtime > 30.0:
            # Give up only if the worker really stopped. Its heartbeat runs on
            # a dedicated thread, so a stale one means the process is gone --
            # not merely busy loading the model.
            print("refine: worker heartbeat stopped, falling back to the subprocess", flush=True)
            (worker_dir / "requests" / f"{rid}.json").unlink(missing_ok=True)
            return False
        time.sleep(0.5)
    print("refine: worker timed out, falling back to the subprocess", flush=True)
    (worker_dir / "requests" / f"{rid}.json").unlink(missing_ok=True)
    return False


def run_query(reference_images: list[Path], query: Path, out_root: Path,
              config_path: Path, reference_scene, skip_refine: bool,
              refine_worker_dir: Path | None = None) -> dict:
    import numpy as np
    from PIL import Image

    from ocmask_pipeline.config import load_config
    from ocmask_pipeline.reconstruction import localize_and_render_query

    config = load_config(config_path)
    stem = query.stem
    q_dir = out_root / stem
    recon_dir = q_dir / "reconstruction"
    recon_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== query: {stem} ===", flush=True)
    result = localize_and_render_query(
        t0_image_paths=reference_images,
        query_image_path=query,
        reference_scene=reference_scene,
        t0_reference_index=0,
        config=config,
    )
    # The sanity metric for this whole protocol: large residual means the two
    # VGGT-Omega calls disagree on the reference geometry and nothing
    # downstream should be trusted.
    print(f"  alignment_residual: {result.alignment_residual}", flush=True)

    Image.fromarray(result.render_t0).save(recon_dir / "render_t0.png")
    Image.fromarray(result.clean_render).save(recon_dir / "clean_render.png")
    Image.fromarray(result.image_t1).save(recon_dir / "image_t1.png")
    np.save(recon_dir / "render_t0_positions.npy", result.render_t0_positions)
    np.save(recon_dir / "clean_render_positions.npy", result.clean_render_positions)
    np.save(recon_dir / "image_t1_positions.npy", result.image_t1_positions)
    np.save(recon_dir / "render_t0_coverage.npy", result.render_t0_coverage)
    np.save(recon_dir / "render_t0_confidence.npy", result.render_t0_confidence)
    np.save(recon_dir / "render_t0_corroboration.npy", result.render_t0_corroboration)
    (recon_dir / "scene_scale.json").write_text(json.dumps({"scene_scale": result.scene_scale}))

    paths = {k: recon_dir / f"{k}.png" for k in ("render_t0", "clean_render", "image_t1")}
    if not skip_refine and config.get("refine", {}).get("enabled", False):
        refined_dir = q_dir / "refined"
        # A live refine_worker.py (model already loaded) does it in seconds;
        # otherwise the per-run subprocess, which reloads DI2FIX each time.
        if not (refine_worker_dir and refine_via_worker(refine_worker_dir, paths, config_path, refined_dir)):
            subprocess.run(
                ["conda", "run", "--no-capture-output", "-n", DIFIX_ENV, "python", str(REPO / "scripts/refine.py"),
                 "--render-t0", str(paths["render_t0"]), "--clean-render", str(paths["clean_render"]),
                 "--image-t1", str(paths["image_t1"]), "--config", str(config_path),
                 "--output-dir", str(refined_dir)],
                check=True,
            )
        paths["render_t0"] = refined_dir / "render_t0.png"
        paths["clean_render"] = refined_dir / "clean_render.png"

    detect_dir = q_dir / "detect"
    subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", DETECT_ENV, "python", str(REPO / "scripts/detect.py"),
         "--render-t0", str(paths["render_t0"]), "--clean-render", str(paths["clean_render"]),
         "--image-t1", str(paths["image_t1"]), "--config", str(config_path),
         "--output-dir", str(detect_dir),
         "--render-t0-positions", str(recon_dir / "render_t0_positions.npy"),
         "--clean-render-positions", str(recon_dir / "clean_render_positions.npy"),
         "--image-t1-positions", str(recon_dir / "image_t1_positions.npy"),
         "--scene-scale-path", str(recon_dir / "scene_scale.json"),
         "--render-t0-coverage", str(recon_dir / "render_t0_coverage.npy"),
         "--render-t0-confidence", str(recon_dir / "render_t0_confidence.npy"),
         "--render-t0-corroboration", str(recon_dir / "render_t0_corroboration.npy"),
         # the per-object masks live only in the stage dump; inference.json
         # carries decisions and labels.png carries only the decision classes
         # (added/moved/removed), not which pixels belong to WHICH object.
         # A robot acting on one changed object needs that mask.
         "--dump-stages", str(detect_dir / "stages")],
        check=True,
    )

    save_panel(paths, detect_dir, q_dir / "panel.png")
    inference = json.loads((detect_dir / "inference.json").read_text())
    summary = {
        "query": str(query),
        "alignment_residual": result.alignment_residual,
        "decision_counts": inference.get("decision_counts"),
        "output_dir": str(q_dir),
    }
    print(f"  decisions: {summary['decision_counts']}", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=REPO / "data/demo1")
    ap.add_argument("--out", type=Path, default=REPO / "results/demo1_pipeline")
    # The demo runs the SHIPPED solution, ablate_v10_no_dino.yaml -- pointed at
    # directly, not copied. An earlier demo1_no_dino.yaml copied only the
    # use_dino_features: false part and silently lacked v10's ceiling/sky and
    # occlusion-aware removal suppression; a copy drifts, a pointer cannot.
    ap.add_argument("--config", type=Path, default=REPO / "configs/ablate_v10_no_dino.yaml")
    ap.add_argument("--max-reference-images", type=int, default=None,
                    help="uniformly subsample the reference set (VRAM/quality tradeoff)")
    ap.add_argument("--skip-refine", action="store_true")
    ap.add_argument("--reference-scene-cache", type=Path, default=None,
                    help="reuse the isolated reference reconstruction from this .npz "
                         "(built and saved on first use; invalidated automatically when "
                         "the reference images or the reconstruction config change)")
    ap.add_argument("--rebuild-reference", action="store_true",
                    help="ignore an existing --reference-scene-cache and rebuild it")
    ap.add_argument("--refine-worker-dir", type=Path, default=None,
                    help="use a resident scripts/refine_worker.py in this directory when one is "
                         "alive (falls back to the per-run refine subprocess otherwise)")
    args = ap.parse_args()

    from ocmask_pipeline.config import load_config
    from ocmask_pipeline.reconstruction import reconstruct_reference_scene

    reference_raw = collect_images(args.data_root / "target")
    queries_raw = collect_images(args.data_root / "inference")
    if not reference_raw:
        raise FileNotFoundError(f"no reference images under {args.data_root / 'target'}")
    if not queries_raw:
        raise FileNotFoundError(f"no query images under {args.data_root / 'inference'}")
    if args.max_reference_images:
        reference_raw = even_subsample(reference_raw, args.max_reference_images)

    work = args.out / "inputs"
    queries = stage_inputs(queries_raw, work / "queries")
    config = load_config(args.config)
    if args.reference_scene_cache:
        reference_scene, reference_images = cached_reference_scene(
            reference_raw, args.reference_scene_cache, args.config, config, args.rebuild_reference)
    else:
        reference_images = stage_inputs(reference_raw, work / "reference")
        print("reconstructing reference scene (once, in isolation from queries)...", flush=True)
        reference_scene = reconstruct_reference_scene(reference_images, config)
    print(f"{len(reference_images)} reference images, {len(queries)} query image(s)", flush=True)

    summaries = [run_query(reference_images, q, args.out, args.config, reference_scene, args.skip_refine,
                           args.refine_worker_dir)
                 for q in queries]
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {args.out / 'summary.json'}")
    print("DEMO1_PIPELINE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
