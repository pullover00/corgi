#!/usr/bin/env python3
"""SceneDiff diagnostic runner: reference-set -> single-query, with every
intermediate stage cached under a provenance-tracked layout.

Layout (ocmask_pipeline.artifacts):

    results/scenediff_diagnostic/SceneDiff/<pair>/shared/reference_reconstruction/
    results/scenediff_diagnostic/SceneDiff/<pair>/<query>/shared/{localization,render,refine}/
    results/scenediff_diagnostic/SceneDiff/<pair>/<query>/<experiment>/{proposals,descriptors,
                                                   tracking,resolution,labels,metrics}/

Stages up to refine depend only on the reconstruction/refine config, so they
live under the experiment name "shared" and are reused by every detect-side
experiment (m1..m5 differ only in three_image_comparison). Each experiment's
own manifests record the shared stages' hashes as upstream, so changing the
reconstruction config still invalidates everything downstream.

Fixes a silent discrepancy with the older run_scenediff_batch.py, whose
detect manifest carried only the three images: without the position /
coverage / confidence / corroboration buffers the visibility filter,
geometric identity and corroboration all no-op, so every SceneDiff run to
date executed a different effective pipeline from PASLCD. This runner passes
every buffer, plus above_horizon.

Methodology: t0 = ``reconstruction.frames_per_video`` frames sampled from
original_video1 (always including the annotators' representative t0 frame),
reconstructed once in isolation; t1 = exactly ONE query frame from
original_video2 (the representative t1 frame), localized against it. Never
multiple t1 frames jointly.

Run in the vggt-omega conda env; shells out to difix3d / goldilocs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from ocmask_pipeline.artifacts import ArtifactStore  # noqa: E402

DETECT_ENV = os.environ.get("DETECTION_CONDA_ENV", "goldilocs")
DIFIX_ENV = os.environ.get("DIFIX3D_CONDA_ENV", "difix3d")
EVAL_ENV = os.environ.get("EVAL_CONDA_ENV", DETECT_ENV)  # needs pycocotools + cv2
DATASET = "SceneDiff"

RENDER_FILES = {
    "render_t0": "render_t0.png", "clean_render": "clean_render.png", "image_t1": "image_t1.png",
    "render_t0_positions": "render_t0_positions.npy", "clean_render_positions": "clean_render_positions.npy",
    "image_t1_positions": "image_t1_positions.npy", "scene_scale_path": "scene_scale.json",
    "render_t0_coverage": "render_t0_coverage.npy", "render_t0_confidence": "render_t0_confidence.npy",
    "render_t0_corroboration": "render_t0_corroboration.npy", "above_horizon": "above_horizon.npy",
}
DETECT_STAGES = ("proposals", "descriptors", "tracking", "resolution", "labels")


def stores(root: Path, pair: str, query: str, experiment: str,
           shared_name: str = "shared") -> tuple[ArtifactStore, ArtifactStore]:
    shared = ArtifactStore(root, DATASET, pair, shared_name, query=query)
    own = ArtifactStore(root, DATASET, pair, experiment, query=query, fallback=shared)
    return shared, own


def nested_reference_subset(n_available: int, n_views: int) -> list[int]:
    """Positions (into the existing baseline T0 frame list) of ``n_views``
    evenly spaced frames, fixed before any run and independent of content:
    fraction f_i = i/(n-1) of the way through the list (f = 1/2 for n = 1),
    position = floor(f * (K-1) + 0.5). Nested for the ablation's ladder --
    K=10: {5} < {0,5,9} < {0,2,5,7,9}; K=11: {5} < {0,5,10} < {0,3,5,8,10} --
    so every smaller reference set is contained in the larger ones. The
    annotators' representative T0 frame is NOT forced in (it would break
    both the even spacing and the nesting)."""
    if n_views >= n_available:
        return list(range(n_available))
    fractions = [0.5] if n_views == 1 else [i / (n_views - 1) for i in range(n_views)]
    return sorted({int(math.floor(f * (n_available - 1) + 0.5)) for f in fractions})


def prepare_frames(pair_dir: Path, frames_root: Path, frames_per_video: int):
    from run_scenediff_batch import (extract_frames, representative_frame_index,
                                     resolve_original_video, sample_frame_indices)
    import cv2

    objects = pickle.loads((pair_dir / "segments.pkl").read_bytes())["objects"]
    t0_rep = representative_frame_index(objects, "in_video1", "video1_frame_idx")
    # The query frame is fixed up front by scenediff_select_query_frames.py
    # (max usable in-scope GT objects; relative position for removed-only
    # pairs). The old most-common-index rule tie-broke table_5_table_6 to a
    # frame with no decodable mask while frame 85 carried both objects.
    queries_file = pair_dir.parents[1] / "diagnostic_subset_queries.json"
    preset = json.loads(queries_file.read_text()).get(pair_dir.name) if queries_file.exists() else None
    t1_rep = int(preset["t1_idx"]) if preset else representative_frame_index(objects, "in_video2", "video2_frame_idx")
    video1, video2 = resolve_original_video(pair_dir, 1), resolve_original_video(pair_dir, 2)
    cap = cv2.VideoCapture(str(video1)); n1 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    t0_indices = sample_frame_indices(n1, frames_per_video)
    if t0_rep not in t0_indices:
        t0_indices = sorted(t0_indices + [t0_rep])
    t0_frames = extract_frames(video1, t0_indices, frames_root, "t0")
    t1_frames = extract_frames(video2, [t1_rep], frames_root, "t1")
    return t0_frames, t1_frames[0], t0_indices, t1_rep


def reconstruct_stages(pair: str, pair_dir: Path, root: Path, config: dict, experiment: str,
                       skip_refine: bool, inventory_source_experiment: str | None = None,
                       dump_inventory: bool = False, reference_views: int | None = None,
                       refine_worker_dir: Path | None = None) -> dict | None:
    import numpy as np
    from PIL import Image
    from ocmask_pipeline.reconstruction import localize_and_render_query, reconstruct_reference_scene

    frames_root = root / DATASET / pair / "frames"
    t0_frames, query_frame, t0_indices, t1_idx = prepare_frames(
        pair_dir, frames_root, int(config["reconstruction"]["frames_per_video"]))
    query = f"t1_{t1_idx:04d}"
    shared_name = "shared"
    subset_positions = None
    if reference_views is not None:
        # Reference-view-count ablation: a deterministic, nested, evenly
        # spaced subset of the BASELINE run's own T0 frame list (which must
        # already exist), so the only thing that changes is how many of the
        # same frames the reference reconstruction sees. Everything that
        # depends on the reference set (reconstruction, localization,
        # render, refine) lives under its own shared_ref<N> cell; the T1
        # query is untouched.
        baseline = root / DATASET / pair / "shared" / "reference_reconstruction" / "t0_frames.json"
        if not baseline.exists():
            raise FileNotFoundError(f"baseline T0 frame list required for --reference-views: {baseline}")
        baseline_frames = json.loads(baseline.read_text())
        subset_positions = nested_reference_subset(len(baseline_frames["indices"]), reference_views)
        t0_indices = [baseline_frames["indices"][i] for i in subset_positions]
        t0_frames = [Path(baseline_frames["paths"][i]) for i in subset_positions]
        missing = [str(p) for p in t0_frames if not p.exists()]
        if missing:
            raise FileNotFoundError(f"baseline T0 frames missing on disk: {missing}")
        shared_name = f"shared_ref{reference_views}"
    shared, own = stores(root, pair, query, experiment, shared_name)

    # --- reference reconstruction (scene-level, shared) ---
    ref_dir = shared.stage_dir("reference_reconstruction")
    if shared.is_fresh("reference_reconstruction", config, ["reference_scene.pkl"]):
        reference_scene = pickle.loads((ref_dir / "reference_scene.pkl").read_bytes())
        print(f"  [{pair}] reference reconstruction: cached", flush=True)
    else:
        print(f"  [{pair}] reference reconstruction: {shared.stale_reason('reference_reconstruction', config)}", flush=True)
        started = time.perf_counter()
        reference_scene = reconstruct_reference_scene(t0_frames, config)
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / "reference_scene.pkl").write_bytes(pickle.dumps(reference_scene))
        (ref_dir / "t0_frames.json").write_text(json.dumps({"indices": t0_indices, "paths": [str(p) for p in t0_frames]}))
        shared.commit("reference_reconstruction", config, ["reference_scene.pkl", "t0_frames.json"],
                      extra={"seconds": time.perf_counter() - started, "n_frames": len(t0_frames),
                             "reference_views": reference_views,
                             "subset_positions_in_baseline_list": subset_positions})

    # --- localization + render (query-level, shared) ---
    render_dir = shared.stage_dir("render")
    if shared.is_fresh("render", config, list(RENDER_FILES.values())) and shared.is_fresh("localization", config):
        print(f"  [{pair}] localization/render: cached", flush=True)
    else:
        print(f"  [{pair}] localization/render: {shared.stale_reason('render', config)}", flush=True)
        started = time.perf_counter()
        result = localize_and_render_query(t0_frames, query_frame, reference_scene, 0, config)
        loc_dir = shared.stage_dir("localization"); loc_dir.mkdir(parents=True, exist_ok=True)
        (loc_dir / "alignment.json").write_text(json.dumps({
            "alignment_residual": result.alignment_residual, "t1_frame_idx": t1_idx,
            "query_frame": str(query_frame), "scene_scale": result.scene_scale,
        }, indent=2))
        shared.commit("localization", config, ["alignment.json"], extra={"seconds": time.perf_counter() - started})
        render_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.render_t0).save(render_dir / "render_t0.png")
        Image.fromarray(result.clean_render).save(render_dir / "clean_render.png")
        Image.fromarray(result.image_t1).save(render_dir / "image_t1.png")
        for key in ("render_t0_positions", "clean_render_positions", "image_t1_positions",
                    "render_t0_coverage", "render_t0_confidence", "render_t0_corroboration"):
            np.save(render_dir / f"{key}.npy", getattr(result, key))
        if result.above_horizon is not None:
            np.save(render_dir / "above_horizon.npy", result.above_horizon)
        (render_dir / "scene_scale.json").write_text(json.dumps({"scene_scale": result.scene_scale}))
        shared.commit("render", config, [n for n in RENDER_FILES.values() if (render_dir / n).exists()],
                      extra={"alignment_residual": result.alignment_residual,
                             "above_horizon_fraction": None if result.above_horizon is None else float(result.above_horizon.mean())})
        print(f"  [{pair}] alignment residual {result.alignment_residual:.5f}", flush=True)

    # --- refine (query-level, shared) ---
    refine_dir = shared.stage_dir("refine")
    refine_on = not skip_refine and config.get("refine", {}).get("enabled", False)
    if refine_on:
        if shared.is_fresh("refine", config, ["render_t0.png", "clean_render.png"]):
            print(f"  [{pair}] refine: cached", flush=True)
        else:
            print(f"  [{pair}] refine: {shared.stale_reason('refine', config)}", flush=True)
            started = time.perf_counter()
            # A live scripts/refine_worker.py (model already resident) does
            # the same DifixPipeline call in seconds; otherwise the per-run
            # subprocess reloads the model (~2.5 min). Output is identical.
            via_worker = False
            if refine_worker_dir is not None:
                from run_demo1 import refine_via_worker
                via_worker = refine_via_worker(
                    refine_worker_dir,
                    {"render_t0": render_dir / "render_t0.png", "clean_render": render_dir / "clean_render.png",
                     "image_t1": render_dir / "image_t1.png"},
                    Path(config["_config_path"]), refine_dir)
            if not via_worker:
                subprocess.run(
                    ["conda", "run", "--no-capture-output", "-n", DIFIX_ENV, "python", str(REPO / "scripts/refine.py"),
                     "--render-t0", str(render_dir / "render_t0.png"), "--clean-render", str(render_dir / "clean_render.png"),
                     "--image-t1", str(render_dir / "image_t1.png"), "--config", str(config["_config_path"]),
                     "--output-dir", str(refine_dir)],
                    check=True,
                )
            shared.commit("refine", config, ["render_t0.png", "clean_render.png"],
                          extra={"seconds": time.perf_counter() - started, "via_worker": via_worker})
    render_t0 = (refine_dir if refine_on else render_dir) / "render_t0.png"
    clean_render = (refine_dir if refine_on else render_dir) / "clean_render.png"

    entry = {"render_t0": str(render_t0), "clean_render": str(clean_render),
             "image_t1": str(render_dir / "image_t1.png"),
             "output_dir": str(own.stage_dir("labels")),
             "dump_stages": str(own.stage_dir("labels").parent)}
    for key, name in RENDER_FILES.items():
        if key in ("render_t0", "clean_render", "image_t1"):
            continue
        if (render_dir / name).exists():
            entry[key] = str(render_dir / name)
    experiment_dir = own.stage_dir("labels").parent
    if dump_inventory:
        entry["dump_inventory_to"] = str(experiment_dir / "inventory")
    if inventory_source_experiment is not None:
        inventory_dir = experiment_dir.parent / inventory_source_experiment / "inventory"
        if not (inventory_dir / "bundle.pkl").exists():
            raise FileNotFoundError(f"missing replay inventory bundle: {inventory_dir / 'bundle.pkl'}")
        entry["load_inventory_from"] = str(inventory_dir)
    return {"pair": pair, "query": query, "t1_idx": t1_idx, "entry": entry, "own": own, "shared": shared}


def evaluate(item: dict, pair_dir: Path, config: dict) -> dict:
    """Shelled out to EVAL_CONDA_ENV: scenediff_gt_eval needs pycocotools,
    which this runner's vggt-omega env lacks. See scenediff_diag_eval.py."""
    own: ArtifactStore = item["own"]
    metrics_dir = own.stage_dir("metrics"); metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / "metrics.json"
    subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", EVAL_ENV, "python", str(REPO / "scripts/scenediff_diag_eval.py"),
         "--pair-dir", str(pair_dir), "--t1-frame-idx", str(item["t1_idx"]),
         "--labels-dir", str(own.stage_dir("labels")), "--out", str(out)],
        check=True,
    )
    own.commit("metrics", config, ["metrics.json"])
    return json.loads(out.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-root", type=Path, default=REPO / "data/scenediff_benchmark")
    ap.add_argument("--pair-ids-file", type=Path, default=REPO / "data/scenediff_benchmark/diagnostic_subset.txt")
    ap.add_argument("--experiment", required=True, help="experiment version name, e.g. m1_full")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=REPO / "results/scenediff_diagnostic")
    ap.add_argument("--skip-refine", action="store_true")
    ap.add_argument("--dump-inventory", action="store_true",
                    help="persist stages 1-3 plus dense recovery features for later state-only replay")
    ap.add_argument("--inventory-source-experiment", default=None,
                    help="load a prior experiment's inventory bundle and rerun only stages 4+")
    ap.add_argument("--reference-views", type=int, default=None,
                    help="reference-view-count ablation: reconstruct from this many evenly spaced, nested "
                         "frames of the baseline's own T0 frame list (see nested_reference_subset); "
                         "shared stages go under shared_ref<N>")
    ap.add_argument("--refine-worker-dir", type=Path, default=None,
                    help="use a live scripts/refine_worker.py in this directory for the refine stage "
                         "(falls back to the per-run subprocess when none is alive)")
    ap.add_argument("--detect-extra-args", default="",
                    help="extra CLI args passed through to detect_batch.py, e.g. '--sequential-model-lifecycle'")
    ap.add_argument("--through", choices=("refine", "detect"), default="detect",
                    help="'refine' stops after the shared reconstruction/render/refine stages -- lets the "
                         "base-independent GPU work run before the detect-side variant is decided")
    args = ap.parse_args()

    from ocmask_pipeline.config import load_config
    config = load_config(args.config)
    config["_config_path"] = str(args.config)
    # keep a verbatim copy of the config file with the experiment, not just the hashed subsets
    exp_root = args.root / DATASET / "_experiments" / args.experiment
    exp_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, exp_root / "config.yaml")
    (exp_root / "invocation.json").write_text(json.dumps({
        "argv": sys.argv, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference_views": args.reference_views, "inventory_source_experiment": args.inventory_source_experiment,
    }, indent=2))

    pairs = [l.strip() for l in args.pair_ids_file.read_text().splitlines() if l.strip()]
    print(f"experiment {args.experiment}: {len(pairs)} pairs, config {args.config}", flush=True)

    items, failed = [], {}
    for pair in pairs:
        pair_dir = args.benchmark_root / "data" / pair
        try:
            item = reconstruct_stages(
                pair, pair_dir, args.root, config, args.experiment, args.skip_refine,
                inventory_source_experiment=args.inventory_source_experiment,
                dump_inventory=args.dump_inventory, reference_views=args.reference_views,
                refine_worker_dir=args.refine_worker_dir,
            )
            if item:
                items.append(item)
        except Exception as e:  # noqa: BLE001 -- one bad pair must not sink the batch; it is recorded
            failed[pair] = f"{type(e).__name__}: {e}"
            print(f"  [{pair}] RECONSTRUCTION FAILED: {failed[pair]}", flush=True)

    reference_sets = {}
    for it in items:
        ref = it["shared"].stage_dir("reference_reconstruction")
        reference_sets[it["pair"]] = {
            "t0_frames": json.loads((ref / "t0_frames.json").read_text())["indices"],
            "reference_seconds": (json.loads((ref / "manifest.json").read_text()).get("extra") or {}).get("seconds"),
            "t1_idx": it["t1_idx"],
            "alignment_residual": json.loads((it["shared"].stage_dir("localization") / "alignment.json").read_text())["alignment_residual"],
        }
    (exp_root / "reference_sets.json").write_text(json.dumps(reference_sets, indent=2))

    if args.through == "refine":
        residuals = {it["pair"]: reference_sets[it["pair"]]["alignment_residual"] for it in items}
        (exp_root / "shared_stages.json").write_text(json.dumps({"alignment_residuals": residuals, "failed": failed}, indent=2))
        print(f"\nshared stages complete for {len(items)} pairs; residuals: "
              + ", ".join(f"{k.split('_')[0]}={v:.4f}" for k, v in residuals.items()) + f"; failed={list(failed)}")
        print("SCENEDIFF_SHARED_DONE", flush=True)
        return 0

    todo = [it for it in items if not it["own"].is_fresh("labels", config, ["labels.png", "inference.json"])]
    print(f"\ndetect: {len(todo)} to run, {len(items) - len(todo)} cached", flush=True)
    if todo:
        manifest = exp_root / "detect_manifest.json"
        manifest.write_text(json.dumps([it["entry"] for it in todo], indent=2))
        started = time.perf_counter()
        subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", DETECT_ENV, "python", str(REPO / "scripts/detect_batch.py"),
             "--manifest", str(manifest), "--config", str(args.config), *shlex.split(args.detect_extra_args)],
            check=True,
        )
        per_query_seconds = (time.perf_counter() - started) / max(len(todo), 1)
        for it in todo:
            own = it["own"]
            for stage in DETECT_STAGES:
                d = own.stage_dir(stage)
                if d.exists():
                    own.commit(stage, config, [p.name for p in d.iterdir() if p.is_file() and p.name not in ("manifest.json", "config_used.json")],
                               extra={"batch_seconds_per_query": per_query_seconds} if stage == "labels" else None)

    rows = {}
    for it in items:
        try:
            rows[it["pair"]] = evaluate(it, args.benchmark_root / "data" / it["pair"], config)
        except Exception as e:  # noqa: BLE001
            failed[it["pair"]] = f"eval {type(e).__name__}: {e}"
            print(f"  [{it['pair']}] EVAL FAILED: {failed[it['pair']]}", flush=True)

    tp = sum(r.get("tp", 0) for r in rows.values()); fp = sum(r.get("fp", 0) for r in rows.values()); fn = sum(r.get("fn", 0) for r in rows.values())
    summary = {
        "experiment": args.experiment, "config": str(args.config), "n_pairs": len(rows), "failed": failed,
        "pooled_iou_t1_only": tp / (tp + fp + fn) if (tp + fp + fn) else None,
        "pooled_precision": tp / (tp + fp) if (tp + fp) else None,
        "pooled_recall": tp / (tp + fn) if (tp + fn) else None,
        "mean_iou": sum(r.get("iou", 0) or 0 for r in rows.values()) / len(rows) if rows else None,
        "per_pair": rows,
    }
    (exp_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[{args.experiment}] n={len(rows)} pooled IoU={summary['pooled_iou_t1_only']} "
          f"P={summary['pooled_precision']} R={summary['pooled_recall']}  failed={list(failed)}")
    print("SCENEDIFF_DIAGNOSTIC_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
