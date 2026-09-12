#!/usr/bin/env python3
"""Package the SceneDiff-paired single-query (P1) results into one analysis-ready table set.

For every completed detect arm under <root>/SceneDiff/_experiments/ (baseline refine-OFF,
refine-ON, movable-object gate, ...), writes per-pair rows and pooled summaries for BOTH
color-replacement states, so either number can be reported:

  replacement=off   REPLACED (label 5) stripped back to UNCHANGED before scoring
  replacement=on    the pipeline's own final pass (find_color_replacement_regions, unchanged)
                    re-run post-hoc on the cached inputs for arms that ran with it off

Both transformations are exact, not approximations: the replacement pass is the LAST stage
of run_object_state_resolution and is strictly additive on pixels no earlier stage explained
(already_changed = labels != 0), consuming only render_t0 / image_t1 / coverage / confidence --
so stripping label 5 reproduces the "off" run, and re-running the pass on an "off" run's
labels reproduces the "on" run, byte for byte in the label map.

Per-pair rows carry everything a component or per-class analysis needs: subset (SD-V/SD-K),
gt_empty (tp=fn=0 by construction -- see the empty-GT partition note in docs), tp/fp/fn,
IoU/P/R, per-predicted-class landing counts (ADDED/MOVED/REMOVED/REPLACED pixels on GT-added /
GT-moved-bucket / background), added_recall, moved_bucket_recall, decision_counts, the
suppression counters (visibility/horizon/ceiling-sky/movable-gate/corroboration), and the
whitelist coverage when a mask root is given. Summaries are pooled (sum of pixel counts, never
mean of ratios) three ways -- all pairs, non-empty-GT pairs, empty-GT pairs as FP volume --
overall and per subset.

Runs in the detection env (goldilocs: numpy/PIL/scipy + the evaluator's pycocotools). CPU only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

CLASS_NAMES = {1: "ADDED", 2: "REMOVED", 3: "MOVED", 5: "REPLACED"}


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def load_manifest(root: Path) -> dict:
    return {r["pair"]: r for r in json.load(open(root / "manifest_test_top1.json"))["queries"]}


def arm_dirs(root: Path) -> list[str]:
    return sorted(p.name for p in (root / "SceneDiff" / "_experiments").iterdir()
                  if p.is_dir() and (p / "logs").exists() and list((p / "logs").glob("chunk_*.done")))


def run_replacement_pass(labels: np.ndarray, qdir: Path, exp: str, settings) -> tuple[np.ndarray, int]:
    """Exactly the pipeline's final stage, on the same inputs the run used."""
    from ocmask_pipeline.change_detection import find_color_replacement_regions
    from ocmask_pipeline.types import Label
    render_t0 = np.asarray(Image.open(qdir / exp / "labels" / "render_t0.png").convert("RGB"))  # the run's own input
    render_dir = qdir / "shared" / "render"
    image_t1 = np.asarray(Image.open(render_dir / "image_t1.png").convert("RGB"))
    coverage = np.load(render_dir / "render_t0_coverage.npy") if (render_dir / "render_t0_coverage.npy").exists() else None
    confidence = np.load(render_dir / "render_t0_confidence.npy") if (render_dir / "render_t0_confidence.npy").exists() else None
    out = labels.copy()
    replaced = find_color_replacement_regions(render_t0, image_t1, coverage, out != int(Label.UNCHANGED), settings, confidence)
    for item in replaced:
        out[item.mask] = int(Label.REPLACED)
    return out, len(replaced)


def score(labels_dir: Path, pair_dir: Path, t1_idx: int, eval_env: str) -> dict:
    out = labels_dir / "metrics.json"
    if not out.exists():
        r = subprocess.run(["conda", "run", "-n", eval_env, "python", str(REPO / "scripts/scenediff_diag_eval.py"),
                            "--pair-dir", str(pair_dir), "--t1-frame-idx", str(t1_idx),
                            "--labels-dir", str(labels_dir), "--out", str(out)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip().splitlines()[-1][:300])
    return json.loads(out.read_text())


def row_from_metrics(m: dict, extra: dict) -> dict:
    g = m["_gt_pixel_counts"]
    row = {**extra, "gt_empty": int((g["added"] + g["moved_bucket"]) == 0),
           "gt_added_px": g["added"], "gt_moved_px": g["moved_bucket"],
           "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "iou": m["iou"], "precision": m["precision"], "recall": m["recall"],
           "added_recall": m["_recall"].get("added_recall"), "moved_bucket_recall": m["_recall"].get("moved_bucket_recall"),
           "pred_changed_fraction": m.get("pred_changed_fraction")}
    for name in ("ADDED", "MOVED", "REMOVED", "REPLACED"):
        c = m.get(name, {})
        row[f"{name}_px"] = c.get("pixels", 0)
        for k in ("on_gt_added", "on_gt_moved_bucket", "on_gt_background"):
            row[f"{name}_{k}"] = c.get(k, 0)
    dc = m.get("decision_counts") or {}
    for k in ("added", "removed", "moved", "replaced", "unchanged"):
        row[f"decisions_{k}"] = dc.get(k, 0)
    for k in ("visibility_filter_rejected", "horizon_suppressed", "ceiling_sky_suppressed",
              "movable_object_gate_rejected", "corroboration_rejected", "tracking_recoveries"):
        row[k] = m.get(k)
    return row


def pooled(rows: list[dict]) -> dict:
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
    return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0}


def summarize(rows: list[dict]) -> dict:
    out = {}
    for subset in ("ALL", "SD-V", "SD-K"):
        rs = rows if subset == "ALL" else [r for r in rows if r["subset"] == subset]
        ne = [r for r in rs if not r["gt_empty"]]; em = [r for r in rs if r["gt_empty"]]
        out[subset] = {"all": pooled(rs), "non_empty": pooled(ne),
                       "empty_fp_volume": {"n": len(em), "fp": sum(r["fp"] for r in em)}}
        # per-class landing (pooled)
        cls = {}
        for name in ("ADDED", "MOVED", "REMOVED", "REPLACED"):
            px = sum(r[f"{name}_px"] for r in rs)
            cls[name] = {"pixels": px, **{k: sum(r[f"{name}_{k}"] for r in rs) for k in ("on_gt_added", "on_gt_moved_bucket", "on_gt_background")}}
        out[subset]["per_class_landing"] = cls
        ga = sum(r["gt_added_px"] for r in rs); gm = sum(r["gt_moved_px"] for r in rs)
        out[subset]["gt_bucket_recall"] = {
            "added": sum((r["added_recall"] or 0) * r["gt_added_px"] for r in rs) / ga if ga else None,
            "moved_bucket": sum((r["moved_bucket_recall"] or 0) * r["gt_moved_px"] for r in rs) / gm if gm else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO / "results/scenediff_single_query_covis_v1")
    ap.add_argument("--arms", nargs="*", default=None, help="experiment names; default = every completed arm")
    ap.add_argument("--baseline-config", type=Path, default=REPO / "configs/scenediff_v10_no_dino_no_refine.yaml",
                    help="config whose color_replacement_* thresholds the post-hoc replacement pass uses")
    ap.add_argument("--mask-root", type=Path, default=None, help="scenediff_movable_masks.py output for coverage columns")
    ap.add_argument("--out", type=Path, default=None, help="default <root>/analysis")
    ap.add_argument("--eval-env", default=os.environ.get("EVAL_CONDA_ENV", "goldilocs"))
    args = ap.parse_args()

    from ocmask_pipeline.change_detection import ThreeImageSettings
    from ocmask_pipeline.config import load_config
    os.environ.setdefault("SAM3_SOURCE", "unused"); os.environ.setdefault("SAM3_IMAGE_CHECKPOINT", "unused")
    cfg = load_config(args.baseline_config)
    cfg["three_image_comparison"]["enable_color_replacement_detection"] = True
    repl_settings = ThreeImageSettings.from_config(cfg)

    root = args.root; out_root = args.out or (root / "analysis"); out_root.mkdir(parents=True, exist_ok=True)
    man = load_manifest(root)
    coverage = {}
    if args.mask_root and (args.mask_root / "union_coverage.csv").exists():
        for r in csv.DictReader(open(args.mask_root / "union_coverage.csv")):
            coverage[r["pair"]] = float(r["union_coverage_fraction"])
    arms = args.arms or arm_dirs(root)
    print(f"arms: {arms}", flush=True)

    provenance = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "code_version": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip(),
                  "manifest_sha": sha16(root / "manifest_test_top1.json"),
                  "replacement_pass_settings": {k: getattr(repl_settings, k) for k in
                                                ("color_replacement_minimum_colorfulness", "color_replacement_residual_percentile",
                                                 "minimum_color_replacement_area", "color_replacement_minimum_confidence")},
                  "arms": {}}

    all_summaries = {}
    for exp in arms:
        exp_dir = root / "SceneDiff" / "_experiments" / exp
        exp_cfg = exp_dir / "config.yaml"
        ran_with = None
        if exp_cfg.exists():
            c = load_config(exp_cfg); ran_with = bool(c["three_image_comparison"].get("enable_color_replacement_detection", False))
        provenance["arms"][exp] = {"config_sha": sha16(exp_cfg) if exp_cfg.exists() else None, "ran_with_replacement": ran_with}
        rows = {"off": [], "on": []}; fails = []
        work = out_root / "_labels" / exp; t0 = time.perf_counter()
        for i, (pair, r) in enumerate(man.items()):
            q = f"t1_{r['t1_annotation_idx']:04d}"; qdir = root / "SceneDiff" / pair / q
            src = qdir / exp / "labels" / "labels.png"
            if not src.exists():
                fails.append((pair, "no labels.png")); continue
            base = np.asarray(Image.open(src))
            variants = {"off": base.copy(), "on": None}
            variants["off"][variants["off"] == 5] = 0
            if ran_with:                       # the run's own labels already ARE the "on" state
                variants["on"] = base.copy(); n_rep = int((base == 5).sum() > 0)
            else:                              # reproduce the pipeline's final pass on the run's labels
                variants["on"], n_rep = run_replacement_pass(base, qdir, exp, repl_settings)
            for state, lab in variants.items():
                d = work / state / pair / q; d.mkdir(parents=True, exist_ok=True)
                if not (d / "labels.png").exists():
                    Image.fromarray(lab).save(d / "labels.png"); shutil.copy(qdir / exp / "labels" / "inference.json", d / "inference.json")
                try:
                    m = score(d, REPO / "data/scenediff_benchmark/data" / pair, r["t1_annotation_idx"], args.eval_env)
                except RuntimeError as e:
                    fails.append((pair, f"{state}: {e}")); continue
                rows[state].append(row_from_metrics(m, {"arm": exp, "replacement": state, "pair": pair, "query": q,
                                                       "subset": r["subset"], "t1_annotation_idx": r["t1_annotation_idx"],
                                                       "whitelist_coverage": coverage.get(pair),
                                                       "replacement_objects_added_posthoc": (None if ran_with else n_rep)}))
            if (i + 1) % 50 == 0:
                print(f"  [{exp}] {i+1}/{len(man)} pairs, {time.perf_counter()-t0:.0f}s", flush=True)
        for state in ("off", "on"):
            if not rows[state]:
                continue
            with open(out_root / f"per_pair_{exp}_replacement_{state}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[state][0].keys())); w.writeheader(); w.writerows(rows[state])
            all_summaries[f"{exp}|replacement={state}"] = summarize(rows[state])
            s = all_summaries[f"{exp}|replacement={state}"]["ALL"]
            print(f"{exp:45s} replacement={state}: all IoU={s['all']['iou']:.4f} P={s['all']['precision']:.4f} R={s['all']['recall']:.4f} "
                  f"(n={s['all']['n']}) | non-empty IoU={s['non_empty']['iou']:.4f} | empty FP={s['empty_fp_volume']['fp']:,}", flush=True)
        provenance["arms"][exp]["failed"] = fails
        if fails:
            print(f"  {exp}: {len(fails)} pairs could not be scored: {fails[:3]}", flush=True)

    json.dump({"provenance": provenance, "summaries": all_summaries}, open(out_root / "summary.json", "w"), indent=2)
    with open(out_root / "README.md", "w") as fh:
        fh.write(__doc__)
        fh.write("\n\nFiles: per_pair_<arm>_replacement_{off,on}.csv (one row per pair), summary.json (pooled, partitioned, per-class), "
                 "_labels/ (the exact label maps each row was scored on).\n")
    print(f"wrote {out_root}/summary.json + per-pair CSVs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
