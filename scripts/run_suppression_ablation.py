#!/usr/bin/env python3
"""Detect-only ablation of the pipeline's change-SUPPRESSION mechanisms (plus
color-replacement detection) on cached, already-refined PASLCD queries.

Motivation (2026-09-08 diagnosis on 375 refine-complete PASLCD queries):
62% of GT-changed pixels are missed, 96.5% of those misses are in regions
where render_t0 HAD reconstructed geometry, and per query the pipeline
discards ~30 candidate changes (12.9 visibility-filter rejections + 16.9
recall-recovery reclassifications) while keeping ~26.5. So the recall loss
is a matching/suppression problem, not a small-object or reconstruction-hole
problem (objects <20px hold only ~11% of missed pixels).

Runs every variant in configs/ablate_v*.yaml over every query found under
results/paslcd_ablation_cache/<scene>/intermediate/<stem>/ (rsynced from the
remote 500-query run: reconstruction/ + refined/), one batched
detect_batch.py call per variant, then scores each against PASLCD GT with the
official binary convention (ocmask_pipeline.metrics). Resumable: a query
whose labels.png already exists for a variant is not recomputed.

Run inside the detection env (goldilocs), with SAM3_SOURCE and
SAM3_IMAGE_CHECKPOINT exported, from the repo root.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CACHE = REPO / "results/paslcd_ablation_cache"
OUT = REPO / "results/paslcd_suppression_ablation"
DATA_ROOT = REPO / "data/PASLCD"

VARIANTS = [
    ("v0_baseline", "configs/ablate_v0_baseline.yaml"),
    ("v1_no_color_replacement", "configs/ablate_v1_no_color_replacement.yaml"),
    ("v2_no_recall_recovery", "configs/ablate_v2_no_recall_recovery.yaml"),
    ("v3_no_visibility_filter", "configs/ablate_v3_no_visibility_filter.yaml"),
    ("v4_no_color_no_recovery", "configs/ablate_v4_no_color_no_recovery.yaml"),
    # Needs above_horizon.npy in each cached query's reconstruction/ dir --
    # run scripts/backfill_above_horizon.py first, or this variant silently
    # scores identically to v0 (the filter no-ops without its input).
    ("v5_horizon_suppression", "configs/ablate_v5_horizon_suppression.yaml"),
    # Successor to v5: same suppression logic, but the input mask comes from
    # SAM3 grounded text-prompt detection ("ceiling"/"sky" on image_t1)
    # rather than RANSAC-plane + camera-pose geometry. Needs no backfill --
    # computed in-process during detect from a genuinely separate SAM3 model
    # (Sam3TextPromptDetector); see that class's docstring for why it can't
    # share generator's model.
    ("v6_ceiling_sky_suppression", "configs/ablate_v6_ceiling_sky_suppression.yaml"),
    # Exact v6 model-removal ablation: all v6 settings are preserved and
    # DINOv2 alone is removed from every appearance/identity evidence gate.
    ("v6_no_dino", "configs/model_ablation_m2_no_dino.yaml"),
    # Controlled state-handling ablation on v6_no_dino: preserve established
    # identities across an ambiguous location test and make the existing
    # visibility-based UNKNOWN handling explicit for unmatched hypotheses.
    ("v6_no_dino_state_resolver", "configs/ablate_v6_no_dino_state_resolver.yaml"),
    # v6 base + occlusion-aware REMOVED suppression (a REMOVED decision
    # mostly covered by an ADDED object's footprint is reclassified, since
    # it is occlusion by the addition, not a separate removal) + REPLACED
    # disabled entirely, per user request. Cache-only, no reconstruction
    # needed -- pure detect-side logic.
    ("v7_occlusion_and_no_replace", "configs/ablate_v7_occlusion_and_no_replace.yaml"),
    # Isolates occlusion suppression from v7's second change (REPLACED
    # disabled) -- exactly v6 plus enable_occlusion_aware_removal_suppression,
    # REPLACED left on. Requested after v7's aggregate result (roughly flat)
    # turned out to hide occlusion suppression's real gain being offset by
    # REPLACED's loss.
    ("v8_occlusion_only", "configs/ablate_v8_occlusion_only.yaml"),
    # v9: same flags as v8, but the occlusion test now uses the depth ordering
    # of render_t0_positions vs image_t1_positions (2D overlap only as fallback)
    # and merges an occluded footprint into the addition instead of clearing it.
    ("v9_occlusion_depth", "configs/ablate_v9_occlusion_depth.yaml"),
    # v10: v9 with DINOv2 removed from both computation and every identity
    # decision (use_dino_features: false) -- user decision 2026-09-09.
    ("v10_no_dino", "configs/ablate_v10_no_dino.yaml"),
]

GEOMETRY_KEYS = {
    "render_t0_positions": "render_t0_positions.npy",
    "clean_render_positions": "clean_render_positions.npy",
    "image_t1_positions": "image_t1_positions.npy",
    "scene_scale_path": "scene_scale.json",
    "render_t0_coverage": "render_t0_coverage.npy",
    "render_t0_confidence": "render_t0_confidence.npy",
    "render_t0_corroboration": "render_t0_corroboration.npy",
    "above_horizon": "above_horizon.npy",
}


def discover_queries() -> list[tuple[str, str, Path]]:
    found = []
    for scene_dir in sorted(p for p in CACHE.iterdir() if p.is_dir()):
        inter = scene_dir / "intermediate"
        if not inter.exists():
            continue
        for stem_dir in sorted(p for p in inter.iterdir() if p.is_dir()):
            needed = [stem_dir / "refined/render_t0.png", stem_dir / "refined/clean_render.png",
                      stem_dir / "reconstruction/image_t1.png"]
            if all(p.exists() for p in needed):
                found.append((scene_dir.name, stem_dir.name, stem_dir))
            else:
                print(f"  skipping incomplete cache entry {scene_dir.name}/{stem_dir.name}", flush=True)
    return found


def inventory_dir(stem_dir: Path) -> Path:
    """Where a query's dumped stages-1-3 bundle lives (see
    change_detection._dump_inventory_bundle). Only valid to load for a
    config whose sam3_proposals/dinov2_features/sam2_tracking sections
    match whatever config produced the dump."""
    return stem_dir / "inventory"


def manifest_entry(stem_dir: Path, out_dir: Path, *, dump_inventory: bool = False,
                   fast_replay: bool = False) -> tuple[dict, bool]:
    """Returns (entry, used_fast_replay). fast_replay silently falls back to
    the normal slow entry when this query has no dump yet -- a partially-
    populated cache degrades to correct-but-slow, never to a crash or a
    silently-wrong result."""
    recon = stem_dir / "reconstruction"
    entry = {
        "render_t0": str(stem_dir / "refined/render_t0.png"),
        "clean_render": str(stem_dir / "refined/clean_render.png"),
        "image_t1": str(recon / "image_t1.png"),
        "output_dir": str(out_dir),
    }
    for key, name in GEOMETRY_KEYS.items():
        if (recon / name).exists():
            entry[key] = str(recon / name)
    used_fast = False
    inv_dir = inventory_dir(stem_dir)
    if fast_replay and (inv_dir / "bundle.pkl").exists():
        entry["load_inventory_from"] = str(inv_dir)
        used_fast = True
    if dump_inventory:
        entry["dump_inventory_to"] = str(inv_dir)
    return entry, used_fast


def evaluate(scene: str, stem: str, labels_path: Path):
    import cv2
    import numpy as np
    from ocmask_pipeline.metrics import binarize_prediction, compute_binary_metrics, load_paslcd_gt

    dataset, inst = scene.rsplit("_Instance_", 1)
    gt = load_paslcd_gt(DATA_ROOT / dataset / f"Instance_{inst}" / "gt_mask" / f"{stem}.png")
    labels = cv2.imread(str(labels_path), cv2.IMREAD_GRAYSCALE)
    if labels is None:
        return None
    if labels.shape != gt.shape:
        labels = cv2.resize(labels, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred = binarize_prediction((labels != 0).astype(np.uint8) * 255, gt.shape)
    m = compute_binary_metrics(gt, pred)
    return {"iou": m.iou, "f1": m.f1, "precision": m.precision, "recall": m.recall,
            "tp": m.tp, "fp": m.fp, "fn": m.fn}


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=None, help="subset of variant names to run (default: all)")
    parser.add_argument("--dump-inventory", action="store_true",
                        help="also save stages 1-3's output per query (results/paslcd_ablation_cache/.../inventory/) "
                             "for a later --fast-replay run to reuse -- adds negligible time to this run")
    parser.add_argument("--fast-replay", action="store_true",
                        help="skip stages 1-3 (SAM3/DINOv2/SAM2, ~95%% of wall time) using a prior --dump-inventory "
                             "run's saved bundle -- ONLY VALID when this variant's config leaves sam3_proposals/"
                             "dinov2_features/sam2_tracking unchanged from whatever config produced the dump; "
                             "a query with no dump yet falls back to the normal slow path automatically")
    args = parser.parse_args()

    variants = [(n, c) for n, c in VARIANTS if not args.variants or n in args.variants]
    queries = discover_queries()
    if not queries:
        print(f"no complete cached queries under {CACHE}", flush=True)
        return 1
    print(f"{len(queries)} cached queries; running {len(variants)} variant(s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    summary = {"n_queries": len(queries), "variants": {}}
    per_query_all: dict[str, dict[str, dict]] = {}

    for name, cfg in variants:
        started = time.perf_counter()
        entries = []
        n_fast = 0
        for scene, stem, stem_dir in queries:
            out_dir = OUT / name / scene / stem / "detect"
            if (out_dir / "labels.png").exists():
                continue
            entry, used_fast = manifest_entry(stem_dir, out_dir, dump_inventory=args.dump_inventory, fast_replay=args.fast_replay)
            entries.append(entry)
            n_fast += used_fast
        manifest_path = OUT / name / "detect_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(entries, indent=2))
        fast_note = f", {n_fast}/{len(entries)} via fast replay" if args.fast_replay else ""
        print(f"=== [{name}] {len(entries)} queries to run ({len(queries) - len(entries)} already done){fast_note}, config={cfg} ===", flush=True)
        if entries:
            subprocess.run(
                [sys.executable, str(REPO / "scripts/detect_batch.py"),
                 "--manifest", str(manifest_path), "--config", str(REPO / cfg)],
                check=True,
            )

        per_query = {}
        for scene, stem, _ in queries:
            m = evaluate(scene, stem, OUT / name / scene / stem / "detect" / "labels.png")
            if m is not None:
                per_query[f"{scene}/{stem}"] = m
        per_query_all[name] = per_query
        agg = {k: mean([m[k] for m in per_query.values()]) for k in ("iou", "f1", "precision", "recall")}
        per_scene = {}
        for key, m in per_query.items():
            per_scene.setdefault(key.split("/")[0], []).append(m["iou"])
        summary["variants"][name] = {
            "config": cfg, "n": len(per_query), **agg,
            "per_scene_iou": {s: mean(v) for s, v in per_scene.items()},
            "per_query": per_query,
            "wall_seconds": time.perf_counter() - started,
        }
        (OUT / name / "eval.json").write_text(json.dumps(summary["variants"][name], indent=2))
        print(f"=== [{name}] n={len(per_query)} mIoU={agg['iou']:.4f} F1={agg['f1']:.4f} "
              f"P={agg['precision']:.4f} R={agg['recall']:.4f} ({time.perf_counter() - started:.0f}s) ===", flush=True)

    print("\n" + "=" * 78)
    print(f"{'variant':28s} {'n':>3s} {'mIoU':>8s} {'F1':>8s} {'prec':>8s} {'recall':>8s} {'dIoU':>8s} {'dF1':>8s}")
    base = summary["variants"].get("v0_baseline")
    for name, v in summary["variants"].items():
        d_iou = v["iou"] - base["iou"] if base else float("nan")
        d_f1 = v["f1"] - base["f1"] if base else float("nan")
        print(f"{name:28s} {v['n']:3d} {v['iou']:8.4f} {v['f1']:8.4f} {v['precision']:8.4f} {v['recall']:8.4f} {d_iou:+8.4f} {d_f1:+8.4f}")
    if base:
        print("\nper-query IoU deltas vs baseline (wins / losses / ties, |delta| > 0.005):")
        for name, pq in per_query_all.items():
            if name == "v0_baseline":
                continue
            deltas = [pq[k]["iou"] - base["per_query"][k]["iou"] for k in pq if k in base["per_query"]]
            wins = sum(d > 0.005 for d in deltas)
            losses = sum(d < -0.005 for d in deltas)
            print(f"  {name:28s} wins={wins:2d} losses={losses:2d} ties={len(deltas) - wins - losses:2d}")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT / 'summary.json'}")
    print("SUPPRESSION_ABLATION_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
