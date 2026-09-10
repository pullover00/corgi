#!/usr/bin/env python3
"""Aggregate the SceneDiff overnight ablations (2026-09-10) into one JSON +
markdown tables. Reads only what run_scenediff_diagnostic.py already wrote:
each experiment's summary.json (per-pair TP/FP/FN + per-class pixel
attribution from scenediff_diag_eval.py), each query's inference.json
(decisions, recovery, object counts, timings), the shared render coverage
buffers and the stage manifests (reconstruction / refine seconds).

Nothing here re-scores a prediction; a "tied" pair is one whose labels.png
is pixel-identical to the baseline's.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/scenediff_diagnostic/SceneDiff"
PAIRS = [l.strip() for l in (REPO / "data/scenediff_benchmark/diagnostic_subset.txt").read_text().split()]
IDENTITY_EVIDENCE = ("direct_identity", "clean_bridge_identity")
SHORT = {
    "P01-20240203-184214_0030_P01-20240203-184214_0032": "P01 184214 0030→0032",
    "P01-20240204-095114_0001_P01-20240204-095114_0011": "P01 095114 0001→0011",
}


def short(pair: str) -> str:
    if pair in SHORT:
        return SHORT[pair]
    parts = pair.split("_")  # <scene>_<a>_<scene>_<b>
    half = len(parts) // 2
    return "_".join(parts[:half]) + "→" + "_".join(parts[half:])


def query_dir(pair: str) -> Path:
    qs = sorted(p for p in (ROOT / pair).iterdir() if p.name.startswith("t1_"))
    assert len(qs) == 1, (pair, qs)
    return qs[0]


def shared_name(exp_root: Path) -> str:
    inv = exp_root / "invocation.json"
    n = json.loads(inv.read_text()).get("reference_views") if inv.exists() else None
    return f"shared_ref{n}" if n else "shared"


def decision_stats(inference: dict) -> dict:
    decisions = inference.get("decisions") or []
    consumed_t0 = {d["t0_object_id"] for d in decisions
                   if d.get("decision") in ("unchanged", "moved") and d.get("evidence") in IDENTITY_EVIDENCE}
    consumed_t1 = {d["t1_object_id"] for d in decisions
                   if d.get("decision") in ("unchanged", "moved") and d.get("evidence") in IDENTITY_EVIDENCE}
    counts = inference.get("object_counts") or {}
    rec = inference.get("tracking_recovery") or {}
    by_ev = {}
    for d in decisions:
        if d.get("decision") in ("unchanged", "moved"):
            by_ev[d.get("evidence") or "none"] = by_ev.get(d.get("evidence") or "none", 0) + 1
    return {
        "n_t0_objects": counts.get("render_t0"), "n_t1_objects": counts.get("image_t1"),
        "accepted_identities_direct": sum(1 for d in decisions if d.get("decision") in ("unchanged", "moved") and d.get("evidence") == "direct_identity"),
        "accepted_identities_bridge": sum(1 for d in decisions if d.get("decision") in ("unchanged", "moved") and d.get("evidence") == "clean_bridge_identity"),
        "accepted_identities": len(consumed_t0),
        "location_mismatch_rejected": sum(1 for d in decisions if d.get("decision") == "location_mismatch_rejected"),
        "unmatched_t0_after_identity": (counts.get("render_t0") or 0) - len(consumed_t0),
        "unmatched_t1_after_identity": (counts.get("image_t1") or 0) - len(consumed_t1),
        "recovery_events": rec.get("tracking_recoveries", 0),
        "recovered_t0_ids": rec.get("tracking_recovered_t0_ids", []),
        "recovered_t1_ids": rec.get("tracking_recovered_t1_ids", []),
        "visibility_filter_rejected": inference.get("visibility_filter_rejected"),
        "ceiling_sky_suppressed": inference.get("ceiling_sky_suppressed"),
        "occlusion_suppressed": inference.get("occlusion_suppressed"),
        "final_decision_counts": inference.get("decision_counts"),
        "confirmed_by_evidence": by_ev,
        "timings": inference.get("timings"),
    }


def collect(experiment: str) -> dict:
    exp_root = ROOT / "_experiments" / experiment
    summary = json.loads((exp_root / "summary.json").read_text())
    shared = shared_name(exp_root)
    rows = {}
    for pair in PAIRS:
        q = query_dir(pair)
        m = summary["per_pair"].get(pair)
        row = {"evaluated": m is not None, "reconstructed": (q / shared / "render" / "manifest.json").exists()}
        if m:
            row.update({k: m.get(k) for k in ("tp", "fp", "fn", "iou", "precision", "recall")})
            row["fp_by_class"] = {c: (m.get(c) or {}).get("on_gt_background", 0) for c in ("ADDED", "REMOVED", "MOVED", "REPLACED")}
            row["pixels_by_class"] = {c: (m.get(c) or {}).get("pixels", 0) for c in ("ADDED", "REMOVED", "MOVED", "REPLACED")}
            row["tp_by_class"] = {c: (m.get(c) or {}).get("on_gt_added", 0) + (m.get(c) or {}).get("on_gt_moved_bucket", 0)
                                  for c in ("ADDED", "REMOVED", "MOVED", "REPLACED")}
        cov = q / shared / "render" / "render_t0_coverage.npy"
        row["coverage"] = float(np.load(cov).mean()) if cov.exists() else None
        for stage, key in (("localization", "localization_seconds"), ("refine", "refine_seconds")):
            man = q / shared / stage / "manifest.json"
            row[key] = ((json.loads(man.read_text()).get("extra") or {}).get("seconds")) if man.exists() else None
        refman = ROOT / pair / shared / "reference_reconstruction" / "manifest.json"
        if refman.exists():
            extra = json.loads(refman.read_text()).get("extra") or {}
            row["reference_seconds"] = extra.get("seconds"); row["n_reference_frames"] = extra.get("n_frames")
            row["t0_frames"] = json.loads((ROOT / pair / shared / "reference_reconstruction" / "t0_frames.json").read_text())["indices"]
        alig = q / shared / "localization" / "alignment.json"
        row["alignment_residual"] = json.loads(alig.read_text())["alignment_residual"] if alig.exists() else None
        inf = q / experiment / "labels" / "inference.json"
        if inf.exists():
            row.update(decision_stats(json.loads(inf.read_text())))
        labman = q / experiment / "labels" / "manifest.json"
        row["detect_seconds_per_query"] = ((json.loads(labman.read_text()).get("extra") or {}).get("batch_seconds_per_query")) if labman.exists() else None
        rows[pair] = row
    ev = [r for r in rows.values() if r["evaluated"]]
    tp = sum(r["tp"] for r in ev); fp = sum(r["fp"] for r in ev); fn = sum(r["fn"] for r in ev)
    p = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    agg = {
        "experiment": experiment, "config": summary.get("config"), "failed": summary.get("failed"),
        "n_reconstructed": sum(r["reconstructed"] for r in rows.values()), "n_evaluated": len(ev),
        "tp": tp, "fp": fp, "fn": fn,
        "pooled_iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "mean_iou": float(np.mean([r["iou"] for r in ev])) if ev else None,
        "precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
        "fp_by_class": {c: sum(r["fp_by_class"][c] for r in ev) for c in ("ADDED", "REMOVED", "MOVED", "REPLACED")},
        "pixels_by_class": {c: sum(r["pixels_by_class"][c] for r in ev) for c in ("ADDED", "REMOVED", "MOVED", "REPLACED")},
        "mean_coverage": float(np.mean([r["coverage"] for r in rows.values() if r["coverage"] is not None])),
        "reference_seconds_total": sum(r.get("reference_seconds") or 0 for r in rows.values()),
        "localization_seconds_total": sum(r.get("localization_seconds") or 0 for r in rows.values()),
        "refine_seconds_total": sum(r.get("refine_seconds") or 0 for r in rows.values()),
        "detect_seconds_per_query_mean": float(np.mean([r["detect_seconds_per_query"] for r in rows.values() if r.get("detect_seconds_per_query")])) if any(r.get("detect_seconds_per_query") for r in rows.values()) else None,
        "detect_stage_seconds_mean": None,
        "final_decision_counts": {},
        "accepted_identities": sum(r.get("accepted_identities", 0) for r in ev),
        "location_mismatch_rejected": sum(r.get("location_mismatch_rejected", 0) for r in ev),
        "unmatched_t0_after_identity": sum(r.get("unmatched_t0_after_identity", 0) for r in ev),
        "unmatched_t1_after_identity": sum(r.get("unmatched_t1_after_identity", 0) for r in ev),
        "recovery_events": sum(r.get("recovery_events", 0) for r in ev),
        "visibility_filter_rejected": sum(r.get("visibility_filter_rejected") or 0 for r in ev),
        "occlusion_suppressed": sum(r.get("occlusion_suppressed") or 0 for r in ev),
        "n_t0_objects": sum(r.get("n_t0_objects") or 0 for r in ev), "n_t1_objects": sum(r.get("n_t1_objects") or 0 for r in ev),
        "per_pair": rows,
    }
    timing_keys = sorted({k for r in ev for k in (r.get("timings") or {})})
    agg["detect_stage_seconds_mean"] = {k: float(np.mean([r["timings"][k] for r in ev if k in (r.get("timings") or {})])) for k in timing_keys}
    for r in ev:
        for k, v in (r.get("final_decision_counts") or {}).items():
            agg["final_decision_counts"][k] = agg["final_decision_counts"].get(k, 0) + v
    return agg


def compare(base: dict, other: dict, base_exp: str, other_exp: str) -> dict:
    out = {"better": [], "worse": [], "tied": [], "pixel_identical": []}
    for pair in PAIRS:
        b, o = base["per_pair"][pair], other["per_pair"][pair]
        if not (b["evaluated"] and o["evaluated"]):
            continue
        q = query_dir(pair)
        lb = np.asarray(Image.open(q / base_exp / "labels/labels.png")); lo = np.asarray(Image.open(q / other_exp / "labels/labels.png"))
        identical = lb.shape == lo.shape and bool((lb == lo).all())
        if identical:
            out["pixel_identical"].append(pair)
        d = o["iou"] - b["iou"]
        (out["tied"] if abs(d) < 1e-6 else out["better"] if d > 0 else out["worse"]).append(pair)
    return out


def fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="scenediff_v10_no_dino")
    ap.add_argument("--experiments", nargs="*", default=[
        "scenediff_v10_no_dino", "scenediff_v10_no_dino_ref1", "scenediff_v10_no_dino_ref3",
        "scenediff_v10_no_dino_ref5", "scenediff_v10_no_dino_no_refine",
        "scenediff_v10_no_dino_no_recovery", "scenediff_v10_no_dino_no_appearance"])
    ap.add_argument("--out", type=Path, default=ROOT / "_experiments" / "overnight_ablations_report.json")
    args = ap.parse_args()

    aggs = {}
    for exp in args.experiments:
        if (ROOT / "_experiments" / exp / "summary.json").exists():
            aggs[exp] = collect(exp)
        else:
            print(f"(skipping {exp}: no summary.json yet)")
    comparisons = {}
    if args.baseline in aggs:
        for exp in aggs:
            if exp != args.baseline:
                comparisons[exp] = compare(aggs[args.baseline], aggs[exp], args.baseline, exp)
    args.out.write_text(json.dumps({"experiments": aggs, "comparisons_vs_baseline": comparisons}, indent=2, default=str))

    print("| Variant | n recon | n eval | Pooled IoU | Mean IoU | Precision | Recall | F1 | Coverage | TP | FP | FN | ADDED FP | REMOVED FP | MOVED FP | REPLACED FP |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for exp, a in aggs.items():
        f = a["fp_by_class"]
        print(f"| {exp.replace('scenediff_v10_no_dino', 'v10_no_dino')} | {a['n_reconstructed']}/10 | {a['n_evaluated']}/10 | {fmt(a['pooled_iou'])} | {fmt(a['mean_iou'])} | "
              f"{fmt(a['precision'])} | {fmt(a['recall'])} | {fmt(a['f1'])} | {fmt(a['mean_coverage'], 3)} | {a['tp']:,} | {a['fp']:,} | {a['fn']:,} | "
              f"{f['ADDED']:,} | {f['REMOVED']:,} | {f['MOVED']:,} | {f['REPLACED']:,} |")
    print()
    print("| Variant | identities | loc-mismatch rej | unmatched T0 | unmatched T1 | recoveries | vis-filtered | occl-merged | final decisions |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for exp, a in aggs.items():
        print(f"| {exp.replace('scenediff_v10_no_dino', 'v10_no_dino')} | {a['accepted_identities']} | {a['location_mismatch_rejected']} | {a['unmatched_t0_after_identity']} | "
              f"{a['unmatched_t1_after_identity']} | {a['recovery_events']} | {a['visibility_filter_rejected']} | {a['occlusion_suppressed']} | {a['final_decision_counts']} |")
    print()
    header = "| Pair | " + " | ".join(e.replace("scenediff_v10_no_dino", "v10") or "v10" for e in aggs) + " |"
    print(header); print("|---|" + "---:|" * len(aggs))
    for pair in PAIRS:
        cells = []
        for a in aggs.values():
            r = a["per_pair"][pair]
            cells.append(fmt(r["iou"]) if r["evaluated"] else ("FAIL" if not r["reconstructed"] else "no-eval"))
        print(f"| {short(pair)} | " + " | ".join(cells) + " |")
    print()
    for exp, c in comparisons.items():
        print(f"{exp} vs {args.baseline}: better {len(c['better'])} / worse {len(c['worse'])} / tied {len(c['tied'])} "
              f"(pixel-identical {len(c['pixel_identical'])})")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
