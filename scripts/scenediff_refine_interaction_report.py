#!/usr/bin/env python3
"""Analyze the 2x4 SceneDiff factorial: reference views N in {1,3,5,10} x
DI2FIX render refinement {ON, OFF}.

Reads only what the runs already wrote (each experiment's summary.json with
per-pair TP/FP/FN and per-class pixel attribution, each query's inference.json
for object/decision counts, and the shared render coverage buffers). Nothing
is re-scored here.

The ON and OFF cells at a given N share the SAME shared_ref<N> reconstruction/
localization/render artifacts, so render coverage is identical within a column
by construction -- reported explicitly as a control check, since DI2FIX only
changes RGB appearance and cannot change coverage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/scenediff_diagnostic/SceneDiff"
EXPERIMENTS = ROOT / "_experiments"
PAIRS = [l.strip() for l in (REPO / "data/scenediff_benchmark/diagnostic_subset.txt").read_text().split()]
CLASSES = ("ADDED", "REMOVED", "MOVED", "REPLACED")
ZERO_COVERAGE_EPS = 1e-6

# (N, refine) -> (experiment name, shared-cell name holding reconstruction/render)
CELLS = {
    (1, "ON"):  ("scenediff_v10_no_dino_ref1", "shared_ref1"),
    (3, "ON"):  ("scenediff_v10_no_dino_ref3", "shared_ref3"),
    (5, "ON"):  ("scenediff_v10_no_dino_ref5", "shared_ref5"),
    (10, "ON"): ("scenediff_v10_no_dino", "shared"),
    (1, "OFF"):  ("scenediff_v10_no_dino_ref1_no_refine", "shared_ref1"),
    (3, "OFF"):  ("scenediff_v10_no_dino_ref3_no_refine", "shared_ref3"),
    (5, "OFF"):  ("scenediff_v10_no_dino_ref5_no_refine", "shared_ref5"),
    (10, "OFF"): ("scenediff_v10_no_dino_no_refine", "shared"),
}
SHORT = {
    "P01-20240203-184214_0030_P01-20240203-184214_0032": "P01 184214 0030-0032",
    "P01-20240204-095114_0001_P01-20240204-095114_0011": "P01 095114 0001-0011",
}


def short(pair: str) -> str:
    if pair in SHORT:
        return SHORT[pair]
    parts = pair.split("_")
    half = len(parts) // 2
    return "_".join(parts[:half]) + "-" + "_".join(parts[half:])


def query_dir(pair: str) -> Path:
    return sorted(p for p in (ROOT / pair).iterdir() if p.name.startswith("t1_"))[0]


def collect(experiment: str, shared: str) -> dict | None:
    summary_path = EXPERIMENTS / experiment / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    per_pair = {}
    for pair in PAIRS:
        q = query_dir(pair)
        m = summary["per_pair"].get(pair)
        cov_path = q / shared / "render" / "render_t0_coverage.npy"
        coverage = float(np.load(cov_path).mean()) if cov_path.exists() else None
        row = {"evaluated": m is not None, "coverage": coverage,
               "zero_coverage": coverage is not None and coverage <= ZERO_COVERAGE_EPS}
        if m:
            row.update({k: m.get(k) for k in ("tp", "fp", "fn", "iou", "precision", "recall")})
            row["fp_by_class"] = {c: (m.get(c) or {}).get("on_gt_background", 0) for c in CLASSES}
            row["pixels_by_class"] = {c: (m.get(c) or {}).get("pixels", 0) for c in CLASSES}
            row["decision_counts"] = m.get("decision_counts")
            row["object_counts"] = m.get("object_counts")
            row["visibility_filter_rejected"] = m.get("visibility_filter_rejected")
            row["tracking_recoveries"] = m.get("tracking_recoveries")
        alig = q / shared / "localization" / "alignment.json"
        row["alignment_residual"] = json.loads(alig.read_text())["alignment_residual"] if alig.exists() else None
        per_pair[pair] = row

    ev = [r for r in per_pair.values() if r["evaluated"]]
    tp = sum(r["tp"] for r in ev); fp = sum(r["fp"] for r in ev); fn = sum(r["fn"] for r in ev)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {
        "experiment": experiment, "shared_cell": shared, "config": summary.get("config"),
        "failed": summary.get("failed"), "n_evaluated": len(ev),
        "tp": tp, "fp": fp, "fn": fn,
        "pooled_iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "mean_iou": float(np.mean([r["iou"] for r in ev])) if ev else None,
        "precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
        "fp_by_class": {c: sum(r["fp_by_class"][c] for r in ev) for c in CLASSES},
        "mean_coverage": float(np.mean([r["coverage"] for r in per_pair.values() if r["coverage"] is not None])),
        "n_zero_coverage": sum(1 for r in per_pair.values() if r["zero_coverage"]),
        "zero_coverage_pairs": [p_ for p_, r in per_pair.items() if r["zero_coverage"]],
        "per_pair": per_pair,
    }


def fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def main() -> int:
    cells = {}
    missing = []
    for key, (exp, shared) in CELLS.items():
        got = collect(exp, shared)
        if got is None:
            missing.append(f"N={key[0]} refine={key[1]} ({exp})")
        else:
            cells[key] = got
    if missing:
        print("MISSING CELLS (not yet run):")
        for m in missing:
            print("  ", m)
        print()
    if not cells:
        return 1

    Ns = [n for n in (1, 3, 5, 10) if (n, "ON") in cells and (n, "OFF") in cells]

    lines = []
    def emit(s=""):
        lines.append(s); print(s)

    emit("## Main 2x4 comparison")
    emit()
    emit("| N | Coverage | IoU refine ON | IoU refine OFF | dIoU OFF-ON | F1 ON | F1 OFF | dFP OFF-ON |")
    emit("|---|---:|---:|---:|---:|---:|---:|---:|")
    for n in Ns:
        on, off = cells[(n, "ON")], cells[(n, "OFF")]
        cov = on["mean_coverage"]
        emit(f"| {n} | {fmt(cov,3)} | {fmt(on['pooled_iou'])} | {fmt(off['pooled_iou'])} | "
             f"{off['pooled_iou']-on['pooled_iou']:+.4f} | {fmt(on['f1'])} | {fmt(off['f1'])} | "
             f"{off['fp']-on['fp']:+,} |")
    emit()

    emit("## Full per-cell metrics")
    emit()
    emit("| N | refine | pooled IoU | mean IoU | P | R | F1 | TP | FP | FN | coverage | n eval | zero-cov |")
    emit("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in Ns:
        for state in ("ON", "OFF"):
            c = cells[(n, state)]
            emit(f"| {n} | {state} | {fmt(c['pooled_iou'])} | {fmt(c['mean_iou'])} | {fmt(c['precision'])} | "
                 f"{fmt(c['recall'])} | {fmt(c['f1'])} | {c['tp']:,} | {c['fp']:,} | {c['fn']:,} | "
                 f"{fmt(c['mean_coverage'],3)} | {c['n_evaluated']}/10 | {c['n_zero_coverage']} |")
    emit()

    emit("## False positives by predicted class")
    emit()
    emit("| N | refine | ADDED FP | REMOVED FP | MOVED FP | REPLACED FP | total FP |")
    emit("|---|---|---:|---:|---:|---:|---:|")
    for n in Ns:
        for state in ("ON", "OFF"):
            c = cells[(n, state)]; f = c["fp_by_class"]
            emit(f"| {n} | {state} | {f['ADDED']:,} | {f['REMOVED']:,} | {f['MOVED']:,} | {f['REPLACED']:,} | {c['fp']:,} |")
    emit()

    emit("## Per-pair IoU (ON -> OFF, delta)")
    emit()
    header = "| Pair | cov (N=10) |" + "".join(f" N={n} ON | N={n} OFF | N={n} d |" for n in Ns)
    emit(header)
    emit("|---|---:|" + "---:|" * (3 * len(Ns)))
    for pair in PAIRS:
        cov10 = cells[(10, "ON")]["per_pair"][pair]["coverage"] if (10, "ON") in cells else None
        row = f"| {short(pair)} | {fmt(cov10,3)} |"
        for n in Ns:
            a = cells[(n, "ON")]["per_pair"][pair]
            b = cells[(n, "OFF")]["per_pair"][pair]
            if a["evaluated"] and b["evaluated"]:
                row += f" {a['iou']:.4f} | {b['iou']:.4f} | {b['iou']-a['iou']:+.4f} |"
            else:
                row += " — | — | — |"
        emit(row)
    emit()

    # render coverage per (pair, N).  Identical between ON and OFF by construction
    # (both read the same shared_ref{N} render cell), so one column per N suffices.
    emit("## Per-pair render coverage by N")
    emit()
    emit("| Pair |" + "".join(f" N={n} |" for n in Ns))
    emit("|---|" + "---:|" * len(Ns))
    for pair in PAIRS:
        row = f"| {short(pair)} |"
        for n in Ns:
            c = cells.get((n, "ON")) or cells.get((n, "OFF"))
            cov = c["per_pair"][pair]["coverage"] if c else None
            row += f" {fmt(cov,3)} |"
        emit(row)
    emit()

    # coverage vs delta-IoU across all N x pair cells
    emit("## Coverage vs refinement benefit (all N x pair cells)")
    emit()
    xs, ys, rows = [], [], []
    for n in Ns:
        for pair in PAIRS:
            a = cells[(n, "ON")]["per_pair"][pair]
            b = cells[(n, "OFF")]["per_pair"][pair]
            if not (a["evaluated"] and b["evaluated"] and a["coverage"] is not None):
                continue
            d = b["iou"] - a["iou"]           # OFF - ON  (positive => refinement HURT)
            rows.append({"N": n, "pair": pair, "coverage": a["coverage"],
                         "iou_on": a["iou"], "iou_off": b["iou"], "delta_off_minus_on": d,
                         "zero_coverage": a["zero_coverage"]})
            xs.append(a["coverage"]); ys.append(d)
    if len(xs) >= 3:
        r = float(np.corrcoef(xs, ys)[0, 1])
        emit(f"n = {len(xs)} (N x pair) cells. Pearson r(coverage, dIoU_OFF-ON) = {r:+.3f}")
        nz = [(x, y) for x, y in zip(xs, ys) if x > ZERO_COVERAGE_EPS]
        if len(nz) >= 3:
            r2 = float(np.corrcoef([x for x, _ in nz], [y for _, y in nz])[0, 1])
            emit(f"excluding zero-coverage cells: n = {len(nz)}, r = {r2:+.3f}")
        emit()
        emit("Positive dIoU means the RAW render scored higher, i.e. refinement HURT that cell.")
        emit()
        # binned view
        emit("| coverage bin | n cells | mean dIoU (OFF-ON) | refinement helped (d<0) | hurt (d>0) | tied |")
        emit("|---|---:|---:|---:|---:|---:|")
        bins = [(0.0, 0.001, "0 (failed render)"), (0.001, 0.4, "0-0.4"), (0.4, 0.6, "0.4-0.6"),
                (0.6, 0.8, "0.6-0.8"), (0.8, 1.01, "0.8-1.0")]
        for lo, hi, name in bins:
            sel = [r_ for r_ in rows if lo <= r_["coverage"] < hi]
            if not sel:
                continue
            d = [r_["delta_off_minus_on"] for r_ in sel]
            emit(f"| {name} | {len(sel)} | {np.mean(d):+.4f} | {sum(1 for x in d if x < -1e-9)} | "
                 f"{sum(1 for x in d if x > 1e-9)} | {sum(1 for x in d if abs(x) <= 1e-9)} |")
    emit()

    emit("## Zero-coverage / failed-render cells")
    emit()
    any_zero = False
    for n in Ns:
        for state in ("ON", "OFF"):
            c = cells[(n, state)]
            for pair in c["zero_coverage_pairs"]:
                any_zero = True
                r_ = c["per_pair"][pair]
                emit(f"- N={n} refine={state} `{short(pair)}`: coverage=0, IoU={fmt(r_.get('iou'))}, "
                     f"TP={r_.get('tp')}, FP={r_.get('fp')}, alignment_residual={fmt(r_.get('alignment_residual'),5)}")
    if not any_zero:
        emit("- none")
    emit()

    emit("## Strongest help / hurt per N")
    emit()
    for n in Ns:
        deltas = []
        for pair in PAIRS:
            a = cells[(n, "ON")]["per_pair"][pair]; b = cells[(n, "OFF")]["per_pair"][pair]
            if a["evaluated"] and b["evaluated"]:
                deltas.append((b["iou"] - a["iou"], pair, a, b))
        if not deltas:
            continue
        helped = min(deltas, key=lambda t: t[0])   # most negative OFF-ON => refinement helped most
        hurt = max(deltas, key=lambda t: t[0])
        emit(f"**N={n}**")
        emit(f"- refinement HELPED most: `{short(helped[1])}` IoU {helped[2]['iou']:.4f} (ON) vs "
             f"{helped[3]['iou']:.4f} (OFF), d={helped[0]:+.4f}, coverage={fmt(helped[2]['coverage'],3)}, "
             f"objects ON={helped[2].get('object_counts')} OFF={helped[3].get('object_counts')}")
        emit(f"- refinement HURT most: `{short(hurt[1])}` IoU {hurt[2]['iou']:.4f} (ON) vs "
             f"{hurt[3]['iou']:.4f} (OFF), d={hurt[0]:+.4f}, coverage={fmt(hurt[2]['coverage'],3)}, "
             f"objects ON={hurt[2].get('object_counts')} OFF={hurt[3].get('object_counts')}")
        emit()

    # control check: coverage identical within an N column
    emit("## Control check: coverage identical between ON and OFF at each N")
    emit()
    for n in Ns:
        a = cells[(n, "ON")]; b = cells[(n, "OFF")]
        same = all(
            (a["per_pair"][p_]["coverage"] is None and b["per_pair"][p_]["coverage"] is None)
            or abs(a["per_pair"][p_]["coverage"] - b["per_pair"][p_]["coverage"]) < 1e-12
            for p_ in PAIRS
        )
        emit(f"- N={n}: shared cell `{a['shared_cell']}` vs `{b['shared_cell']}` -> "
             f"per-pair coverage identical: {same}")
    emit()

    # ---- derived interaction statistics (paired, per N) ----------------------
    # Sign counts and the paired Wilcoxon are computed on the 10 per-pair deltas
    # within each N.  Ties (identical ON/OFF IoU, e.g. zero-coverage cells) are
    # dropped from the Wilcoxon but kept in the sign counts.
    emit("## Interaction statistics per N")
    emit()
    emit("| N | helped (d<0) | hurt (d>0) | tied | mean dIoU | median dIoU | Wilcoxon p (non-tied) |")
    emit("|---|---:|---:|---:|---:|---:|---:|")
    interaction = {}
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None
    for n in Ns:
        ds = [r_["delta_off_minus_on"] for r_ in rows if r_["N"] == n]
        if not ds:
            continue
        nzd = [x for x in ds if abs(x) > 1e-9]
        pval = None
        if wilcoxon is not None and len(nzd) >= 1:
            try:
                pval = float(wilcoxon(nzd)[1])
            except Exception:
                pval = None
        rec = {"n_helped": sum(1 for x in ds if x < -1e-9),
               "n_hurt": sum(1 for x in ds if x > 1e-9),
               "n_tied": sum(1 for x in ds if abs(x) <= 1e-9),
               "mean_delta": float(np.mean(ds)), "median_delta": float(np.median(ds)),
               "wilcoxon_p_nontied": pval, "n_nontied": len(nzd)}
        interaction[n] = rec
        emit(f"| {n} | {rec['n_helped']} | {rec['n_hurt']} | {rec['n_tied']} | "
             f"{rec['mean_delta']:+.4f} | {rec['median_delta']:+.4f} | "
             f"{'n/a' if pval is None else f'{pval:.3f}'} |")
    emit()
    trend_r = None
    if len(interaction) >= 3:
        ks = sorted(interaction)
        trend_r = float(np.corrcoef(ks, [interaction[k]["mean_delta"] for k in ks])[0, 1])
        emit(f"Trend of mean dIoU against N: r = {trend_r:+.3f} "
             f"(positive = refinement becomes more harmful as the reference grows).")
        emit()

    # ---- render-side SAM3 proposal counts ------------------------------------
    # DI2FIX only rewrites the RGB of render_t0, so the render_t0 proposal count
    # is the most direct downstream measurement of what refinement changed.
    emit("## Render-side SAM3 proposal counts (render_t0)")
    emit()
    emit("| N | proposals ON | proposals OFF | delta | REMOVED ON | REMOVED OFF | ADDED ON | ADDED OFF |")
    emit("|---|---:|---:|---:|---:|---:|---:|---:|")
    prop_rows, prop_totals = [], {}
    for n in Ns:
        tot = {"prop_on": 0, "prop_off": 0, "rem_on": 0, "rem_off": 0, "add_on": 0, "add_off": 0}
        for pair in PAIRS:
            a = cells[(n, "ON")]["per_pair"][pair]
            b = cells[(n, "OFF")]["per_pair"][pair]
            oa, ob = a.get("object_counts") or {}, b.get("object_counts") or {}
            da, db = a.get("decision_counts") or {}, b.get("decision_counts") or {}
            pa, pb = oa.get("render_t0"), ob.get("render_t0")
            prop_rows.append({"N": n, "pair": pair, "prop_on": pa, "prop_off": pb,
                              "delta_prop_on_minus_off": (pa - pb) if (pa is not None and pb is not None) else None,
                              "delta_iou_off_minus_on": b["iou"] - a["iou"] if (a["evaluated"] and b["evaluated"]) else None,
                              "coverage": a["coverage"]})
            tot["prop_on"] += pa or 0; tot["prop_off"] += pb or 0
            tot["rem_on"] += da.get("removed", 0); tot["rem_off"] += db.get("removed", 0)
            tot["add_on"] += da.get("added", 0); tot["add_off"] += db.get("added", 0)
        prop_totals[n] = tot
        emit(f"| {n} | {tot['prop_on']} | {tot['prop_off']} | {tot['prop_on'] - tot['prop_off']:+d} | "
             f"{tot['rem_on']} | {tot['rem_off']} | {tot['add_on']} | {tot['add_off']} |")
    emit()
    prop_corr = None
    usable = [r_ for r_ in prop_rows
              if r_["delta_prop_on_minus_off"] is not None and r_["delta_iou_off_minus_on"] is not None
              and r_["coverage"] is not None and r_["coverage"] > ZERO_COVERAGE_EPS]
    if len(usable) >= 3:
        prop_corr = float(np.corrcoef([r_["delta_prop_on_minus_off"] for r_ in usable],
                                      [r_["delta_iou_off_minus_on"] for r_ in usable])[0, 1])
        emit(f"Pearson r(extra render_t0 proposals from refinement, dIoU_OFF-ON) = {prop_corr:+.3f} "
             f"(n = {len(usable)} non-zero-coverage cells). Positive = the more proposals "
             f"refinement adds on the render side, the more it hurts.")
        emit()

    out_json = EXPERIMENTS / "refinement_reference_interaction.json"
    out_json.write_text(json.dumps({
        "cells": {f"N{n}_{s}": cells[(n, s)] for (n, s) in cells},
        "coverage_vs_delta_rows": rows,
        "missing_cells": missing,
        "interaction_stats_per_n": interaction,
        "trend_r_meandelta_vs_n": trend_r,
        "proposal_counts_per_n": prop_totals,
        "proposal_rows": prop_rows,
        "proposal_delta_vs_iou_delta_r": prop_corr,
    }, indent=2, default=str))
    print(f"\nwrote {out_json}")
    (EXPERIMENTS / "refinement_reference_interaction_tables.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {EXPERIMENTS / 'refinement_reference_interaction_tables.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
