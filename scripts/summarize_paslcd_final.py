#!/usr/bin/env python3
"""Summarize a finished (or partial) run_paslcd_benchmark.py output root into
one report: query counts, success/failure, mean-of-scene-means and pooled
metrics, summed TP/FP/FN pixels, decision-count statistics, per-scene and
per-query tables, runtime from the run logs, and failure reasons.

Reads only what the benchmark wrote (metrics.csv, paslcd_result.json,
run_logs/*.log, provenance/); does not recompute predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASETS = ["Cantina", "Garden", "Lounge", "Lunch_room", "Meeting_room", "Playground", "Porch", "Pots", "Printing_area", "Zen"]
INSTANCES = ["Instance_1", "Instance_2"]


def expected_queries(data_root: Path) -> dict[str, list[str]]:
    out = {}
    for d in DATASETS:
        for i in INSTANCES:
            gt = data_root / d / i / "gt_mask"
            if gt.exists():
                out[f"{d}_{i}"] = sorted(p.stem for p in gt.glob("*.png"))
    return out


def load_rows(root: Path) -> dict[str, list[dict]]:
    rows = defaultdict(list)
    for metrics in sorted(root.glob("*_Instance_*/metrics.csv")):
        with metrics.open() as handle:
            for row in csv.DictReader(handle):
                rows[metrics.parent.name].append(row)
    return rows


def decision_stats(root: Path) -> tuple[Counter, int]:
    totals: Counter = Counter()
    n = 0
    for result in root.glob("*_Instance_*/intermediate/*/paslcd_result.json"):
        counts = json.loads(result.read_text()).get("decision_counts") or {}
        totals.update({k: int(v) for k, v in counts.items()})
        n += 1
    return totals, n


def runtime(root: Path) -> tuple[str | None, str | None, float | None]:
    starts, ends = [], []
    for log in sorted((root / "run_logs").glob("*.log")) + sorted(root.glob("launcher*.log")):
        for line in log.read_text(errors="replace").splitlines():
            m = re.search(r"=== (FINAL_START|PASS \d+ start) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line)
            if m:
                starts.append(m.group(2))
            m = re.search(r"=== (FINAL_END|PASS \d+ exit=\d+) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line)
            if m:
                ends.append(m.group(2))
    if not starts:
        return None, None, None
    t0 = min(datetime.fromisoformat(s) for s in starts)
    t1 = max(datetime.fromisoformat(e) for e in ends) if ends else None
    hours = (t1 - t0).total_seconds() / 3600 if t1 else None
    return t0.isoformat(sep=" "), t1.isoformat(sep=" ") if t1 else None, hours


def failure_reasons(root: Path, missing: dict[str, list[str]]) -> dict[str, str]:
    reasons = {}
    text = "\n".join(p.read_text(errors="replace") for p in sorted((root / "run_logs").glob("*.log")))
    for scene, stems in missing.items():
        if not stems:
            continue
        dataset, inst = scene.rsplit("_Instance_", 1)
        tag = f"[{dataset}/Instance_{inst}]"
        block = "\n".join(line for line in text.splitlines() if tag in line or "Error" in line or "Traceback" in line or "Killed" in line)
        errs = re.findall(r"^(\w*(?:Error|Exception)[^\n]*)$", block, flags=re.M)
        reasons[scene] = errs[-1] if errs else "no traceback found in run logs (query never reached evaluation)"
    return reasons


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=REPO / "results/paslcd_final_v10_no_dino")
    ap.add_argument("--data-root", type=Path, default=REPO / "data/PASLCD")
    ap.add_argument("--out", type=Path, default=None, help="markdown report path (default <root>/PASLCD_FINAL_V10_NO_DINO.md)")
    args = ap.parse_args()
    root = args.root
    out = args.out or root / "PASLCD_FINAL_V10_NO_DINO.md"

    expected = expected_queries(args.data_root)
    rows_by_scene = load_rows(root)
    all_rows = [r for rows in rows_by_scene.values() for r in rows]
    done = {scene: {r["test_image"] for r in rows} for scene, rows in rows_by_scene.items()}
    missing = {scene: [s for s in stems if s not in done.get(scene, set())] for scene, stems in expected.items()}
    n_expected = sum(len(v) for v in expected.values())
    n_done = len(all_rows)
    n_missing = sum(len(v) for v in missing.values())

    def mean(rows, field):
        return sum(float(r[field]) for r in rows) / len(rows) if rows else float("nan")

    scene_table = []
    for scene in sorted(expected):
        rows = rows_by_scene.get(scene, [])
        scene_table.append({
            "scene": scene, "n": len(rows), "expected": len(expected[scene]),
            "iou": mean(rows, "iou"), "f1": mean(rows, "f1"), "precision": mean(rows, "precision"), "recall": mean(rows, "recall"),
            "tp": sum(int(r["tp"]) for r in rows), "fp": sum(int(r["fp"]) for r in rows), "fn": sum(int(r["fn"]) for r in rows),
        })
    scenes_with_rows = [s for s in scene_table if s["n"]]
    mean_of_scenes = {k: (sum(s[k] for s in scenes_with_rows) / len(scenes_with_rows) if scenes_with_rows else float("nan")) for k in ("iou", "f1", "precision", "recall")}
    pooled = {k: mean(all_rows, k) for k in ("iou", "f1", "precision", "recall")}
    TP = sum(int(r["tp"]) for r in all_rows); FP = sum(int(r["fp"]) for r in all_rows); FN = sum(int(r["fn"]) for r in all_rows)
    pixel_level = {
        "iou": TP / (TP + FP + FN) if TP + FP + FN else float("nan"),
        "precision": TP / (TP + FP) if TP + FP else float("nan"),
        "recall": TP / (TP + FN) if TP + FN else float("nan"),
    }
    pixel_level["f1"] = 2 * TP / (2 * TP + FP + FN) if 2 * TP + FP + FN else float("nan")
    decisions, n_decision_files = decision_stats(root)
    t0, t1, hours = runtime(root)
    reasons = failure_reasons(root, missing)
    residuals = [float(r["alignment_residual"]) for r in all_rows if r.get("alignment_residual") not in (None, "", "nan")]

    prov = root / "provenance"
    git_head = (prov / "git_head.txt").read_text().strip() if (prov / "git_head.txt").exists() else "n/a"
    machine = (prov / "machine.txt").read_text().strip() if (prov / "machine.txt").exists() else "n/a"
    command = (prov / "command.txt").read_text().strip() if (prov / "command.txt").exists() else "n/a"

    lines = []
    lines += [f"# PASLCD final evaluation — v10_no_dino", ""]
    lines += [f"Output root: `{root}`  ", f"Config: `provenance/config_used.yaml` (copy of `configs/ablate_v10_no_dino.yaml`)  ",
              f"Git HEAD: `{git_head}` + uncommitted changes in `provenance/git_diff_uncommitted.patch` (code hashes in `provenance/code_sha256.txt`)  ",
              f"Command: `{command}`", ""]
    lines += ["## Protocol", "",
              "- `scripts/run_paslcd_benchmark.py` defaults: all 10 datasets × 2 instances × 25 queries = 500, **full reference set per instance** (no `--max-reference-images`), no `--limit`.",
              "- Multi-view T0 reference reconstruction (VGGT-Omega) built once per instance and saved to `reference_scenes/`; exactly one T1 query per inference, localised by a joint VGGT-Omega call over references + query.",
              "- DI²FIX render refinement **enabled** (`refine.enabled: true`, no `--skip-refine`); detection runs on `refined/render_t0.png` and `refined/clean_render.png`.",
              "- Detection: SAM3 proposals + SAM3 features, SAM2 bidirectional tracking, geometric identity, visibility filter, recall recovery, colour replacement, SAM3 text-prompt ceiling/sky suppression, depth-ordered occlusion merge. `use_dino_features: false` (DINOv2 neither computed nor consulted). No conservative state resolver.",
              "- Metric: PASLCD binary changed/unchanged, GT thresholded at 127, prediction resized to GT resolution with nearest-neighbour, per-query IoU/F1/precision/recall from `ocmask_pipeline.metrics.compute_binary_metrics`.", ""]
    lines += ["## Totals", "", f"| | |", "|---|---|",
              f"| PASLCD queries (expected) | {n_expected} |", f"| evaluated successfully | {n_done} |", f"| failed / missing | {n_missing} |",
              f"| scene instances with all queries | {sum(1 for s in scene_table if s['n'] == s['expected'])}/{len(scene_table)} |",
              f"| run start | {t0 or 'n/a'} |", f"| run end | {t1 or 'n/a (in progress)'} |", f"| wall time | {f'{hours:.1f} h' if hours is not None else 'n/a'} |",
              f"| alignment residual (median / max) | {f'{sorted(residuals)[len(residuals)//2]:.5f} / {max(residuals):.5f}' if residuals else 'n/a'} |", ""]
    lines += ["## Headline metrics", "", "| aggregation | mIoU | F1 | precision | recall |", "|---|---|---|---|---|",
              f"| mean of scene means (benchmark_summary convention) | {fmt(mean_of_scenes['iou'])} | {fmt(mean_of_scenes['f1'])} | {fmt(mean_of_scenes['precision'])} | {fmt(mean_of_scenes['recall'])} |",
              f"| pooled mean over queries | {fmt(pooled['iou'])} | {fmt(pooled['f1'])} | {fmt(pooled['precision'])} | {fmt(pooled['recall'])} |",
              f"| pixel-level (summed TP/FP/FN) | {fmt(pixel_level['iou'])} | {fmt(pixel_level['f1'])} | {fmt(pixel_level['precision'])} | {fmt(pixel_level['recall'])} |", "",
              f"Summed pixels at GT resolution: TP = {TP:,}, FP = {FP:,}, FN = {FN:,}.", ""]
    if decisions:
        total_changes = sum(v for k, v in decisions.items() if k != "unchanged")
        lines += ["## Prediction statistics (object decisions, summed over queries)", "", "| class | decisions | per query | share of change decisions |", "|---|---|---|---|"]
        for k in ("added", "removed", "moved", "replaced", "unchanged"):
            v = decisions.get(k, 0)
            share = f"{100 * v / total_changes:.1f}%" if k != "unchanged" and total_changes else "—"
            lines.append(f"| {k} | {v:,} | {v / n_decision_files:.2f} | {share} |")
        lines += ["", f"From {n_decision_files} `paslcd_result.json` files. `unchanged` counts matched-and-unchanged object pairs, not pixels.", ""]
    lines += ["## Per-scene", "", "| scene | n | mIoU | F1 | precision | recall | TP | FP | FN |", "|---|---|---|---|---|---|---|---|---|"]
    for s in scene_table:
        if s["n"]:
            lines.append(f"| {s['scene']} | {s['n']}/{s['expected']} | {fmt(s['iou'])} | {fmt(s['f1'])} | {fmt(s['precision'])} | {fmt(s['recall'])} | {s['tp']:,} | {s['fp']:,} | {s['fn']:,} |")
        else:
            lines.append(f"| {s['scene']} | 0/{s['expected']} | — | — | — | — | — | — | — |")
    lines += ["", "## Failures", ""]
    if n_missing == 0:
        lines.append("None — every expected query produced a prediction and a metrics row.")
    else:
        for scene, stems in missing.items():
            if stems:
                lines.append(f"- **{scene}**: {len(stems)} missing ({', '.join(stems[:5])}{'…' if len(stems) > 5 else ''}) — {reasons.get(scene, 'unknown')}")
    lines += ["", "## Per-query", "", "| scene | query | IoU | F1 | precision | recall | TP | FP | FN | residual |", "|---|---|---|---|---|---|---|---|---|---|"]
    for scene in sorted(rows_by_scene):
        for r in sorted(rows_by_scene[scene], key=lambda r: r["test_image"]):
            res = r.get("alignment_residual", "")
            res = f"{float(res):.5f}" if res not in ("", None, "nan") else "cached"
            lines.append(f"| {scene} | {r['test_image']} | {float(r['iou']):.4f} | {float(r['f1']):.4f} | {float(r['precision']):.4f} | {float(r['recall']):.4f} | {int(r['tp']):,} | {int(r['fp']):,} | {int(r['fn']):,} | {res} |")
    lines += ["", "## Machine", "", "```", machine, "```", ""]
    out.write_text("\n".join(lines))
    (root / "final_summary.json").write_text(json.dumps({
        "n_expected": n_expected, "n_done": n_done, "n_missing": n_missing, "mean_of_scene_means": mean_of_scenes, "pooled": pooled,
        "pixel_level": pixel_level, "TP": TP, "FP": FP, "FN": FN, "decisions": dict(decisions), "run_start": t0, "run_end": t1, "hours": hours,
        "per_scene": scene_table, "missing": {k: v for k, v in missing.items() if v}, "failure_reasons": reasons,
    }, indent=2))
    print(f"wrote {out}"); print(f"{n_done}/{n_expected} queries; mean-of-scenes mIoU={mean_of_scenes['iou']:.4f} pooled mIoU={pooled['iou']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
