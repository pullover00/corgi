#!/usr/bin/env python3
"""Diagnostic comparison of the PASLCD suppression-ablation variants
(results/paslcd_suppression_ablation/v*), beyond aggregate mIoU/F1.

Per variant:
  A. overall IoU / F1 / precision / recall / changed-pixel fraction
  B. per predicted class (ADDED, REMOVED, MOVED, REPLACED, WARPED): TP, FP,
     precision, share of all FP pixels, share of GT pixels it recovers.
     GT is binary, so FN cannot be attributed to a predicted class -- it is
     reported once, overall.
  C. stage-wise counters averaged over queries, from each inference.json:
     proposals per frame, decisions by kind, confirmed identities by
     evidence tag (direct / clean-bridge / tracking-recovery / geometric),
     visibility-filter rejections, horizon suppressions, corroboration
     rejections, tracking recoveries -- plus FP attribution to render holes
     and to the top-of-image band (ceiling/sky proxy).

Writes analysis.json and analysis.md next to the variant directories. CPU
only; safe to run while the GPU is busy.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ocmask_pipeline.metrics import load_paslcd_gt  # noqa: E402

ROOT = REPO / "results/paslcd_suppression_ablation"
CACHE = REPO / "results/paslcd_ablation_cache"
DATA = REPO / "data/PASLCD"
CLASSES = {1: "ADDED", 2: "REMOVED", 3: "MOVED", 4: "WARPED", 5: "REPLACED"}
TOP_BAND = 0.20


def analyze_variant(vdir: Path) -> dict | None:
    eval_path = vdir / "eval.json"
    if not eval_path.exists():
        return None
    ev = json.loads(eval_path.read_text())
    px = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0, "total": 0, "fp_hole": 0, "fp_top": 0, "gt_top": 0}
    cls = {c: {"tp": 0, "fp": 0} for c in CLASSES}
    stage = defaultdict(float)
    evidence = Counter()
    kinds = Counter()
    n = 0
    for key in ev["per_query"]:
        scene, stem = key.split("/")
        ds, inst = scene.rsplit("_Instance_", 1)
        lab = cv2.imread(str(vdir / scene / stem / "detect" / "labels.png"), cv2.IMREAD_GRAYSCALE)
        info_path = vdir / scene / stem / "detect" / "inference.json"
        if lab is None or not info_path.exists():
            continue
        gt = load_paslcd_gt(DATA / ds / f"Instance_{inst}" / "gt_mask" / f"{stem}.png")
        h, w = lab.shape
        gtg = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        pred = lab != 0
        fp = pred & ~gtg
        px["tp"] += int((pred & gtg).sum()); px["fp"] += int(fp.sum()); px["fn"] += int((~pred & gtg).sum())
        px["gt"] += int(gtg.sum()); px["pred"] += int(pred.sum()); px["total"] += h * w
        top = np.zeros((h, w), bool); top[: int(h * TOP_BAND)] = True
        px["fp_top"] += int((fp & top).sum()); px["gt_top"] += int((gtg & top).sum())
        cov_path = CACHE / scene / "intermediate" / stem / "reconstruction" / "render_t0_coverage.npy"
        if cov_path.exists():
            cov = np.load(cov_path)
            if cov.shape != (h, w):
                cov = cv2.resize(cov.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            px["fp_hole"] += int((fp & ~(cov > 0)).sum())
        for c in CLASSES:
            m = lab == c
            cls[c]["tp"] += int((m & gtg).sum()); cls[c]["fp"] += int((m & ~gtg).sum())

        info = json.loads(info_path.read_text())
        n += 1
        oc = info.get("object_counts") or {}
        for frame, key in (("t0", "render_t0"), ("clean", "clean_render"), ("t1", "image_t1")):
            stage[f"proposals_{frame}"] += float(oc.get(key, 0) or 0)
        for name, secs in (info.get("timings") or {}).items():
            stage[f"sec_{name}"] += float(secs or 0)
        stage["visibility_filter_rejected"] += float(info.get("visibility_filter_rejected", 0) or 0)
        stage["horizon_suppressed"] += float(info.get("horizon_suppressed", 0) or 0)
        stage["corroboration_rejected"] += float(info.get("corroboration_rejected", 0) or 0)
        stage["tracking_recoveries"] += float((info.get("tracking_recovery") or {}).get("tracking_recoveries", 0) or 0)
        for d in info.get("decisions") or []:
            kinds[d.get("decision", "?")] += 1
            if d.get("decision") in ("unchanged", "moved"):
                evidence[d.get("evidence") or "none"] += 1

    if n == 0:
        return None
    # stability / catastrophic-FP view: per-query precision and per-scene IoU
    per_query = ev["per_query"]
    catastrophic = sorted((k for k, m in per_query.items() if m["fp"] > 50_000 and m["precision"] < 0.10),
                          key=lambda k: -per_query[k]["fp"])
    scene_iou = {}
    for k, m in per_query.items():
        scene_iou.setdefault(k.split("/")[0], []).append(m["iou"])
    scene_iou = {s: sum(v) / len(v) for s, v in scene_iou.items()}
    total_fp = max(px["fp"], 1)
    per_class = {}
    for c, name in CLASSES.items():
        tp, fp = cls[c]["tp"], cls[c]["fp"]
        per_class[name] = {"tp": tp, "fp": fp,
                           "precision": tp / (tp + fp) if (tp + fp) else None,
                           "share_of_all_fp": fp / total_fp,
                           "share_of_gt_recovered": tp / max(px["gt"], 1)}
    return {
        "n_queries": n, "mIoU": ev["iou"], "F1": ev["f1"], "precision": ev["precision"], "recall": ev["recall"],
        "pooled_iou": px["tp"] / max(px["tp"] + px["fp"] + px["fn"], 1),
        "changed_pixel_fraction": px["pred"] / max(px["total"], 1),
        "fn_pixels": px["fn"], "fp_pixels": px["fp"],
        "fp_share_in_render_holes": px["fp_hole"] / total_fp,
        "fp_share_in_top_band": px["fp_top"] / total_fp,
        "gt_share_in_top_band": px["gt_top"] / max(px["gt"], 1),
        "per_class": per_class,
        "per_scene_iou": scene_iou,
        "scene_iou_std": float(np.std(list(scene_iou.values()))),
        "n_zero_iou_queries": sum(1 for m in per_query.values() if m["iou"] == 0.0),
        "catastrophic_fp_queries": catastrophic,
        "stage_means": {k: v / n for k, v in stage.items()},
        "decisions_by_kind_mean": {k: v / n for k, v in kinds.items()},
        "confirmed_identity_by_evidence_mean": {k: v / n for k, v in evidence.items()},
    }


def main() -> int:
    out = {}
    for vdir in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("v")):
        res = analyze_variant(vdir)
        if res:
            out[vdir.name] = res
            print(f"analyzed {vdir.name}: n={res['n_queries']}")
    (ROOT / "analysis.json").write_text(json.dumps(out, indent=2))

    lines = ["# PASLCD variant diagnostic analysis\n",
             "## A. Overall\n",
             "| variant | n | mIoU | F1 | prec | recall | pooled IoU | changed-px frac | FP px | FN px |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for v, r in out.items():
        lines.append(f"| {v} | {r['n_queries']} | {r['mIoU']:.4f} | {r['F1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
                     f"{r['pooled_iou']:.4f} | {r['changed_pixel_fraction']:.4f} | {r['fp_pixels']:,} | {r['fn_pixels']:,} |")
    lines += ["\n## B. Per predicted class (precision, share of all FP pixels, share of GT pixels recovered)\n"]
    for v, r in out.items():
        lines.append(f"\n**{v}**\n\n| class | TP px | FP px | precision | share of FP | GT recovered |\n|---|---|---|---|---|---|")
        for name, c in r["per_class"].items():
            p = "–" if c["precision"] is None else f"{c['precision']:.3f}"
            lines.append(f"| {name} | {c['tp']:,} | {c['fp']:,} | {p} | {100*c['share_of_all_fp']:.1f}% | {100*c['share_of_gt_recovered']:.1f}% |")
    lines += ["\n## C. Stage-wise (means per query)\n"]
    keys = sorted({k for r in out.values() for k in r["stage_means"]})
    lines.append("| variant | " + " | ".join(keys) + " | FP in holes | FP in top band | GT in top band |")
    lines.append("|---|" + "---|" * (len(keys) + 3))
    for v, r in out.items():
        lines.append(f"| {v} | " + " | ".join(f"{r['stage_means'].get(k, 0):.1f}" for k in keys)
                     + f" | {100*r['fp_share_in_render_holes']:.1f}% | {100*r['fp_share_in_top_band']:.1f}% | {100*r['gt_share_in_top_band']:.1f}% |")
    lines += ["\n### Decisions by kind (mean per query)\n"]
    for v, r in out.items():
        lines.append(f"- **{v}**: " + ", ".join(f"{k} {x:.1f}" for k, x in sorted(r["decisions_by_kind_mean"].items())))
    lines += ["\n### Confirmed identities by evidence (mean per query)\n"]
    for v, r in out.items():
        lines.append(f"- **{v}**: " + ", ".join(f"{k} {x:.1f}" for k, x in sorted(r["confirmed_identity_by_evidence_mean"].items())))

    # D. deltas against v0, per class, so "which error class changed" is read off directly
    base = out.get("v0_baseline")
    if base:
        lines += ["\n## D. Change vs v0_baseline, by class (FP px and TP px deltas; negative FP = fewer false positives)\n",
                  "| variant | ΔmIoU | Δprec | Δrecall | ΔFP ADDED | ΔFP REMOVED | ΔFP REPLACED | ΔTP ADDED | ΔTP REMOVED | ΔTP REPLACED |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for v, r in out.items():
            if v == "v0_baseline":
                continue
            d = lambda cls, k: r["per_class"][cls][k] - base["per_class"][cls][k]  # noqa: E731
            lines.append(f"| {v} | {r['mIoU']-base['mIoU']:+.4f} | {r['precision']-base['precision']:+.4f} | {r['recall']-base['recall']:+.4f} | "
                         f"{d('ADDED','fp'):+,} | {d('REMOVED','fp'):+,} | {d('REPLACED','fp'):+,} | "
                         f"{d('ADDED','tp'):+,} | {d('REMOVED','tp'):+,} | {d('REPLACED','tp'):+,} |")
    lines += ["\n## E. Stability across scenes\n",
              "| variant | scene-IoU std | zero-IoU queries | catastrophic-FP queries (FP>50k px & prec<0.10) |", "|---|---|---|---|"]
    for v, r in out.items():
        lines.append(f"| {v} | {r['scene_iou_std']:.4f} | {r['n_zero_iou_queries']} | {len(r['catastrophic_fp_queries'])}: "
                     + ", ".join(k.split('/')[0][:14] for k in r["catastrophic_fp_queries"][:6]) + " |")
    scenes = sorted({s for r in out.values() for s in r["per_scene_iou"]})
    lines += ["\n### Per-scene mean IoU\n", "| scene | " + " | ".join(out) + " |", "|---|" + "---|" * len(out)]
    for s in scenes:
        lines.append(f"| {s} | " + " | ".join(f"{r['per_scene_iou'].get(s, float('nan')):.3f}" for r in out.values()) + " |")
    (ROOT / "analysis.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {ROOT / 'analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
