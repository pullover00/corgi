#!/usr/bin/env python3
"""Aggregate the chunked 250-pair SceneDiff test-split run into one summary.

The runner writes _experiments/<exp>/summary.json per invocation, so with
chunking each chunk overwrites the last. This reads the authoritative
per-pair files instead -- <pair>/t1_*/<exp>/metrics/metrics.json -- and pools
TP/FP/FN over everything present. Reports two aggregates:
  * all evaluated pairs of the official 250-pair test split;
  * the same minus the diagnostic-subset pairs that were visible during
    development (9 of the 10 are in the test split), i.e. the strictly
    held-out number.
Reads only; recomputes nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "data/scenediff_benchmark"


def pooled(rows: dict) -> dict:
    tp = sum(r.get("tp", 0) or 0 for r in rows.values())
    fp = sum(r.get("fp", 0) or 0 for r in rows.values())
    fn = sum(r.get("fn", 0) or 0 for r in rows.values())
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    ious = [x.get("iou", 0) or 0 for x in rows.values()]
    return {
        "n_evaluated": len(rows), "tp": tp, "fp": fp, "fn": fn,
        "pooled_iou_t1_only": tp / (tp + fp + fn) if tp + fp + fn else None,
        "pooled_precision": p, "pooled_recall": r,
        "pooled_f1": (2 * p * r / (p + r)) if (p is not None and r is not None and p + r) else None,
        "mean_iou": sum(ious) / len(ious) if ious else None,
        "n_zero_iou": sum(1 for v in ious if v == 0),
        "n_empty_gt": sum(1 for x in rows.values() if (x.get("tp", 0) or 0) + (x.get("fn", 0) or 0) == 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=REPO / "results/scenediff_test250")
    ap.add_argument("--experiment", default="scenediff_v10_no_dino_no_refine")
    ap.add_argument("--pair-ids-file", type=Path, default=BENCH / "test_split_250.txt")
    ap.add_argument("--exclude-file", type=Path, default=BENCH / "diagnostic_subset.txt",
                    help="pairs visible during development; reported separately")
    args = ap.parse_args()

    pairs = [l.strip() for l in args.pair_ids_file.read_text().splitlines() if l.strip()]
    seen = {l.strip() for l in args.exclude_file.read_text().splitlines() if l.strip()}
    tree = args.root / "SceneDiff"
    exp_root = tree / "_experiments" / args.experiment

    rows, missing = {}, []
    for pair in pairs:
        hits = sorted(tree.glob(f"{pair}/t1_*/{args.experiment}/metrics/metrics.json"))
        if not hits:
            missing.append(pair); continue
        m = json.loads(hits[0].read_text())
        m["query"] = hits[0].parents[2].name
        rows[pair] = m

    failed = {}
    for f in sorted((exp_root / "logs").glob("chunk_*.summary.json")) if (exp_root / "logs").exists() else []:
        failed.update(json.loads(f.read_text()).get("failed") or {})

    held_out = {k: v for k, v in rows.items() if k not in seen}
    # Pre-registered (2026-09-10, before launch): 46/250 test pairs are
    # removed-only with no in-scope GT at any frame, so their IoU is 0 by
    # construction and they can only add FP. Report the evaluable subset
    # (tp+fn>0, i.e. some GT pixels exist) alongside the full split.
    evaluable = {k: v for k, v in rows.items() if (v.get("tp", 0) or 0) + (v.get("fn", 0) or 0) > 0}
    held_out_evaluable = {k: v for k, v in evaluable.items() if k not in seen}
    out = {
        "experiment": args.experiment, "root": str(args.root),
        "n_pairs_in_split": len(pairs), "n_missing_outputs": len(missing), "missing": missing,
        "failed": failed,
        "all_evaluated": pooled(rows),
        "evaluable_only": pooled(evaluable),
        "held_out_excluding_diagnostic": pooled(held_out),
        "held_out_evaluable_only": pooled(held_out_evaluable),
        "diagnostic_pairs_in_split": sorted(k for k in rows if k in seen),
        "per_pair": rows,
    }
    exp_root.mkdir(parents=True, exist_ok=True)
    (exp_root / "summary_all.json").write_text(json.dumps(out, indent=2, default=str))

    def fmt(x, d=4):
        return "—" if x is None else (f"{x:,}" if isinstance(x, int) else f"{x:.{d}f}")
    lines = [f"# {args.experiment} on the 250-pair SceneDiff test split", "",
             f"outputs present for {len(rows)}/{len(pairs)} pairs; missing {len(missing)}; failed {len(failed)}", "",
             "| aggregate | n | pooled IoU | P | R | F1 | mean IoU | zero-IoU | empty-GT | TP | FP | FN |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, agg in (("all evaluated (official split)", out["all_evaluated"]),
                      ("evaluable only (pairs with in-scope GT)", out["evaluable_only"]),
                      ("held-out (minus diagnostic pairs)", out["held_out_excluding_diagnostic"]),
                      ("held-out, evaluable only", out["held_out_evaluable_only"])):
        lines.append(f"| {name} | {agg['n_evaluated']} | {fmt(agg['pooled_iou_t1_only'])} | {fmt(agg['pooled_precision'])} | "
                     f"{fmt(agg['pooled_recall'])} | {fmt(agg['pooled_f1'])} | {fmt(agg['mean_iou'])} | {agg['n_zero_iou']} | "
                     f"{agg['n_empty_gt']} | {fmt(agg['tp'])} | {fmt(agg['fp'])} | {fmt(agg['fn'])} |")
    lines += ["", "| pair | query | IoU | TP | FP | FN | seen in dev |", "|---|---|---:|---:|---:|---:|---|"]
    for pair in pairs:
        r = rows.get(pair)
        if r is None:
            lines.append(f"| {pair} | — | (no output) | | | | {'yes' if pair in seen else ''} |"); continue
        lines.append(f"| {pair} | {r['query']} | {fmt(r.get('iou'))} | {fmt(r.get('tp'))} | {fmt(r.get('fp'))} | "
                     f"{fmt(r.get('fn'))} | {'yes' if pair in seen else ''} |")
    (exp_root / "summary_all.md").write_text("\n".join(lines) + "\n")
    a = out["all_evaluated"]; h = out["held_out_excluding_diagnostic"]; e = out["evaluable_only"]
    print(f"[{args.experiment}] evaluated {a['n_evaluated']}/{len(pairs)} (missing {len(missing)}, failed {len(failed)})")
    print(f"  all      : pooled IoU={fmt(a['pooled_iou_t1_only'])} P={fmt(a['pooled_precision'])} R={fmt(a['pooled_recall'])} "
          f"F1={fmt(a['pooled_f1'])} meanIoU={fmt(a['mean_iou'])}")
    print(f"  evaluable: pooled IoU={fmt(e['pooled_iou_t1_only'])} P={fmt(e['pooled_precision'])} R={fmt(e['pooled_recall'])} "
          f"F1={fmt(e['pooled_f1'])} meanIoU={fmt(e['mean_iou'])} (n={e['n_evaluated']})")
    print(f"  held-out : pooled IoU={fmt(h['pooled_iou_t1_only'])} P={fmt(h['pooled_precision'])} R={fmt(h['pooled_recall'])} "
          f"F1={fmt(h['pooled_f1'])} meanIoU={fmt(h['mean_iou'])} (n={h['n_evaluated']})")
    print(f"wrote {exp_root/'summary_all.json'} and summary_all.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
