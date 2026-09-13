#!/usr/bin/env python3
"""How much does CORGI's single-query protocol cost, and is its query frame flattering?

Consumes two official-evaluator runs over the SAME 25 pairs:
  --single  rank-1 only  (the protocol the paper reports)
  --multi   ranks 1-8    (the same method, eight query frames per pair)
and the top-8 manifest (rank + co-visibility per query frame).

Reports
  1. pooled pixel IoU single vs multi -- the headline handicap number;
  2. the per-rank curve, from official_eval_summary.json's per_scene.per_frame
     (tp/fp/fn per evaluated frame, keyed by the evaluator's resampled index),
     joined back to rank and co-visibility through the manifest. This is the
     honest answer to "did you report your easiest frame?": if IoU is flat in
     rank, the single-query choice is not cherry-picking; if it falls, the
     paper number is optimistic by exactly that much;
  3. the rank-1-vs-rest split inside the multi-query run, which isolates (2)
     from any pooling artefact;
  4. GT coverage: how much of each scene's annotated change a single frame can
     see at all -- the part of the gap no detector can close.
Writes a markdown block ready to paste into the paper's protocol section.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pooled(rows):
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
    d = tp + fp + fn
    return {"iou": tp / d if d else None, "p": tp / (tp + fp) if tp + fp else None,
            "r": tp / (tp + fn) if tp + fn else None, "tp": tp, "fp": fp, "fn": fn, "n": len(rows)}


def fmt(d):
    f = lambda x: "—" if x is None else f"{x:.4f}"
    return f"IoU={f(d['iou'])} P={f(d['p'])} R={f(d['r'])} (frames={d['n']})"


def frame_rows(summary, rank_of, subset_of):
    """One row per evaluated frame, tagged with its rank/covisibility/subset."""
    rows = []
    for scene, s in summary["per_scene"].items():
        for fi, m in s.get("per_frame", {}).items():
            key = (scene, int(fi))
            meta = rank_of.get(key)
            rows.append({"scene": scene, "resampled_idx": int(fi), **m,
                         "rank": meta["rank"] if meta else None,
                         "covis": meta["covisibility"] if meta else None,
                         "subset": subset_of.get(scene)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--single", type=Path, required=True, help="official_eval_summary.json for the rank-1 run")
    ap.add_argument("--multi", type=Path, required=True, help="official_eval_summary.json for the ranks 1-8 run")
    ap.add_argument("--manifest", type=Path, required=True, help="manifest_multiquery_top8.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text())
    rank_of, subset_of, covis_by_rank = {}, {}, {}
    for q in man["queries"]:
        rank_of[(q["pair"], q["t1_annotation_idx"] // 30)] = q
        subset_of[q["pair"]] = q["subset"]
        covis_by_rank.setdefault(q["rank"], []).append(q["covisibility"])

    S = json.loads(args.single.read_text()); M = json.loads(args.multi.read_text())
    srows, mrows = frame_rows(S, rank_of, subset_of), frame_rows(M, rank_of, subset_of)
    sp, mp = pooled(srows), pooled(mrows)

    L = []
    W = L.append
    W("## Single-query vs multi-query on the same 25 pairs (official SceneDiff evaluator)\n")
    W(f"Same method, same config, same whitelist -- the only variable is which frame is queried.")
    W(f"Subset: {sum(1 for v in subset_of.values() if v == 'SD-V')} SD-V + "
      f"{sum(1 for v in subset_of.values() if v == 'SD-K')} SD-K pairs, drawn before any per-pair score was read.\n")
    W("| protocol | frames scored | pooled IoU | precision | recall |")
    W("|---|---|---|---|---|")
    for name, d in (("single-query (rank 1)", sp), ("multi-query (ranks 1-8)", mp)):
        g = lambda x: "—" if x is None else f"{x:.4f}"
        W(f"| {name} | {d['n']} | **{g(d['iou'])}** | {g(d['p'])} | {g(d['r'])} |")
    if sp["iou"] and mp["iou"]:
        delta = mp["iou"] - sp["iou"]
        W(f"\nHandicap: **{delta:+.4f} IoU** ({delta / sp['iou'] * 100:+.1f}% relative) when the same method "
          f"is asked for eight frames per scene instead of one.\n")

    W("### Per-rank curve (inside the multi-query run)\n")
    W("Rank 1 is the most co-visible annotated frame with the reconstruction; rank 8 the least of the top eight.\n")
    W("| rank | frames | median co-visibility | pooled IoU | precision | recall |")
    W("|---|---|---|---|---|---|")
    by_rank = {}
    for r in mrows:
        by_rank.setdefault(r["rank"], []).append(r)
    for k in sorted(x for x in by_rank if x is not None):
        d = pooled(by_rank[k]); cv = sorted(covis_by_rank.get(k, []))
        med = f"{cv[len(cv) // 2]:.3f}" if cv else "—"
        g = lambda x: "—" if x is None else f"{x:.4f}"
        W(f"| {k} | {d['n']} | {med} | {g(d['iou'])} | {g(d['p'])} | {g(d['r'])} |")
    r1 = pooled(by_rank.get(1, [])); rest = pooled([r for k, v in by_rank.items() if k and k > 1 for r in v])
    if r1["iou"] is not None and rest["iou"] is not None:
        W(f"\nrank 1 alone {fmt(r1)}\nranks 2-8   {fmt(rest)}\n"
          f"=> the frame the single-query protocol picks scores **{r1['iou'] - rest['iou']:+.4f} IoU** "
          f"above the frames it does not pick.\n")

    W("### By subset\n")
    W("| subset | single-query IoU | multi-query IoU | delta |")
    W("|---|---|---|---|")
    for sub in ("SD-V", "SD-K"):
        a = pooled([r for r in srows if r["subset"] == sub]); b = pooled([r for r in mrows if r["subset"] == sub])
        g = lambda x: "—" if x is None else f"{x:.4f}"
        d = f"{b['iou'] - a['iou']:+.4f}" if a["iou"] is not None and b["iou"] is not None else "—"
        W(f"| {sub} | {g(a['iou'])} | {g(b['iou'])} | {d} |")

    W("\n### Ground-truth reachable from one frame\n")
    sg = sum(r["tp"] + r["fn"] for r in srows); mg = sum(r["tp"] + r["fn"] for r in mrows)
    W(f"Annotated change pixels visible to the scored frames: single-query {sg:,.0f}, multi-query {mg:,.0f} "
      f"({sg / mg * 100:.1f}% of the eight-frame total if the protocol scores one frame).")
    W("Removed objects are annotated only in the before video and are outside both numbers; that restriction is "
      "documented separately and is a property of the protocol, not of the detector.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    json.dump({"single": sp, "multi": mp, "by_rank": {str(k): pooled(v) for k, v in by_rank.items() if k}},
              open(args.out.with_suffix(".json"), "w"), indent=1)
    print("\n".join(L))
    print(f"\nwrote {args.out} and {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
