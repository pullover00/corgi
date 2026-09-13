#!/usr/bin/env python3
"""MOVED-state resolver probe: enable_location_mismatch_state_resolver vs the final arm.

Answers the four questions the PASLCD session asked, on SceneDiff P1 (250 test queries),
with the diagnostic evaluator (internal comparison against our own arm, same evaluator
both sides).

IMPORTANT mechanism note, verified in change_detection.py before running:
  reject_low_confidence_moved does NOT discard the pair's objects. record_rejected()
  only appends a decision record; neither object enters consumed_t0/consumed_t1, so the
  t0 instance falls through to REMOVED and the t1 instance to ADDED (the comment at
  :1595 says so explicitly). resolve_low_location_identity() has two outcomes and BOTH
  consume the pair: geometry-resolvable + not-same-location -> one MOVED object replacing
  that ADDED+REMOVED pair; otherwise -> consumed with NO output object at all
  ("ambiguous_location_suppressed"). So the variant is mostly an OUTPUT-SUPPRESSING
  change, not an output-adding one -- which is why the empty-GT partition matters.
"""
from __future__ import annotations
import glob, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/tessa/change_pipeline/results/scenediff_single_query_covis_v1")
FINAL = "scenediff_single_query_covis_v1_gate_refine"
VAR = "scenediff_single_query_covis_v1_gate_refine_movedresolver"


def load(exp):
    d = {}
    for f in sorted(glob.glob(str(ROOT / "SceneDiff/_experiments" / exp / "logs/chunk_*.summary.json"))):
        d.update(json.load(open(f))["per_pair"])
    return d


def empty(m):
    g = m["_gt_pixel_counts"]; return (g["added"] + g["moved_bucket"]) == 0


def pooled(ms):
    tp = sum(m["tp"] for m in ms); fp = sum(m["fp"] for m in ms); fn = sum(m["fn"] for m in ms)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(ms), "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0, "p": p, "r": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def fmt(d):
    return f"IoU={d['iou']:.4f} P={d['p']:.4f} R={d['r']:.4f} F1={d['f1']:.4f} (n={d['n']})"


def decisions_for(exp, man):
    """Raw decision records; n_decisions_by_kind has a fixed key list that omits these."""
    kinds, resolver = Counter(), Counter()
    n_files = 0
    for pair, q in man.items():
        f = ROOT / "SceneDiff" / pair / f"t1_{q:04d}" / exp / "labels" / "inference.json"
        if not f.exists():
            continue
        n_files += 1
        for d in json.load(open(f)).get("decisions", []):
            kinds[d.get("decision", "?")] += 1
            if d.get("state_resolver"):
                resolver[d["state_resolver"]] += 1
    return kinds, resolver, n_files


def main():
    man = {r["pair"]: r["t1_annotation_idx"] for r in json.load(open(ROOT / "manifest_test_top1.json"))["queries"]}
    sub = {r["pair"]: r["subset"] for r in json.load(open(ROOT / "manifest_test_top1.json"))["queries"]}
    A, B = load(FINAL), load(VAR)
    common = sorted(set(A) & set(B))
    print(f"final arm {len(A)} pairs | variant {len(B)} pairs | paired on {len(common)}\n")
    if len(B) < 250:
        print(f"NOTE: variant incomplete ({len(B)}/250) -- numbers below are provisional\n")

    print("=" * 78)
    print("1. POOLED METRICS  (final arm -> resolver variant)")
    print("=" * 78)
    for title, keys in (("ALL", common), ("SD-V", [k for k in common if sub[k] == "SD-V"]),
                        ("SD-K", [k for k in common if sub[k] == "SD-K"])):
        a_all, b_all = pooled([A[k] for k in keys]), pooled([B[k] for k in keys])
        ne = [k for k in keys if not empty(A[k])]
        a_ne, b_ne = pooled([A[k] for k in ne]), pooled([B[k] for k in ne])
        print(f"\n  {title}")
        print(f"    all       final {fmt(a_all)}")
        print(f"              var   {fmt(b_all)}   dIoU={b_all['iou']-a_all['iou']:+.4f} dF1={b_all['f1']-a_all['f1']:+.4f}")
        print(f"    non-empty final {fmt(a_ne)}")
        print(f"              var   {fmt(b_ne)}   dIoU={b_ne['iou']-a_ne['iou']:+.4f} dF1={b_ne['f1']-a_ne['f1']:+.4f}")

    print("\n" + "=" * 78)
    print("2. MOVED-BUCKET SPECIFICS")
    print("=" * 78)
    for name, D in (("final", A), ("variant", B)):
        mv_px = sum(D[k].get("MOVED", {}).get("pixels", 0) for k in common)
        on_mb = sum(D[k].get("MOVED", {}).get("on_gt_moved_bucket", 0) for k in common)
        on_ad = sum(D[k].get("MOVED", {}).get("on_gt_added", 0) for k in common)
        on_bg = sum(D[k].get("MOVED", {}).get("on_gt_background", 0) for k in common)
        gm = sum(D[k]["_gt_pixel_counts"]["moved_bucket"] for k in common)
        prec = (on_mb + on_ad) / mv_px if mv_px else None
        # pooled moved-bucket recall, weighted by each pair's moved-bucket GT (all predictions, any class)
        rec = sum((D[k]["_recall"].get("moved_bucket_recall") or 0) * D[k]["_gt_pixel_counts"]["moved_bucket"]
                  for k in common) / gm if gm else None
        print(f"\n  {name}:")
        print(f"    MOVED predicted px {mv_px:,}" + ("" if mv_px else "   (the class never fires)"))
        if mv_px:
            print(f"      on GT moved-bucket {on_mb:,} ({on_mb/mv_px*100:.1f}%) | on GT added {on_ad:,} "
                  f"({on_ad/mv_px*100:.1f}%) | on GT background {on_bg:,} ({on_bg/mv_px*100:.1f}%)")
            print(f"      MOVED-class pixel precision (lands on ANY changed GT): {prec:.4f}")
        print(f"    moved-bucket GT px {gm:,}; pooled moved-bucket recall (all classes) {rec:.4f}")

    print("\n" + "=" * 78)
    print("3. DECISION ACCOUNTING")
    print("=" * 78)
    ka, ra, na = decisions_for(FINAL, man)
    kb, rb, nb = decisions_for(VAR, {p: man[p] for p in common})
    print(f"\n  final arm ({na} inference.json read):")
    for k, v in ka.most_common(8):
        print(f"    {k:34s} {v}")
    print(f"\n  variant ({nb} read):")
    for k, v in kb.most_common(8):
        print(f"    {k:34s} {v}")
    print(f"\n  state_resolver outcomes in the variant: {dict(rb) or '(none)'}")
    rej = ka.get("location_mismatch_rejected", 0)
    promoted = rb.get("geometry_supported_moved", 0)
    suppressed = rb.get("ambiguous_location_suppressed", 0)
    print(f"\n  final arm rejected pairs           : {rej}")
    print(f"  variant promoted to MOVED          : {promoted}" +
          (f"  ({promoted/rej*100:.1f}% of them)" if rej else ""))
    print(f"  variant suppressed as UNKNOWN      : {suppressed}" +
          (f"  ({suppressed/rej*100:.1f}% of them)" if rej else ""))
    print("  NOTE: a suppressed pair removes an ADDED and a REMOVED prediction that the final")
    print("        arm emits, so suppression alone changes the score without any MOVED appearing.")

    print("\n" + "=" * 78)
    print("4. EMPTY / NO-VISIBLE-GT PAIRS (tp=fn=0 by construction; enter pooled IoU only via FP)")
    print("=" * 78)
    em = [k for k in common if empty(A[k])]
    fa = sum(A[k]["fp"] for k in em); fb = sum(B[k]["fp"] for k in em)
    ta = sum(A[k]["fp"] for k in common); tb = sum(B[k]["fp"] for k in common)
    print(f"\n  empty pairs n={len(em)}")
    print(f"    FP px  final {fa:,} -> variant {fb:,}  ({(fb-fa)/fa*100:+.1f}%)" if fa else "")
    print(f"    share of all FP: final {fa/ta*100:.1f}% -> variant {fb/tb*100:.1f}%")
    print(f"  all pairs total FP: final {ta:,} -> variant {tb:,}  ({(tb-ta)/ta*100:+.1f}%)")

    d = [(k, B[k]["iou"] - A[k]["iou"]) for k in common]
    better = sum(1 for _, x in d if x > 1e-9); worse = sum(1 for _, x in d if x < -1e-9)
    print(f"\n  per-pair IoU: variant better on {better}, worse on {worse}, tied {len(d)-better-worse}")
    d.sort(key=lambda t: t[1])
    print("    biggest losses:"); [print(f"      {k:48s} {A[k]['iou']:.4f} -> {B[k]['iou']:.4f} ({x:+.4f})") for k, x in d[:4]]
    print("    biggest gains:");  [print(f"      {k:48s} {A[k]['iou']:.4f} -> {B[k]['iou']:.4f} ({x:+.4f})") for k, x in d[-4:][::-1]]

    json.dump({"n_paired": len(common), "per_pair_delta": dict(d),
               "resolver_outcomes": dict(rb), "final_rejected": rej},
              open(ROOT / "analysis/moved_resolver_probe.json", "w"), indent=1)
    print(f"\nwrote {ROOT/'analysis/moved_resolver_probe.json'}")


if __name__ == "__main__":
    sys.exit(main())
