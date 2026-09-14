"""Class precision for the relabel proposal: of the t1 endpoints of candidate pairs,
what fraction of their ADDED pixels land on moved-bucket GT?

Exact, no ordering guesswork: inventory_t1.objects is indexed by t1_object_id-1
(record_pair uses target+1), and the GT is built by scenediff_gt_eval.load_query_gt,
the same function the evaluator uses (moved_bucket = in_video1 and in_video2)."""
import json, pickle, sys
from pathlib import Path
import numpy as np, cv2
REPO = Path("/home/tessa/change_pipeline"); sys.path.insert(0, str(REPO/"src")); sys.path.insert(0, str(REPO/"scripts"))
from scenediff_gt_eval import load_query_gt
R = REPO/"results/scenediff_single_query_covis_v1"; EXP = "scenediff_single_query_covis_v1_gate_refine"
man = {q["pair"]: q["t1_annotation_idx"] for q in json.load(open(R/"manifest_test_top1.json"))["queries"]}
cands = json.load(open(R/"analysis/relabel_candidates.json"))["per_query_candidates"]

tot_px = on_mb = on_add = on_bg = 0; n_obj = 0; n_q = 0; per_obj = []
for p, n in sorted(cands.items()):
    if not n: continue
    t1 = man[p]; d = R/"SceneDiff"/p/f"t1_{t1:04d}"/EXP
    try:
        decs = json.load(open(d/"labels"/"inference.json"))["decisions"]
        bundle = pickle.load(open(d/"inventory"/"bundle.pkl", "rb"))
    except Exception as e:
        print(f"  skip {p}: {e}"); continue
    last_t0, last_t1 = {}, {}
    for x in decs:
        if x.get("t0_object_id") is not None: last_t0[x["t0_object_id"]] = x.get("decision")
        if x.get("t1_object_id") is not None: last_t1[x["t1_object_id"]] = x.get("decision")
    ids = [x["t1_object_id"] for x in decs if x.get("decision") == "location_mismatch_rejected"
           and last_t0.get(x["t0_object_id"]) == "removed" and last_t1.get(x["t1_object_id"]) == "added"]
    if not ids: continue
    gt = load_query_gt(REPO/"data/scenediff_benchmark/data"/p, t1)
    objs = bundle["inventory_t1"].objects
    n_q += 1
    for oid in ids:
        i = oid - 1
        if not (0 <= i < len(objs)): continue
        m = np.asarray(objs[i].mask, bool)
        if m.shape != gt.label.shape:
            m = cv2.resize(m.astype(np.uint8), (gt.label.shape[1], gt.label.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        px = int(m.sum())
        if not px: continue
        mb = int((m & (gt.label == 2)).sum()); ad = int((m & (gt.label == 1)).sum())
        tot_px += px; on_mb += mb; on_add += ad; on_bg += px - mb - ad; n_obj += 1
        per_obj.append(mb / px)
print(f"\ncandidate t1 endpoints measured: {n_obj} objects across {n_q} queries")
print(f"  total pixels            {tot_px:,}")
print(f"  on moved-bucket GT      {on_mb:,}  ({on_mb/tot_px*100:.1f}%)   <-- class precision for the relabel")
print(f"  on added GT             {on_add:,}  ({on_add/tot_px*100:.1f}%)")
print(f"  on background           {on_bg:,}  ({on_bg/tot_px*100:.1f}%)")
a = np.array(per_obj)
print(f"\n  per-object moved-bucket fraction: median {np.median(a):.3f}, "
      f">50% on {(a>0.5).sum()}/{len(a)} objects, >0 on {(a>0).sum()}/{len(a)}")
print(f"\n  for reference, ALL ADDED predictions pooled land 18.4% on moved-bucket GT")
