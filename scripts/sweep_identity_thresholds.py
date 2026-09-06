#!/usr/bin/env python3
"""Offline threshold sweep for sam_cosine / dino_cosine / track_iou /
same_location_iou (spatial IoU), against already-completed PASLCD runs --
without re-running SAM2 tracking (the expensive part) for every candidate
threshold setting.

Each already-completed query's inference.json already has the full
candidate-pair evidence matrices (direct_sam_cosine, direct_dino_cosine,
direct_bidirectional_track_iou, direct_spatial_iou -- see
change_detection.resolve_three_image_changes's diagnostics["association_evidence"]).
This script reuses those matrices directly and only needs one light,
SAM3+DINOv2-only rerun per query (no SAM2) to regenerate the actual object
masks (not saved as arrays, only as visualization PNGs) needed to rasterize
a candidate threshold setting's decisions into pixels for scoring against GT.

Two-phase design so the (cheap) sweep itself never touches the GPU again:
  1. `--collect`: for each completed query, regenerate objects_t0/clean/t1
     and cache {masks, evidence, gt} to a .npz per query (SAM3+DINOv2 only).
  2. `--sweep`: pure-numpy grid search over the cached data.

Known simplification: only the *direct* (render_t0-vs-image_t1) identity
path is replayed here -- clean_bridge_identity and tracking_recovery are not
(they need dense feature pooling at track-implied locations that isn't
cached). Per the pixel-loss analysis, direct_identity is ~90% of the
observed loss, so this still targets the dominant failure mode; treat any
chosen thresholds as a starting point to re-validate with a real (full) run,
not a final answer.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def collect(detect_dirs: list[Path], gt_masks: list[Path], config_path: Path, cache_dir: Path) -> None:
    import cv2
    import numpy as np
    from PIL import Image

    from ocmask_pipeline.adapters.dinov2 import Dinov2FeatureExtractor
    from ocmask_pipeline.change_detection import (
        ThreeImageSettings,
        _build_inventory,
        _proposal_kwargs,
        _suppress_feature_matched_parts,
        select_object_proposals,
    )
    from ocmask_pipeline.config import load_config
    from ocmask_pipeline.metrics import load_paslcd_gt
    from ocmask_pipeline.stages.sam3_proposals import Sam3AutomaticMaskGenerator

    config = load_config(config_path)
    settings = ThreeImageSettings.from_config(config)
    sam_cfg = config["sam3_proposals"]
    generator = Sam3AutomaticMaskGenerator(sam_cfg["sam3_image_checkpoint"], source=sam_cfg["sam3_source"], **_proposal_kwargs(config))
    dino_extractor = Dinov2FeatureExtractor(config["dinov2_features"])

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        for detect_dir, gt_mask_path in zip(detect_dirs, gt_masks):
            query_id = detect_dir.parent.name
            out_path = cache_dir / f"{query_id}.npz"
            if out_path.exists():
                print(f"{query_id}: cached, skipping")
                continue
            inference_path = detect_dir / "inference.json"
            if not inference_path.exists():
                print(f"{query_id}: no inference.json, skipping")
                continue
            inference = json.loads(inference_path.read_text())
            evidence = inference.get("association_evidence")
            if not evidence:
                print(f"{query_id}: no association_evidence, skipping")
                continue

            render_t0 = np.asarray(Image.open(detect_dir / "render_t0.png").convert("RGB"))
            image_t1 = np.asarray(Image.open(detect_dir / "target.png").convert("RGB"))

            raw_t0, sam_t0 = generator.generate_with_feature_map(render_t0)
            raw_t1, sam_t1 = generator.generate_with_feature_map(image_t1)
            dino_t0 = dino_extractor.feature_map(render_t0)
            dino_t1 = dino_extractor.feature_map(image_t1)

            objects_t0 = select_object_proposals(raw_t0, settings)
            objects_t1 = select_object_proposals(raw_t1, settings)
            inv_t0 = _suppress_feature_matched_parts(_build_inventory(objects_t0, sam_t0, dino_t0, settings), settings)
            inv_t1 = _suppress_feature_matched_parts(_build_inventory(objects_t1, sam_t1, dino_t1, settings), settings)
            final_t0 = list(inv_t0.objects)
            final_t1 = list(inv_t1.objects)

            n0, n1 = len(evidence["direct_sam_cosine"]), len(evidence["direct_sam_cosine"][0]) if evidence["direct_sam_cosine"] else 0
            if len(final_t0) != n0 or len(final_t1) != n1:
                print(f"{query_id}: WARNING object-count mismatch after part-suppression rerun ({len(final_t0)},{len(final_t1)}) vs saved evidence ({n0},{n1}) -- skipping (nondeterminism or upstream code change)")
                continue

            height, width = image_t1.shape[:2]
            gt_native = load_paslcd_gt(gt_mask_path)
            gt = cv2.resize(gt_native, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)

            masks_t0 = np.stack([np.asarray(o.mask, dtype=bool) for o in final_t0]) if final_t0 else np.zeros((0, height, width), bool)
            masks_t1 = np.stack([np.asarray(o.mask, dtype=bool) for o in final_t1]) if final_t1 else np.zeros((0, height, width), bool)

            np.savez_compressed(
                out_path,
                masks_t0=np.packbits(masks_t0, axis=None) if masks_t0.size else masks_t0,
                masks_t0_shape=np.array(masks_t0.shape),
                masks_t1=np.packbits(masks_t1, axis=None) if masks_t1.size else masks_t1,
                masks_t1_shape=np.array(masks_t1.shape),
                gt=np.packbits(gt, axis=None),
                gt_shape=np.array(gt.shape),
                sam=np.asarray(evidence["direct_sam_cosine"], dtype=np.float32),
                dino=np.asarray(evidence["direct_dino_cosine"], dtype=np.float32),
                track=np.asarray(evidence["direct_bidirectional_track_iou"], dtype=np.float32),
                spatial=np.asarray(evidence["direct_spatial_iou"], dtype=np.float32),
            )
            print(f"{query_id}: cached ({len(final_t0)} t0 objs, {len(final_t1)} t1 objs)")
    finally:
        generator.release()
        dino_extractor.release()


def _unpack(data, key_prefix):
    import numpy as np

    shape = tuple(data[f"{key_prefix}_shape"])
    if int(np.prod(shape)) == 0:
        return np.zeros(shape, dtype=bool)
    packed = data[key_prefix]
    return np.unpackbits(packed)[: int(np.prod(shape))].reshape(shape).astype(bool)


def _replay_direct_only(sam, dino, track, spatial, sam_thr, dino_thr, track_thr, spatial_thr, margin=0.02):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    n0, n1 = sam.shape
    identity = (sam >= sam_thr) & (dino >= dino_thr)
    feature_score = np.minimum(sam, dino)

    # reciprocal-with-margin (mirrors change_detection._reciprocal_with_margin)
    reciprocal = np.zeros_like(identity, dtype=bool)
    if identity.any():
        masked = np.where(identity, feature_score, -np.inf)
        row_best = np.argmax(masked, axis=1) if n1 else np.array([], int)
        column_best = np.argmax(masked, axis=0) if n0 else np.array([], int)
        for row, column in zip(*np.nonzero(identity)):
            if row_best[row] != column or column_best[column] != row:
                continue
            row_values = np.delete(masked[row], column)
            column_values = np.delete(masked[:, column], row)
            row_second = float(row_values.max(initial=-1.0))
            column_second = float(column_values.max(initial=-1.0))
            if feature_score[row, column] - max(row_second, column_second) >= margin:
                reciprocal[row, column] = True

    tracked = track >= track_thr
    feasible = identity & (tracked | reciprocal)
    score = feature_score + 0.20 * track

    pairs = []
    if feasible.any():
        rows, cols = linear_sum_assignment(np.where(feasible, -score, 1e6))
        pairs = [(int(r), int(c)) for r, c in zip(rows, cols) if feasible[r, c]]

    matched_t0 = {r for r, _ in pairs}
    matched_t1 = {c for _, c in pairs}
    unchanged_or_moved = []
    for r, c in pairs:
        decision = "unchanged" if spatial[r, c] >= spatial_thr else "moved"
        unchanged_or_moved.append((r, c, decision))
    removed = [r for r in range(n0) if r not in matched_t0]
    added = [c for c in range(n1) if c not in matched_t1]
    return unchanged_or_moved, removed, added


def sweep(cache_dir: Path, output: Path) -> None:
    import numpy as np

    cached = sorted(cache_dir.glob("*.npz"))
    if not cached:
        raise SystemExit(f"no cached queries in {cache_dir} -- run --collect first")

    grid = {
        "sam_thr": [0.45, 0.55, 0.65, 0.75],
        "dino_thr": [0.40, 0.50, 0.60, 0.70],
        "track_thr": [0.10, 0.20, 0.35, 0.50],
        "spatial_thr": [0.30, 0.45, 0.60],
    }
    combos = list(itertools.product(*grid.values()))
    print(f"sweeping {len(combos)} threshold combinations over {len(cached)} cached queries")

    results = []
    loaded = []
    for path in cached:
        with np.load(path) as data:
            masks_t0 = _unpack(data, "masks_t0")
            masks_t1 = _unpack(data, "masks_t1")
            gt = np.unpackbits(data["gt"])[: int(np.prod(data["gt_shape"]))].reshape(tuple(data["gt_shape"])).astype(bool)
            loaded.append((path.stem, masks_t0, masks_t1, gt, data["sam"], data["dino"], data["track"], data["spatial"]))

    for sam_thr, dino_thr, track_thr, spatial_thr in combos:
        ious, f1s, precisions, recalls = [], [], [], []
        for name, masks_t0, masks_t1, gt, sam, dino, track, spatial in loaded:
            pairs, removed, added = _replay_direct_only(sam, dino, track, spatial, sam_thr, dino_thr, track_thr, spatial_thr)
            height, width = gt.shape
            pred = np.zeros((height, width), dtype=bool)
            for r, c, decision in pairs:
                if decision == "moved":
                    pred |= masks_t1[c]
            for c in added:
                pred |= masks_t1[c]
            # removed objects live in render_t0's frame, not image_t1's --
            # they don't contribute pixels to this (image_t1-aligned) mask.
            tp = int((gt & pred).sum())
            fp = int((~gt & pred).sum())
            fn = int((gt & ~pred).sum())
            union = tp + fp + fn
            iou = tp / union if union else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            ious.append(iou); f1s.append(f1); precisions.append(precision); recalls.append(recall)
        results.append({
            "sam_thr": sam_thr, "dino_thr": dino_thr, "track_thr": track_thr, "spatial_thr": spatial_thr,
            "mean_iou": sum(ious) / len(ious), "mean_f1": sum(f1s) / len(f1s),
            "mean_precision": sum(precisions) / len(precisions), "mean_recall": sum(recalls) / len(recalls),
            "n_queries": len(loaded),
        })

    results.sort(key=lambda r: r["mean_f1"], reverse=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"wrote {len(results)} results to {output}")
    print("\ntop 10 by mean F1:")
    for r in results[:10]:
        print(f"  sam>={r['sam_thr']} dino>={r['dino_thr']} track>={r['track_thr']} spatial>={r['spatial_thr']}  "
              f"-> iou={r['mean_iou']:.4f} f1={r['mean_f1']:.4f} prec={r['mean_precision']:.4f} rec={r['mean_recall']:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    collect_p = sub.add_parser("collect")
    collect_p.add_argument("--results-root", type=Path, default=REPO / "results" / "paslcd")
    collect_p.add_argument("--data-root", type=Path, default=REPO / "data" / "PASLCD")
    collect_p.add_argument("--config", type=Path, default=REPO / "configs/pipeline.yaml")
    collect_p.add_argument("--cache-dir", type=Path, default=REPO / "results" / "paslcd" / "threshold_sweep_cache")
    collect_p.add_argument("--limit", type=int, default=None)

    sweep_p = sub.add_parser("sweep")
    sweep_p.add_argument("--cache-dir", type=Path, default=REPO / "results" / "paslcd" / "threshold_sweep_cache")
    sweep_p.add_argument("--output", type=Path, default=REPO / "results" / "paslcd" / "threshold_sweep_results.json")

    args = parser.parse_args()

    if args.command == "collect":
        detect_dirs, gt_masks = [], []
        for scene_dir in sorted(args.results_root.iterdir()):
            if not scene_dir.is_dir() or "_Instance_" not in scene_dir.name:
                continue
            # scene dirs are named "<Dataset>_<Instance>" where Instance is
            # itself "Instance_1"/"Instance_2" -- the last two "_"-parts are
            # always the instance, everything before is the dataset (which
            # may itself contain underscores, e.g. "Lunch_room").
            parts = scene_dir.name.split("_")
            instance = "_".join(parts[-2:])
            dataset = "_".join(parts[:-2])
            intermediate = scene_dir / "intermediate"
            if not intermediate.exists():
                continue
            for query_dir in sorted(intermediate.iterdir()):
                detect_dir = query_dir / "detect"
                gt_path = args.data_root / dataset / instance / "gt_mask" / f"{query_dir.name}.png"
                if detect_dir.exists() and gt_path.exists():
                    detect_dirs.append(detect_dir)
                    gt_masks.append(gt_path)
        if args.limit:
            detect_dirs, gt_masks = detect_dirs[: args.limit], gt_masks[: args.limit]
        print(f"found {len(detect_dirs)} completed queries with GT")
        collect(detect_dirs, gt_masks, args.config, args.cache_dir)
    else:
        sweep(args.cache_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
