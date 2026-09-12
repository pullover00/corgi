#!/usr/bin/env python3
"""Ground-truth extraction and evaluation for SceneDiff pairs, restricted to
image_t1's own pixel space.

Scope (deliberately narrower than SceneDiff's official whole-video AP
protocol -- see docs/METHODS.md and chat history for why):

video1 and video2 are separate camera walkthroughs of the same real space,
so a video1-frame object mask and a video2-frame object mask are NOT in a
shared pixel coordinate system -- comparing their positions directly would
require the same 3D reconstruction+reprojection our own pipeline already
does, which would make any resulting "ground truth" circular (validating
our reconstruction against itself). image_t1 in our pipeline is an
unmodified extracted frame from video2, though -- so a GT object mask
decoded from video2_objects at that *exact* frame index is already
pixel-aligned with our image_t1, no reconstruction needed. That is the only
GT this module produces:

  - ADDED objects (in_video1=False, in_video2=True): fully in scope.
  - "moved-bucket" objects (in_video1=True and in_video2=True -- this is
    SceneDiff's own official definition of "moved": present in both videos,
    NOT verified to have actually changed position, see
    scene_diff/scripts/evaluate_multiview.py:extract_ground_truth): their
    video2-side appearance is in scope.
  - REMOVED-only objects (in_video1=True, in_video2=False): OUT of scope.
    They never appear in image_t1's frame at all; their true old-location
    footprint in image_t1's coordinate space is unknowable without the same
    reconstruction step this is trying to independently validate.

So this cannot evaluate our REMOVED recall against real GT, and it cannot
verify our MOVED calls represent genuine spatial displacement (SceneDiff
itself doesn't distinguish that within its "moved" bucket). What it CAN do,
non-circularly: check whether our predicted ADDED/MOVED pixels in image_t1
land on real SceneDiff-tracked objects at all (vs. background/noise), and
which GT bucket they land on.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ocmask_pipeline.types import Label  # noqa: E402


class QueryGT:
    """GT for one query: (pair_dir, t1_frame_idx)."""

    def __init__(self, shape: tuple[int, int]):
        self.shape = shape
        # 0 = background/untracked, 1 = ADDED, 2 = moved-bucket (present in both)
        self.label = np.zeros(shape, dtype=np.uint8)
        self.objects: list[dict] = []  # per-object metadata for diagnostics


def _nearest_available_frame(frames: dict, target: int) -> int | None:
    keys = sorted(int(k) for k in frames.keys())
    if not keys:
        return None
    return min(keys, key=lambda k: abs(k - target))


def _video2_frame_shape(pair_dir: Path, frame_idx: int) -> tuple[int, int]:
    """Read the native query-frame shape for a removed-only empty GT.

    Such pairs intentionally have no ``video2_objects`` RLE from which to
    infer a canvas, but their prediction must still be scored against an
    all-background T1 frame as preregistered.  This is evaluation plumbing
    only; it does not synthesize an object mask.
    """
    candidates = sorted(pair_dir.glob("original_video2.*")) or sorted(pair_dir.glob("video2.*"))
    if not candidates:
        raise FileNotFoundError(f"no video2 file in {pair_dir}")
    # Only the canvas SHAPE is needed here, and every frame of one video shares
    # it -- so read it from the stream metadata instead of seeking to and
    # decoding ``frame_idx``. The seek+decode form failed on 13/250 P1 pairs
    # (all of them this empty-GT path; every one decoded fine when retried in
    # isolation), i.e. a transient OpenCV decode hiccup under load in a path
    # with no retry, not bad data. Metadata cannot hiccup that way; decoding
    # frame 0 is the fallback for containers that report 0 there.
    capture = cv2.VideoCapture(str(candidates[0]))
    try:
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        if height <= 0 or width <= 0:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"could not determine video2 frame shape in {pair_dir} (metadata empty, frame 0 undecodable)")
            height, width = (int(value) for value in frame.shape[:2])
    finally:
        capture.release()
    return (height, width)


def load_query_gt(pair_dir: Path, t1_frame_idx: int, max_frame_distance: int = 15, allow_empty_frame: bool = True) -> QueryGT:
    """Build image_t1-space GT for one query. Objects whose nearest available
    video2 frame is more than ``max_frame_distance`` away from
    ``t1_frame_idx`` are skipped (their footprint at our actual query frame
    is too uncertain to trust)."""
    segments = pickle.loads((pair_dir / "segments.pkl").read_bytes())
    video2_objects = segments["video2_objects"]
    meta_by_id = {obj["original_obj_idx"]: obj for obj in segments["objects"]}

    shape = None
    gt = None
    for obj_id, frames in video2_objects.items():
        meta = meta_by_id.get(obj_id)
        if meta is None or not meta.get("in_video2", False):
            continue
        nearest = _nearest_available_frame(frames, t1_frame_idx)
        if nearest is None or abs(nearest - t1_frame_idx) > max_frame_distance:
            continue
        rle = frames[str(nearest)]
        mask = mask_utils.decode(rle).astype(bool)
        if shape is None:
            shape = mask.shape
            gt = QueryGT(shape)

        status = "moved_bucket" if meta.get("in_video1", False) else "added"
        value = 2 if status == "moved_bucket" else 1
        # later objects don't overwrite earlier ones already marked -- first
        # writer wins, consistent with "don't silently drop signal", ties
        # are rare (checked: SceneDiff objects are largely non-overlapping)
        write_mask = mask & (gt.label == 0)
        gt.label[write_mask] = value
        gt.objects.append({
            "obj_id": obj_id,
            "label": meta.get("label"),
            "status": status,
            "frame_used": nearest,
            "frame_requested": t1_frame_idx,
            "pixels": int(mask.sum()),
        })

    if gt is None and not any(obj.get("in_video2", False) for obj in segments["objects"]):
        # Removed-only diagnostic: there is deliberately no in-scope T1
        # object.  Score every predicted pixel as background/FP instead of
        # treating the absence of video2 RLEs as an evaluator failure.
        return QueryGT(_video2_frame_shape(pair_dir, t1_frame_idx))
    if gt is None and allow_empty_frame:
        # The pair has after-video objects, but none has a usable mask within
        # max_frame_distance of THIS query frame -- they are simply not visible
        # here. Under SceneDiff's own evaluator a frame with no annotation is an
        # empty frame (every prediction is FP), not an error. Co-visibility-based
        # query selection (2026-09-11) lands on such frames routinely, since it
        # never consults the annotations.
        return QueryGT(_video2_frame_shape(pair_dir, t1_frame_idx))
    if gt is None:
        raise ValueError(f"no video2 objects with a usable frame near {t1_frame_idx} in {pair_dir}")
    return gt


def compare_to_prediction(gt: QueryGT, labels_path: Path) -> dict:
    """Compare our pipeline's labels.png against the image_t1-space GT.
    Resizes our prediction to GT resolution with nearest-neighbor, matching
    the PASLCD evaluation convention (see metrics.py)."""
    pred = cv2.imread(str(labels_path), cv2.IMREAD_GRAYSCALE)
    if pred is None:
        raise FileNotFoundError(labels_path)
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    result = {}
    gt_added = gt.label == 1
    gt_moved_bucket = gt.label == 2
    gt_background = gt.label == 0

    for label_value, label_name in [(int(Label.ADDED), "ADDED"), (int(Label.MOVED), "MOVED"),
                                      (int(Label.REMOVED), "REMOVED"), (int(Label.REPLACED), "REPLACED")]:
        pred_mask = pred == label_value
        n = int(pred_mask.sum())
        if n == 0:
            result[label_name] = {"pixels": 0}
            continue
        result[label_name] = {
            "pixels": n,
            "on_gt_added": int((pred_mask & gt_added).sum()),
            "on_gt_moved_bucket": int((pred_mask & gt_moved_bucket).sum()),
            "on_gt_background": int((pred_mask & gt_background).sum()),
        }

    # recall on the in-scope GT buckets only
    pred_changed = pred != 0
    result["_recall"] = {
        "added_recall": float((pred_changed & gt_added).sum() / gt_added.sum()) if gt_added.sum() else None,
        "moved_bucket_recall": float((pred_changed & gt_moved_bucket).sum() / gt_moved_bucket.sum()) if gt_moved_bucket.sum() else None,
    }
    result["_gt_pixel_counts"] = {
        "added": int(gt_added.sum()), "moved_bucket": int(gt_moved_bucket.sum()), "background": int(gt_background.sum()),
    }
    result["_objects"] = gt.objects
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--t1-frame-idx", type=int, required=True)
    parser.add_argument("--labels", type=Path, help="path to our pipeline's labels.png for this query; if omitted, only GT stats are printed")
    parser.add_argument("--dump-gt-png", type=Path, help="optional: write the GT label map as a visualizable PNG (0/85/170 for background/added/moved-bucket)")
    args = parser.parse_args()

    pair_dir = args.benchmark_root / "data" / args.pair_id
    gt = load_query_gt(pair_dir, args.t1_frame_idx)

    print(f"=== {args.pair_id} @ t1_frame={args.t1_frame_idx} ===")
    print(f"GT shape: {gt.shape}")
    for obj in gt.objects:
        print(f"  {obj['status']:12s} obj_id={obj['obj_id']:3d} label={obj['label']:20s} "
              f"frame_used={obj['frame_used']:4d} (requested {obj['frame_requested']}) pixels={obj['pixels']}")
    n_added = sum(1 for o in gt.objects if o["status"] == "added")
    n_moved = sum(1 for o in gt.objects if o["status"] == "moved_bucket")
    print(f"objects in scope: {n_added} added, {n_moved} moved-bucket")

    if args.dump_gt_png:
        vis = (gt.label.astype(np.float32) * 127.5).astype(np.uint8)
        cv2.imwrite(str(args.dump_gt_png), vis)
        print(f"wrote {args.dump_gt_png}")

    if args.labels:
        result = compare_to_prediction(gt, args.labels)
        print("\n--- comparison vs our prediction ---")
        print(json.dumps({k: v for k, v in result.items() if not k.startswith("_") or k in ("_recall", "_gt_pixel_counts")}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
