#!/usr/bin/env python3
"""Run the legacy multi-script research DAG (not the production pipeline).

This historical audit tool sequences the original stage scripts in dependency
order for fixed10/new15. It has different cache/resume/numerical semantics,
loads an unused SAM3.1 branch, and must not be used for a new result. Use
``ocmask evaluate changesim --full-pipeline`` or
``scripts/run_eval_resilient.sh`` for the fused production method.

The explicit ``--allow-legacy-research-dag`` acknowledgement is required to
make an accidental launch fail before any GPU work or output mutation.

Stage 2/3 run twice per split: once at points_per_side=96 ("grid96", the
main path used by every downstream stage) and once at the original
points_per_side=64 ("grid64", needed only because the DINOv2 feature stage's
config was frozen against that earlier lineage -- see README.md's "Two SAM3
grid densities" note).

Requires scripts/bootstrap_models.sh to have been run, checkpoints/ to be
populated, and SAM3_SOURCE/SAM3_IMAGE_CHECKPOINT/SAM31_CHECKPOINT to be
exported. Requires the ChangeSim dataset under data/changesim/. See
README.md before running this against real data -- it launches MASt3R,
SAM2, SAM3, SAM3.1, and DINOv2 inference across every pair in both splits.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]

MANIFEST = {
    "fixed10": "data/changesim/manifest-table3.jsonl",
    "new15": "data/changesim/manifest-new15.jsonl",
}

# Exact config filenames per (stage, split); see README.md's pipeline-stage
# table for what each stage does and why grid64 configs exist alongside the
# grid96 ones used by every other stage.
CONFIGS = {
    "s02_grid96": {"fixed10": "changesim-sam3-masks-sam31-tracking-no-splat-densegrid96.yaml",
                   "new15": "changesim-sam3-masks-sam31-tracking-new15-densegrid96.yaml"},
    "s02_grid64": {"fixed10": "changesim-sam3-masks-sam31-tracking-no-splat.yaml",
                   "new15": "changesim-sam3-masks-sam31-tracking-new15.yaml"},
    "s03_grid96": {"fixed10": "changesim-sam3-masks-sam2-tracking-v4-no-splat-densegrid96.yaml",
                   "new15": "changesim-sam3-masks-sam2-tracking-v4-new15-densegrid96.yaml"},
    "s03_grid64": {"fixed10": "changesim-sam3-masks-sam2-tracking-v4-no-splat.yaml",
                   "new15": "changesim-sam3-masks-sam2-tracking-v4-new15.yaml"},
    "s04": {"fixed10": "changesim-sam3-identity-location-no-splat-densegrid96.yaml",
            "new15": "changesim-sam3-identity-location-new15-densegrid96.yaml"},
    "s05": {"fixed10": "changesim-dinov2-identity-location-no-splat-hires.yaml",
            "new15": "changesim-dinov2-identity-location-new15-hires.yaml"},
    "s06": {"fixed10": "changesim-sam3-sam2-moved-association-no-splat-densegrid96.yaml",
            "new15": "changesim-sam3-sam2-moved-association-new15-densegrid96.yaml"},
    "s07": {"fixed10": "changesim-sam3-guarded-hybrid-fixed10-densegrid96.yaml",
            "new15": "changesim-sam3-guarded-hybrid-new15-densegrid96.yaml"},
    "s08": {"fixed10": "changesim-sam3-conservative-a3-fixed10-densegrid96.yaml",
            "new15": "changesim-sam3-conservative-a3-new15-densegrid96.yaml"},
    "s09": {"fixed10": "changesim-obvious-object-sentinel-fixed10-densegrid96.yaml",
            "new15": "changesim-obvious-object-sentinel-new15-densegrid96.yaml"},
    "s10": {"fixed10": "changesim-real-image-association-resolver-fixed10-densegrid96.yaml",
            "new15": "changesim-real-image-association-resolver-new15-densegrid96.yaml"},
}


def _config(stage: str, split: str) -> str:
    return f"configs/stages/{CONFIGS[stage][split]}"


def _run(*args: str) -> None:
    print("+ " + " ".join(args), flush=True)
    subprocess.run(list(args), cwd=REPOSITORY, check=True)


def run_split(split: str, *, skip_html: bool) -> None:
    py = sys.executable
    manifest = MANIFEST[split]

    _run(py, "-m", "ocmask.cli", "--config", "configs/stage01_reconstruction.yaml",
         "evaluate", "changesim", "--manifest", manifest,
         "--output", f"outputs/s01_reconstruction/{split}", "--artifact-level", "full")

    for grid in ("grid96", "grid64"):
        _run(py, "scripts/run_sam3_pairwise_experiment.py",
             "--config", _config(f"s02_{grid}", split),
             "--output", f"outputs/s02_sam3_proposals/{split}_{grid}")

    for grid in ("grid96", "grid64"):
        _run(py, "scripts/run_sam3_pairwise_experiment.py",
             "--config", _config(f"s03_{grid}", split),
             "--output", f"outputs/s03_sam2_tracking_v4/{split}_{grid}")

    _run(py, "scripts/run_sam3_identity_location_experiment.py",
         "--config", _config("s04", split), "--output", f"outputs/s04_sam3_features/{split}")

    _run(py, "scripts/run_dinov2_identity_location_experiment.py",
         "--config", _config("s05", split), "--output", f"outputs/s05_dinov2_features/{split}")

    step6 = [py, "scripts/run_sam3_moved_association_experiment.py",
             "--config", _config("s06", split), "--output", f"outputs/s06_moved_association/{split}"]
    if skip_html:
        step6.append("--skip-html")
    _run(*step6)

    _run(py, "scripts/run_sam3_guarded_hybrid_experiment.py",
         "--config", _config("s07", split), "--output", f"outputs/s07_guarded_hybrid/{split}")

    step8 = [py, "scripts/run_sam3_feature_veto_gate_experiment.py",
             "--config", _config("s08", split), "--output", f"outputs/s08_feature_veto_gate/{split}"]
    if skip_html:
        step8.append("--skip-html")
    _run(*step8)

    _run(py, "scripts/run_obvious_object_sentinel_experiment.py",
         "--config", _config("s09", split), "--output", f"outputs/s09_obvious_object_sentinel/{split}")

    _run(py, "scripts/run_real_image_association_resolver.py",
         "--config", _config("s10", split), "--output", f"outputs/s10_association_resolver/{split}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--allow-legacy-research-dag",
        action="store_true",
        help="Acknowledge that this is the superseded research DAG, not production evaluation.",
    )
    parser.add_argument("--splits", default="fixed10,new15")
    parser.add_argument(
        "--with-html", action="store_true",
        help="Also build the (slow) diagnostic index.html for stages 6 and 8.",
    )
    parser.add_argument(
        "--stage11-only", action="store_true",
        help="Skip stages 1-10 and run only the final method over already-computed caches.",
    )
    args = parser.parse_args()
    if not args.allow_legacy_research_dag:
        raise SystemExit(
            "scripts/run_pipeline.py is a superseded research DAG and is blocked "
            "by default. Use scripts/run_eval_resilient.sh for production, or pass "
            "--allow-legacy-research-dag only for an intentional historical audit."
        )
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    for split in splits:
        if split not in MANIFEST:
            raise SystemExit(f"unknown split {split!r}; choose from {sorted(MANIFEST)}")

    if not args.stage11_only:
        for split in splits:
            run_split(split, skip_html=not args.with_html)

    _run(
        sys.executable, "scripts/run_slot_inconsistency_replacement_experiment.py",
        "--splits", ",".join(splits),
    )


if __name__ == "__main__":
    main()
