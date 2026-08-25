"""Unit tests for ocmask.weekend_cache against small synthetic fixtures.

No GPU, no model, no real a3_overnight.py cache -- these build a minimal
fake job directory on disk (tmp_path) with the same shape the real one has
(see docs/cache_audit.md), then check ChangesimWeekendCache.lookup()
validates/rejects it correctly. The real cache was audited separately by
read-only, model/GPU-free inspection (including the semantic preprocessing
helper), documented in docs/cache_audit.md; these tests instead prove the
*validation logic*
itself is correct against cases we control, including the failure modes a
stale/incompatible real cache could hit.
"""

from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image, PngImagePlugin

from ocmask.weekend_cache import (
    ChangesimWeekendCache,
    changed_proposal_ids_from_tracking_attempts,
    mast3r_preprocessed_rgb,
)
from ocmask.reproducibility import measure_cache_assets


def test_changed_proposal_ids_from_tracking_attempts_filters_by_gate() -> None:
    attempts = {
        "stages": {
            "source_to_clean": {
                "attempts": [
                    {"proposal_id": 1, "post_consistency_gate_accepted": False},
                    {"proposal_id": 2, "post_consistency_gate_accepted": True},
                    {"proposal_id": 3, "post_consistency_gate_accepted": False},
                ]
            },
            "target_to_clean": {"attempts": []},
        }
    }
    assert changed_proposal_ids_from_tracking_attempts(attempts, "source_to_clean") == (1, 3)
    assert changed_proposal_ids_from_tracking_attempts(attempts, "target_to_clean") == ()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


LIVE_CONFIG = {
    "reconstruction": {
        "seed": 2026,
        "image": {"width": 64, "height": 48},
        "geometry": {"depth_epsilon": 0.001},
        "mast3r": {
            "checkpoint": "checkpoints/mast3r.pth",
            "checkpoint_sha256": "mast3r-current-sha",
            "source": "src/mast3r",
            "source_commit": "mast3r-current-commit",
        },
        "sam2": {
            "checkpoint": "checkpoints/sam2.pt",
            "checkpoint_sha256": "sam2-current-sha",
            "source_commit": "sam2-current-commit",
        },
        "tracking": {"minimum_mask_area": 32, "minimum_track_iou": 0.2, "visibility_alpha": 0.8},
    },
    "sam3_proposals": {
        "sam3_image_checkpoint_sha256": "abc123",
        "proposals": {"points_per_side": 96, "minimum_mask_area": 32},
    },
    "sam3_features": {
        "sam3": {"checkpoint_sha256": "def456", "minimum_feature_cells": 4.0},
        "matching": {"fallback_minimum_cosine": 0.65},
        "classification": {"unchanged_mask_iou": 0.5},
    },
}


def _write_test_image(path: Path, offset: int) -> None:
    y, x = np.mgrid[:48, :64]
    rgb = np.stack(
        (
            (3 * x + 5 * y + offset) % 256,
            (11 * x + 7 * y + 2 * offset) % 256,
            (13 * x + 17 * y + 3 * offset) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _build_fake_cache(root: Path, *, pair_id: str = "Warehouse_1_Seq_0_0", with_stages_2to4: bool = True) -> Path:
    """A minimal but structurally faithful fake a3_overnight.py job directory."""

    image0 = root / "a.png"
    image1 = root / "b.png"
    _write_test_image(image0, 19)
    _write_test_image(image1, 71)
    shard_dir = root / "shards" / "shard-0000"
    _write_json(
        root / "job.json",
        {"schema_version": 1, "shards": [{"index": 0, "path": str(shard_dir), "count": 1}]},
    )
    (shard_dir).mkdir(parents=True, exist_ok=True)
    (shard_dir / "manifest.jsonl").write_text(
        json.dumps(
            {
                "id": pair_id,
                "image0": str(image0.resolve()),
                "image1": str(image1.resolve()),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Stage 1: a_baseline_cache, content-hash pair dir + progress.jsonl.
    stage1_pair_dir = shard_dir / "outputs" / "a_baseline_cache" / "pairs" / "deadbeefcafe"
    stage1_pair_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stage1_pair_dir / "reconstruction.npz",
        image0=mast3r_preprocessed_rgb(image0, LIVE_CONFIG["reconstruction"]),
        image1=mast3r_preprocessed_rgb(image1, LIVE_CONFIG["reconstruction"]),
    )
    for name in ("geometry.npz", "render_0_to_1.png", "render_clean_to_1.png"):
        (stage1_pair_dir / name).write_bytes(b"fake")
    # Faithfully model the legacy stage-1 config: provenance identity fields
    # were added to the live config later, but paths/settings were already pinned.
    cached_reconstruction = json.loads(json.dumps(LIVE_CONFIG["reconstruction"]))
    cached_reconstruction["mast3r"].pop("checkpoint_sha256")
    cached_reconstruction["mast3r"].pop("source")
    cached_reconstruction["mast3r"].pop("source_commit")
    cached_reconstruction["sam2"].pop("checkpoint_sha256")
    cached_reconstruction["sam2"].pop("source_commit")
    _write_json(
        stage1_pair_dir / "config.json",
        {**cached_reconstruction, "schema_version": 1},
    )
    _write_json(
        stage1_pair_dir / "inputs.json",
        {"image0": str(image0.resolve()), "image1": str(image1.resolve())},
    )
    progress_path = shard_dir / "outputs" / "a_baseline_cache" / "progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps({"id": pair_id, "status": "success", "artifacts": str(stage1_pair_dir)}) + "\n",
        encoding="utf-8",
    )

    if not with_stages_2to4:
        return root

    _write_yaml(shard_dir / "configs" / "b.yaml", {
        "proposals": LIVE_CONFIG["sam3_proposals"]["proposals"],
        "sam3_image_checkpoint_sha256": LIVE_CONFIG["sam3_proposals"]["sam3_image_checkpoint_sha256"],
    })
    _write_yaml(shard_dir / "configs" / "c.yaml", {
        "proposals": LIVE_CONFIG["sam3_proposals"]["proposals"],
        "baseline_protocol": {"tracking": LIVE_CONFIG["reconstruction"]["tracking"]},
    })
    _write_yaml(shard_dir / "configs" / "d.yaml", {
        "sam3": LIVE_CONFIG["sam3_features"]["sam3"],
        "matching": LIVE_CONFIG["sam3_features"]["matching"],
        "classification": LIVE_CONFIG["sam3_features"]["classification"],
    })

    stage2_pair_dir = shard_dir / "outputs" / "b_sam3_sam31" / "pairs" / pair_id
    (stage2_pair_dir / "proposal_cache").mkdir(parents=True, exist_ok=True)
    (stage2_pair_dir / "proposal_cache" / "source.npz").write_bytes(b"fake")
    (stage2_pair_dir / "proposal_cache" / "target.npz").write_bytes(b"fake")

    stage3_pair_dir = shard_dir / "outputs" / "c_sam3_sam2" / "pairs" / pair_id
    stage3_pair_dir.mkdir(parents=True, exist_ok=True)
    for name in ("labels.png", "diagnostics.json", "tracking_attempts.json", "target.png"):
        (stage3_pair_dir / name).write_bytes(b"fake")
    _write_json(shard_dir / "outputs" / "c_sam3_sam2" / "report.json", {"failures": []})

    stage4_pair_dir = shard_dir / "outputs" / "d_identity" / "pairs" / pair_id
    stage4_pair_dir.mkdir(parents=True, exist_ok=True)
    for name in ("sam3_features.npz", "decisions.json"):
        (stage4_pair_dir / name).write_bytes(b"fake")
    _write_json(shard_dir / "outputs" / "d_identity" / "report.json", {"failures": []})

    return root


def _lookup(
    cache: ChangesimWeekendCache,
    root: Path,
    pair_id: str = "Warehouse_1_Seq_0_0",
):
    return cache.lookup(pair_id, root / "a.png", root / "b.png")


def _inference_inputs(root: Path, pair_id: str) -> dict[str, dict[str, str]]:
    return {
        pair_id: {
            str((root / "a.png").resolve()): "image-zero-sha",
            str((root / "b.png").resolve()): "image-one-sha",
        }
    }


def test_mast3r_preprocessing_is_model_free_and_matches_pinned_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "input.png"
    _write_test_image(image, 23)
    real_import = builtins.__import__

    def reject_model_imports(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in {"torch", "torchvision", "mast3r", "dust3r"}:
            raise AssertionError(f"unexpected model-stack import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_model_imports)
    actual = mast3r_preprocessed_rgb(image, LIVE_CONFIG["reconstruction"])

    assert actual.shape == (384, 512, 3)
    assert actual.dtype == np.uint8
    assert hashlib.sha256(actual.tobytes()).hexdigest() == (
        "91c83ffe0a20e7df2845060bb5b706274d2fb2badbc7034016e377e150f68223"
    )


def test_lookup_returns_all_four_stages_for_a_fully_valid_pair(tmp_path: Path) -> None:
    root = _build_fake_cache(tmp_path)
    cache = ChangesimWeekendCache(root, LIVE_CONFIG)
    result = _lookup(cache, root)
    assert result.stage1_dir is not None
    assert result.stage2_dir is not None
    assert result.stage3_dir is not None
    assert result.stage4_dir is not None


def test_lookup_returns_nothing_for_an_unknown_pair_id(tmp_path: Path) -> None:
    root = _build_fake_cache(tmp_path)
    cache = ChangesimWeekendCache(root, LIVE_CONFIG)
    result = _lookup(cache, root, "Warehouse_99_Seq_0_0")
    assert result.stage1_dir is None
    assert result.stage2_dir is None


def test_lookup_rejects_pair_id_hit_for_different_current_inputs(tmp_path: Path) -> None:
    root = _build_fake_cache(tmp_path)
    other0 = tmp_path / "other0.png"
    other1 = tmp_path / "other1.png"
    other0.write_bytes(b"other-zero")
    other1.write_bytes(b"other-one")

    result = ChangesimWeekendCache(root, LIVE_CONFIG).lookup(
        "Warehouse_1_Seq_0_0", other0, other1
    )

    assert result.stage1_dir is None
    assert result.stop_reason == "cache_manifest_input_mismatch"


def test_lookup_rejects_stage1_with_mismatched_recorded_inputs(tmp_path: Path) -> None:
    root = _build_fake_cache(tmp_path)
    stage1 = (
        root
        / "shards"
        / "shard-0000"
        / "outputs"
        / "a_baseline_cache"
        / "pairs"
        / "deadbeefcafe"
    )
    _write_json(
        stage1 / "inputs.json",
        {"image0": str((root / "b.png").resolve()), "image1": str((root / "a.png").resolve())},
    )

    result = _lookup(ChangesimWeekendCache(root, LIVE_CONFIG), root)

    assert result.stage1_dir is None
    assert result.stop_reason == "stage1_recorded_input_mismatch"


def test_lookup_rejects_same_paths_when_current_semantic_pixels_changed(
    tmp_path: Path,
) -> None:
    root = _build_fake_cache(tmp_path)
    image0 = root / "a.png"
    with Image.open(image0) as opened:
        changed = np.asarray(opened.convert("RGB")).copy()
    changed[10:20, 12:24] = 255 - changed[10:20, 12:24]
    Image.fromarray(changed, mode="RGB").save(image0)

    result = _lookup(ChangesimWeekendCache(root, LIVE_CONFIG), root)

    assert result.stage1_dir is None
    assert result.stop_reason == "stage1_semantic_input_mismatch"


def test_lookup_accepts_raw_byte_change_with_identical_preprocessed_pixels(
    tmp_path: Path,
) -> None:
    root = _build_fake_cache(tmp_path)
    image0 = root / "a.png"
    original_bytes = image0.read_bytes()
    with Image.open(image0) as opened:
        pixels = opened.convert("RGB").copy()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("cache-test", "different raw bytes; identical RGB")
    pixels.save(image0, pnginfo=metadata, compress_level=9)

    assert image0.read_bytes() != original_bytes
    result = _lookup(ChangesimWeekendCache(root, LIVE_CONFIG), root)
    assert result.stage1_dir is not None
    assert result.stop_reason == "all_available"


@pytest.mark.parametrize(
    ("component", "field"),
    (
        ("mast3r", "checkpoint_sha256"),
        ("mast3r", "source"),
        ("mast3r", "source_commit"),
        ("sam2", "checkpoint_sha256"),
        ("sam2", "source_commit"),
    ),
)
def test_lookup_rejects_recorded_provenance_identity_when_it_disagrees(
    tmp_path: Path, component: str, field: str
) -> None:
    root = _build_fake_cache(tmp_path)
    stage1 = (
        root
        / "shards"
        / "shard-0000"
        / "outputs"
        / "a_baseline_cache"
        / "pairs"
        / "deadbeefcafe"
    )
    cached_config = json.loads((stage1 / "config.json").read_text(encoding="utf-8"))
    cached_config[component][field] = "wrong-recorded-identity"
    _write_json(stage1 / "config.json", cached_config)

    result = _lookup(ChangesimWeekendCache(root, LIVE_CONFIG), root)

    assert result.stage1_dir is None
    assert result.stop_reason == "stage1_config_mismatch"


def test_lookup_falls_back_to_stage1_only_when_stages_2to4_are_missing(tmp_path: Path) -> None:
    root = _build_fake_cache(tmp_path, with_stages_2to4=False)
    cache = ChangesimWeekendCache(root, LIVE_CONFIG)
    result = _lookup(cache, root)
    assert result.stage1_dir is not None
    assert result.stage2_dir is None
    assert result.stage3_dir is None
    assert result.stage4_dir is None


def test_lookup_rejects_stage1_on_config_mismatch(tmp_path: Path) -> None:
    root = _build_fake_cache(tmp_path)
    mismatched_config = json.loads(json.dumps(LIVE_CONFIG))
    mismatched_config["reconstruction"]["tracking"]["minimum_track_iou"] = 0.99  # differs from the cache
    cache = ChangesimWeekendCache(root, mismatched_config)
    result = _lookup(cache, root)
    assert result.stage1_dir is None


def test_lookup_rejects_stage1_when_a_required_file_is_missing(tmp_path: Path) -> None:
    root = _build_fake_cache(tmp_path)
    shard_dir = root / "shards" / "shard-0000"
    stage1_pair_dir = shard_dir / "outputs" / "a_baseline_cache" / "pairs" / "deadbeefcafe"
    (stage1_pair_dir / "geometry.npz").unlink()
    cache = ChangesimWeekendCache(root, LIVE_CONFIG)
    result = _lookup(cache, root)
    assert result.stage1_dir is None


def test_lookup_rejects_stage1_when_progress_status_is_not_success(tmp_path: Path) -> None:
    root = _build_fake_cache(tmp_path)
    shard_dir = root / "shards" / "shard-0000"
    progress_path = shard_dir / "outputs" / "a_baseline_cache" / "progress.jsonl"
    progress_path.write_text(
        json.dumps({"id": "Warehouse_1_Seq_0_0", "status": "failure", "artifacts": "/nonexistent"}) + "\n",
        encoding="utf-8",
    )
    cache = ChangesimWeekendCache(root, LIVE_CONFIG)
    result = _lookup(cache, root)
    assert result.stage1_dir is None


def test_lookup_prefers_the_last_progress_entry_for_a_retried_pair(tmp_path: Path) -> None:
    """A pair that failed then succeeded on retry (both logged) should count as cached."""

    root = _build_fake_cache(tmp_path)
    shard_dir = root / "shards" / "shard-0000"
    stage1_pair_dir = shard_dir / "outputs" / "a_baseline_cache" / "pairs" / "deadbeefcafe"
    progress_path = shard_dir / "outputs" / "a_baseline_cache" / "progress.jsonl"
    progress_path.write_text(
        json.dumps({"id": "Warehouse_1_Seq_0_0", "status": "failure", "type": "CUDAOutOfMemoryError"})
        + "\n"
        + json.dumps({"id": "Warehouse_1_Seq_0_0", "status": "success", "artifacts": str(stage1_pair_dir)})
        + "\n",
        encoding="utf-8",
    )
    cache = ChangesimWeekendCache(root, LIVE_CONFIG)
    result = _lookup(cache, root)
    assert result.stage1_dir == stage1_pair_dir


def test_lookup_rejects_stage3_when_pair_id_is_in_that_stages_failures(tmp_path: Path) -> None:
    root = _build_fake_cache(tmp_path)
    shard_dir = root / "shards" / "shard-0000"
    _write_json(
        shard_dir / "outputs" / "c_sam3_sam2" / "report.json",
        {"failures": [{"id": "Warehouse_1_Seq_0_0"}]},
    )
    cache = ChangesimWeekendCache(root, LIVE_CONFIG)
    result = _lookup(cache, root)
    assert result.stage1_dir is not None
    assert result.stage2_dir is not None
    assert result.stage3_dir is None
    assert result.stage4_dir is None  # chain stops once stage 3 is invalid


def test_lookup_rejects_stage2_on_proposal_config_mismatch(tmp_path: Path) -> None:
    root = _build_fake_cache(tmp_path)
    shard_dir = root / "shards" / "shard-0000"
    _write_yaml(
        shard_dir / "configs" / "b.yaml",
        {
            "proposals": {**LIVE_CONFIG["sam3_proposals"]["proposals"], "points_per_side": 64},
            "sam3_image_checkpoint_sha256": LIVE_CONFIG["sam3_proposals"]["sam3_image_checkpoint_sha256"],
        },
    )
    cache = ChangesimWeekendCache(root, LIVE_CONFIG)
    result = _lookup(cache, root)
    assert result.stage1_dir is not None
    assert result.stage2_dir is None
    assert result.stage3_dir is None


def test_stage4_cache_identity_uses_source_commit_and_checkpoint_hash_not_paths(
    tmp_path: Path,
) -> None:
    root = _build_fake_cache(tmp_path)
    config_path = root / "shards" / "shard-0000" / "configs" / "d.yaml"
    cached = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cached["sam3"]["source"] = "/equivalent/checkout/location"
    cached["sam3"]["checkpoint"] = "/equivalent/content-addressed/blob"
    _write_yaml(config_path, cached)

    assert _lookup(ChangesimWeekendCache(root, LIVE_CONFIG), root).stage4_dir

    cached["sam3"]["checkpoint_sha256"] = "different-weights"
    _write_yaml(config_path, cached)
    result = _lookup(ChangesimWeekendCache(root, LIVE_CONFIG), root)
    assert result.stage3_dir is not None
    assert result.stage4_dir is None
    assert result.stop_reason == "stage4_config_mismatch"


def test_missing_job_json_raises_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ChangesimWeekendCache(tmp_path, LIVE_CONFIG)


def test_reproducibility_identity_hashes_consumed_cache_tensors(
    tmp_path: Path,
) -> None:
    pair_id = "Warehouse_1_Seq_0_0"
    root = _build_fake_cache(tmp_path, pair_id=pair_id)

    inputs = _inference_inputs(root, pair_id)
    before = measure_cache_assets(root, LIVE_CONFIG, [pair_id], inputs)
    stage2 = before["consumed_pair_assets"][pair_id]["stage2"]
    source_record = next(
        record for record in stage2 if record["path"].endswith("source.npz")
    )

    Path(source_record["path"]).write_bytes(b"different cached tensor")
    after = measure_cache_assets(root, LIVE_CONFIG, [pair_id], inputs)
    after_stage2 = after["consumed_pair_assets"][pair_id]["stage2"]
    changed_record = next(
        record for record in after_stage2 if record["path"].endswith("source.npz")
    )

    assert source_record["sha256"] != changed_record["sha256"]
