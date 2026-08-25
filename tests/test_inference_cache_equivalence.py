from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import ocmask.inference as inference
from ocmask.config import load_config
from ocmask.io import save_image, save_json
from ocmask.stages.sam3_identity_location import (
    AppearanceFeatures,
    FeatureDescriptorBatch,
    SimilarityCalibration,
)
from ocmask.stages.sam3_proposals import Sam3Proposal
from ocmask.types import ObjectMask
from ocmask.weekend_cache import StageCacheDirs


def _object(mask: np.ndarray, proposal_id: int = 1) -> ObjectMask:
    return ObjectMask(
        mask=np.asarray(mask, dtype=bool),
        score=1.0,
        source="test",
        metadata={"automatic_proposal_id": proposal_id},
    )


def test_stage4_cache_recomputes_current_decisions_instead_of_trusting_json(
    tmp_path: Path,
) -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    source = [_object(mask.copy())]
    target = [_object(mask.copy())]
    # Identical, finite dense tensors make the one source/target object an
    # unambiguous same-identity/same-location pair under current logic.
    feature_map = np.zeros((2, 2, 2), dtype=np.float16)
    feature_map[0] = 1.0
    np.savez_compressed(
        tmp_path / "sam3_features.npz", source=feature_map, target=feature_map
    )
    stale_calibration = {
        "threshold": 0.99,
        "valid": True,
        "positive_count": 99,
        "negative_count": 99,
        "positive_acceptance": 1.0,
        "negative_acceptance": 0.0,
        "inferred_threshold": 0.99,
        "invalid_reason": None,
    }
    stale_matches = [
        {
            "decision": "moved",
            "source_index": 0,
            "target_index": 0,
            "source_proposal_id": 1,
            "target_proposal_id": 1,
        }
    ]
    save_json(
        tmp_path / "decisions.json",
        {
            "diagnostics": {
                "calibration": stale_calibration,
                "source_gate_changed_count": 0,
                "target_gate_changed_count": 0,
                "source_valid_descriptor_count": 1,
                "target_valid_descriptor_count": 1,
            },
            "matches": stale_matches,
        },
    )
    cfg = deepcopy(load_config("configs/pipeline.yaml")["sam3_features"])
    cfg["sam3"]["minimum_feature_cells"] = 0.1

    appearance, diagnostics = inference._appearance_features_from_cache(
        tmp_path,
        source,
        target,
        np.asarray([False]),
        np.asarray([False]),
        cfg,
    )

    assert appearance is not None
    assert appearance.calibration.threshold == cfg["matching"][
        "fallback_minimum_cosine"
    ]
    assert [record["decision"] for record in appearance.match_records] == [
        "unchanged"
    ]
    assert appearance.match_records != stale_matches
    assert diagnostics["dense_maps_reused"] is True
    assert diagnostics["decision_policy"] == "current_cpu_recomputed"
    assert diagnostics["historical_decision_audit"] == {
        "calibration_equal": False,
        "match_records_equal": False,
    }


def test_stage4_cache_count_mismatch_falls_back_before_using_dense_maps(
    tmp_path: Path,
) -> None:
    mask = np.ones((4, 4), dtype=bool)
    feature_map = np.ones((2, 2, 2), dtype=np.float16)
    np.savez_compressed(
        tmp_path / "sam3_features.npz", source=feature_map, target=feature_map
    )
    save_json(
        tmp_path / "decisions.json",
        {
            "diagnostics": {
                "source_gate_changed_count": 1,
                "target_gate_changed_count": 0,
            },
            "matches": [],
        },
    )

    appearance, diagnostics = inference._appearance_features_from_cache(
        tmp_path,
        [_object(mask.copy())],
        [_object(mask.copy())],
        np.asarray([False]),
        np.asarray([False]),
        load_config("configs/pipeline.yaml")["sam3_features"],
    )

    assert appearance is None
    assert diagnostics["dense_maps_reused"] is False
    assert diagnostics["fallback_reason"] == "source_gate_count_mismatch"


def test_run_pair_separates_tracker_lifetimes_around_sam3(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.ones((4, 4), dtype=bool)
    proposal = Sam3Proposal(mask, 0.9, 0.9, (1.0, 1.0), (0, 0, 4, 4))
    artifact = tmp_path / "stage1"
    stage2 = tmp_path / "stage2"
    artifact.mkdir()
    (stage2 / "proposal_cache").mkdir(parents=True)

    class FakeState:
        def to_dict(self):
            return {"state": "test"}

    monkeypatch.setattr(inference, "capture_torch_numerical_state", FakeState)
    monkeypatch.setattr(inference, "apply_post_reconstruction_numerics", lambda: None)
    monkeypatch.setattr(
        inference,
        "load_cached_inputs",
        lambda *_args, **_kwargs: SimpleNamespace(
            source_render=rgb,
            target_image=rgb,
            clean_render=rgb,
            cross_coverage=np.ones((4, 4), dtype=bool),
            clean_coverage=np.ones((4, 4), dtype=bool),
        ),
    )
    monkeypatch.setattr(
        inference,
        "load_reconstruction",
        lambda *_args: SimpleNamespace(images=(rgb, rgb)),
    )
    monkeypatch.setattr(inference, "load_proposal_cache", lambda *_args: [proposal])

    tracker_count = 0

    class FakeTracker:
        def __init__(self, _config):
            nonlocal tracker_count
            tracker_count += 1
            self.number = tracker_count
            events.append(f"tracker{self.number}:init")

        def track(self, masks, _source, _target):
            events.append(f"tracker{self.number}:track")
            return [
                SimpleNamespace(
                    mask=np.asarray(item, dtype=bool),
                    accepted=True,
                    rejection_reasons=(),
                )
                for item in masks
            ]

        def release(self):
            events.append(f"tracker{self.number}:release")

    monkeypatch.setattr(inference, "Sam2MaskTracker", FakeTracker)

    def fake_stage3(_artifact, output, tracker, *_args):
        events.append(f"stage3:tracker{tracker.number}")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        save_image(output / "target.png", rgb)
        save_json(
            output / "tracking_attempts.json",
            {"stages": {"target_to_clean": {"attempts": []}}},
        )
        return np.zeros((4, 4), dtype=np.uint8), {
            "source_changed_proposal_ids": (1,),
            "target_changed_proposal_ids": (1,),
        }

    monkeypatch.setattr(inference, "run_cached_pair", fake_stage3)

    class FakeFeatureExtractor:
        def __init__(self, *_args):
            events.append("sam3_features:init")

        def release(self):
            events.append("sam3_features:release")

    monkeypatch.setattr(inference, "Sam3FeatureExtractor", FakeFeatureExtractor)

    descriptors = FeatureDescriptorBatch(
        vectors=np.ones((1, 1), dtype=np.float32),
        valid=np.asarray([True]),
        effective_cells=np.asarray([1.0], dtype=np.float32),
    )

    def fake_appearance(*_args, **_kwargs):
        events.append("sam3_features:compute")
        return AppearanceFeatures(
            source_map=np.ones((1, 2, 2), dtype=np.float16),
            target_map=np.ones((1, 2, 2), dtype=np.float16),
            source_features=descriptors,
            target_features=descriptors,
            calibration=SimilarityCalibration(
                0.65, False, 0, 0, None, None, None, "test"
            ),
            match_records=[],
        )

    monkeypatch.setattr(inference, "compute_appearance_features", fake_appearance)

    class FakeDino:
        def __init__(self, _config):
            events.append("dino:init")

        def feature_map(self, _image):
            return np.ones((1, 2, 2), dtype=np.float32)

        def release(self):
            events.append("dino:release")

    monkeypatch.setattr(inference, "Dinov2FeatureExtractor", FakeDino)
    monkeypatch.setattr(
        inference,
        "refine_with_motion_and_replacement_evidence",
        lambda labels, *_args, **_kwargs: (labels.copy(), {}),
    )

    def fake_feature_veto(*args, **_kwargs):
        tracker = args[8]
        events.append(f"feature_veto:tracker{tracker.number}")
        return args[6].copy(), {}

    monkeypatch.setattr(
        inference, "apply_feature_veto_direct_replacement", fake_feature_veto
    )

    class FakeSentinel:
        def __init__(self, *_args, **_kwargs):
            events.append("sentinel:init")

        def generate_with_feature_map(self, _image):
            events.append("sentinel:generate")
            return [proposal], np.ones((1, 2, 2), dtype=np.float16)

        def release(self):
            events.append("sentinel:release")

    monkeypatch.setattr(inference, "Sam3AutomaticMaskGenerator", FakeSentinel)
    monkeypatch.setattr(inference, "resolver_settings_from_config", lambda *_args: object())

    def fake_resolver(*args, **_kwargs):
        tracker = args[7]
        events.append(f"resolver:tracker{tracker.number}")
        return args[5].copy(), {}

    monkeypatch.setattr(inference, "resolve_real_image_associations", fake_resolver)
    monkeypatch.setattr(
        inference,
        "resolve_object_consistent_labels",
        lambda _artifact, labels, *_args, **_kwargs: (
            labels.copy(),
            labels.copy(),
            {},
        ),
    )

    class FakeCache:
        def lookup(self, _pair_id, _image0, _image1):
            return StageCacheDirs(
                stage1_dir=artifact,
                stage2_dir=stage2,
                stop_reason="stage3_artifact_unavailable",
            )

    cfg = load_config("configs/pipeline.yaml")
    result = inference.run_pair(
        tmp_path / "image0.png",
        tmp_path / "image1.png",
        tmp_path / "output",
        cfg,
        pair_id="pair",
        cache=FakeCache(),
    )

    assert tracker_count == 3
    assert events.index("tracker1:release") < events.index("sam3_features:init")
    assert events.index("tracker2:release") < events.index("sentinel:init")
    assert events.index("sentinel:release") < events.index("tracker3:init")
    assert "stage3:tracker1" in events
    assert "feature_veto:tracker2" in events
    assert "resolver:tracker3" in events
    assert result.diagnostics["cache"]["stages"]["stage1"]["used"] is True
    assert result.diagnostics["cache"]["stages"]["stage2"]["used"] is True
    assert result.diagnostics["cache"]["stages"]["stage3"]["used"] is False
    assert result.diagnostics["cache"]["stages"]["stage4"]["used"] is False
