"""Cheap, GPU-free checks that configs/pipeline.yaml still matches what
ocmask.inference.run_pair and the stage functions it calls expect.

These don't exercise any model -- they exist to catch config-schema drift
(a renamed/missing key) without needing GPU compute or checkpoints, as a
fast complement to the real end-to-end validation described in
docs/rewrite_plan.md.
"""

from __future__ import annotations

from pathlib import Path

from ocmask.config import load_config
from ocmask.inference import _proposal_generator_kwargs
from ocmask.stages.real_image_association_resolver import (
    AssociationResolverSettings,
    resolver_settings_from_config,
)
from ocmask.stages.sam3_proposals import Sam3AutomaticMaskGenerator

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml"


def _load() -> dict:
    return load_config(CONFIG_PATH)


def test_pipeline_config_has_every_stage_section() -> None:
    config = _load()
    for section in (
        "reconstruction",
        "sam3_proposals",
        "sam3_features",
        "dinov2_features",
        "motion_and_replacement_evidence",
        "feature_veto",
        "obvious_object_sentinel",
        "association_resolver",
        "object_consistent_masks",
    ):
        assert section in config, f"missing config section: {section}"


def test_resolver_settings_from_config_builds_from_the_real_config() -> None:
    settings = resolver_settings_from_config(_load())
    assert isinstance(settings, AssociationResolverSettings)
    # Spot-check a couple of values against the pinned config so a silent
    # renumbering (not just a missing key) is also caught.
    assert settings.minimum_valid_depth == 1.0e-06
    assert settings.different_identity_margin == 0.1
    assert settings.minimum_directional_iou == 0.3
    assert settings.reject_frame_border is True


def test_proposal_generator_kwargs_construct_both_generators() -> None:
    config = _load()
    stage2_kwargs = _proposal_generator_kwargs(config["sam3_proposals"]["proposals"])
    Sam3AutomaticMaskGenerator("unused-checkpoint-path.pt", **stage2_kwargs)

    stage9_kwargs = _proposal_generator_kwargs(
        config["obvious_object_sentinel"]["sam3"]["proposal_generation"],
        minimum_mask_area_key="minimum_mask_area_pixels",
    )
    Sam3AutomaticMaskGenerator("unused-checkpoint-path.pt", **stage9_kwargs)
