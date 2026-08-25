from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


REFERENCE = Path(__file__).resolve().parents[1] / "reference" / "headline_68_51"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_recovered_headline_bundle_checksums_and_frozen_metrics() -> None:
    for line in (REFERENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split(maxsplit=1)
        assert _sha256(REFERENCE / relative) == expected

    report = json.loads(
        (REFERENCE / "replay" / "report.json").read_text(encoding="utf-8")
    )
    frozen = json.loads(
        (REFERENCE / "replay" / "predictions_frozen.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["ground_truth_after_prediction_freeze"] is True
    assert frozen["ground_truth_used"] is False
    assert sum(len(split["pairs"]) for split in frozen["splits"].values()) == 25

    headline = report["overall"]["full_mask_candidate"]
    assert headline["binary"]["changed"]["iou"] == pytest.approx(
        0.4257805047750508
    )
    assert headline["binary"]["unchanged"]["iou"] == pytest.approx(
        0.944506676835935
    )
    assert headline["binary_miou"] == pytest.approx(0.6851435908054929)
    assert headline["multiclass"]["added"]["iou"] == pytest.approx(
        0.38612515380396967
    )
    assert headline["multiclass"]["removed"]["iou"] == pytest.approx(
        0.3151772017923002
    )
    assert headline["multiclass"]["moved"]["iou"] == pytest.approx(
        0.17554384188351024
    )
    assert headline["multiclass"]["replaced"]["iou"] == pytest.approx(
        0.20451374185173102
    )
    assert headline["multiclass_miou"] == pytest.approx(0.4051733232334892)
