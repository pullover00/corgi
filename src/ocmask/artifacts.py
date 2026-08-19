"""Small, pure I/O helpers shared by the stage-runner scripts.

These read pinned intermediate artifacts (manifests, prior-stage selections,
and cached SAM3/DINOv2 dense feature maps) without touching ground truth or
running any model. They were originally duplicated inside a "Branch B2"
diagnostic runner that also carried an unused reciprocal-matching/rigid-pose
estimator (see ``ocmask.stages.branch_b2`` for what was dropped); only these
functions were ever load-bearing for the final pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def resolve_path(path: str | Path, repository: Path) -> Path:
    """Resolve a config path relative to the repository root unless absolute."""

    value = Path(path)
    return value.resolve() if value.is_absolute() else (repository / value).resolve()


def load_feature_maps(root: Path, pair_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the dense source/target feature maps cached by a features stage."""

    path = root / "pairs" / pair_id / "sam3_features.npz"
    with np.load(path) as cache:
        return np.asarray(cache["source"], np.float32), np.asarray(cache["target"], np.float32)


def manifest_targets(path: Path) -> dict[str, Path]:
    """Read ``{pair_id: ground_truth_path}`` from a ChangeSim manifest without opening images."""

    base = path.parent
    output: dict[str, Path] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        target = Path(row["target"])
        output[str(row["id"])] = target if target.is_absolute() else (base / target).resolve()
    return output


def parent_records(root: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Load a prior stage's frozen selection and per-pair report records."""

    selection = list(json.loads((root / "selection.json").read_text(encoding="utf-8"))["ids"])
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    records = {str(record["id"]): record for record in report["pairs"]}
    if report.get("failures") or set(selection) != set(records):
        raise RuntimeError(f"incomplete parent: {root}")
    return selection, records
