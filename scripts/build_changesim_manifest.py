#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ocmask.changesim import decode_target_array


def numeric_key(path: Path) -> tuple[int, str]:
    """Sort numeric frame stems naturally while tolerating unusual names."""
    try:
        return int(path.stem), path.name
    except ValueError:
        return 2**63 - 1, path.name


def relative_or_absolute(path: Path, manifest_dir: Path) -> str:
    """Prefer readable paths relative to the resulting manifest."""
    try:
        return str(path.resolve().relative_to(manifest_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def build_manifest(dataset_root: Path, output: Path, paper_test: bool = False) -> list[dict]:
    """Validate ChangeSim's paired layout and build one row per query frame."""
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)
    if paper_test:
        sequences = [
            dataset_root / f"Warehouse_{warehouse}" / f"Seq_{sequence}"
            for warehouse in range(6, 10)
            for sequence in range(2)
        ]
    else:
        sequences = sorted(path for path in dataset_root.glob("Seq_*") if path.is_dir())
    for sequence in sequences:
        if not sequence.is_dir():
            raise FileNotFoundError(f"Required sequence directory is missing: {sequence}")
        for image1 in sorted((sequence / "rgb").glob("*.png"), key=numeric_key):
            image0 = sequence / "t0" / "rgb" / image1.name
            target = sequence / "change_segmentation" / image1.name
            missing = [path for path in (image0, target) if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{sequence.name}/{image1.name} lacks paired files: "
                    + ", ".join(str(path) for path in missing)
                )
            raw = decode_target_array(np.asarray(Image.open(target)))
            classes = [int(value) for value in np.unique(raw) if int(value) != 0]
            rows.append(
                {
                    "id": f"{sequence.parent.name}_{sequence.name}_{image1.stem}",
                    "image0": relative_or_absolute(image0, output.parent),
                    "image1": relative_or_absolute(image1, output.parent),
                    "target": relative_or_absolute(target, output.parent),
                    "classes": classes,
                }
            )
    if not rows:
        raise ValueError(f"No Seq_*/rgb/*.png frames found below {dataset_root}")
    output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a GOLDILOCS ChangeSim JSONL manifest")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--paper-test",
        action="store_true",
        help="use Warehouse_6..9 Seq_0/Seq_1 (the paper's 8,212-pair protocol)",
    )
    args = parser.parse_args()
    output = args.output or args.dataset_root / "manifest.jsonl"
    rows = build_manifest(args.dataset_root, output, paper_test=args.paper_test)
    print(f"Wrote {len(rows)} pairs to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
