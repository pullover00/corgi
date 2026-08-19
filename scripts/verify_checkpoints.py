#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    """Hash large checkpoint files incrementally to bound memory use."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    """Record trusted local hashes or verify files against recorded values."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="checkpoints/manifest.json")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    path = Path(args.manifest)
    manifest = json.loads(path.read_text())
    failed = False
    for name, record in manifest["models"].items():
        checkpoint = Path(record["path"])
        if not checkpoint.exists():
            print(f"MISSING {name}: {checkpoint}")
            failed = True
            continue
        actual = sha256(checkpoint)
        if args.record:
            record["sha256"] = actual
            print(f"RECORDED {name}: {actual}")
        elif record["sha256"] != actual:
            print(f"MISMATCH {name}: expected {record['sha256']}, got {actual}")
            failed = True
        else:
            print(f"OK {name}: {actual}")
    if args.record:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
