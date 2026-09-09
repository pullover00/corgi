#!/usr/bin/env python3
"""Port a method config to another machine WITHOUT touching method settings.

Copies only the machine-specific keys (checkpoint/source paths, and the SAM2
``vos_optimized`` flag, which must be false on Blackwell GPUs) from a config
that already works on the target machine into a copy of the method config,
then prints every key that changed so the port is auditable. Thresholds and
every other method setting come from --base untouched.

    python scripts/make_machine_config.py --base configs/ablate_v10_no_dino.yaml \
        --machine configs/pipeline_remote.yaml --out configs/ablate_v10_no_dino_remote.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

MACHINE_KEYS = [
    ("reconstruction", "vggt_omega_checkpoint"),
    ("reconstruction", "vggt_omega_root"),
    ("reconstruction_mast3r", "mast3r_root"),
    ("reconstruction_mast3r", "mast3r_checkpoint"),
    ("refine", "di2fix_root"),
    ("sam3_proposals", "sam3_source"),
    ("sam3_proposals", "sam3_image_checkpoint"),
    ("dinov2_features", "dinov2", "checkpoint"),
    ("dinov2_features", "dinov2", "source"),
    ("sam2_tracking", "sam2", "checkpoint"),
    ("sam2_tracking", "sam2", "model_cfg"),
    ("sam2_tracking", "sam2", "vos_optimized"),
]


def get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None, False
        d = d[k]
    return d, True


def put(d, path, value):
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="method config (thresholds etc. are taken from here)")
    ap.add_argument("--machine", type=Path, required=True, help="a config known to work on the target machine (paths are taken from here)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    base = yaml.safe_load(args.base.read_text())
    machine = yaml.safe_load(args.machine.read_text())
    changed = []
    for path in MACHINE_KEYS:
        new, present = get(machine, path)
        if not present:
            continue
        old, _ = get(base, path)
        if old != new:
            put(base, path, new)
            changed.append((".".join(path), old, new))
    header = (f"# Machine port of {args.base.name} -- method settings unchanged; only the keys below\n"
              f"# were copied from {args.machine.name} (see scripts/make_machine_config.py).\n")
    for key, old, new in changed:
        header += f"#   {key}: {old!r} -> {new!r}\n"
    args.out.write_text(header + yaml.safe_dump(base, sort_keys=False))
    print(f"wrote {args.out}; {len(changed)} machine key(s) changed:")
    for key, old, new in changed:
        print(f"  {key}: {old!r} -> {new!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
