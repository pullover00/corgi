from __future__ import annotations

import sys
from pathlib import Path


def mast3r_checkout() -> Path:
    """Return the expected pinned MASt3R checkout inside this repository."""
    return Path(__file__).resolve().parents[1] / "mast3r"


def configure_mast3r_paths() -> Path:
    """Expose upstream MASt3R and its DUSt3R submodule for normal imports.

    MASt3R deliberately ships without Python packaging metadata. Its own demos
    run from the repository root, which implicitly adds these directories to
    ``sys.path``. We perform the equivalent operation explicitly and locally.
    """
    checkout = mast3r_checkout()
    package = checkout / "mast3r"
    dust3r_package = checkout / "dust3r" / "dust3r"
    if not package.is_dir() or not dust3r_package.is_dir():
        raise RuntimeError(
            f"MASt3R sources are incomplete at {checkout}. "
            "Run scripts/bootstrap_models.sh first."
        )

    # Package discovery needs the parents of ``mast3r/`` and ``dust3r/``.
    for source_root in (checkout, checkout / "dust3r"):
        value = str(source_root)
        if value not in sys.path:
            sys.path.insert(0, value)
    return checkout

