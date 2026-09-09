"""Structured, provenance-tracked cache for every expensive pipeline stage.

Layout::

    <root>/<dataset>/<scene>/<experiment>/reference_reconstruction/   (scene-level)
    <root>/<dataset>/<scene>/<query>/<experiment>/<stage>/            (query-level)

Each stage directory holds its own artifacts plus a ``manifest.json``
recording *what produced them*: the hash of the config subset the stage
actually depends on, the hashes of its upstream stages, the git revision,
and the file list. ``is_fresh`` refuses a cached stage whose config subset
or upstream chain has changed, so an intermediate is never silently reused
across an incompatible configuration -- while a change confined to, say,
``three_image_comparison`` leaves the reconstruction and refine caches
valid, which is the whole point of caching them.

Stage dependencies are declared in ``STAGE_CONFIG_DEPS`` / ``STAGE_UPSTREAM``
rather than hashing the whole config, because hashing everything would
invalidate a 40-minute VGGT-Omega reconstruction every time a matching
threshold moves.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Ordered coarse-to-fine. Each entry names the config sections whose contents
# can change that stage's output; anything not listed is irrelevant to it.
STAGE_CONFIG_DEPS: dict[str, tuple[str, ...]] = {
    "reference_reconstruction": ("reconstruction",),
    "localization": ("reconstruction",),
    "render": ("reconstruction",),
    "refine": ("refine",),
    "proposals": ("sam3_proposals",),
    "descriptors": ("sam3_proposals", "dinov2_features"),
    "tracking": ("sam2_tracking",),
    "resolution": ("three_image_comparison",),
    "labels": ("three_image_comparison",),
    "metrics": (),
}

# Which earlier stages each stage's output depends on. A change anywhere
# upstream propagates through the recorded hashes and invalidates this stage.
STAGE_UPSTREAM: dict[str, tuple[str, ...]] = {
    "reference_reconstruction": (),
    "localization": ("reference_reconstruction",),
    "render": ("localization",),
    "refine": ("render",),
    "proposals": ("refine",),
    "descriptors": ("proposals",),
    "tracking": ("proposals",),
    "resolution": ("descriptors", "tracking"),
    "labels": ("resolution",),
    "metrics": ("labels",),
}

SCENE_LEVEL_STAGES = frozenset({"reference_reconstruction"})


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def code_version() -> str:
    """Git revision, with a -dirty suffix when the tree has uncommitted
    changes -- a cached stage produced from edited-but-uncommitted code must
    be identifiable as such."""
    try:
        root = Path(__file__).resolve().parents[2]
        rev = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if rev.returncode != 0:
            return "unknown"
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10)
        return rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


@dataclass
class ArtifactStore:
    """Addresses one (dataset, scene, query, experiment) cell of the cache."""

    root: Path
    dataset: str
    scene: str
    experiment: str
    query: str | None = None
    # Where to look for an upstream stage this experiment does not own.
    # Reconstruction/render/refine depend only on the reconstruction and
    # refine config, so several detect-side experiments (m1..m5) share one
    # copy under experiment "shared" rather than each recomputing VGGT-Omega;
    # their manifests still record the shared stages' hashes as upstream, so
    # a reconstruction change invalidates every experiment built on it.
    fallback: "ArtifactStore | None" = None

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def stage_dir(self, stage: str) -> Path:
        if stage not in STAGE_CONFIG_DEPS:
            raise KeyError(f"unknown stage {stage!r}; known: {sorted(STAGE_CONFIG_DEPS)}")
        base = self.root / self.dataset / self.scene
        if stage in SCENE_LEVEL_STAGES:
            return base / self.experiment / stage
        if self.query is None:
            raise ValueError(f"stage {stage!r} is query-level but this store has no query")
        return base / self.query / self.experiment / stage

    def manifest_path(self, stage: str) -> Path:
        return self.stage_dir(stage) / "manifest.json"

    def resolve(self, stage: str) -> "ArtifactStore":
        """The store that actually owns ``stage``: this one if it has a
        manifest for it, else the fallback (recursively)."""
        if self.manifest_path(stage).exists() or self.fallback is None:
            return self
        return self.fallback.resolve(stage)

    def read_manifest(self, stage: str) -> dict[str, Any] | None:
        path = self.resolve(stage).manifest_path(stage)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    def config_hash(self, stage: str, config: dict[str, Any]) -> str:
        subset = {k: config.get(k) for k in STAGE_CONFIG_DEPS[stage]}
        return _stable_hash(subset)

    def upstream_hashes(self, stage: str) -> dict[str, str | None]:
        """Recorded config hashes of this stage's inputs, as they currently
        sit on disk. A missing upstream manifest yields None, which will not
        match anything recorded, so the stage reads as stale."""
        out: dict[str, str | None] = {}
        for name in STAGE_UPSTREAM[stage]:
            manifest = self.read_manifest(name)
            out[name] = manifest.get("config_hash") if manifest else None
        return out

    def is_fresh(self, stage: str, config: dict[str, Any], required_files: Iterable[str] = ()) -> bool:
        """True only when the stage was produced by this exact config subset
        AND its whole ancestor chain is itself fresh AND its files exist.

        The recursion is the point: checking only the immediate parent's
        recorded hash lets a change deep in the chain pass unnoticed, because
        an untouched intermediate keeps matching the hash its child recorded.
        Staleness has to propagate all the way down or a reconstruction
        config change would silently leave stale refined renders in use.
        """
        return self.stale_reason(stage, config, required_files) is None

    def stale_reason(self, stage: str, config: dict[str, Any],
                     required_files: Iterable[str] = ()) -> str | None:
        """Human-readable explanation for a cache miss, for run logs.
        Returns None when the stage is usable."""
        for name in STAGE_UPSTREAM[stage]:
            upstream_reason = self.resolve(name).stale_reason(name, config)
            if upstream_reason is not None:
                return f"upstream {name} is stale ({upstream_reason})"
        manifest = self.read_manifest(stage)
        if manifest is None:
            return "no manifest (never computed)"
        if manifest.get("config_hash") != self.config_hash(stage, config):
            return (f"config for {STAGE_CONFIG_DEPS[stage]} changed "
                    f"({manifest.get('config_hash')} -> {self.config_hash(stage, config)})")
        if manifest.get("upstream") != self.upstream_hashes(stage):
            return f"upstream hashes changed ({manifest.get('upstream')} -> {self.upstream_hashes(stage)})"
        directory = self.resolve(stage).stage_dir(stage)
        names = list(required_files) or manifest.get("files", [])
        missing = [n for n in names if not (directory / n).exists()]
        if missing:
            return f"missing files: {missing}"
        return None

    def commit(self, stage: str, config: dict[str, Any], files: Iterable[str],
               extra: dict[str, Any] | None = None) -> Path:
        """Record the manifest for a stage whose files have just been written.
        The full config is stored alongside, not just the hashed subset, so an
        old run stays interpretable after the config file itself has moved on."""
        directory = self.stage_dir(stage)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config_used.json").write_text(json.dumps(config, indent=2, default=str))
        manifest = {
            "stage": stage,
            "dataset": self.dataset,
            "scene": self.scene,
            "query": self.query,
            "experiment": self.experiment,
            "config_hash": self.config_hash(stage, config),
            "config_sections_hashed": list(STAGE_CONFIG_DEPS[stage]),
            "upstream": self.upstream_hashes(stage),
            "code_version": code_version(),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": sorted(files),
        }
        if extra:
            manifest["extra"] = extra
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return directory
