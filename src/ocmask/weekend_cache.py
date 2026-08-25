"""Optional reuse of a pre-computed ``a3_overnight.py``-style cache.

This lets :func:`ocmask.inference.run_pair` reuse stages 1-3 and the stage-4
SAM3 dense features (the current CPU calibration/classification still reruns)
for a ChangeSim pair when a previous, independent job already
computed them with an identical configuration, instead of recomputing them
from scratch. It is strictly opt-in (nothing calls this module unless a
caller explicitly builds a :class:`ChangesimWeekendCache` and passes it to
``run_pair``) and every lookup is validated at the point of use -- a
missing file, a failed pair, or a configuration mismatch silently falls
back to "no cache for this pair/stage", never to a stale or incompatible
result. See ``docs/cache_audit.md`` for the full audit this module's
validation logic is based on: which stages the cache actually covers
(stages 1-4 only -- nothing for stages 5-11), how complete each stage is
per shard, and the exact config-compatibility checks that were run against
real cached artifacts before this module was written.

Stage 3 needs one field the cache predates: ``run_cached_pair``'s
``source_changed_proposal_ids``/``target_changed_proposal_ids`` diagnostics
were added after this cache was generated, so its per-pair
``diagnostics.json`` does not have them. They are instead re-derived from
``tracking_attempts.json`` (present in every cached stage-3 pair, in both
the cache and a fresh run, since ``run_cached_pair`` always writes it) --
see :func:`changed_proposal_ids_from_tracking_attempts`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .io import load_rgb

# Config keys a cached artifact's own recorded config is allowed to carry
# that the live merged pipeline.yaml does not (or vice versa) without that
# alone being treated as an incompatibility -- purely structural/bookkeeping
# differences between "one YAML file per stage" (the cache's origin) and
# "one merged pipeline.yaml" (this repository), never an algorithmic
# parameter.
_IGNORED_CONFIG_KEYS = frozenset({"schema_version"})


def mast3r_preprocessed_rgb(
    path: str | Path,
    reconstruction_config: Mapping[str, Any],
) -> np.ndarray:
    """Reproduce the RGB bytes stored for a current MASt3R input.

    This is the pinned, model-free equivalent of the image path used by
    :class:`~ocmask.adapters.mast3r.Mast3rAdapter`: ``load_rgb`` first makes
    the configured pipeline-size RGB image, DUSt3R resizes its long edge to
    512 and center-crops to patch-16 dimensions, and ``ImgNorm`` is converted
    back through ``SparseGA.imgs`` before the reconstruction is serialized.
    The float32 round trip is intentional; converting the resized PIL image
    directly to uint8 differs by one for many pixels.

    Only NumPy and Pillow are used.  In particular, validating a legacy cache
    never imports torch, MASt3R, or DUSt3R.
    """

    image_config = reconstruction_config["image"]
    pipeline_size = (int(image_config["width"]), int(image_config["height"]))
    image = Image.fromarray(load_rgb(path, pipeline_size))

    # Exact copy of DUSt3R's size=512, square_ok=False, patch_size=16 path.
    long_edge = max(image.size)
    resampling = (
        Image.Resampling.LANCZOS
        if long_edge > 512
        else Image.Resampling.BICUBIC
    )
    resized_size = tuple(int(round(value * 512 / long_edge)) for value in image.size)
    image = image.resize(resized_size, resampling)
    width, height = image.size
    center_x, center_y = width // 2, height // 2
    half_width = ((2 * center_x) // 16) * 16 / 2
    half_height = ((2 * center_y) // 16) * 16 / 2
    if width == height:
        half_height = 3 * half_width / 4
    image = image.crop(
        (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        )
    )

    # torchvision ToTensor + Normalize((.5,) * 3, (.5,) * 3), followed by
    # SparseGA's ``tensor * .5 + .5`` and the adapter's ``* 255`` truncation.
    tensor = np.asarray(image, dtype=np.uint8).astype(np.float32) / np.float32(255)
    normalized = (tensor - np.float32(0.5)) / np.float32(0.5)
    rgb = normalized * np.float32(0.5) + np.float32(0.5)
    return np.clip(rgb * np.float32(255), 0, 255).astype(np.uint8)


def _config_subset_matches(live: Mapping[str, Any], cached: Mapping[str, Any]) -> bool:
    """True if every key in ``live`` is present in ``cached`` with an equal value.

    Deliberately one-directional (``cached`` may have extra keys ``live``
    doesn't, e.g. per-stage bookkeeping fields like ``experiment_id`` or
    ``recommended_output`` that never affect computation) -- see this
    module's docstring and ``docs/cache_audit.md`` for the real diff this
    was verified against.
    """

    for key, value in live.items():
        if key in _IGNORED_CONFIG_KEYS:
            continue
        if cached.get(key) != value:
            return False
    return True


def changed_proposal_ids_from_tracking_attempts(
    tracking_attempts: Mapping[str, Any], stage: str
) -> tuple[int, ...]:
    """Recover ``run_cached_pair``'s changed-proposal-ID set from its ledger.

    ``stage`` is ``"source_to_clean"`` or ``"target_to_clean"``. A proposal
    counts as changed exactly when the clean-render gate rejected it
    (``post_consistency_gate_accepted`` is false) -- the same condition
    ``run_cached_pair`` itself uses to build ``source_changed``/
    ``target_changed`` before this diagnostic field existed.
    """

    attempts = tracking_attempts.get("stages", {}).get(stage, {}).get("attempts", [])
    return tuple(
        int(attempt["proposal_id"]) for attempt in attempts if not attempt["post_consistency_gate_accepted"]
    )


@dataclass(frozen=True)
class StageCacheDirs:
    """Validated, ready-to-use cached artifact directories for one pair.

    Each field is either a directory known to contain everything the
    corresponding ``run_pair`` stage needs and to match the live config, or
    ``None`` (no valid cache -- compute that stage normally). Later stages
    are only ever populated if every earlier one was too: stage 2's
    proposals were generated from stage 1's *exact* cached reconstruction
    bytes, not a freshly recomputed (and possibly floating-point-nonidentical)
    one, so reusing stage 2+ without also reusing stage 1 for the same pair
    would risk a silent proposal-ordering/count mismatch downstream.
    """

    stage1_dir: Path | None = None
    stage2_dir: Path | None = None
    stage3_dir: Path | None = None
    stage4_dir: Path | None = None
    stop_reason: str = "cache_not_requested"
    """Why the validated cache prefix stops where it does."""


class ChangesimWeekendCache:
    """Look up and validate cached stages 1-4 artifacts from a job directory.

    ``cache_root`` is a job directory in the shape ``a3_overnight.py``
    writes (``job.json`` + ``shards/shard-NNNN/{manifest.jsonl,
    configs/*.yaml, outputs/<stage>/...}``). ``config`` is the live merged
    ``pipeline.yaml`` mapping ``run_pair`` is about to use -- every lookup
    checks the cached artifact's own recorded configuration against it
    before returning a path.
    """

    def __init__(self, cache_root: str | Path, config: Mapping[str, Any]) -> None:
        self.root = Path(cache_root)
        self.config = config
        job_path = self.root / "job.json"
        if not job_path.is_file():
            raise FileNotFoundError(f"not an a3_overnight.py-style cache (missing job.json): {self.root}")
        self._job = json.loads(job_path.read_text(encoding="utf-8"))
        self._pair_shard: dict[str, Path] = {}
        self._pair_manifest: dict[str, dict[str, Any]] = {}
        for shard in self._job.get("shards", []):
            shard_dir = Path(shard["path"])
            manifest_path = shard_dir / "manifest.jsonl"
            if not manifest_path.is_file():
                continue
            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self._pair_shard[row["id"]] = shard_dir
                self._pair_manifest[row["id"]] = row
        self._shard_config_ok: dict[Path, dict[str, bool]] = {}
        self._stage1_progress: dict[Path, dict[str, dict]] = {}

    # -- shard-level config validation, cached per shard -----------------

    def _stage1_config_matches(self, cached_config_json: Mapping[str, Any]) -> bool:
        live = dict(self.config["reconstruction"])
        # The historical stage-1 artifacts predate these provenance-only
        # fields.  If a cache does record one, require it to match; otherwise
        # omit only that absent identity field from the config comparison.
        # Checkpoint paths and every setting that affects computation remain
        # mandatory, while the current run manifest independently fingerprints
        # the checkpoint files used by a fresh run.
        legacy_optional_fields = {
            "mast3r": ("checkpoint_sha256", "source", "source_commit"),
            "sam2": ("checkpoint_sha256", "source_commit"),
        }
        for component, optional_fields in legacy_optional_fields.items():
            live_component = live.get(component)
            cached_component = cached_config_json.get(component)
            if not isinstance(live_component, Mapping) or not isinstance(
                cached_component, Mapping
            ):
                continue
            live_component = dict(live_component)
            for field in optional_fields:
                if field not in cached_component:
                    live_component.pop(field, None)
            live[component] = live_component
        return _config_subset_matches(live, cached_config_json)

    def _validate_shard_config(self, shard_dir: Path, stage: str) -> bool:
        cache = self._shard_config_ok.setdefault(shard_dir, {})
        if stage in cache:
            return cache[stage]
        ok = self._compute_shard_config_validity(shard_dir, stage)
        cache[stage] = ok
        return ok

    def _compute_shard_config_validity(self, shard_dir: Path, stage: str) -> bool:
        import yaml

        if stage == "stage2":
            path = shard_dir / "configs" / "b.yaml"
            if not path.is_file():
                return False
            cached = yaml.safe_load(path.read_text(encoding="utf-8"))
            live_proposals = self.config["sam3_proposals"]
            return (
                cached.get("proposals") == live_proposals["proposals"]
                and cached.get("sam3_image_checkpoint_sha256") == live_proposals.get("sam3_image_checkpoint_sha256")
            )
        if stage == "stage3":
            path = shard_dir / "configs" / "c.yaml"
            if not path.is_file():
                return False
            cached = yaml.safe_load(path.read_text(encoding="utf-8"))
            live_tracking = self.config["reconstruction"]["tracking"]
            cached_tracking = cached.get("baseline_protocol", {}).get("tracking", {})
            return (
                cached.get("proposals") == self.config["sam3_proposals"]["proposals"]
                and all(live_tracking.get(k) == v for k, v in cached_tracking.items())
            )
        if stage == "stage4":
            path = shard_dir / "configs" / "d.yaml"
            if not path.is_file():
                return False
            cached = yaml.safe_load(path.read_text(encoding="utf-8"))
            live = self.config["sam3_features"]
            cached_sam3 = dict(cached.get("sam3") or {})
            live_sam3 = dict(live["sam3"])
            # Filesystem locations are deployment details. The model identity
            # is the required checkpoint SHA plus source commit and the
            # remaining inference settings; a Hugging Face snapshot symlink
            # and its content-addressed blob must be cache-equivalent.
            for location_key in ("source", "checkpoint"):
                cached_sam3.pop(location_key, None)
                live_sam3.pop(location_key, None)
            return (
                cached_sam3 == live_sam3
                and cached.get("matching") == live["matching"]
                and cached.get("classification") == live["classification"]
            )
        raise ValueError(f"unknown stage {stage!r}")

    # -- stage 1: a_baseline_cache (progress.jsonl, content-hash dirs) ---

    def _stage1_progress_for_shard(self, shard_dir: Path) -> dict[str, dict]:
        if shard_dir in self._stage1_progress:
            return self._stage1_progress[shard_dir]
        progress_path = shard_dir / "outputs" / "a_baseline_cache" / "progress.jsonl"
        best: dict[str, dict] = {}
        if progress_path.is_file():
            for line in progress_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pair_id = record.get("id") or record.get("pair_id")
                if pair_id and record.get("status") == "success":
                    best[pair_id] = record
        self._stage1_progress[shard_dir] = best
        return best

    @staticmethod
    def _same_resolved_path(first: str | Path, second: str | Path) -> bool:
        return Path(first).resolve() == Path(second).resolve()

    def _manifest_inputs_match(
        self,
        pair_id: str,
        image0_path: str | Path,
        image1_path: str | Path,
    ) -> bool:
        """Bind a pair ID to the manifest's paths (semantic bytes are checked later)."""

        row = self._pair_manifest.get(pair_id)
        if row is None:
            return False
        try:
            return self._same_resolved_path(row["image0"], image0_path) and self._same_resolved_path(
                row["image1"], image1_path
            )
        except (KeyError, TypeError):
            return False

    def _stage1_dir(
        self,
        pair_id: str,
        shard_dir: Path,
        image0_path: str | Path,
        image1_path: str | Path,
    ) -> tuple[Path | None, str | None]:
        record = self._stage1_progress_for_shard(shard_dir).get(pair_id)
        if record is None:
            return None, "stage1_progress_unavailable"
        artifacts = record.get("artifacts")
        if not artifacts:
            return None, "stage1_artifact_unavailable"
        pair_dir = Path(artifacts)
        required = (
            "reconstruction.npz",
            "geometry.npz",
            "render_0_to_1.png",
            "render_clean_to_1.png",
            "config.json",
            "inputs.json",
        )
        if not pair_dir.is_dir() or not all((pair_dir / name).is_file() for name in required):
            return None, "stage1_artifact_unavailable"
        try:
            cached_config = json.loads((pair_dir / "config.json").read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None, "stage1_config_unreadable"
        if not self._stage1_config_matches(cached_config):
            return None, "stage1_config_mismatch"
        try:
            cached_inputs = json.loads(
                (pair_dir / "inputs.json").read_text(encoding="utf-8")
            )
            inputs_match = self._same_resolved_path(
                cached_inputs["image0"], image0_path
            ) and self._same_resolved_path(cached_inputs["image1"], image1_path)
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return None, "stage1_recorded_inputs_unreadable"
        if not inputs_match:
            return None, "stage1_recorded_input_mismatch"

        # Legacy inputs.json records paths but no content hash.  Bind those
        # paths to the actual computation by reproducing the current pinned
        # MASt3R preprocessing and comparing the resulting uint8 arrays with
        # the exact images serialized into the cached reconstruction.
        try:
            expected0 = mast3r_preprocessed_rgb(image0_path, self.config["reconstruction"])
            expected1 = mast3r_preprocessed_rgb(image1_path, self.config["reconstruction"])
            with np.load(pair_dir / "reconstruction.npz", allow_pickle=False) as reconstruction:
                image0_matches = np.array_equal(reconstruction["image0"], expected0)
                image1_matches = np.array_equal(reconstruction["image1"], expected1)
        except Exception:
            # Corrupt archives, unreadable/invalid images, and malformed config
            # are ordinary cache misses, never reasons to abort fresh inference.
            return None, "stage1_semantic_input_validation_failed"
        if not image0_matches or not image1_matches:
            return None, "stage1_semantic_input_mismatch"
        return pair_dir, None

    # -- stages 2-4: pair_id-named directories, report.json/selection.json

    def _stage234_dir(self, pair_id: str, shard_dir: Path, stage_name: str, required: tuple[str, ...]) -> Path | None:
        pair_dir = shard_dir / "outputs" / stage_name / "pairs" / pair_id
        if not pair_dir.is_dir() or not all((pair_dir / name).is_file() for name in required):
            return None
        report_path = shard_dir / "outputs" / stage_name / "report.json"
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
            failed_ids = {failure.get("id") for failure in report.get("failures", [])}
            if pair_id in failed_ids:
                return None
        return pair_dir

    def lookup(
        self,
        pair_id: str,
        image0_path: str | Path,
        image1_path: str | Path,
    ) -> StageCacheDirs:
        """Return whichever prefix of stages 1-4 has a valid cached result.

        Never raises for an ordinary "not cached"/"cache invalid" outcome
        (a missing file, a config mismatch, a failed pair) -- those all
        just produce ``None`` for that stage and every stage after it. The
        lookup is deliberately bound to both resolved input paths and to their
        exact post-preprocessing uint8 arrays stored in ``reconstruction.npz``:
        a matching pair ID/path alone is not enough to reuse stale artifacts.
        Raw files that decode and preprocess identically are semantically safe.
        """

        shard_dir = self._pair_shard.get(pair_id)
        if shard_dir is None:
            return StageCacheDirs(stop_reason="pair_id_not_found")
        if not self._manifest_inputs_match(pair_id, image0_path, image1_path):
            return StageCacheDirs(stop_reason="cache_manifest_input_mismatch")

        stage1_dir, stage1_stop_reason = self._stage1_dir(
            pair_id, shard_dir, image0_path, image1_path
        )
        if stage1_dir is None:
            return StageCacheDirs(stop_reason=stage1_stop_reason or "stage1_invalid")

        if not self._validate_shard_config(shard_dir, "stage2"):
            return StageCacheDirs(
                stage1_dir=stage1_dir, stop_reason="stage2_config_mismatch"
            )
        stage2_dir = self._stage234_dir(
            pair_id, shard_dir, "b_sam3_sam31", ("proposal_cache/source.npz", "proposal_cache/target.npz")
        )
        if stage2_dir is None:
            return StageCacheDirs(
                stage1_dir=stage1_dir, stop_reason="stage2_artifact_unavailable"
            )

        if not self._validate_shard_config(shard_dir, "stage3"):
            return StageCacheDirs(
                stage1_dir=stage1_dir,
                stage2_dir=stage2_dir,
                stop_reason="stage3_config_mismatch",
            )
        stage3_dir = self._stage234_dir(
            pair_id, shard_dir, "c_sam3_sam2", ("labels.png", "diagnostics.json", "tracking_attempts.json", "target.png")
        )
        if stage3_dir is None:
            return StageCacheDirs(
                stage1_dir=stage1_dir,
                stage2_dir=stage2_dir,
                stop_reason="stage3_artifact_unavailable",
            )

        if not self._validate_shard_config(shard_dir, "stage4"):
            return StageCacheDirs(
                stage1_dir=stage1_dir,
                stage2_dir=stage2_dir,
                stage3_dir=stage3_dir,
                stop_reason="stage4_config_mismatch",
            )
        stage4_dir = self._stage234_dir(
            pair_id, shard_dir, "d_identity", ("sam3_features.npz", "decisions.json")
        )
        return StageCacheDirs(
            stage1_dir=stage1_dir,
            stage2_dir=stage2_dir,
            stage3_dir=stage3_dir,
            stage4_dir=stage4_dir,
            stop_reason=(
                "all_available" if stage4_dir is not None else "stage4_artifact_unavailable"
            ),
        )
