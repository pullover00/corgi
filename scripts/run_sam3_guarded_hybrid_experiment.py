#!/usr/bin/env python3
"""Run the fixed-ten guarded SAM3/SAM2 hybrid experiment from frozen caches.

No model is loaded.  Phase one writes and hashes every prediction from the
immutable baseline labels, SAM3 feature decisions, proposal masks, and exact
SAM2 track rasters.  ChangeSim ground truth is opened only after the complete
prediction freeze has been written and verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from ocmask.changesim import MetricAccumulator, normalize_target
from ocmask.config import load_config
from ocmask.stages.sam3_guarded_hybrid import (
    HybridEvidence,
    compose_guarded_variant,
    make_hybrid_evidence,
)
from ocmask.stages.sam3_pairwise import load_proposal_cache
from ocmask.io import save_image, save_json
from ocmask.types import Label


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY
    / "configs/stages/changesim-sam3-guarded-hybrid-fixed10-densegrid96.yaml"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to the config's recommended_output directory.",
    )
    parser.add_argument(
        "--predictions-only",
        action="store_true",
        help="Freeze predictions without opening ChangeSim ground truth.",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _implementation_hash() -> str:
    paths = (
        Path(__file__).resolve(),
        REPOSITORY / "src/ocmask/stages/sam3_guarded_hybrid.py",
        REPOSITORY / "src/ocmask/stages/sam3_identity_location.py",
        REPOSITORY / "src/ocmask/stages/sam3_pairwise.py",
        REPOSITORY / "src/ocmask/masks.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPOSITORY)).encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _load_report_root(path: Path, name: str) -> tuple[dict, list[str]]:
    report_path = path / "report.json"
    selection_path = path / "selection.json"
    if not report_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError(f"{name} report/selection is incomplete: {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = list(
        json.loads(selection_path.read_text(encoding="utf-8"))["ids"]
    )
    if report.get("failures"):
        raise RuntimeError(f"{name} contains failed pairs")
    if not selected or len(set(selected)) != len(selected):
        raise RuntimeError(f"{name} must contain unique, non-empty pairs")
    return report, selected


def _load_proposal_cache_root(path: Path, name: str) -> tuple[dict, list[str]]:
    """Like ``_load_report_root``, but also accepts a proposals-only cache.

    ``run_sam3_pairwise_experiment.py --proposals-only`` (used by
    ``a3_overnight.py`` for this stage) intentionally exits before computing
    metrics, so it writes ``proposal_cache_complete.json`` instead of
    ``report.json``. This root only ever needs the cached proposal masks and
    the pair-id list -- never the metrics -- so a bare completion marker is
    just as valid a signal here as a full report.
    """
    report_path = path / "report.json"
    if report_path.is_file():
        return _load_report_root(path, name)

    marker_path = path / "proposal_cache_complete.json"
    selection_path = path / "selection.json"
    if not marker_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError(f"{name} report/selection is incomplete: {path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    selected = list(
        json.loads(selection_path.read_text(encoding="utf-8"))["ids"]
    )
    if not selected or len(set(selected)) != len(selected):
        raise RuntimeError(f"{name} must contain unique, non-empty pairs")
    if set(marker.get("pairs", [])) != set(selected):
        raise RuntimeError(f"{name} completion marker disagrees with its own selection")
    return {"failures": []}, selected


def _validate_inputs(config: dict) -> dict[str, Any]:
    roots = {
        "baseline": Path(config["parent_evaluation"]).resolve(),
        "identity": Path(config["identity_evaluation"]).resolve(),
        "proposals": Path(config["proposal_cache_parent"]).resolve(),
        "tracks": Path(config["moved_track_cache_parent"]).resolve(),
    }
    baseline_report, selection = _load_report_root(roots["baseline"], "baseline")
    identity_report, identity_selection = _load_report_root(
        roots["identity"], "identity"
    )
    track_report, track_selection = _load_report_root(roots["tracks"], "track cache")
    proposal_report, proposal_selection = _load_proposal_cache_root(
        roots["proposals"], "proposal cache"
    )
    for name, current in (
        ("identity", identity_selection),
        ("track cache", track_selection),
        ("proposal cache", proposal_selection),
    ):
        if current != selection:
            raise RuntimeError(f"{name} pair order differs from the frozen baseline")
    records = {record["id"]: record for record in baseline_report["pairs"]}
    if list(records) != selection and set(records) != set(selection):
        raise RuntimeError("baseline report and selection contain different pairs")
    return {
        "roots": roots,
        "selection": selection,
        "baseline_report": baseline_report,
        "identity_report": identity_report,
        "track_report": track_report,
        "proposal_report": proposal_report,
        "baseline_records": records,
    }


def _prepare_output(output: Path, config_hash: str) -> None:
    marker = output / "experiment.json"
    if output.exists() and any(output.iterdir()):
        if not marker.is_file():
            raise RuntimeError(f"refusing non-empty unowned output: {output}")
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("config_sha256") != config_hash:
            raise RuntimeError("output belongs to a different hybrid configuration")
    output.mkdir(parents=True, exist_ok=True)
    (output / "pairs").mkdir(exist_ok=True)


def _unpack_tracks(
    cache_dir: Path,
    parent_ledger: Mapping,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict]:
    """Load exact accepted SAM2 rasters and validate their parent decisions."""

    metadata_path = cache_dir / "metadata.json"
    arrays_path = cache_dir / "tracks.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("ground_truth_used"):
        raise RuntimeError(f"invalid tracking-cache protocol: {cache_dir}")
    if metadata.get("arrays_sha256") != _sha256_file(arrays_path):
        raise RuntimeError(f"tracking-cache array hash mismatch: {cache_dir}")
    with np.load(arrays_path) as cache:
        height, width = int(cache["height"]), int(cache["width"])

        def restore(
            prefix: str, stage: str
        ) -> dict[int, np.ndarray]:
            packed = np.asarray(cache[f"{prefix}_masks_packed"], np.uint8)
            masks = np.unpackbits(packed, axis=2, count=width).astype(bool)
            masks = masks[:, :height, :width]
            accepted = np.asarray(cache[f"{prefix}_accepted"], bool)
            rows = metadata[stage]
            parent_rows = parent_ledger["stages"][stage]["attempts"]
            if not (len(masks) == len(accepted) == len(rows) == len(parent_rows)):
                raise RuntimeError(f"{stage}: cache and parent record counts differ")
            output: dict[int, np.ndarray] = {}
            for mask, keep, row, parent in zip(
                masks, accepted, rows, parent_rows, strict=True
            ):
                proposal_id = int(row["proposal_id"])
                if proposal_id != int(parent["proposal_id"]):
                    raise RuntimeError(f"{stage}: proposal order changed")
                if bool(keep) != bool(row["accepted"]):
                    raise RuntimeError(f"{stage}: cache acceptance metadata differs")
                if bool(keep) != bool(parent["tracker_accepted"]):
                    raise RuntimeError(f"{stage}: parent acceptance differs")
                if _sha256_array(mask) != row["mask_sha256"]:
                    raise RuntimeError(f"{stage}: cached mask hash differs")
                if keep:
                    if proposal_id in output:
                        raise RuntimeError(f"{stage}: duplicate proposal ID")
                    output[proposal_id] = mask.copy()
            return output

        return (
            restore("forward", "source_to_target"),
            restore("backward", "target_to_source"),
            metadata,
        )


def _pair_paths(context: dict, pair_id: str) -> dict[str, Path]:
    roots = context["roots"]
    return {
        "baseline_labels": roots["baseline"] / "pairs" / pair_id / "labels.png",
        "baseline_ledger": roots["baseline"]
        / "pairs"
        / pair_id
        / "tracking_attempts.json",
        "baseline_result": roots["baseline"] / "pairs" / pair_id / "result.json",
        "identity_decisions": roots["identity"]
        / "pairs"
        / pair_id
        / "decisions.json",
        "identity_result": roots["identity"] / "pairs" / pair_id / "result.json",
        "identity_features": roots["identity"]
        / "pairs"
        / pair_id
        / "sam3_features.npz",
        "identity_feature_metadata": roots["identity"]
        / "pairs"
        / pair_id
        / "sam3_features.json",
        "source_proposals": roots["proposals"]
        / "pairs"
        / pair_id
        / "proposal_cache/source.npz",
        "target_proposals": roots["proposals"]
        / "pairs"
        / pair_id
        / "proposal_cache/target.npz",
        "proposal_metadata": roots["proposals"]
        / "pairs"
        / pair_id
        / "proposal_cache/metadata.json",
        "track_arrays": roots["tracks"]
        / "pairs"
        / pair_id
        / "tracking_cache/tracks.npz",
        "track_metadata": roots["tracks"]
        / "pairs"
        / pair_id
        / "tracking_cache/metadata.json",
    }


def _pair_fingerprint(paths: Mapping[str, Path], config_hash: str) -> str:
    """Hash every inference input, explicitly excluding ChangeSim ground truth."""

    digest = hashlib.sha256(bytes.fromhex(config_hash))
    for name, path in sorted(paths.items()):
        if not path.is_file():
            raise FileNotFoundError(f"missing hybrid input {name}: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _evidence_summary(evidence: HybridEvidence) -> dict[str, Any]:
    return {
        "replacement_pair_count": evidence.replacement_pair_count,
        "moved_feature_pair_count": evidence.moved_feature_pair_count,
        "forward_owner_count": evidence.forward_owner_count,
        "reverse_owner_count": evidence.reverse_owner_count,
        "confirmed_forward_count": evidence.confirmed_forward_count,
        "confirmed_reverse_count": evidence.confirmed_reverse_count,
        "replacement_candidate_pixels": int(evidence.replacement.sum()),
        "moved_forward_only_unverified_pixels": int(
            evidence.moved_forward_only_unverified.sum()
        ),
        "moved_reverse_only_unverified_pixels": int(
            evidence.moved_reverse_only_unverified.sum()
        ),
        "moved_confirmed_pixels": int(evidence.moved_confirmed.sum()),
        "moved_bidirectional_pixels": int(evidence.moved_bidirectional.sum()),
        "moved_unowned_parent_pixels": int(evidence.moved_unowned.sum()),
    }


def _table3(metrics: dict) -> dict[str, dict[str, float]]:
    return {
        "binary": {
            "changed": 100.0 * metrics["binary"]["changed"]["iou"],
            "unchanged": 100.0 * metrics["binary"]["unchanged"]["iou"],
            "miou": 100.0 * metrics["binary_miou"],
        },
        "multiclass": {
            **{
                name: 100.0 * metrics["multiclass"][name]["iou"]
                for name in ("added", "removed", "moved", "replaced", "unchanged")
            },
            "miou": 100.0 * metrics["multiclass_miou"],
        },
    }


def _manifest_targets(path: Path) -> dict[str, Path]:
    """Resolve target paths after the prediction freeze, without stratifying."""

    base = path.resolve().parent
    output: dict[str, Path] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        target = Path(row["target"])
        output[str(row["id"])] = (
            target if target.is_absolute() else (base / target).resolve()
        )
    return output


def _verify_prediction_freeze(freeze: dict, output: Path) -> None:
    for pair in freeze["pairs"]:
        for variant, record in pair["predictions"].items():
            path = output / record["relative_path"]
            if _sha256_file(path) != record["file_sha256"]:
                raise RuntimeError(
                    f"prediction changed after freeze: {pair['id']}/{variant}"
                )


def main() -> int:
    args = _arguments()
    config = load_config(args.config)
    context = _validate_inputs(config)
    variants = list(config["variants"])
    expected_variants = {
        "replacement_only",
        "moved_verification",
        "combined_guarded_hybrid",
    }
    if set(variants) != expected_variants or len(variants) != 3:
        raise ValueError("config must declare each guarded variant exactly once")
    output = (
        args.output.resolve()
        if args.output
        else Path(config["recommended_output"]).resolve()
    )
    immutable = [path.resolve() for path in context["roots"].values()]
    if any(output == root or root in output.parents for root in immutable):
        raise ValueError("hybrid output must be outside every immutable parent")

    config_hash = _sha256_json(config)
    implementation_hash = _implementation_hash()
    parent_hashes = {
        name: _sha256_file(
            root / "report.json"
            if (root / "report.json").is_file()
            else root / "proposal_cache_complete.json"
        )
        for name, root in context["roots"].items()
    }
    execution_hash = _sha256_json(
        {
            "config_sha256": config_hash,
            "implementation_sha256": implementation_hash,
            "parent_report_sha256": parent_hashes,
        }
    )
    _prepare_output(output, config_hash)
    save_json(
        output / "experiment.json",
        {
            "experiment_id": config["experiment_id"],
            "config": str(args.config.resolve()),
            "config_sha256": config_hash,
            "implementation_sha256": implementation_hash,
            "execution_sha256": execution_hash,
            "parent_report_sha256": parent_hashes,
            "ground_truth_used_in_inference": False,
            "selection_role": config["protocol"]["selection_role"],
        },
    )
    save_json(output / "selection.json", {"ids": context["selection"]})

    frozen_pairs = []
    run_started = time.perf_counter()
    for number, pair_id in enumerate(context["selection"], 1):
        pair_output = output / "pairs" / pair_id
        pair_output.mkdir(parents=True, exist_ok=True)
        paths = _pair_paths(context, pair_id)
        fingerprint = _pair_fingerprint(paths, config_hash)
        baseline = np.asarray(Image.open(paths["baseline_labels"]), np.uint8)
        if baseline.ndim != 2 or baseline.dtype != np.uint8:
            raise RuntimeError(f"{pair_id}: invalid parent label map")
        if not np.all(np.isin(baseline, [int(item) for item in Label])):
            raise RuntimeError(f"{pair_id}: parent label map contains unknown classes")

        source = load_proposal_cache(paths["source_proposals"])
        target = load_proposal_cache(paths["target_proposals"])
        source_masks = [np.asarray(item.mask, bool) for item in source]
        target_masks = [np.asarray(item.mask, bool) for item in target]
        decisions_payload = json.loads(
            paths["identity_decisions"].read_text(encoding="utf-8")
        )
        decisions = list(decisions_payload["matches"])
        ledger = json.loads(paths["baseline_ledger"].read_text(encoding="utf-8"))
        forward, reverse, track_metadata = _unpack_tracks(
            paths["track_arrays"].parent, ledger
        )
        # Validate that the track cache itself is the exact file included in
        # this pair fingerprint, not an adjacent stale directory.
        if track_metadata["arrays_sha256"] != _sha256_file(paths["track_arrays"]):
            raise RuntimeError(f"{pair_id}: track metadata points to other arrays")

        evidence = make_hybrid_evidence(
            baseline,
            decisions,
            source_masks,
            target_masks,
            forward,
            reverse,
            minimum_track_candidate_iou=float(
                config["moved_verification"]["minimum_track_candidate_iou"]
            ),
        )
        np.savez_compressed(
            pair_output / "evidence.npz",
            replacement=evidence.replacement,
            moved_forward_only_unverified=evidence.moved_forward_only_unverified,
            moved_reverse_only_unverified=evidence.moved_reverse_only_unverified,
            moved_confirmed=evidence.moved_confirmed,
            moved_bidirectional=evidence.moved_bidirectional,
            moved_unowned=evidence.moved_unowned,
        )

        predictions: dict[str, dict[str, Any]] = {}
        variant_diagnostics: dict[str, dict] = {}
        for variant in variants:
            labels, diagnostics = compose_guarded_variant(
                baseline, evidence, variant
            )
            variant_dir = pair_output / variant
            variant_dir.mkdir(exist_ok=True)
            labels_path = variant_dir / "labels.png"
            save_image(labels_path, labels)
            predictions[variant] = {
                "relative_path": str(labels_path.relative_to(output)),
                "file_sha256": _sha256_file(labels_path),
                "array_sha256": _sha256_array(labels),
                "binary_array_sha256": _sha256_array(
                    labels != int(Label.UNCHANGED)
                ),
            }
            variant_diagnostics[variant] = diagnostics.to_dict()

        baseline_binary_hash = _sha256_array(
            baseline != int(Label.UNCHANGED)
        )
        if any(
            record["binary_array_sha256"] != baseline_binary_hash
            for record in predictions.values()
        ):
            raise AssertionError(f"{pair_id}: a variant changed binary support")
        decision_record = {
            "id": pair_id,
            "execution_sha256": execution_hash,
            "pair_fingerprint_sha256": fingerprint,
            "baseline_labels_sha256": _sha256_file(paths["baseline_labels"]),
            "baseline_binary_array_sha256": baseline_binary_hash,
            "evidence": _evidence_summary(evidence),
            "variants": variant_diagnostics,
            "predictions": predictions,
            "ground_truth_used": False,
        }
        save_json(pair_output / "decisions.json", decision_record)
        frozen_pairs.append(decision_record)
        print(
            f"[predict {number}/{len(context['selection'])}] {pair_id}: "
            f"replace={variant_diagnostics['replacement_only']['total_relabelled']}, "
            f"moved={variant_diagnostics['moved_verification']['total_relabelled']}, "
            f"combined={variant_diagnostics['combined_guarded_hybrid']['total_relabelled']}",
            flush=True,
        )

    freeze = {
        "schema_version": 1,
        "execution_sha256": execution_hash,
        "selection": context["selection"],
        "variants": variants,
        "pairs": frozen_pairs,
        "prediction_count": len(frozen_pairs) * len(variants),
        "ground_truth_opened_before_freeze": False,
    }
    save_json(output / "predictions_frozen.json", freeze)
    _verify_prediction_freeze(freeze, output)
    print(f"[freeze] {freeze['prediction_count']} predictions hashed before GT", flush=True)
    if args.predictions_only:
        return 0

    # Evaluation phase: this is the first point at which target images open.
    targets = _manifest_targets(Path(config["manifest"]))
    accumulators = {
        "baseline": MetricAccumulator(),
        **{variant: MetricAccumulator() for variant in variants},
    }
    pair_reports = []
    for number, pair in enumerate(frozen_pairs, 1):
        _verify_prediction_freeze(freeze, output)
        pair_id = pair["id"]
        if pair_id not in targets:
            raise RuntimeError(f"{pair_id}: missing from manifest")
        target = normalize_target(targets[pair_id])
        baseline = np.asarray(
            Image.open(_pair_paths(context, pair_id)["baseline_labels"]), np.uint8
        )
        baseline_pair = MetricAccumulator()
        baseline_pair.add(baseline, target)
        accumulators["baseline"].add_confusion(baseline_pair.confusion)
        confusions = {"baseline": baseline_pair.confusion.tolist()}
        for variant in variants:
            record = pair["predictions"][variant]
            labels = np.asarray(Image.open(output / record["relative_path"]), np.uint8)
            if _sha256_file(output / record["relative_path"]) != record["file_sha256"]:
                raise RuntimeError(f"{pair_id}/{variant}: prediction changed")
            pair_accumulator = MetricAccumulator()
            pair_accumulator.add(labels, target)
            accumulators[variant].add_confusion(pair_accumulator.confusion)
            confusions[variant] = pair_accumulator.confusion.tolist()
        pair_reports.append(
            {
                "id": pair_id,
                "status": "success",
                "confusion": confusions,
                "evidence": pair["evidence"],
                "variants": pair["variants"],
                "parent_artifacts": str(
                    (context["roots"]["baseline"] / "pairs" / pair_id).resolve()
                ),
                "artifacts": str((output / "pairs" / pair_id).resolve()),
            }
        )
        print(f"[evaluate {number}/{len(frozen_pairs)}] {pair_id}", flush=True)

    metrics = {name: accumulator.compute() for name, accumulator in accumulators.items()}
    tables = {name: _table3(value) for name, value in metrics.items()}
    frozen_parent_table = context["baseline_report"]["table3_iou_percent"]
    for section in ("binary", "multiclass"):
        for name, expected in frozen_parent_table[section].items():
            if not np.isclose(tables["baseline"][section][name], expected, atol=1e-10):
                raise AssertionError(f"recomputed baseline metric differs: {section}/{name}")
    total_pixels = int(accumulators["baseline"].confusion.sum())
    expected_pixels = len(context["selection"]) * 640 * 480
    if total_pixels != expected_pixels:
        raise AssertionError(f"expected {expected_pixels:,} pixels, got {total_pixels}")

    aggregate_transitions = {
        variant: {
            key: int(sum(pair["variants"][variant][key] for pair in frozen_pairs))
            for key in (
                "added_to_replaced",
                "removed_to_replaced",
                "moved_to_added",
                "moved_to_removed",
                "total_relabelled",
            )
        }
        for variant in variants
    }
    report = {
        "protocol": {
            "dataset": "ChangeSim",
            "experiment": config["experiment_id"],
            "pairs_selected": len(context["selection"]),
            "pairs_succeeded": len(frozen_pairs),
            "evaluated_pixels": total_pixels,
            "seed": context["baseline_report"]["protocol"]["seed"],
            "selection_role": config["protocol"]["selection_role"],
            "predictions_frozen_before_gt": True,
            "ground_truth_used_in_inference": False,
            "binary_support_equal_for_all_variants": True,
        },
        "baseline": {
            "description": "Frozen SAM3 masks + SAM2 tracking parent",
            "metrics": metrics["baseline"],
            "table3_iou_percent": tables["baseline"],
        },
        "variants": {
            variant: {
                "metrics": metrics[variant],
                "table3_iou_percent": tables[variant],
                "transitions": aggregate_transitions[variant],
            }
            for variant in variants
        },
        "pairs": pair_reports,
        "failures": [],
        "elapsed_seconds": time.perf_counter() - run_started,
        "provenance": {
            "config_sha256": config_hash,
            "implementation_sha256": implementation_hash,
            "execution_sha256": execution_hash,
            "parent_report_sha256": parent_hashes,
            "prediction_freeze_sha256": _sha256_file(
                output / "predictions_frozen.json"
            ),
            "ground_truth_used_in_inference": False,
            "tuned_on_test": False,
            "selection_role": config["protocol"]["selection_role"],
        },
    }
    save_json(output / "report.json", report)
    print(json.dumps({name: tables[name] for name in ["baseline", *variants]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

