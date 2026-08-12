"""Validation-only empirical Oracle ceiling over saved fixed-strength outputs.

This diagnostic never performs diffusion generation, training, or test-split
evaluation.  Selection is restricted to the saved pointwise-median prediction
for each fixed-strength action within the same run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.oracle_metrics import metrics_from_embeddings


FIXED_PATTERN = re.compile(r"^fixed_(\d+)$")
REQUIRED_FIELDS = {
    "predictions",
    "targets",
    "evaluation_sample_ids",
    "evaluation_split",
    "query_caption_ids",
    "query_captions",
    "per_sample_cttp",
    "selected_reference_sample_ids",
    "generated_to_reference_distances",
}
ORIGINAL_FIELDS = {
    "predictions",
    "targets",
    "evaluation_sample_ids",
    "evaluation_split",
    "query_caption_ids",
    "query_captions",
    "per_sample_cttp",
}
DEFAULT_LAMBDAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0)
SCHEMA_VERSION = "ri-verbalts-validation-oracle-v1"


class AuditError(RuntimeError):
    pass


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def validate_run_metadata(split, max_batches, sample_ids, expected_count):
    if str(split) != "valid":
        raise AuditError(f"Oracle input split must be valid, got {split!r}")
    if int(max_batches) > 0:
        raise AuditError("Partial validation artifacts (max_batches > 0) are forbidden")
    sample_ids = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    if sample_ids.size != int(expected_count):
        raise AuditError(
            f"Validation sample count {sample_ids.size} != expected {expected_count}"
        )
    if np.unique(sample_ids).size != sample_ids.size:
        raise AuditError("Validation sample IDs contain duplicates")
    expected_ids = np.arange(int(expected_count), dtype=np.int64)
    if not np.array_equal(np.sort(sample_ids), expected_ids):
        raise AuditError("Validation sample IDs are missing or outside the expected range")


def align_action_arrays(actions, allow_joint_reference=False):
    """Align action dictionaries by sample ID and audit invariant query fields."""
    if not actions:
        raise AuditError("No fixed-strength actions were supplied")
    names = list(actions)
    canonical_ids = np.sort(
        np.asarray(actions[names[0]]["evaluation_sample_ids"], dtype=np.int64)
    )
    aligned = {}
    invariant_fields = ("query_caption_ids", "query_captions", "targets")
    sample_fields = (
        "predictions",
        "targets",
        "query_caption_ids",
        "query_captions",
        "per_sample_cttp",
        "selected_reference_sample_ids",
        "generated_to_reference_distances",
    )
    for name in names:
        action = actions[name]
        ids = np.asarray(action["evaluation_sample_ids"], dtype=np.int64).reshape(-1)
        if np.unique(ids).size != ids.size:
            raise AuditError(f"Action {name} has duplicate sample IDs")
        order = np.argsort(ids)
        if not np.array_equal(ids[order], canonical_ids):
            raise AuditError(f"Action {name} sample IDs do not match the fixed grid")
        for field in sample_fields:
            value = np.asarray(action[field])
            if value.ndim == 0 or value.shape[0] != ids.size:
                raise AuditError(
                    f"Action {name} field {field} is not aligned one-to-one with sample IDs"
                )
        for field in ("predictions", "per_sample_cttp", "generated_to_reference_distances"):
            if not np.isfinite(np.asarray(action[field], dtype=np.float64)).all():
                raise AuditError(f"Action {name} field {field} contains non-finite values")
        aligned[name] = {
            key: np.asarray(value)[order]
            if np.asarray(value).ndim > 0 and np.asarray(value).shape[0] == ids.size
            else np.asarray(value)
            for key, value in action.items()
        }
    canonical = aligned[names[0]]
    for name in names[1:]:
        current = aligned[name]
        for field in invariant_fields:
            left, right = canonical[field], current[field]
            equal = (
                np.allclose(left, right, rtol=0, atol=1e-6, equal_nan=True)
                if np.issubdtype(left.dtype, np.number)
                else np.array_equal(left, right)
            )
            if not equal:
                raise AuditError(f"Action {name} differs in invariant field {field}")
        if not np.array_equal(
            canonical["selected_reference_sample_ids"],
            current["selected_reference_sample_ids"],
        ):
            if not allow_joint_reference:
                raise AuditError(
                    "Fixed actions use different references; strength-only Oracle is invalid"
                )
    return canonical_ids, aligned


def _choose_with_ties(scores, distances, strengths, eligible, threshold, tolerance):
    scores = np.asarray(scores, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    strengths = np.asarray(strengths, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    selected = np.full(scores.shape[0], -1, dtype=np.int64)
    for row in range(scores.shape[0]):
        candidates = np.flatnonzero(eligible[row])
        if not candidates.size:
            continue
        best = np.max(scores[row, candidates])
        tied = candidates[scores[row, candidates] >= best - float(tolerance)]
        # Pre-fixed tie break: non-near-reference, larger distance, higher
        # strength, then lower fixed action index.
        selected[row] = min(
            tied,
            key=lambda column: (
                int(distances[row, column] < threshold),
                -float(distances[row, column]),
                -float(strengths[column]),
                int(column),
            ),
        )
    return selected


def select_max_cttp(scores, distances, strengths, threshold, tolerance=1e-6):
    eligible = np.ones_like(np.asarray(scores), dtype=bool)
    return _choose_with_ties(
        scores, distances, strengths, eligible, threshold, tolerance
    )


def select_non_reference(
    scores,
    distances,
    strengths,
    threshold,
    tolerance=1e-6,
    original_available=False,
):
    distances = np.asarray(distances, dtype=np.float64)
    eligible = distances >= float(threshold)
    selected = _choose_with_ties(
        scores, distances, strengths, eligible, threshold, tolerance
    )
    unmet = selected < 0
    if np.any(unmet) and not original_available:
        for row in np.flatnonzero(unmet):
            maximum = np.max(distances[row])
            tied = np.flatnonzero(distances[row] >= maximum - float(tolerance))
            selected[row] = min(
                tied,
                key=lambda column: (-float(strengths[column]), int(column)),
            )
    # -1 represents an explicit Original fallback when it is available.
    return selected, unmet


def select_pareto(
    scores,
    distances,
    strengths,
    strength_lambda,
    threshold,
    constrained=False,
    tolerance=1e-9,
    original_available=False,
):
    scores = np.asarray(scores, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    strengths = np.asarray(strengths, dtype=np.float64)
    score_min = scores.min(axis=1, keepdims=True)
    score_range = scores.max(axis=1, keepdims=True) - score_min
    normalized_score = (scores - score_min) / (score_range + 1e-12)
    strength_range = strengths.max() - strengths.min()
    normalized_strength = (strengths - strengths.min()) / (strength_range + 1e-12)
    utility = normalized_score - float(strength_lambda) * normalized_strength[None, :]
    eligible = (
        distances >= float(threshold)
        if constrained
        else np.ones_like(scores, dtype=bool)
    )
    # Utility is the primary score.  The same deterministic non-reference /
    # distance / strength / action-index tie break is then applied.
    selected = _choose_with_ties(
        utility, distances, strengths, eligible, threshold, tolerance
    )
    unmet = selected < 0
    if np.any(unmet) and not original_available:
        for row in np.flatnonzero(unmet):
            maximum = np.max(distances[row])
            tied = np.flatnonzero(distances[row] >= maximum - float(tolerance))
            selected[row] = min(
                tied,
                key=lambda column: (-float(strengths[column]), int(column)),
            )
    return selected, unmet


def select_hybrid(
    fixed_scores,
    original_scores,
    distances,
    strengths,
    threshold,
    semantic_ratio=0.99,
    tolerance=1e-6,
):
    fixed_scores = np.asarray(fixed_scores, dtype=np.float64)
    original_scores = np.asarray(original_scores, dtype=np.float64).reshape(-1)
    distances = np.asarray(distances, dtype=np.float64)
    strengths = np.asarray(strengths, dtype=np.float64)
    eligible = (fixed_scores >= semantic_ratio * original_scores[:, None]) & (
        distances >= float(threshold)
    )
    selected = np.full(original_scores.size, -1, dtype=np.int64)
    for row in range(original_scores.size):
        candidates = np.flatnonzero(eligible[row])
        if candidates.size:
            minimum_strength = np.min(strengths[candidates])
            tied = candidates[strengths[candidates] <= minimum_strength + tolerance]
            selected[row] = min(
                tied,
                key=lambda column: (
                    -float(fixed_scores[row, column]),
                    -float(distances[row, column]),
                    int(column),
                ),
            )
    return selected, selected < 0


def compose_selected(matrix, selected, original=None):
    matrix = np.asarray(matrix)
    selected = np.asarray(selected, dtype=np.int64)
    output = np.empty((selected.size,) + matrix.shape[2:], dtype=matrix.dtype)
    rows = np.arange(selected.size)
    fixed = selected >= 0
    output[fixed] = matrix[selected[fixed], rows[fixed]]
    if np.any(~fixed):
        if original is None:
            raise ValueError("Original fallback was selected but Original data are unavailable")
        output[~fixed] = np.asarray(original)[~fixed]
    return output


def discover_thresholds(manifest, calibration=None):
    """Read train-only q01/q05/q10 thresholds without validation calibration."""
    discovered = {}
    candidates = []
    for source in (manifest, calibration or {}):
        for key in ("near_reference_thresholds", "copy_thresholds"):
            if isinstance(source.get(key), dict):
                candidates.append(source[key])
    constraint = manifest.get("copy_constraint", {})
    estimation = constraint.get("estimation") or {}
    if constraint.get("threshold") is not None:
        quantile = float(estimation.get("quantile", 0.05))
        discovered[f"q{int(round(quantile * 100)):02d}"] = float(
            constraint["threshold"]
        )
    for candidate in candidates:
        for label in ("q01", "q05", "q10"):
            value = candidate.get(label)
            if isinstance(value, dict):
                value = value.get("threshold")
            if value is not None:
                discovered[label] = float(value)
    if "q05" not in discovered:
        raise AuditError("Train-only q05 near-reference threshold is unavailable")
    return discovered


def symmetric_relative_difference(left, right, epsilon=1e-12):
    left = float(left)
    right = float(right)
    return abs(left - right) / max(abs(left), abs(right), float(epsilon))


def audit_metric_reproduction(
    recomputed,
    recorded,
    runtime_mode,
    absolute_tolerance=1e-3,
    stochastic_relative_tolerance=0.05,
):
    """Audit deterministic metrics exactly and legacy-dropout metrics statistically."""
    difference = float(recomputed) - float(recorded)
    relative_difference = symmetric_relative_difference(recomputed, recorded)
    if runtime_mode == "eval":
        passed = abs(difference) <= float(absolute_tolerance)
        criterion = f"absolute_difference<={float(absolute_tolerance):g}"
    elif runtime_mode == "legacy_train":
        passed = relative_difference <= float(stochastic_relative_tolerance)
        criterion = (
            "symmetric_relative_difference<="
            f"{float(stochastic_relative_tolerance):g}"
        )
    else:
        raise ValueError(f"Unknown CTTP runtime mode: {runtime_mode}")
    return {
        "difference": difference,
        "relative_difference": relative_difference,
        "passed": bool(passed),
        "criterion": criterion,
    }


def _load_yaml(path):
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _condition_results(condition_dir):
    results_path = condition_dir / "results.csv"
    if not results_path.is_file():
        raise AuditError(f"Aggregate validation results are missing: {results_path}")
    rows = read_csv(results_path)
    output = {}
    for row in rows:
        split = row.get("split")
        if split != "valid":
            raise AuditError(f"Condition {condition_dir.name} results are not validation-only")
        run = int(row["run"])
        if run in output:
            raise AuditError(f"Condition {condition_dir.name} has duplicate run {run}")
        output[run] = {
            "cttp": float(row["cttp"]),
            "fid": float(row["fid"]),
            "jftsd": float(row["jftsd"]),
        }
    return output


def _load_npz(path, required_fields):
    path = Path(path)
    before = sha256_file(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = required_fields.difference(archive.files)
        if missing:
            raise AuditError(f"{path} is missing fields: {sorted(missing)}")
        payload = {field: np.array(archive[field]) for field in required_fields}
    if sha256_file(path) != before:
        raise AuditError(f"Input artifact changed while being read: {path}")
    return payload, before


def _discover_fixed(validation_root, expected_runs, expected_count, allow_joint_reference):
    conditions = []
    for directory in sorted(path for path in validation_root.iterdir() if path.is_dir()):
        match = FIXED_PATTERN.match(directory.name)
        if match:
            conditions.append((directory, int(match.group(1)) / 100.0))
    if not conditions:
        raise AuditError("No fixed-strength validation conditions were found")

    result_rows = {}
    config_summaries = {}
    file_records = []
    per_run = {run: {} for run in range(expected_runs)}
    shared_identities = None
    per_run_action_context = {}
    for directory, directory_strength in conditions:
        results = _condition_results(directory)
        if sorted(results) != list(range(expected_runs)):
            raise AuditError(f"{directory.name} does not contain all expected runs")
        result_rows[directory.name] = results
        file_records.append(
            {
                "role": "aggregate_results",
                "condition": directory.name,
                "path": str((directory / "results.csv").resolve()),
                "sha256": sha256_file(directory / "results.csv"),
            }
        )
        for run in range(expected_runs):
            run_dir = directory / str(run)
            config_path = run_dir / "eval_configs.yaml"
            prediction_path = run_dir / "rag_predictions.npz"
            if not config_path.is_file() or not prediction_path.is_file():
                raise AuditError(f"Missing fixed-action artifact in {run_dir}")
            config = _load_yaml(config_path)
            eval_config = config.get("eval", {})
            if eval_config.get("split") != "valid":
                raise AuditError(f"{config_path} is not validation-only")
            if int(eval_config.get("max_batches", -1)) > 0:
                raise AuditError(f"{config_path} records partial validation")
            rag = eval_config.get("rag", {})
            configured_strength = float(rag.get("strength"))
            if abs(configured_strength - directory_strength) > 1e-9:
                raise AuditError(
                    f"Directory/config strength mismatch for {directory.name}: "
                    f"{directory_strength} vs {configured_strength}"
                )
            if not rag.get("enabled") or rag.get("mode", "diffusion") != "diffusion":
                raise AuditError(f"{directory.name} is not a fixed RAG diffusion action")
            if rag.get("adaptive_controller", {}).get("enabled", False):
                raise AuditError(f"{directory.name} unexpectedly enables adaptive control")
            if int(rag.get("start_step", -1)) >= 0:
                raise AuditError(
                    f"{directory.name} uses an explicit start step instead of its fixed strength"
                )

            payload, prediction_hash = _load_npz(prediction_path, REQUIRED_FIELDS)
            split = str(np.asarray(payload["evaluation_split"]).item())
            validate_run_metadata(
                split,
                eval_config.get("max_batches", -1),
                payload["evaluation_sample_ids"],
                expected_count,
            )
            aggregate_cttp = results[run]["cttp"]
            stored_mean = float(np.asarray(payload["per_sample_cttp"]).mean())
            if not np.isclose(stored_mean, aggregate_cttp, rtol=0, atol=1e-4):
                raise AuditError(
                    f"{prediction_path} per-sample CTTP mean {stored_mean} "
                    f"does not reproduce aggregate {aggregate_cttp}"
                )
            per_run[run][directory.name] = {
                **payload,
                "strength": configured_strength,
                "prediction_path": str(prediction_path.resolve()),
            }
            identities = {
                "clip_model_path": str(_resolve(eval_config.get("clip_model_path", ""))),
                "clip_config_path": str(_resolve(eval_config.get("clip_config_path", ""))),
                "cache_folder": str(_resolve(eval_config.get("cache_folder", ""))),
            }
            if shared_identities is None:
                shared_identities = identities
            elif identities != shared_identities:
                raise AuditError("CTTP checkpoint/config/training-stat paths differ across actions")
            action_context = {
                "verbalts_model_path": str(
                    _resolve(eval_config.get("model_path", ""))
                ),
                "retrieval_index_path": str(_resolve(rag.get("index_path", ""))),
                "sampler": str(eval_config.get("sampler", "")),
                "n_samples": int(eval_config.get("n_samples", -1)),
                "retrieval_top_k": int(rag.get("top_k", -1)),
                "retrieval_selection": str(rag.get("selection", "")),
                "retrieval_seed": int(rag.get("seed", -1)),
            }
            if run not in per_run_action_context:
                per_run_action_context[run] = action_context
            elif per_run_action_context[run] != action_context:
                raise AuditError(
                    f"Fixed actions in run {run} differ outside the strength setting"
                )
            config_summaries[f"{directory.name}/run{run}"] = {
                "split": eval_config.get("split"),
                "max_batches": int(eval_config.get("max_batches", -1)),
                "strength": configured_strength,
                "n_samples": int(eval_config.get("n_samples", -1)),
                **identities,
                **action_context,
            }
            file_records.extend(
                [
                    {
                        "role": "fixed_prediction",
                        "condition": directory.name,
                        "run": run,
                        "path": str(prediction_path.resolve()),
                        "sha256": prediction_hash,
                    },
                    {
                        "role": "eval_config",
                        "condition": directory.name,
                        "run": run,
                        "path": str(config_path.resolve()),
                        "sha256": sha256_file(config_path),
                    },
                ]
            )

    aligned_runs = {}
    oracle_type = "reference+strength joint oracle" if allow_joint_reference else "strength-only oracle"
    for run, actions in per_run.items():
        sample_ids, aligned = align_action_arrays(actions, allow_joint_reference)
        ordered = sorted(aligned, key=lambda name: aligned[name]["strength"])
        aligned_runs[run] = {
            "sample_ids": sample_ids,
            "action_names": ordered,
            "actions": aligned,
        }
    return {
        "runs": aligned_runs,
        "results": result_rows,
        "config_summaries": config_summaries,
        "files": file_records,
        "cttp_identity": shared_identities,
        "oracle_type": oracle_type,
        "per_run_action_context": per_run_action_context,
    }


def _load_original(
    validation_root,
    expected_runs,
    expected_count,
    expected_cttp_identity,
    expected_action_context,
):
    directory = validation_root / "original"
    results = _condition_results(directory)
    if sorted(results) != list(range(expected_runs)):
        raise AuditError("Original aggregate results do not contain all expected runs")
    prediction_paths = [directory / str(run) / "rag_predictions.npz" for run in range(expected_runs)]
    available = [path.is_file() for path in prediction_paths]
    if any(available) and not all(available):
        raise AuditError(
            "Original predictions are present for only a subset of runs; hybrid Oracle is invalid"
        )
    payloads = {} if all(available) else None
    config_summaries = {}
    files = [
        {
            "role": "aggregate_results",
            "condition": "original",
            "path": str((directory / "results.csv").resolve()),
            "sha256": sha256_file(directory / "results.csv"),
        }
    ]
    for run in range(expected_runs):
        run_dir = directory / str(run)
        config_path = run_dir / "eval_configs.yaml"
        if not config_path.is_file():
            raise AuditError(f"Original evaluation config is missing: {config_path}")
        config = _load_yaml(config_path)
        eval_config = config.get("eval", {})
        if eval_config.get("split") != "valid":
            raise AuditError(f"Original config is not validation-only: {config_path}")
        if int(eval_config.get("max_batches", -1)) > 0:
            raise AuditError(f"Original config records partial validation: {config_path}")
        if eval_config.get("rag", {}).get("enabled", False):
            raise AuditError(f"Original config unexpectedly enables RAG: {config_path}")
        identity = {
            "clip_model_path": str(_resolve(eval_config.get("clip_model_path", ""))),
            "clip_config_path": str(_resolve(eval_config.get("clip_config_path", ""))),
            "cache_folder": str(_resolve(eval_config.get("cache_folder", ""))),
        }
        if identity != expected_cttp_identity:
            raise AuditError("Original and fixed actions use different CTTP/stat identities")
        original_context = {
            "verbalts_model_path": str(_resolve(eval_config.get("model_path", ""))),
            "sampler": str(eval_config.get("sampler", "")),
            "n_samples": int(eval_config.get("n_samples", -1)),
        }
        for field, value in original_context.items():
            if expected_action_context[run][field] != value:
                raise AuditError(
                    f"Original and fixed actions differ in run {run} field {field}"
                )
        config_summaries[f"original/run{run}"] = {
            "split": eval_config.get("split"),
            "max_batches": int(eval_config.get("max_batches", -1)),
            "rag_enabled": bool(eval_config.get("rag", {}).get("enabled", False)),
            **identity,
            **original_context,
        }
        files.append(
            {
                "role": "eval_config",
                "condition": "original",
                "run": run,
                "path": str(config_path.resolve()),
                "sha256": sha256_file(config_path),
            }
        )
        if payloads is None:
            continue
        path = prediction_paths[run]
        payload, digest = _load_npz(path, ORIGINAL_FIELDS)
        split = str(np.asarray(payload["evaluation_split"]).item())
        validate_run_metadata(split, -1, payload["evaluation_sample_ids"], expected_count)
        for field in ORIGINAL_FIELDS.difference({"evaluation_split"}):
            value = np.asarray(payload[field])
            if value.ndim == 0 or value.shape[0] != expected_count:
                raise AuditError(f"Original field {field} is not sample-aligned")
        for field in ("predictions", "per_sample_cttp"):
            if not np.isfinite(np.asarray(payload[field], dtype=np.float64)).all():
                raise AuditError(f"Original field {field} contains non-finite values")
        if not np.isclose(
            np.asarray(payload["per_sample_cttp"]).mean(),
            results[run]["cttp"],
            rtol=0,
            atol=1e-4,
        ):
            raise AuditError("Original per-sample CTTP does not reproduce aggregate CTTP")
        order = np.argsort(payload["evaluation_sample_ids"])
        payloads[run] = {
            key: np.asarray(value)[order]
            if np.asarray(value).ndim > 0 and np.asarray(value).shape[0] == order.size
            else np.asarray(value)
            for key, value in payload.items()
        }
        files.append(
            {
                "role": "original_prediction",
                "condition": "original",
                "run": run,
                "path": str(path.resolve()),
                "sha256": digest,
            }
        )
    return results, payloads, files, config_summaries


def _load_training_statistics(identity):
    cache = Path(identity["cache_folder"])
    paths = {
        "training_ts_mean": cache / "fid_mean.npy",
        "training_ts_cov": cache / "fid_cov.npy",
        "training_joint_mean": cache / "jftsd_mean.npy",
        "training_joint_cov": cache / "jftsd_cov.npy",
    }
    for path in paths.values():
        if not path.is_file():
            raise AuditError(f"Frozen training statistic is missing: {path}")
    values = {name: np.load(path) for name, path in paths.items()}
    files = [
        {
            "role": name,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    ]
    return values, files


def _load_cttp(identity, device, runtime_mode):
    import torch
    import yaml

    from models.cttp.cttp_model import CTTP

    config_path = Path(identity["clip_config_path"])
    checkpoint_path = Path(identity["clip_model_path"])
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise AuditError("Frozen CTTP config/checkpoint is missing")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["device"] = device
    config["text"]["device"] = device
    config["ts"]["device"] = device
    model = CTTP(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.requires_grad_(False)
    if runtime_mode == "legacy_train":
        # The validation artifacts were produced by BaseEvaluator before it
        # called clip.eval(). Preserve that scorer distribution (including
        # dropout), while keeping every parameter frozen.
        model.train()
    elif runtime_mode == "eval":
        model.eval()
    else:
        raise ValueError(f"Unknown CTTP runtime mode: {runtime_mode}")
    files = [
        {"role": "cttp_config", "path": str(config_path), "sha256": sha256_file(config_path)},
        {"role": "cttp_checkpoint", "path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
    ]
    runtime = {
        "mode": runtime_mode,
        "parameters_frozen": True,
        "ts_dropout": float(config.get("ts", {}).get("dropout", 0.0)),
        "stochastic": runtime_mode == "legacy_train"
        and float(config.get("ts", {}).get("dropout", 0.0)) > 0,
        "reason": (
            "Historical BaseEvaluator left CTTP in train mode; Monte Carlo repeats "
            "estimate the legacy scorer distribution."
            if runtime_mode == "legacy_train"
            else "Deterministic CTTP eval mode."
        ),
    }
    return model, files, runtime


def _seed_scorer(seed):
    import random
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _encode_text(model, captions, batch_size):
    import torch

    chunks = []
    with torch.no_grad():
        for start in range(0, len(captions), batch_size):
            embedding = model.get_text_coemb(list(captions[start : start + batch_size]), None)
            chunks.append(embedding.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _encode_ts(model, predictions, lengths, batch_size):
    import torch

    chunks = []
    with torch.no_grad():
        for start in range(0, predictions.shape[0], batch_size):
            end = min(start + batch_size, predictions.shape[0])
            values = torch.as_tensor(
                predictions[start:end], dtype=torch.float32, device=model.device
            )
            current_lengths = torch.as_tensor(
                lengths[start:end], dtype=torch.int32, device=model.device
            )
            embedding = model.get_ts_coemb(values, current_lengths)
            chunks.append(embedding.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _metric_row(
    run,
    policy,
    selected,
    action_names,
    strengths,
    score_matrix,
    distance_matrix,
    ts_embedding_matrix,
    text_embeddings,
    statistics,
    threshold_label,
    threshold,
    unmet,
    original_ts_embeddings=None,
    original_scores=None,
):
    ts_embedding_matrix = np.asarray(ts_embedding_matrix)
    text_embeddings = np.asarray(text_embeddings)
    if ts_embedding_matrix.ndim != 4 or text_embeddings.ndim != 3:
        raise ValueError("Repeated embeddings must have [R,A,N,D] and [R,N,D] shape")
    if ts_embedding_matrix.shape[0] != text_embeddings.shape[0]:
        raise ValueError("Time-series/text scorer repeat counts differ")
    repeat_metrics = []
    for repeat in range(ts_embedding_matrix.shape[0]):
        original_repeat = (
            None
            if original_ts_embeddings is None
            else np.asarray(original_ts_embeddings)[repeat]
        )
        selected_ts = compose_selected(
            ts_embedding_matrix[repeat], selected, original=original_repeat
        )
        repeat_metrics.append(
            metrics_from_embeddings(
                selected_ts, text_embeddings[repeat], **statistics
            )
        )
    metrics = {}
    for metric in ("cttp", "fid", "jftsd"):
        values = np.asarray([row[metric] for row in repeat_metrics], dtype=np.float64)
        metrics[metric] = float(values.mean())
        metrics[f"{metric}_scorer_repeat_std"] = (
            float(values.std(ddof=1)) if values.size > 1 else 0.0
        )
    rows = np.arange(selected.size)
    fixed = selected >= 0
    selected_scores = np.empty(selected.size, dtype=np.float32)
    selected_distances = np.full(selected.size, np.nan, dtype=np.float32)
    selected_strengths = np.full(selected.size, np.nan, dtype=np.float32)
    selected_actions = np.full(selected.size, "original", dtype="U64")
    selected_scores[fixed] = score_matrix[selected[fixed], rows[fixed]]
    selected_distances[fixed] = distance_matrix[selected[fixed], rows[fixed]]
    selected_strengths[fixed] = strengths[selected[fixed]]
    selected_actions[fixed] = np.asarray(action_names)[selected[fixed]]
    if np.any(~fixed):
        if original_scores is None:
            raise ValueError("Original scores are required for an Original fallback")
        selected_scores[~fixed] = np.asarray(original_scores, dtype=np.float32)[~fixed]
    near = fixed & (selected_distances < float(threshold))
    return {
        "run": int(run),
        "policy": policy,
        "threshold_label": threshold_label,
        "threshold": float(threshold),
        "scorer_repeats": int(ts_embedding_matrix.shape[0]),
        **metrics,
        "near_reference_rate": float(near.mean()),
        "mean_strength": float(np.nanmean(selected_strengths)) if fixed.any() else None,
        "constraint_unmet_rate": float(np.asarray(unmet, dtype=bool).mean()),
    }, {
        "selected_actions": selected_actions,
        "selected_strengths": selected_strengths,
        "selected_scores": selected_scores,
        "selected_distances": selected_distances,
        "near_reference": near,
        "constraint_unmet": np.asarray(unmet, dtype=bool),
    }


def _summaries(metric_rows):
    grouped = {}
    for row in metric_rows:
        key = (row["policy"], row["threshold_label"])
        grouped.setdefault(key, []).append(row)
    output = []
    for (policy, threshold_label), rows in sorted(grouped.items()):
        result = {"policy": policy, "threshold_label": threshold_label, "runs": len(rows)}
        for metric in (
            "cttp",
            "fid",
            "jftsd",
            "near_reference_rate",
            "mean_strength",
            "constraint_unmet_rate",
            "cttp_delta_original",
            "fid_delta_original",
            "jftsd_delta_original",
            "cttp_delta_best_fixed",
            "fid_delta_best_fixed",
            "jftsd_delta_best_fixed",
            "cttp_scorer_repeat_std",
            "fid_scorer_repeat_std",
            "jftsd_scorer_repeat_std",
        ):
            values = np.asarray(
                [row[metric] for row in rows if row[metric] is not None], dtype=float
            )
            result[f"{metric}_mean"] = float(values.mean()) if values.size else None
            result[f"{metric}_std"] = (
                float(values.std(ddof=1)) if values.size > 1 else 0.0
            )
        output.append(result)
    return output


def _choose_pareto(summary_rows, prefix, original_rows, near_cap):
    original_cttp = np.mean([row["cttp"] for row in original_rows.values()])
    candidates = [row for row in summary_rows if row["policy"].startswith(prefix)]
    feasible = [
        row
        for row in candidates
        if row["cttp_mean"] >= 0.99 * original_cttp
        and row["near_reference_rate_mean"] <= near_cap + 1e-12
    ]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda row: (row["jftsd_mean"], row["fid_mean"], -row["cttp_mean"]),
    )


def choose_per_run_pareto(metric_rows, prefix, original_rows, near_caps):
    """Choose an optimistic lambda independently within each run.

    This is deliberately reported separately from the shared-lambda result and
    is never used as the primary research decision.
    """
    output = {}
    for run, original in sorted(original_rows.items()):
        candidates = [
            row
            for row in metric_rows
            if row["run"] == run and row["policy"].startswith(prefix)
        ]
        feasible = [
            row
            for row in candidates
            if row["cttp"] >= 0.99 * original["cttp"]
            and row["near_reference_rate"] <= near_caps[run] + 1e-12
        ]
        output[run] = (
            min(feasible, key=lambda row: (row["jftsd"], row["fid"], -row["cttp"]))
            if feasible
            else None
        )
    return output


def _run_analysis(args):
    validation_root = Path(args.validation_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if "test" in args.dataset_split.lower():
        raise AuditError("Test split is forbidden")
    if args.dataset_split != "valid":
        raise AuditError("Oracle analysis requires --dataset-split valid")
    dataset_path = Path(args.dataset_folder).resolve() / "valid_ts.npy"
    if not dataset_path.is_file():
        raise AuditError(f"Validation dataset shape source is missing: {dataset_path}")
    validation_values = np.load(dataset_path, mmap_mode="r")
    if validation_values.dtype == object or validation_values.ndim < 2:
        raise AuditError("Variable-length validation data require explicit lengths; unsupported")
    expected_count = int(validation_values.shape[0])
    sequence_length = int(validation_values.shape[1])

    teacher_manifest_path = Path(args.teacher_manifest).resolve()
    teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    if teacher_manifest.get("split") != "train":
        raise AuditError("Near-reference threshold manifest must be train-only")
    calibration = (
        json.loads(Path(args.threshold_calibration).read_text(encoding="utf-8"))
        if args.threshold_calibration
        else None
    )
    if calibration is not None:
        calibration_split = calibration.get("split", calibration.get("source_split"))
        if calibration_split != "train":
            raise AuditError("Near-reference calibration must explicitly declare split=train")
    thresholds = discover_thresholds(teacher_manifest, calibration)
    fixed = _discover_fixed(
        validation_root,
        args.expected_runs,
        expected_count,
        args.allow_joint_reference,
    )
    (
        original_results,
        original_payloads,
        original_files,
        original_config_summaries,
    ) = _load_original(
        validation_root,
        args.expected_runs,
        expected_count,
        fixed["cttp_identity"],
        fixed["per_run_action_context"],
    )
    decision_path = Path(args.validation_decision).resolve()
    validation_decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if validation_decision.get("evaluation_split") != "valid":
        raise AuditError("Validation decision comparator is not validation-only")
    best_fixed_name = validation_decision["best_fixed_condition"]
    if best_fixed_name not in fixed["results"]:
        raise AuditError("Validation-best fixed condition is absent from Oracle grid")
    best_fixed_results = fixed["results"][best_fixed_name]

    statistics, statistic_files = _load_training_statistics(fixed["cttp_identity"])
    model, model_files, scorer_runtime = _load_cttp(
        fixed["cttp_identity"], args.device, args.cttp_runtime_mode
    )
    scorer_repeats = (
        int(args.scorer_repeats)
        if args.cttp_runtime_mode == "legacy_train"
        else 1
    )
    if scorer_repeats < 1:
        raise ValueError("--scorer-repeats must be positive")
    print(
        f"CTTP runtime={args.cttp_runtime_mode}, scorer_repeats={scorer_repeats}, "
        f"frozen_parameters={scorer_runtime['parameters_frozen']}",
        flush=True,
    )
    input_files = fixed["files"] + original_files + statistic_files + model_files + [
        {
            "role": "teacher_manifest",
            "path": str(teacher_manifest_path),
            "sha256": sha256_file(teacher_manifest_path),
        },
        {
            "role": "validation_decision",
            "path": str(decision_path),
            "sha256": sha256_file(decision_path),
        },
        {
            "role": "validation_shape_source",
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
    ]
    if args.threshold_calibration:
        calibration_path = Path(args.threshold_calibration).resolve()
        input_files.append(
            {
                "role": "train_only_threshold_calibration",
                "path": str(calibration_path),
                "sha256": sha256_file(calibration_path),
            }
        )

    metric_rows = []
    usage_rows = []
    sensitivity_rows = []
    all_assignments = {}
    reproduction_rows = []
    q05 = thresholds["q05"]
    lambdas = tuple(float(value) for value in args.pareto_lambdas.split(","))
    for run in range(args.expected_runs):
        run_data = fixed["runs"][run]
        names = run_data["action_names"]
        actions = run_data["actions"]
        strengths = np.asarray([actions[name]["strength"] for name in names], dtype=np.float64)
        predictions = np.stack([actions[name]["predictions"] for name in names], axis=0)
        if predictions.shape[2] != sequence_length:
            raise AuditError("Saved prediction length differs from fixed-length validation data")
        scores = np.stack([actions[name]["per_sample_cttp"] for name in names], axis=0)
        distances = np.stack(
            [actions[name]["generated_to_reference_distances"] for name in names], axis=0
        )
        captions = np.asarray(actions[names[0]]["query_captions"]).astype(str)
        caption_ids = np.asarray(actions[names[0]]["query_caption_ids"], dtype=np.int64)
        lengths = np.full(expected_count, sequence_length, dtype=np.int32)
        original_predictions = None
        original_scores = None
        if original_payloads is not None:
            original = original_payloads[run]
            if not np.array_equal(
                original["evaluation_sample_ids"], run_data["sample_ids"]
            ):
                raise AuditError("Original sample IDs do not align with fixed actions")
            for field in ("query_caption_ids", "query_captions", "targets"):
                left = original[field]
                right = actions[names[0]][field]
                equal = (
                    np.allclose(left, right, rtol=0, atol=1e-6, equal_nan=True)
                    if np.issubdtype(np.asarray(left).dtype, np.number)
                    else np.array_equal(left, right)
                )
                if not equal:
                    raise AuditError(f"Original differs from fixed actions in {field}")
            original_predictions = original["predictions"]
            original_scores = np.asarray(original["per_sample_cttp"], dtype=np.float32)

        text_repeat_values = []
        ts_repeat_values = []
        original_repeat_values = []
        for repeat in range(scorer_repeats):
            scorer_seed = int(args.scorer_seed) + run * 10000 + repeat
            print(
                f"[run {run}] CTTP encoding repeat {repeat + 1}/{scorer_repeats} "
                f"seed={scorer_seed}",
                flush=True,
            )
            _seed_scorer(scorer_seed)
            repeat_text = _encode_text(
                model, captions, args.embedding_batch_size
            )
            repeat_ts = np.stack(
                [
                    _encode_ts(
                        model,
                        predictions[column],
                        lengths,
                        args.embedding_batch_size,
                    )
                    for column in range(len(names))
                ],
                axis=0,
            )
            text_repeat_values.append(repeat_text)
            ts_repeat_values.append(repeat_ts)
            if original_predictions is not None:
                original_repeat_values.append(
                    _encode_ts(
                        model,
                        original_predictions,
                        lengths,
                        args.embedding_batch_size,
                    )
                )
        text_embeddings = np.stack(text_repeat_values, axis=0)
        ts_embeddings = np.stack(ts_repeat_values, axis=0)
        original_ts_embeddings = (
            np.stack(original_repeat_values, axis=0)
            if original_repeat_values
            else None
        )

        def audit_action(action_name, embedding_repeats, expected_metrics):
            repeat_rows = [
                metrics_from_embeddings(
                    embedding_repeats[repeat],
                    text_embeddings[repeat],
                    **statistics,
                )
                for repeat in range(scorer_repeats)
            ]
            record = {
                "run": run,
                "action": action_name,
                "scorer_runtime_mode": args.cttp_runtime_mode,
                "scorer_repeats": scorer_repeats,
            }
            failures = []
            for metric in ("cttp", "fid", "jftsd"):
                values = np.asarray(
                    [row[metric] for row in repeat_rows], dtype=np.float64
                )
                recomputed = float(values.mean())
                audit = audit_metric_reproduction(
                    recomputed,
                    expected_metrics[metric],
                    args.cttp_runtime_mode,
                    args.metric_audit_atol,
                    args.stochastic_metric_rtol,
                )
                record[metric] = recomputed
                record[f"{metric}_scorer_repeat_std"] = (
                    float(values.std(ddof=1)) if values.size > 1 else 0.0
                )
                record[f"delta_{metric}"] = audit["difference"]
                record[f"relative_delta_{metric}"] = audit["relative_difference"]
                record[f"{metric}_audit_passed"] = audit["passed"]
                record[f"{metric}_audit_criterion"] = audit["criterion"]
                if not audit["passed"]:
                    failures.append(
                        f"{metric}: delta={audit['difference']:.6g}, "
                        f"relative={audit['relative_difference']:.4%}, "
                        f"criterion={audit['criterion']}"
                    )
            reproduction_rows.append(record)
            if failures:
                raise AuditError(
                    f"Offline scorer reproduction failed for {action_name}/run{run}: "
                    + "; ".join(failures)
                )
            print(
                f"[run {run}] scorer audit PASS: {action_name}", flush=True
            )

        for column, name in enumerate(names):
            audit_action(name, ts_embeddings[:, column], fixed["results"][name][run])
        if original_ts_embeddings is not None:
            audit_action("original", original_ts_embeddings, original_results[run])

        policies = []
        selected = select_max_cttp(scores.T, distances.T, strengths, q05, args.cttp_tolerance)
        policies.append(("max_cttp", "q05", q05, selected, np.zeros(expected_count, bool)))
        for label, threshold in sorted(thresholds.items()):
            selected, unmet = select_non_reference(
                scores.T,
                distances.T,
                strengths,
                threshold,
                args.cttp_tolerance,
                original_payloads is not None,
            )
            policies.append((f"non_reference_max_cttp_{label}", label, threshold, selected, unmet))
        for value in lambdas:
            label = f"{value:g}".replace(".", "p")
            selected, unmet = select_pareto(
                scores.T, distances.T, strengths, value, q05, constrained=False
            )
            policies.append((f"pareto_unconstrained_lambda_{label}", "q05", q05, selected, unmet))
            selected, unmet = select_pareto(
                scores.T,
                distances.T,
                strengths,
                value,
                q05,
                constrained=True,
                original_available=original_payloads is not None,
            )
            policies.append((f"pareto_non_reference_lambda_{label}", "q05", q05, selected, unmet))
        if original_payloads is not None:
            selected, fallback = select_hybrid(
                scores.T, original_scores, distances.T, strengths, q05
            )
            policies.append(("original_hybrid", "q05", q05, selected, fallback))

        assignment_rows = []
        assignment_npz = {
            "evaluation_sample_id": [],
            "query_caption_id": [],
            "selected_action": [],
            "selected_strength": [],
            "selected_per_sample_cttp": [],
            "generated_to_reference_distance": [],
            "near_reference": [],
            "constraint_unmet": [],
            "fallback_reason": [],
            "policy_name": [],
            "run_id": [],
        }
        for policy, threshold_label, threshold, selected, unmet in policies:
            row, assignment = _metric_row(
                run,
                policy,
                selected,
                names,
                strengths,
                scores,
                distances,
                ts_embeddings,
                text_embeddings,
                statistics,
                threshold_label,
                threshold,
                unmet,
                original_ts_embeddings,
                original_scores,
            )
            metric_rows.append(row)
            for action_name, strength in zip(names, strengths):
                usage_rows.append(
                    {
                        "run": run,
                        "policy": policy,
                        "action": action_name,
                        "strength": float(strength),
                        "count": int(np.sum(assignment["selected_actions"] == action_name)),
                        "rate": float(np.mean(assignment["selected_actions"] == action_name)),
                    }
                )
            if np.any(assignment["selected_actions"] == "original"):
                usage_rows.append(
                    {
                        "run": run,
                        "policy": policy,
                        "action": "original",
                        "strength": None,
                        "count": int(np.sum(assignment["selected_actions"] == "original")),
                        "rate": float(np.mean(assignment["selected_actions"] == "original")),
                    }
                )
            fallback_reasons = np.where(
                assignment["constraint_unmet"],
                "constraint_unmet_original" if original_payloads is not None else "constraint_unmet_max_distance",
                "",
            )
            for index, sample_id in enumerate(run_data["sample_ids"]):
                record = {
                    "evaluation_sample_id": int(sample_id),
                    "query_caption_id": int(caption_ids[index]),
                    "selected_action": str(assignment["selected_actions"][index]),
                    "selected_strength": float(assignment["selected_strengths"][index]),
                    "selected_per_sample_cttp": float(assignment["selected_scores"][index]),
                    "generated_to_reference_distance": float(assignment["selected_distances"][index]),
                    "near_reference": bool(assignment["near_reference"][index]),
                    "constraint_unmet": bool(assignment["constraint_unmet"][index]),
                    "fallback_reason": str(fallback_reasons[index]),
                    "policy_name": policy,
                    "run_id": run,
                }
                assignment_rows.append(record)
                for key in assignment_npz:
                    assignment_npz[key].append(record[key])
        run_output = output_dir / f"run_{run}"
        run_output.mkdir(parents=True, exist_ok=True)
        write_csv(run_output / "oracle_assignments.csv", [json_safe(row) for row in assignment_rows])
        np.savez(
            run_output / "oracle_assignments.npz",
            **{key: np.asarray(value) for key, value in assignment_npz.items()},
        )
        all_assignments[run] = assignment_rows

        max_assignment = next(item for item in policies if item[0] == "max_cttp")[3]
        for threshold_label, threshold in ((label, thresholds.get(label)) for label in ("q01", "q05", "q10")):
            for column, name in enumerate(names):
                sensitivity_rows.append(
                    {
                        "run": run,
                        "threshold_label": threshold_label,
                        "threshold": threshold,
                        "available": threshold is not None,
                        "subject": name,
                        "near_reference_rate": None
                        if threshold is None
                        else float(np.mean(distances[column] < threshold)),
                    }
                )
            sensitivity_rows.append(
                {
                    "run": run,
                    "threshold_label": threshold_label,
                    "threshold": threshold,
                    "available": threshold is not None,
                    "subject": "max_cttp",
                    "near_reference_rate": None
                    if threshold is None
                    else float(
                        np.mean(
                            distances[max_assignment, np.arange(expected_count)] < threshold
                        )
                    ),
                }
            )

    best_fixed_near_by_run = {
        run: float(
            np.mean(
                fixed["runs"][run]["actions"][best_fixed_name][
                    "generated_to_reference_distances"
                ]
                < q05
            )
        )
        for run in range(args.expected_runs)
    }
    best_fixed_near = float(np.mean(list(best_fixed_near_by_run.values())))
    for row in metric_rows:
        run = row["run"]
        for metric in ("cttp", "fid", "jftsd"):
            row[f"{metric}_delta_original"] = float(
                row[metric] - original_results[run][metric]
            )
            row[f"{metric}_delta_best_fixed"] = float(
                row[metric] - best_fixed_results[run][metric]
            )
    summary_rows = _summaries(metric_rows)
    shared_constrained = _choose_pareto(
        summary_rows,
        "pareto_non_reference_lambda_",
        original_results,
        best_fixed_near,
    )
    shared_unconstrained = _choose_pareto(
        summary_rows,
        "pareto_unconstrained_lambda_",
        original_results,
        best_fixed_near,
    )
    per_run_constrained = choose_per_run_pareto(
        metric_rows,
        "pareto_non_reference_lambda_",
        original_results,
        best_fixed_near_by_run,
    )
    per_run_unconstrained = choose_per_run_pareto(
        metric_rows,
        "pareto_unconstrained_lambda_",
        original_results,
        best_fixed_near_by_run,
    )
    hybrid_summary = next(
        (row for row in summary_rows if row["policy"] == "original_hybrid"), None
    )

    original_mean = {
        metric: float(np.mean([row[metric] for row in original_results.values()]))
        for metric in ("cttp", "fid", "jftsd")
    }
    best_fixed_mean = {
        metric: float(np.mean([row[metric] for row in best_fixed_results.values()]))
        for metric in ("cttp", "fid", "jftsd")
    }

    def usable(row):
        if row is None:
            return False
        policy_rows = [item for item in metric_rows if item["policy"] == row["policy"]]
        direction_wins = sum(
            item["cttp"] >= 0.99 * original_results[item["run"]]["cttp"]
            and item["fid"] < original_results[item["run"]]["fid"]
            and item["jftsd"] < original_results[item["run"]]["jftsd"]
            and item["jftsd"] < best_fixed_results[item["run"]]["jftsd"]
            and item["near_reference_rate"]
            <= best_fixed_near_by_run[item["run"]] + 1e-12
            for item in policy_rows
        )
        return (
            row["cttp_mean"] >= 0.99 * original_mean["cttp"]
            and row["fid_mean"] < original_mean["fid"]
            and row["jftsd_mean"] < original_mean["jftsd"]
            and row["jftsd_mean"] < best_fixed_mean["jftsd"]
            and row["near_reference_rate_mean"] <= best_fixed_near + 1e-12
            and direction_wins >= math.ceil(2 * args.expected_runs / 3)
        )

    usable_candidates = [
        row for row in (shared_constrained, hybrid_summary) if usable(row)
    ]
    usable_candidate = (
        min(
            usable_candidates,
            key=lambda row: (row["jftsd_mean"], row["fid_mean"], -row["cttp_mean"]),
        )
        if usable_candidates
        else None
    )
    max_cttp_summary = next(row for row in summary_rows if row["policy"] == "max_cttp")
    unconstrained_distribution_headroom = shared_unconstrained is not None and (
        shared_unconstrained["cttp_mean"] >= 0.99 * original_mean["cttp"]
        and shared_unconstrained["fid_mean"] < original_mean["fid"]
        and shared_unconstrained["jftsd_mean"] < original_mean["jftsd"]
        and shared_unconstrained["jftsd_mean"] < best_fixed_mean["jftsd"]
    )
    metric_targeted_headroom = (
        max_cttp_summary["cttp_mean"]
        > max(original_mean["cttp"], best_fixed_mean["cttp"])
    )
    metric_or_copy_only = metric_targeted_headroom or unconstrained_distribution_headroom
    if usable_candidate is not None:
        classification = "USABLE_HEADROOM_PRESENT"
    elif metric_or_copy_only:
        classification = "METRIC_TARGETED_OR_COPY_HEADROOM_ONLY"
    else:
        classification = "NO_USABLE_HEADROOM_IN_EVALUATED_GRID"

    unchanged_inputs = all(
        sha256_file(item["path"]) == item["sha256"] for item in input_files
    )
    if not unchanged_inputs:
        raise AuditError("At least one input artifact changed during Oracle analysis")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "test_split_read_or_run": False,
        "validation_root": str(validation_root),
        "output_dir": str(output_dir),
        "dataset_split": "valid",
        "expected_validation_samples": expected_count,
        "fixed_sequence_length": sequence_length,
        "oracle_type": fixed["oracle_type"],
        "fixed_strength_grid": [
            fixed["runs"][0]["actions"][name]["strength"]
            for name in fixed["runs"][0]["action_names"]
        ],
        "action_source": "saved pointwise-median predictions only; candidates unused",
        "pareto_lambdas": list(lambdas),
        "cttp_tolerance": args.cttp_tolerance,
        "cttp_runtime": scorer_runtime,
        "scorer_repeats": scorer_repeats,
        "scorer_seed": int(args.scorer_seed),
        "deterministic_metric_audit_atol": float(args.metric_audit_atol),
        "stochastic_metric_audit_rtol": float(args.stochastic_metric_rtol),
        "near_reference_thresholds": thresholds,
        "near_reference_threshold_source": str(teacher_manifest_path),
        "hybrid_oracle_available": original_payloads is not None,
        "hybrid_unavailable_reason": None
        if original_payloads is not None
        else "Original per-sample predictions were not saved; regeneration is forbidden.",
        "best_fixed_condition": best_fixed_name,
        "config_summaries": {
            **fixed["config_summaries"],
            **original_config_summaries,
        },
        "input_files": input_files,
        "ignored_prediction_fields": ["candidates"],
    }
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "audit_errors": [],
        "validation_only": True,
        "full_validation": True,
        "sample_id_alignment": "passed",
        "query_target_alignment": "passed",
        "reference_alignment": "passed" if not args.allow_joint_reference else "joint-reference override enabled",
        "per_sample_cttp_reproduction": "passed",
        "offline_metric_reproduction": reproduction_rows,
        "offline_metric_reproduction_mode": (
            "stochastic Monte Carlo compatibility audit"
            if scorer_runtime["stochastic"]
            else "deterministic exact-tolerance audit"
        ),
        "input_files_unchanged": unchanged_inputs,
        "q01_available": "q01" in thresholds,
        "q05_available": True,
        "q10_available": "q10" in thresholds,
        "original_predictions_available": original_payloads is not None,
    }
    decision = {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "decision_scope": "policy-specific empirical ceiling on validation only",
        "diagnostic_only": True,
        "test_split_read_or_run": False,
        "scorer_runtime_mode": args.cttp_runtime_mode,
        "scorer_repeats": scorer_repeats,
        "stochastic_scorer_caveat": (
            "Historical CTTP validation used train-mode dropout. Oracle metrics are "
            "Monte Carlo estimates under that legacy scorer distribution; Original "
            "aggregate metrics are a single historical realization."
            if scorer_runtime["stochastic"]
            else None
        ),
        "original_mean": original_mean,
        "best_fixed_condition": best_fixed_name,
        "best_fixed_mean": best_fixed_mean,
        "best_fixed_near_reference_rate": best_fixed_near,
        "best_fixed_near_reference_rate_per_run": best_fixed_near_by_run,
        "shared_non_reference_pareto": shared_constrained,
        "shared_unconstrained_pareto": shared_unconstrained,
        "per_run_optimistic_non_reference_pareto": per_run_constrained,
        "per_run_optimistic_unconstrained_pareto": per_run_unconstrained,
        "hybrid_oracle_available": original_payloads is not None,
        "hybrid_unavailable_reason": None
        if original_payloads is not None
        else "Original per-sample predictions were not saved; regeneration is forbidden.",
        "hybrid_oracle": hybrid_summary,
        "usable_candidate": usable_candidate,
        "metric_targeted_headroom": metric_targeted_headroom,
        "unconstrained_distribution_headroom": unconstrained_distribution_headroom,
        "interpretation": (
            "Current grid shows usable validation headroom; bottleneck may be controller/teacher/features."
            if classification == "USABLE_HEADROOM_PRESENT"
            else "Observed headroom is metric-targeted or reference-retention dependent."
            if classification == "METRIC_TARGETED_OR_COPY_HEADROOM_ONLY"
            else "No usable headroom was observed under the current fixed grid, references, and validation Oracle policies."
        ),
        "automatic_test_evaluation": False,
    }

    (output_dir / "oracle_manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2), encoding="utf-8"
    )
    (output_dir / "oracle_integrity_report.json").write_text(
        json.dumps(json_safe(integrity), indent=2), encoding="utf-8"
    )
    (output_dir / "oracle_decision.json").write_text(
        json.dumps(json_safe(decision), indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "oracle_metrics_per_run.csv", [json_safe(row) for row in metric_rows])
    write_csv(output_dir / "oracle_summary.csv", [json_safe(row) for row in summary_rows])
    write_csv(output_dir / "oracle_strength_usage.csv", [json_safe(row) for row in usage_rows])
    write_csv(
        output_dir / "oracle_pareto_frontier.csv",
        [json_safe(row) for row in metric_rows if row["policy"].startswith("pareto_")],
    )
    write_csv(
        output_dir / "oracle_near_reference_sensitivity.csv",
        [json_safe(row) for row in sensitivity_rows],
    )
    return decision


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", required=True)
    parser.add_argument("--dataset-folder", required=True)
    parser.add_argument("--dataset-split", choices=["valid"], default="valid")
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--validation-decision", required=True)
    parser.add_argument("--threshold-calibration", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-runs", type=int, default=3)
    parser.add_argument("--pareto-lambdas", default=",".join(map(str, DEFAULT_LAMBDAS)))
    parser.add_argument("--cttp-tolerance", type=float, default=1e-6)
    parser.add_argument("--metric-audit-atol", type=float, default=1e-3)
    parser.add_argument(
        "--cttp-runtime-mode",
        choices=["legacy_train", "eval"],
        default="legacy_train",
        help="Match historical validation scorer state or use deterministic eval mode.",
    )
    parser.add_argument("--scorer-repeats", type=int, default=3)
    parser.add_argument("--scorer-seed", type=int, default=2026)
    parser.add_argument("--stochastic-metric-rtol", type=float, default=0.05)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-joint-reference", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        decision = _run_analysis(args)
    except Exception as error:
        integrity = {
            "schema_version": SCHEMA_VERSION,
            "passed": False,
            "audit_errors": [f"{type(error).__name__}: {error}"],
            "test_split_read_or_run": False,
        }
        decision = {
            "schema_version": SCHEMA_VERSION,
            "classification": "INCONCLUSIVE_ARTIFACTS_MISSING",
            "reason": str(error),
            "diagnostic_only": True,
            "test_split_read_or_run": False,
            "automatic_test_evaluation": False,
        }
        (output_dir / "oracle_integrity_report.json").write_text(
            json.dumps(integrity, indent=2), encoding="utf-8"
        )
        (output_dir / "oracle_decision.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8"
        )
        raise
    print(json.dumps(json_safe(decision), indent=2))
    print(f"Oracle ceiling analysis written to {output_dir}")
    print("Test split was not read or evaluated.")


if __name__ == "__main__":
    main()
