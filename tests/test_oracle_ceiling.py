"""Dependency-light tests for the validation-only Oracle ceiling."""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import numpy as np

from evaluation.oracle_metrics import metrics_from_embeddings
from tools.analyze_oracle_ceiling import (
    ORIGINAL_FIELDS,
    REQUIRED_FIELDS,
    AuditError,
    _load_npz,
    align_action_arrays,
    compose_selected,
    discover_thresholds,
    select_hybrid,
    select_max_cttp,
    select_non_reference,
    select_pareto,
    sha256_file,
    validate_run_metadata,
)


def _raises(error_type, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}")


def _action(ids, references=(10, 11, 12), offset=0.0):
    ids = np.asarray(ids, dtype=np.int64)
    values = ids.astype(np.float32) + float(offset)
    return {
        "evaluation_sample_ids": ids,
        "query_caption_ids": ids + 100,
        "query_captions": np.asarray([f"caption-{item}" for item in ids]),
        "targets": ids.astype(np.float32)[:, None, None],
        "predictions": values[:, None, None],
        "per_sample_cttp": values,
        "selected_reference_sample_ids": np.asarray(references, dtype=np.int64)[ids],
        "generated_to_reference_distances": values / 10,
        "evaluation_split": np.asarray("valid"),
    }


def _statistics(ts_embeddings, text_embeddings):
    joint = np.concatenate([ts_embeddings, text_embeddings], axis=1)
    return {
        "training_ts_mean": ts_embeddings.mean(axis=0),
        "training_ts_cov": np.cov(ts_embeddings, rowvar=False),
        "training_joint_mean": joint.mean(axis=0),
        "training_joint_cov": np.cov(joint, rowvar=False),
    }


def test_01_alignment_uses_sample_ids_not_row_positions():
    first = _action([2, 0, 1])
    second = _action([1, 2, 0], offset=10.0)
    ids, aligned = align_action_arrays({"a": first, "b": second})
    np.testing.assert_array_equal(ids, [0, 1, 2])
    np.testing.assert_array_equal(aligned["a"]["evaluation_sample_ids"], ids)
    np.testing.assert_allclose(aligned["b"]["predictions"].reshape(-1), [10, 11, 12])


def test_02_missing_duplicate_or_inconsistent_ids_fail():
    validate_run_metadata("valid", -1, [0, 1, 2], 3)
    _raises(AuditError, validate_run_metadata, "valid", -1, [0, 1, 1], 3)
    _raises(AuditError, validate_run_metadata, "valid", -1, [0, 1], 3)
    _raises(
        AuditError,
        align_action_arrays,
        {
            "a": _action([0, 1, 2]),
            "b": _action([0, 1, 3], references=(10, 11, 12, 13)),
        },
    )


def test_03_test_split_is_rejected():
    _raises(AuditError, validate_run_metadata, "test", -1, [0, 1], 2)


def test_04_partial_validation_is_rejected():
    _raises(AuditError, validate_run_metadata, "valid", 1, [0, 1], 2)


def test_05_reference_mismatch_fails_strength_only_oracle():
    left = _action([0, 1, 2])
    right = _action([0, 1, 2], references=(20, 21, 22))
    _raises(AuditError, align_action_arrays, {"a": left, "b": right})
    _, joint = align_action_arrays(
        {"a": left, "b": right}, allow_joint_reference=True
    )
    assert set(joint) == {"a", "b"}


def test_06_max_cttp_selects_the_correct_fixed_action():
    scores = np.asarray([[0.1, 0.7, 0.2], [0.9, 0.8, 0.1]])
    distances = np.ones_like(scores)
    selected = select_max_cttp(scores, distances, [0.2, 0.5, 0.9], 0.6)
    np.testing.assert_array_equal(selected, [1, 0])


def test_07_non_reference_oracle_excludes_low_distance_actions():
    scores = np.asarray([[0.99, 0.8], [0.7, 0.9]])
    distances = np.asarray([[0.1, 0.8], [0.9, 0.2]])
    selected, unmet = select_non_reference(
        scores, distances, [0.2, 0.9], threshold=0.6
    )
    np.testing.assert_array_equal(selected, [1, 0])
    assert not unmet.any()


def test_08_infeasible_non_reference_fallback_records_violation():
    scores = np.asarray([[0.9, 0.8]])
    distances = np.asarray([[0.2, 0.4]])
    selected, unmet = select_non_reference(
        scores, distances, [0.2, 0.9], threshold=0.6
    )
    np.testing.assert_array_equal(selected, [1])
    np.testing.assert_array_equal(unmet, [True])
    original, unmet_original = select_non_reference(
        scores, distances, [0.2, 0.9], threshold=0.6, original_available=True
    )
    np.testing.assert_array_equal(original, [-1])
    np.testing.assert_array_equal(unmet_original, [True])


def test_09_hybrid_oracle_falls_back_to_original():
    fixed = np.asarray([[0.95, 0.80], [0.80, 0.85]])
    original = np.asarray([0.90, 0.90])
    distances = np.asarray([[0.8, 0.9], [0.8, 0.9]])
    selected, fallback = select_hybrid(
        fixed, original, distances, [0.2, 0.9], threshold=0.6
    )
    np.testing.assert_array_equal(selected, [0, -1])
    np.testing.assert_array_equal(fallback, [False, True])
    matrix = np.asarray([[[1.0], [2.0]], [[3.0], [4.0]]])
    np.testing.assert_allclose(
        compose_selected(matrix, selected, original=np.asarray([[8.0], [9.0]])),
        [[1.0], [9.0]],
    )


def test_10_tie_break_is_deterministic_and_prefers_non_reference():
    scores = np.asarray([[1.0, 1.0 - 5e-7, 1.0 - 5e-7]])
    distances = np.asarray([[0.2, 0.8, 0.9]])
    strengths = np.asarray([0.2, 0.5, 0.9])
    first = select_max_cttp(scores, distances, strengths, 0.6, tolerance=1e-6)
    second = select_max_cttp(scores, distances, strengths, 0.6, tolerance=1e-6)
    np.testing.assert_array_equal(first, [2])
    np.testing.assert_array_equal(first, second)


def test_11_action_selection_never_mixes_runs():
    distances = np.ones((1, 2))
    strengths = [0.2, 0.9]
    run_zero = select_max_cttp([[1.0, 0.0]], distances, strengths, 0.6)
    run_one = select_max_cttp([[0.0, 1.0]], distances, strengths, 0.6)
    np.testing.assert_array_equal(run_zero, [0])
    np.testing.assert_array_equal(run_one, [1])


def test_12_train_only_q01_q05_q10_thresholds_are_discovered():
    manifest = {
        "split": "train",
        "copy_constraint": {
            "threshold": 0.6,
            "estimation": {"quantile": 0.05},
        },
        "near_reference_thresholds": {
            "q01": {"threshold": 0.4},
            "q10": 0.8,
        },
    }
    assert discover_thresholds(manifest) == {"q01": 0.4, "q05": 0.6, "q10": 0.8}


def test_13_aggregate_cttp_is_recomputed_from_selected_samples():
    ts = np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    text = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    metrics = metrics_from_embeddings(ts, text, **_statistics(ts, text))
    expected = np.mean(np.sum(ts * text, axis=1))
    np.testing.assert_allclose(metrics["cttp"], expected)


def test_14_set_metrics_cannot_average_action_level_fid_or_jftsd():
    signature = inspect.signature(metrics_from_embeddings)
    assert "action_fid" not in signature.parameters
    assert "action_jftsd" not in signature.parameters
    text = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    first = np.asarray([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0]])
    second = np.asarray([[3.0, 3.0], [3.0, 3.0], [0.0, 0.0]])
    reference = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    stats = _statistics(reference, text)
    mixed = compose_selected(np.stack([first, second]), np.asarray([0, 1, 0]))
    mixed_fid = metrics_from_embeddings(mixed, text, **stats)["fid"]
    averaged_fid = np.mean(
        [
            metrics_from_embeddings(first, text, **stats)["fid"],
            metrics_from_embeddings(second, text, **stats)["fid"],
        ]
    )
    assert not np.isclose(mixed_fid, averaged_fid)


def test_15_tiny_set_fid_and_jftsd_match_full_set_statistics():
    ts = np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0]])
    text = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    metrics = metrics_from_embeddings(ts, text, **_statistics(ts, text))
    np.testing.assert_allclose(metrics["fid"], 0.0, atol=1e-8)
    np.testing.assert_allclose(metrics["jftsd"], 0.0, atol=1e-7)


def test_16_policies_repeat_identically():
    rng = np.random.default_rng(7)
    scores = rng.normal(size=(20, 4))
    distances = rng.uniform(size=(20, 4))
    arguments = (scores, distances, [0.2, 0.4, 0.7, 0.95], 0.2, 0.6)
    first = select_pareto(*arguments, constrained=True)
    second = select_pareto(*arguments, constrained=True)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_17_loading_ignores_candidates_and_does_not_modify_input_file():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "artifact.npz"
        config_path = Path(directory) / "eval_configs.yaml"
        config_path.write_text("eval:\n  split: valid\n", encoding="utf-8")
        payload = _action([0, 1, 2])
        payload["candidates"] = np.full((10, 3, 1, 1), 99.0)
        np.savez(path, **payload)
        before = sha256_file(path)
        config_before = sha256_file(config_path)
        loaded, recorded = _load_npz(path, REQUIRED_FIELDS)
        assert "split: valid" in config_path.read_text(encoding="utf-8")
        assert "candidates" not in loaded
        assert recorded == before == sha256_file(path)
        assert config_before == sha256_file(config_path)
        assert ORIGINAL_FIELDS.issubset(loaded)
