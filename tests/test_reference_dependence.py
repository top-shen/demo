import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.reference_dependence import (
    analyze_reference_dependence,
    compute_reference_metrics,
    select_stratified_cases,
    standardized_rmse,
)
from retrieval import INDEX_VERSION


def test_standardized_rmse_respects_variable_scale():
    a = np.asarray([[[0.0, 0.0], [0.0, 0.0]]])
    b = np.asarray([[[1.0, 2.0], [1.0, 2.0]]])
    actual = standardized_rmse(a, b, np.asarray([1.0, 2.0]))
    np.testing.assert_allclose(actual, [1.0])


def _metric_fixture():
    references = np.zeros((2, 2, 1), dtype=np.float32)
    targets = np.asarray([[[2.0], [2.0]], [[2.0], [2.0]]], dtype=np.float32)
    predictions = np.asarray([[[1.0], [1.0]], [[0.25], [0.25]]], dtype=np.float32)
    baseline = np.asarray([[[2.0], [2.0]], [[2.0], [2.0]]], dtype=np.float32)
    candidates = np.stack([predictions - 0.1, predictions + 0.1])
    rag = {
        "predictions": predictions,
        "targets": targets,
        "candidates": candidates,
        "test_sample_ids": np.asarray([10, 11]),
        "query_captions": np.asarray(["query 10", "query 11"]),
    }
    base = {
        "predictions": baseline,
        "targets": targets,
        "candidates": np.stack([baseline, baseline]),
        "test_sample_ids": np.asarray([10, 11]),
    }
    trace = {
        10: {
            "query_caption": "query 10",
            "selected_reference_caption": "ref 0",
            "similarity": 0.9,
        },
        11: {
            "query_caption": "query 11",
            "selected_reference_caption": "ref 1",
            "similarity": 0.5,
        },
    }
    return rag, base, references, trace


def test_metrics_measure_correction_and_reference_attraction():
    rag, base, references, trace = _metric_fixture()
    frame = compute_reference_metrics(
        rag,
        references,
        np.asarray([True, True]),
        np.asarray([0, 1]),
        trace,
        np.asarray([1.0]),
        baseline_artifact=base,
        copy_threshold=0.3,
    )
    np.testing.assert_allclose(frame["reference_target_srmse"], [2.0, 2.0])
    np.testing.assert_allclose(frame["rag_target_srmse"], [1.0, 1.75])
    np.testing.assert_allclose(frame["target_correction_gain"], [1.0, 0.25])
    np.testing.assert_allclose(frame["reference_attraction"], [0.5, 0.875])
    assert frame["near_reference_copy"].tolist() == [False, True]
    assert np.all(frame["candidate_diversity_srmse"] > 0)


def test_stratified_selection_is_deterministic_and_unique():
    rows = []
    for sample_id in range(20):
        rows.append(
            {
                "sample_id": sample_id,
                "fallback": False,
                "reference_sample_id": sample_id % 3,
                "retrieval_similarity": sample_id / 20,
                "target_correction_gain": sample_id - 10,
                "reference_attraction": (20 - sample_id) / 20,
                "rag_reference_srmse": sample_id / 10,
            }
        )
    frame = pd.DataFrame(rows)
    first = select_stratified_cases(frame, per_group=2, seed=7)
    second = select_stratified_cases(frame, per_group=2, seed=7)
    assert first[["sample_id", "selection_reason"]].equals(
        second[["sample_id", "selection_reason"]]
    )
    assert first["sample_id"].is_unique
    assert set(first["selection_reason"]) == {
        "high_similarity",
        "low_similarity",
        "best_target_correction",
        "worst_target_correction",
        "strongest_reference_attraction",
        "random",
    }


def _write_end_to_end_fixture(root: Path):
    train_ts = np.asarray(
        [
            [[0.0], [0.5], [1.0], [1.5]],
            [[1.0], [1.5], [2.0], [2.5]],
            [[2.0], [2.5], [3.0], [3.5]],
        ],
        dtype=np.float32,
    )
    metadata = {
        "index_version": INDEX_VERSION,
        "dataset_name": "fixture",
        "split": "train",
        "embedding_model": "fake",
    }
    index_path = root / "index.npz"
    np.savez(
        index_path,
        embeddings=np.eye(3, dtype=np.float32),
        sample_ids=np.arange(3, dtype=np.int64),
        caption_ids=np.zeros(3, dtype=np.int64),
        captions=np.asarray(["r0", "r1", "r2"]),
        train_ts=train_ts,
        train_ts_sample_ids=np.arange(3, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    targets = train_ts + 0.4
    predictions = train_ts + 0.2
    rag_path = root / "rag.npz"
    np.savez(
        rag_path,
        predictions=predictions,
        targets=targets,
        candidates=np.stack([predictions - 0.05, predictions + 0.05]),
        test_sample_ids=np.asarray([100, 101, 102]),
        query_captions=np.asarray(["q0", "q1", "q2"]),
        reference_sample_ids=np.asarray([[0, 1, 2], [0, 1, 2]]),
    )
    baseline_path = root / "baseline.npz"
    baseline_order = np.asarray([2, 1, 0])
    np.savez(
        baseline_path,
        predictions=targets[baseline_order],
        targets=targets[baseline_order],
        candidates=np.stack([targets[baseline_order], targets[baseline_order]]),
        # Reverse order verifies ID-based rather than row-based alignment.
        test_sample_ids=np.asarray([102, 101, 100]),
    )
    trace_path = root / "trace.jsonl"
    with open(trace_path, "w", encoding="utf-8") as trace_file:
        for row, sample_id in enumerate((100, 101, 102)):
            trace_file.write(
                json.dumps(
                    {
                        "test_sample_id": sample_id,
                        "query_caption": f"q{row}",
                        "selected_reference_sample_id": row,
                        "selected_reference_caption": f"r{row}",
                        "similarity": 0.7 + 0.1 * row,
                        "candidate_id": 0,
                        "fallback": False,
                    }
                )
                + "\n"
            )
    return rag_path, baseline_path, trace_path, index_path


def test_end_to_end_analysis_writes_reproducible_tables(tmp_path):
    rag, baseline, trace, index = _write_end_to_end_fixture(tmp_path)
    output = tmp_path / "analysis"
    result = analyze_reference_dependence(
        str(rag),
        str(trace),
        str(index),
        str(output),
        baseline_predictions_path=str(baseline),
        cases_per_group=1,
        make_plots=False,
    )
    assert len(result["metrics"]) == 3
    assert (output / "reference_dependence_metrics.csv").exists()
    assert (output / "selected_cases.csv").exists()
    assert (output / "similarity_strata_summary.csv").exists()
    assert (output / "reference_dependence_summary.json").exists()
    assert (output / "reference_dependence_report.md").exists()
    assert result["summary"]["num_retrieval_success"] == 3
    # Correct ID alignment makes every baseline exactly equal its target.
    np.testing.assert_allclose(result["metrics"]["baseline_target_srmse"], 0.0)
