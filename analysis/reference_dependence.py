"""Reference-dependence diagnostics for RI-VerbalTS prediction artifacts.

The analysis is intentionally offline: it consumes saved predictions, the
train-only retrieval index, and the retrieval trace.  It never loads a model or
changes an experiment result.
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from retrieval import load_retrieval_index


EPS = 1e-8
ANALYSIS_VERSION = "ri-verbalts-reference-dependence-v1"


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _sample_ids(artifact: Dict[str, np.ndarray]) -> np.ndarray:
    for key in ("evaluation_sample_ids", "test_sample_ids", "validation_sample_ids"):
        if key in artifact:
            ids = np.asarray(artifact[key], dtype=np.int64).reshape(-1)
            if len(np.unique(ids)) != len(ids):
                raise ValueError(f"{key} contains duplicate sample IDs")
            return ids
    raise ValueError("Prediction artifact has no evaluation/test/validation sample IDs")


def _validate_series(name: str, values: np.ndarray, n_samples: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != n_samples:
        raise ValueError(f"{name} must have shape [N,L,V], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return values


def _align_artifact(
    artifact: Dict[str, np.ndarray], target_ids: np.ndarray, artifact_name: str
) -> Dict[str, np.ndarray]:
    source_ids = _sample_ids(artifact)
    row_by_id = {int(sample_id): row for row, sample_id in enumerate(source_ids)}
    missing = [int(sample_id) for sample_id in target_ids if int(sample_id) not in row_by_id]
    if missing:
        raise ValueError(
            f"{artifact_name} is missing {len(missing)} target sample IDs; first={missing[:5]}"
        )
    order = np.asarray([row_by_id[int(sample_id)] for sample_id in target_ids], dtype=np.int64)
    aligned: Dict[str, np.ndarray] = {}
    for key, values in artifact.items():
        values = np.asarray(values)
        if key == "candidates" and values.ndim >= 2 and values.shape[1] == len(source_ids):
            aligned[key] = values[:, order]
        elif values.ndim >= 1 and values.shape[0] == len(source_ids):
            aligned[key] = values[order]
        else:
            aligned[key] = values
    return aligned


def _load_trace(path: str) -> Dict[int, Dict]:
    records: Dict[int, Dict] = {}
    with open(path, "r", encoding="utf-8") as trace_file:
        for line_no, line in enumerate(trace_file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = int(record.get("test_sample_id", record.get("evaluation_sample_id", -1)))
            if sample_id < 0:
                raise ValueError(f"Trace line {line_no} has no valid sample ID")
            candidate_id = int(record.get("candidate_id", 0))
            # Aggregate predictions use the fixed record or candidate 0.
            if sample_id not in records or candidate_id == 0:
                records[sample_id] = record
    return records


def _reference_ids(
    artifact: Dict[str, np.ndarray], sample_ids: np.ndarray, trace: Dict[int, Dict]
) -> np.ndarray:
    if "selected_reference_sample_ids" in artifact:
        reference_ids = np.asarray(artifact["selected_reference_sample_ids"], dtype=np.int64)
    elif "reference_sample_ids" in artifact:
        reference_ids = np.asarray(artifact["reference_sample_ids"], dtype=np.int64)
        if reference_ids.ndim == 2:
            reference_ids = reference_ids[0]
    else:
        reference_ids = np.asarray(
            [
                -1
                if trace.get(int(sample_id), {}).get("selected_reference_sample_id") is None
                else int(trace[int(sample_id)]["selected_reference_sample_id"])
                for sample_id in sample_ids
            ],
            dtype=np.int64,
        )
    reference_ids = reference_ids.reshape(-1)
    if reference_ids.shape[0] != sample_ids.shape[0]:
        raise ValueError("Reference IDs do not align with prediction sample IDs")
    return reference_ids


def _references_from_index(
    index: Dict, reference_ids: np.ndarray, expected_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    row_by_id = {
        int(sample_id): row
        for row, sample_id in enumerate(index["train_ts_sample_ids"].astype(np.int64))
    }
    references = np.full((len(reference_ids),) + expected_shape, np.nan, dtype=np.float32)
    valid = np.zeros(len(reference_ids), dtype=bool)
    for row, sample_id in enumerate(reference_ids):
        if int(sample_id) < 0:
            continue
        if int(sample_id) not in row_by_id:
            raise ValueError(f"Reference sample ID {int(sample_id)} is absent from the index")
        reference = np.asarray(index["train_ts"][row_by_id[int(sample_id)]], dtype=np.float32)
        if reference.ndim == 1:
            reference = reference[:, None]
        if reference.shape != expected_shape:
            raise ValueError(
                f"Reference {int(sample_id)} shape {reference.shape} != prediction shape {expected_shape}"
            )
        references[row] = reference
        valid[row] = True
    return references, valid


def training_variable_scale(train_ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return train-only per-variable mean/std used for all distance metrics."""
    train_ts = np.asarray(train_ts, dtype=np.float64)
    if train_ts.ndim == 2:
        train_ts = train_ts[:, :, None]
    if train_ts.ndim != 3:
        raise ValueError(f"train_ts must be [N,L,V], got {train_ts.shape}")
    mean = train_ts.mean(axis=(0, 1))
    scale = train_ts.std(axis=(0, 1))
    finite_positive = scale[np.isfinite(scale) & (scale > EPS)]
    fallback = float(np.median(finite_positive)) if finite_positive.size else 1.0
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, fallback)
    return mean.astype(np.float32), scale.astype(np.float32)


def standardized_rmse(a: np.ndarray, b: np.ndarray, scale: np.ndarray) -> np.ndarray:
    diff = (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) / scale
    return np.sqrt(np.mean(np.square(diff), axis=(-2, -1)))


def standardized_mae(a: np.ndarray, b: np.ndarray, scale: np.ndarray) -> np.ndarray:
    diff = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) / scale
    return np.mean(diff, axis=(-2, -1))


def mean_variable_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean Pearson correlation across nonconstant variables, one value per row."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    output = np.full(a.shape[0], np.nan, dtype=np.float64)
    for row in range(a.shape[0]):
        correlations = []
        for variable in range(a.shape[2]):
            av = a[row, :, variable]
            bv = b[row, :, variable]
            if np.std(av) <= EPS or np.std(bv) <= EPS:
                continue
            correlations.append(float(np.corrcoef(av, bv)[0, 1]))
        if correlations:
            output[row] = float(np.mean(correlations))
    return output


def candidate_diversity(candidates: np.ndarray, scale: np.ndarray) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.ndim != 4:
        raise ValueError(f"candidates must be [S,N,L,V], got {candidates.shape}")
    if candidates.shape[0] < 2:
        return np.zeros(candidates.shape[1], dtype=np.float64)
    total = np.zeros(candidates.shape[1], dtype=np.float64)
    pairs = 0
    for left in range(candidates.shape[0]):
        for right in range(left + 1, candidates.shape[0]):
            total += standardized_rmse(candidates[left], candidates[right], scale)
            pairs += 1
    return total / max(pairs, 1)


def _safe_field(artifact: Dict[str, np.ndarray], name: str, n: int, default=np.nan):
    if name not in artifact:
        return np.full(n, default)
    values = np.asarray(artifact[name]).reshape(-1)
    return values if len(values) == n else np.full(n, default)


def compute_reference_metrics(
    rag_artifact: Dict[str, np.ndarray],
    references: np.ndarray,
    valid_reference: np.ndarray,
    reference_ids: np.ndarray,
    trace: Dict[int, Dict],
    scale: np.ndarray,
    baseline_artifact: Optional[Dict[str, np.ndarray]] = None,
    copy_threshold: float = 0.05,
) -> pd.DataFrame:
    sample_ids = _sample_ids(rag_artifact)
    n_samples = len(sample_ids)
    prediction = _validate_series("predictions", rag_artifact["predictions"], n_samples)
    target = _validate_series("targets", rag_artifact["targets"], n_samples)
    candidates = np.asarray(rag_artifact["candidates"], dtype=np.float32)
    if candidates.ndim != 4 or candidates.shape[1:] != prediction.shape:
        raise ValueError(
            f"candidates must be [S,N,L,V] matching predictions; got {candidates.shape}"
        )

    rag_reference = np.full(n_samples, np.nan)
    rag_reference_mae = np.full(n_samples, np.nan)
    rag_reference_corr = np.full(n_samples, np.nan)
    reference_target = np.full(n_samples, np.nan)
    candidate_reference = np.full(n_samples, np.nan)
    if valid_reference.any():
        rows = np.flatnonzero(valid_reference)
        rag_reference[rows] = standardized_rmse(
            prediction[rows], references[rows], scale
        )
        rag_reference_mae[rows] = standardized_mae(
            prediction[rows], references[rows], scale
        )
        rag_reference_corr[rows] = mean_variable_correlation(
            prediction[rows], references[rows]
        )
        reference_target[rows] = standardized_rmse(
            references[rows], target[rows], scale
        )
        candidate_reference[rows] = np.mean(
            [
                standardized_rmse(candidate[rows], references[rows], scale)
                for candidate in candidates
            ],
            axis=0,
        )

    rag_target = standardized_rmse(prediction, target, scale)
    correction_gain = reference_target - rag_target
    correction_fraction = correction_gain / np.maximum(reference_target, EPS)
    ref_target_ratio = rag_reference / np.maximum(rag_target, EPS)
    diversity = candidate_diversity(candidates, scale)
    candidate_to_median = np.mean(
        [standardized_rmse(candidate, prediction, scale) for candidate in candidates], axis=0
    )

    similarities = []
    query_captions = []
    reference_captions = []
    fallback = []
    for sample_id in sample_ids:
        record = trace.get(int(sample_id), {})
        similarity = record.get("similarity_top1", record.get("similarity"))
        similarities.append(np.nan if similarity is None else float(similarity))
        query_captions.append(str(record.get("query_caption", "")))
        reference_captions.append(str(record.get("selected_reference_caption", "")))
        fallback.append(bool(record.get("fallback", False)))

    if "query_captions" in rag_artifact:
        artifact_captions = np.asarray(rag_artifact["query_captions"]).astype(str)
        query_captions = [
            query_captions[row] or str(artifact_captions[row]) for row in range(n_samples)
        ]

    frame = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "query_caption": query_captions,
            "reference_sample_id": reference_ids,
            "reference_caption": reference_captions,
            "retrieval_similarity": similarities,
            "fallback": fallback,
            "rag_reference_srmse": rag_reference,
            "rag_reference_smae": rag_reference_mae,
            "rag_reference_correlation": rag_reference_corr,
            "reference_target_srmse": reference_target,
            "rag_target_srmse": rag_target,
            "target_correction_gain": correction_gain,
            "target_correction_fraction": correction_fraction,
            "reference_vs_target_distance_ratio": ref_target_ratio,
            "candidate_diversity_srmse": diversity,
            "candidate_reference_srmse_mean": candidate_reference,
            "candidate_to_aggregate_srmse_mean": candidate_to_median,
            "near_reference_copy": valid_reference & (rag_reference <= copy_threshold),
            "rag_cttp": _safe_field(rag_artifact, "per_sample_cttp", n_samples),
            "strength": _safe_field(rag_artifact, "controller_strengths", n_samples),
            "start_step": _safe_field(
                rag_artifact, "controller_start_steps", n_samples, default=-1
            ),
        }
    )

    if baseline_artifact is not None:
        baseline = _validate_series(
            "baseline predictions", baseline_artifact["predictions"], n_samples
        )
        baseline_target = standardized_rmse(baseline, target, scale)
        baseline_reference = np.full(n_samples, np.nan)
        if valid_reference.any():
            rows = np.flatnonzero(valid_reference)
            baseline_reference[rows] = standardized_rmse(
                baseline[rows], references[rows], scale
            )
        frame["baseline_target_srmse"] = baseline_target
        frame["baseline_reference_srmse"] = baseline_reference
        frame["rag_baseline_srmse"] = standardized_rmse(prediction, baseline, scale)
        frame["reference_attraction"] = 1.0 - rag_reference / np.maximum(
            baseline_reference, EPS
        )
        frame["target_gain_vs_baseline"] = baseline_target - rag_target
        frame["baseline_cttp"] = _safe_field(
            baseline_artifact, "per_sample_cttp", n_samples
        )
        frame["cttp_gain_vs_baseline"] = frame["rag_cttp"] - frame["baseline_cttp"]
    return frame


def select_stratified_cases(
    metrics: pd.DataFrame, per_group: int = 2, seed: int = 42
) -> pd.DataFrame:
    """Predefined, deduplicated selection to reduce visual cherry-picking."""
    eligible = metrics[(~metrics["fallback"]) & metrics["reference_sample_id"].ge(0)].copy()
    if eligible.empty:
        return pd.DataFrame(columns=list(metrics.columns) + ["selection_reason"])
    rules = [
        ("high_similarity", "retrieval_similarity", False),
        ("low_similarity", "retrieval_similarity", True),
        ("best_target_correction", "target_correction_gain", False),
        ("worst_target_correction", "target_correction_gain", True),
    ]
    if "reference_attraction" in eligible:
        rules.append(("strongest_reference_attraction", "reference_attraction", False))
    else:
        rules.append(("closest_to_reference", "rag_reference_srmse", True))

    selected_rows: List[pd.Series] = []
    used = set()
    for reason, column, ascending in rules:
        ranked = eligible.dropna(subset=[column]).sort_values(
            [column, "sample_id"], ascending=[ascending, True]
        )
        count = 0
        for _, row in ranked.iterrows():
            sample_id = int(row["sample_id"])
            if sample_id in used:
                continue
            selected = row.copy()
            selected["selection_reason"] = reason
            selected_rows.append(selected)
            used.add(sample_id)
            count += 1
            if count >= per_group:
                break

    remaining = eligible[~eligible["sample_id"].isin(used)]
    if not remaining.empty:
        random_rows = remaining.sample(
            n=min(per_group, len(remaining)), random_state=int(seed), replace=False
        )
        for _, row in random_rows.iterrows():
            selected = row.copy()
            selected["selection_reason"] = "random"
            selected_rows.append(selected)
    if not selected_rows:
        return pd.DataFrame(columns=list(metrics.columns) + ["selection_reason"])
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _numeric_summary(values: pd.Series) -> Dict[str, Optional[float]]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {key: None for key in ("mean", "median", "q05", "q25", "q75", "q95")}
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "q05": float(values.quantile(0.05)),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
        "q95": float(values.quantile(0.95)),
    }


def build_summary(metrics: pd.DataFrame, copy_threshold: float, dataset_name: str) -> Dict:
    numeric_columns = [
        "retrieval_similarity",
        "rag_reference_srmse",
        "rag_reference_correlation",
        "reference_target_srmse",
        "rag_target_srmse",
        "target_correction_gain",
        "target_correction_fraction",
        "candidate_diversity_srmse",
        "reference_attraction",
        "baseline_target_srmse",
        "target_gain_vs_baseline",
    ]
    valid_reference = metrics["reference_sample_id"].ge(0) & metrics[
        "rag_reference_srmse"
    ].notna()
    valid_metrics = metrics[valid_reference]
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "dataset_name": dataset_name,
        "num_samples": int(len(metrics)),
        "num_retrieval_success": int(valid_reference.sum()),
        "num_fallback": int((~valid_reference).sum()),
        "copy_threshold_srmse": float(copy_threshold),
        "near_reference_copy_rate": float(valid_metrics["near_reference_copy"].mean())
        if len(valid_metrics)
        else None,
        "target_correction_positive_rate": float(
            (valid_metrics["target_correction_gain"] > 0).mean()
        )
        if len(valid_metrics)
        else None,
        "reference_closer_than_target_rate": float(
            (valid_metrics["reference_vs_target_distance_ratio"] < 1).mean()
        )
        if len(valid_metrics)
        else None,
        "metrics": {},
        "interpretation_guardrails": [
            "Distances are standardized by train-only per-variable standard deviations.",
            "Near-copy rate is operational and depends on the reported threshold.",
            "Reference attraction is a dependence indicator, not a causal estimate.",
            "Ground truth is used only for post-hoc evaluation, never retrieval.",
        ],
    }
    for column in numeric_columns:
        if column in metrics:
            summary["metrics"][column] = _numeric_summary(metrics[column])
    return summary


def _format_report_value(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _write_markdown_report(path: Path, summary: Dict):
    metric_rows = []
    for name, values in summary["metrics"].items():
        metric_rows.append(
            "| {} | {} | {} | {} | {} |".format(
                name,
                _format_report_value(values["mean"]),
                _format_report_value(values["median"]),
                _format_report_value(values["q25"]),
                _format_report_value(values["q75"]),
            )
        )
    lines = [
        "# Reference-dependence analysis report",
        "",
        "## Material Passport",
        "",
        f"- Artifact ID: `{ANALYSIS_VERSION}`",
        "- Artifact type: post-hoc experiment analysis",
        f"- Dataset: `{summary['dataset_name']}`",
        "- Verification status: ANALYZED",
        "- Retrieval source: train-only index recorded below",
        "",
        "## Key rates",
        "",
        f"- Samples: {summary['num_samples']}",
        f"- Successful references: {summary['num_retrieval_success']}",
        f"- Fallbacks: {summary['num_fallback']}",
        f"- Near-reference copy rate (threshold={summary['copy_threshold_srmse']}): "
        f"{_format_report_value(summary['near_reference_copy_rate'])}",
        "- Positive target-correction rate: "
        f"{_format_report_value(summary['target_correction_positive_rate'])}",
        "- Output-closer-to-reference-than-target rate: "
        f"{_format_report_value(summary['reference_closer_than_target_rate'])}",
        "",
        "## Metric distribution",
        "",
        "| Metric | Mean | Median | Q25 | Q75 |",
        "|---|---:|---:|---:|---:|",
        *metric_rows,
        "",
        "## Interpretation guardrails",
        "",
        *[f"- {item}" for item in summary["interpretation_guardrails"]],
        "",
        "## Provenance",
        "",
        f"- RAG predictions: `{summary['rag_predictions_path']}`",
        f"- Baseline predictions: `{summary['baseline_predictions_path'] or 'not provided'}`",
        f"- Retrieval trace: `{summary['retrieval_trace_path']}`",
        f"- Retrieval index: `{summary['retrieval_index_path']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def similarity_strata(metrics: pd.DataFrame) -> pd.DataFrame:
    valid = metrics.dropna(subset=["retrieval_similarity"]).copy()
    if valid.empty:
        return pd.DataFrame()
    ranks = valid["retrieval_similarity"].rank(method="first", pct=True)
    valid["similarity_stratum"] = pd.cut(
        ranks,
        bins=[0.0, 1 / 3, 2 / 3, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )
    columns = [
        "retrieval_similarity",
        "rag_reference_srmse",
        "reference_target_srmse",
        "rag_target_srmse",
        "target_correction_gain",
        "target_correction_fraction",
        "candidate_diversity_srmse",
        "near_reference_copy",
    ]
    if "reference_attraction" in valid:
        columns.append("reference_attraction")
    return valid.groupby("similarity_stratum", observed=True)[columns].agg(
        ["count", "mean", "median"]
    )


def _shorten(text: str, width: int = 180) -> str:
    text = " ".join(str(text).split())
    return textwrap.shorten(text, width=width, placeholder=" …")


def _plot_case(
    output_path: Path,
    row: pd.Series,
    prediction: np.ndarray,
    target: np.ndarray,
    reference: np.ndarray,
    candidates: np.ndarray,
    train_mean: np.ndarray,
    train_scale: np.ndarray,
    baseline: Optional[np.ndarray],
    max_line_variables: int,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_steps, n_variables = prediction.shape
    time = np.arange(n_steps)
    metadata = (
        f"Case {int(row['sample_id'])} · {row['selection_reason']} · "
        f"sim={row['retrieval_similarity']:.3f} · ref={int(row['reference_sample_id'])} · "
        f"d(RAG,ref)={row['rag_reference_srmse']:.3f} · "
        f"correction={row['target_correction_gain']:+.3f}\n"
        f"Query: {_shorten(row['query_caption'])}\n"
        f"Retrieved: {_shorten(row['reference_caption'])}"
    )

    if n_variables <= max_line_variables:
        columns = 2 if n_variables > 1 else 1
        rows = int(math.ceil(n_variables / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(7.0 * columns, 3.2 * rows), squeeze=False)
        for variable, axis in enumerate(axes.flat):
            if variable >= n_variables:
                axis.axis("off")
                continue
            for candidate in candidates:
                axis.plot(time, candidate[:, variable], color="#e57373", alpha=0.12, linewidth=0.8)
            axis.plot(time, target[:, variable], color="black", linewidth=2.0, label="Ground truth")
            axis.plot(time, reference[:, variable], color="#777777", linestyle="--", linewidth=1.8, label="Reference")
            axis.plot(time, prediction[:, variable], color="#d62728", linewidth=2.0, label="RAG+VerbalTS")
            if baseline is not None:
                axis.plot(time, baseline[:, variable], color="#1f77b4", linewidth=1.8, label="VerbalTS")
            axis.set_title(f"Variable {variable}")
            axis.set_xlabel("Time")
            axis.grid(alpha=0.2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False)
        fig.suptitle(metadata, fontsize=10, y=0.995)
        fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    else:
        standardized = {
            "Ground truth": (target - train_mean) / train_scale,
            "Reference": (reference - train_mean) / train_scale,
            "RAG+VerbalTS": (prediction - train_mean) / train_scale,
        }
        if baseline is not None:
            standardized["VerbalTS"] = (baseline - train_mean) / train_scale
        variable_score = np.mean(
            np.abs(standardized["Reference"] - standardized["Ground truth"])
            + np.abs(standardized["RAG+VerbalTS"] - standardized["Reference"]),
            axis=0,
        )
        chosen = np.argsort(-variable_score)[:max_line_variables]
        heatmap_count = len(standardized)
        line_rows = int(math.ceil(len(chosen) / 3))
        fig = plt.figure(figsize=(17, 4.0 + 3.0 * line_rows))
        grid = fig.add_gridspec(1 + line_rows, 12, height_ratios=[1.25] + [1.0] * line_rows)
        vmax = max(float(np.nanpercentile(np.abs(value), 98)) for value in standardized.values())
        vmax = max(vmax, 1.0)
        width = 12 // heatmap_count
        image = None
        for index, (name, values) in enumerate(standardized.items()):
            axis = fig.add_subplot(grid[0, index * width : (index + 1) * width])
            image = axis.imshow(
                values.T,
                aspect="auto",
                interpolation="nearest",
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
            )
            axis.set_title(name)
            axis.set_xlabel("Time")
            if index == 0:
                axis.set_ylabel("Variable")
        if image is not None:
            fig.colorbar(image, ax=fig.axes[:heatmap_count], shrink=0.75, label="Train-standardized value")
        for index, variable in enumerate(chosen):
            line_row = 1 + index // 3
            line_col = (index % 3) * 4
            axis = fig.add_subplot(grid[line_row, line_col : line_col + 4])
            for candidate in candidates:
                axis.plot(time, candidate[:, variable], color="#e57373", alpha=0.10, linewidth=0.7)
            axis.plot(time, target[:, variable], color="black", linewidth=1.8, label="Ground truth")
            axis.plot(time, reference[:, variable], color="#777777", linestyle="--", linewidth=1.5, label="Reference")
            axis.plot(time, prediction[:, variable], color="#d62728", linewidth=1.8, label="RAG+VerbalTS")
            if baseline is not None:
                axis.plot(time, baseline[:, variable], color="#1f77b4", linewidth=1.5, label="VerbalTS")
            axis.set_title(f"Variable {int(variable)}")
            axis.grid(alpha=0.2)
        handles, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False)
        fig.suptitle(metadata, fontsize=10, y=0.995)
        fig.subplots_adjust(top=0.83, bottom=0.08, hspace=0.5, wspace=0.55)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(output_path: Path, metrics: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    valid = metrics.dropna(subset=["retrieval_similarity", "rag_reference_srmse"])
    axes[0, 0].scatter(
        valid["retrieval_similarity"], valid["rag_reference_srmse"], s=10, alpha=0.35
    )
    axes[0, 0].set(xlabel="Retrieval similarity", ylabel="sRMSE(RAG, reference)", title="Similarity vs reference movement")

    valid = metrics.dropna(subset=["reference_target_srmse", "rag_target_srmse"])
    axes[0, 1].scatter(
        valid["reference_target_srmse"], valid["rag_target_srmse"], s=10, alpha=0.35
    )
    limit = max(
        float(valid[["reference_target_srmse", "rag_target_srmse"]].max().max())
        if not valid.empty
        else 1.0,
        1e-3,
    )
    axes[0, 1].plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
    axes[0, 1].set(
        xlabel="sRMSE(reference, target)",
        ylabel="sRMSE(RAG, target)",
        title="Points below diagonal are corrected toward target",
    )

    axes[1, 0].hist(metrics["target_correction_gain"].dropna(), bins=40, color="#d62728", alpha=0.8)
    axes[1, 0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set(xlabel="Target correction gain", ylabel="Samples", title="Positive means RAG improves over reference")

    axes[1, 1].scatter(
        metrics["rag_reference_srmse"], metrics["candidate_diversity_srmse"], s=10, alpha=0.35
    )
    axes[1, 1].set(
        xlabel="sRMSE(RAG, reference)",
        ylabel="Candidate pairwise sRMSE",
        title="Reference movement vs generation diversity",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_reference_dependence(
    rag_predictions_path: str,
    retrieval_trace_path: str,
    retrieval_index_path: str,
    output_dir: str,
    baseline_predictions_path: Optional[str] = None,
    copy_threshold: float = 0.05,
    cases_per_group: int = 2,
    selection_seed: int = 42,
    max_line_variables: int = 6,
    make_plots: bool = True,
) -> Dict:
    if copy_threshold < 0:
        raise ValueError("copy_threshold must be nonnegative")
    if cases_per_group < 0:
        raise ValueError("cases_per_group must be nonnegative")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rag = _load_npz(rag_predictions_path)
    sample_ids = _sample_ids(rag)
    trace = _load_trace(retrieval_trace_path)
    index = load_retrieval_index(retrieval_index_path)
    prediction = _validate_series("predictions", rag["predictions"], len(sample_ids))
    references_ids = _reference_ids(rag, sample_ids, trace)
    references, valid_reference = _references_from_index(
        index, references_ids, tuple(prediction.shape[1:])
    )
    train_mean, train_scale = training_variable_scale(index["train_ts"])

    baseline = None
    if baseline_predictions_path:
        baseline = _align_artifact(
            _load_npz(baseline_predictions_path), sample_ids, "baseline artifact"
        )
        baseline_targets = _validate_series("baseline targets", baseline["targets"], len(sample_ids))
        if not np.allclose(baseline_targets, rag["targets"], atol=1e-6, rtol=1e-6):
            raise ValueError("RAG and baseline artifacts contain different targets")

    metrics = compute_reference_metrics(
        rag,
        references,
        valid_reference,
        references_ids,
        trace,
        train_scale,
        baseline_artifact=baseline,
        copy_threshold=copy_threshold,
    )
    metrics.to_csv(output / "reference_dependence_metrics.csv", index=False)
    selected = select_stratified_cases(metrics, cases_per_group, selection_seed)
    selected.to_csv(output / "selected_cases.csv", index=False)
    strata = similarity_strata(metrics)
    strata.to_csv(output / "similarity_strata_summary.csv")

    dataset_name = str(index["metadata"].get("dataset_name", "unknown"))
    summary = build_summary(metrics, copy_threshold, dataset_name)
    summary.update(
        {
            "rag_predictions_path": str(rag_predictions_path),
            "baseline_predictions_path": baseline_predictions_path,
            "retrieval_trace_path": str(retrieval_trace_path),
            "retrieval_index_path": str(retrieval_index_path),
            "selected_case_count": int(len(selected)),
        }
    )
    with open(output / "reference_dependence_summary.json", "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2, allow_nan=False)
    _write_markdown_report(output / "reference_dependence_report.md", summary)

    if make_plots:
        figures = output / "figures"
        figures.mkdir(exist_ok=True)
        _plot_summary(figures / "reference_dependence_summary.png", metrics)
        row_by_id = {int(sample_id): row for row, sample_id in enumerate(sample_ids)}
        baseline_predictions = None if baseline is None else np.asarray(baseline["predictions"])
        candidates = np.asarray(rag["candidates"])
        targets = np.asarray(rag["targets"])
        for _, case in selected.iterrows():
            row = row_by_id[int(case["sample_id"])]
            filename = (
                f"case_{int(case['sample_id']):05d}_{case['selection_reason']}.png"
            )
            _plot_case(
                figures / filename,
                case,
                prediction[row],
                targets[row],
                references[row],
                candidates[:, row],
                train_mean,
                train_scale,
                None if baseline_predictions is None else baseline_predictions[row],
                max_line_variables,
            )
    return {
        "metrics": metrics,
        "selected_cases": selected,
        "summary": summary,
        "output_dir": str(output),
    }
