"""Summarize teacher, controller-validation, and adaptive evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.adaptive_strength_controller import (
    monotonic_violation_rate,
    similarity_confidence,
)
from retrieval.strength_teacher import load_teacher_dataset


def json_safe(value):
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relationship_rows(similarity, strength, label):
    similarity = np.asarray(similarity, dtype=float)
    strength = np.asarray(strength, dtype=float)
    valid = np.isfinite(similarity) & np.isfinite(strength)
    return [
        {"source": label, "similarity": float(x), "strength": float(y)}
        for x, y in zip(similarity[valid], strength[valid])
    ]


def decile_rows(similarity, values, prefix="cttp"):
    similarity = np.asarray(similarity, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(similarity) & np.isfinite(values)
    similarity, values = similarity[valid], values[valid]
    if similarity.size == 0:
        return []
    edges = np.quantile(similarity, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges[1:-1], similarity, side="right"), 0, 9)
    rows = []
    for decile in range(10):
        mask = bins == decile
        if mask.any():
            rows.append(
                {
                    "similarity_decile": decile + 1,
                    "count": int(mask.sum()),
                    "similarity_mean": float(similarity[mask].mean()),
                    f"{prefix}_mean": float(values[mask].mean()),
                }
            )
    return rows


def bounded_monotonic_rate(strength, confidence, max_rows=2048):
    strength = np.asarray(strength)
    confidence = np.asarray(confidence)
    if strength.size > max_rows:
        order = np.argsort(confidence)
        positions = np.linspace(0, order.size - 1, max_rows).round().astype(np.int64)
        selected = order[positions]
        strength = strength[selected]
        confidence = confidence[selected]
    return monotonic_violation_rate(strength, confidence)


def load_trace(path):
    if not path:
        return []
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    # Fixed-reference traces contain one row per query. If a legacy/diverse
    # trace has candidates, keep candidate 0 for per-query analysis.
    return [record for record in records if int(record.get("candidate_id", 0)) == 0]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-npz", default="")
    parser.add_argument("--teacher-manifest", default="")
    parser.add_argument("--controller-predictions", default="")
    parser.add_argument("--evaluation-npz", default="")
    parser.add_argument("--retrieval-trace", default="")
    parser.add_argument("--fixed-sweep", default="")
    parser.add_argument("--copy-threshold", type=float)
    parser.add_argument("--output-dir", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    relationship = []
    histogram_rows = []

    teacher = manifest = None
    if args.teacher_npz:
        teacher, manifest = load_teacher_dataset(
            args.teacher_npz, args.teacher_manifest or None
        )
        similarity = teacher["top_k_similarities"][:, 0]
        target = teacher["soft_strength_targets"]
        relationship.extend(relationship_rows(similarity, target, "teacher"))
        valid = teacher["gate_targets"] > 0.5
        confidence = similarity_confidence(
            similarity, manifest["similarity_q05"], manifest["similarity_q95"]
        )
        summary["teacher"] = {
            "rows": int(similarity.size),
            "gate_rate": float(teacher["gate_targets"].mean()),
            "target_strength_mean": float(np.nanmean(target)),
            "target_strength_std": float(np.nanstd(target)),
            "monotonic_violation_rate": bounded_monotonic_rate(
                target[valid], confidence[valid]
            )
            if valid.any()
            else 0.0,
            "copy_rate_by_strength": (
                teacher["candidate_copy_distances"] < args.copy_threshold
            ).mean(axis=0).astype(float).tolist()
            if args.copy_threshold is not None
            else None,
        }

    if args.controller_predictions:
        with np.load(args.controller_predictions, allow_pickle=False) as archive:
            controller = {name: np.array(archive[name]) for name in archive.files}
        predicted = controller["predicted_strength"]
        target = controller["strength_targets"]
        valid = controller["gate_targets"] > 0.5
        summary["controller_validation"] = {
            "rows": int(predicted.size),
            "strength_mae": float(np.mean(np.abs(predicted[valid] - target[valid])))
            if valid.any()
            else None,
            "start_step_mae": float(
                np.mean(
                    np.abs(
                        controller["predicted_start_step"][valid]
                        - controller["target_start_step"][valid]
                    )
                )
            )
            if valid.any()
            else None,
            "gate_rate": float(
                (controller["predicted_gate_probability"] >= 0.5).mean()
            ),
            "teacher_target_error": float(np.mean(np.abs(predicted[valid] - target[valid])))
            if valid.any()
            else None,
        }
        if teacher is not None and "row_indices" in controller:
            rows = controller["row_indices"].astype(np.int64)
            relationship.extend(
                relationship_rows(
                    teacher["top_k_similarities"][rows, 0], predicted, "predicted_validation"
                )
            )

    trace = load_trace(args.retrieval_trace)
    if args.evaluation_npz:
        with np.load(args.evaluation_npz, allow_pickle=False) as archive:
            evaluation = {name: np.array(archive[name]) for name in archive.files}
        strengths = evaluation.get("controller_strengths", np.asarray([]))
        finite_strengths = strengths[np.isfinite(strengths)]
        if finite_strengths.size:
            counts, edges = np.histogram(finite_strengths, bins=10)
            histogram_rows = [
                {
                    "left": float(edges[index]),
                    "right": float(edges[index + 1]),
                    "count": int(count),
                }
                for index, count in enumerate(counts)
            ]
        fallback = evaluation.get("fallback_reasons", np.asarray([], dtype=str)).astype(str)
        actions = evaluation.get("controller_actions", np.asarray([], dtype=str)).astype(str)
        distances = evaluation.get(
            "generated_to_reference_distances", np.asarray([], dtype=float)
        )
        summary["evaluation"] = {
            "rows": int(evaluation["predictions"].shape[0]),
            "gate_rate": float(np.mean(actions == "adaptive_retrieval_diffusion"))
            if actions.size
            else None,
            "fallback_rate": float(np.mean(fallback != "")) if fallback.size else None,
            "copy_rate": float(np.nanmean(distances < args.copy_threshold))
            if distances.size and args.copy_threshold is not None
            else None,
            "strength_mean": float(finite_strengths.mean()) if finite_strengths.size else None,
            "strength_std": float(finite_strengths.std()) if finite_strengths.size else None,
        }
        if trace and len(trace) == evaluation["per_sample_cttp"].shape[0]:
            trace_similarity = np.asarray(
                [record.get("similarity_top1", np.nan) for record in trace], dtype=float
            )
            write_csv(
                output_dir / "similarity_decile_cttp.csv",
                decile_rows(trace_similarity, evaluation["per_sample_cttp"]),
            )
            relationship.extend(
                relationship_rows(trace_similarity, strengths, "predicted_test")
            )

    if args.fixed_sweep:
        sweep = json.loads(Path(args.fixed_sweep).read_text(encoding="utf-8"))
        summary["fixed_strength_sweep"] = sweep
        write_csv(output_dir / "fixed_strength_sweep.csv", sweep.get("rows", []))

    write_csv(output_dir / "similarity_strength.csv", relationship)
    write_csv(output_dir / "strength_histogram.csv", histogram_rows)
    (output_dir / "controller_analysis.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Analysis written to {output_dir}")


if __name__ == "__main__":
    main()
