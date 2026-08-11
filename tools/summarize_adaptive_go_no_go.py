"""Aggregate the final train-only adaptive-strength go/no-go experiment."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


SEED_PATTERN = re.compile(r"seed(\d+)$")


def _mean(rows, key):
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Metric {key} contains a missing or non-finite value")
    return float(np.mean(values))


def _std(rows, key):
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Metric {key} contains a missing or non-finite value")
    return float(np.std(values))


def _seed(row):
    match = SEED_PATTERN.search(row["name"])
    if not match:
        raise ValueError(f"Controller name does not end in seedNN: {row['name']}")
    return int(match.group(1))


def summarize_rows(rows, experiment=None):
    pair = [
        row
        for row in rows
        if row["name"].startswith("pair_direct_seed")
        and "shuffled" not in row["name"]
    ]
    score = [row for row in rows if row["name"].startswith("score_only_direct_seed")]
    shuffled = [row for row in rows if row["name"].startswith("pair_direct_shuffled_seed")]
    if not pair or len(pair) != len(score) or len(pair) != len(shuffled):
        raise ValueError("Expected equally sized pair, score-only, and shuffled groups")
    if len(pair) < 3:
        raise ValueError("Stable go/no-go requires at least three controller seeds")
    pair_by_seed = {_seed(row): row for row in pair}
    score_by_seed = {_seed(row): row for row in score}
    shuffled_by_seed = {_seed(row): row for row in shuffled}
    if any(
        len(group) != len(pair)
        for group in (pair_by_seed, score_by_seed, shuffled_by_seed)
    ):
        raise ValueError("Controller comparison contains duplicate seed names")
    if set(pair_by_seed) != set(score_by_seed) or set(pair_by_seed) != set(shuffled_by_seed):
        raise ValueError("Controller comparison groups do not contain the same seeds")

    groups = {}
    for name, group in (("pair", pair), ("score_only", score), ("shuffled", shuffled)):
        groups[name] = {
            "seeds": sorted(_seed(row) for row in group),
            "strength_mae_mean": _mean(group, "learned_strength_mae"),
            "strength_mae_std": _std(group, "learned_strength_mae"),
            "start_step_mae_mean": _mean(group, "start_step_mae"),
            "start_step_mae_std": _std(group, "start_step_mae"),
            "gate_auroc_mean": _mean(group, "gate_auroc"),
            "gate_auroc_std": _std(group, "gate_auroc"),
            "gate_balanced_accuracy_mean": _mean(group, "gate_balanced_accuracy"),
            "beats_constant_strength_count": int(
                sum(bool(row["beats_constant"]) for row in group)
            ),
            "beats_constant_step_count": int(
                sum(bool(row["beats_constant_start_step"]) for row in group)
            ),
        }

    constant_strength_mae = _mean(pair, "constant_strength_mae")
    constant_step_mae = _mean(pair, "constant_start_step_mae")
    seeds = sorted(pair_by_seed)
    pair_vs_shuffled_strength = [
        shuffled_by_seed[seed]["learned_strength_mae"]
        - pair_by_seed[seed]["learned_strength_mae"]
        for seed in seeds
    ]
    pair_vs_score_strength = [
        score_by_seed[seed]["learned_strength_mae"]
        - pair_by_seed[seed]["learned_strength_mae"]
        for seed in seeds
    ]
    gate_advantage = [
        pair_by_seed[seed]["gate_auroc"] - shuffled_by_seed[seed]["gate_auroc"]
        for seed in seeds
    ]
    comparisons = {
        "constant_strength_mae": constant_strength_mae,
        "constant_start_step_mae": constant_step_mae,
        "pair_mean_strength_improvement_over_constant": float(
            constant_strength_mae - groups["pair"]["strength_mae_mean"]
        ),
        "pair_mean_step_improvement_over_constant": float(
            constant_step_mae - groups["pair"]["start_step_mae_mean"]
        ),
        "pair_strength_advantage_over_shuffled_by_seed": [
            float(value) for value in pair_vs_shuffled_strength
        ],
        "pair_strength_advantage_over_shuffled_mean": float(
            np.mean(pair_vs_shuffled_strength)
        ),
        "pair_strength_advantage_over_score_only_by_seed": [
            float(value) for value in pair_vs_score_strength
        ],
        "pair_strength_advantage_over_score_only_mean": float(
            np.mean(pair_vs_score_strength)
        ),
        "pair_gate_auroc_advantage_over_shuffled_by_seed": [
            float(value) for value in gate_advantage
        ],
        "pair_gate_auroc_advantage_over_shuffled_mean": float(np.mean(gate_advantage)),
    }
    required_wins = len(seeds) // 2 + 1
    strength_criteria = {
        "pair_strength_beats_constant_majority": (
            groups["pair"]["beats_constant_strength_count"] >= required_wins
        ),
        "pair_step_beats_constant_majority": (
            groups["pair"]["beats_constant_step_count"] >= required_wins
        ),
        "pair_mean_strength_beats_constant": (
            comparisons["pair_mean_strength_improvement_over_constant"] > 0
        ),
        "pair_mean_step_beats_constant": (
            comparisons["pair_mean_step_improvement_over_constant"] > 0
        ),
        "pair_mean_strength_beats_shuffled": (
            comparisons["pair_strength_advantage_over_shuffled_mean"] > 0
        ),
        "pair_strength_beats_shuffled_majority": (
            sum(value > 0 for value in pair_vs_shuffled_strength) >= required_wins
        ),
    }
    diagnostic_criteria = {
        "pair_mean_strength_beats_score_only": (
            comparisons["pair_strength_advantage_over_score_only_mean"] > 0
        ),
        "pair_strength_beats_score_only_majority": (
            sum(value > 0 for value in pair_vs_score_strength) >= required_wins
        ),
        "pair_gate_auroc_at_least_0_65": groups["pair"]["gate_auroc_mean"] >= 0.65,
        "pair_gate_auroc_advantage_over_shuffled_at_least_0_05": (
            comparisons["pair_gate_auroc_advantage_over_shuffled_mean"] >= 0.05
        ),
        "pair_gate_advantage_over_shuffled_majority": (
            sum(value > 0 for value in gate_advantage) >= required_wins
        ),
    }
    decision = "GO" if all(strength_criteria.values()) else "NO_GO"
    result = {
        "schema_version": "ri-verbalts-adaptive-go-no-go-v1",
        "groups": groups,
        "comparisons": comparisons,
        "adaptive_strength_criteria": strength_criteria,
        "diagnostic_criteria": diagnostic_criteria,
        "decision": decision,
        "failed_adaptive_strength_criteria": [
            name for name, passed in strength_criteria.items() if not passed
        ],
        "pair_feature_increment": (
            "SUPPORTED"
            if diagnostic_criteria["pair_mean_strength_beats_score_only"]
            and diagnostic_criteria["pair_strength_beats_score_only_majority"]
            else "NOT_SUPPORTED"
        ),
        "gate_signal": (
            "SUPPORTED"
            if diagnostic_criteria["pair_gate_auroc_at_least_0_65"]
            and diagnostic_criteria[
                "pair_gate_auroc_advantage_over_shuffled_at_least_0_05"
            ]
            and diagnostic_criteria["pair_gate_advantage_over_shuffled_majority"]
            else "NOT_SUPPORTED"
        ),
        "decision_scope": "train-only controller development; not a test-set conclusion",
        "recommended_next_step": (
            "lock the selected controller family and proceed to dataset validation"
            if decision == "GO"
            else "stop the current text-only per-sample strength design; retain gate/fixed-strength baselines or redesign features"
        ),
    }
    if experiment is not None:
        result["experiment"] = experiment
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--teacher-npz", default="")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--samples-per-action", type=int, default=0)
    parser.add_argument("--generation-seed", type=int, default=0)
    parser.add_argument("--controller-seeds", default="")
    args = parser.parse_args()
    comparison = json.loads(Path(args.comparison_json).read_text(encoding="utf-8"))
    experiment = {
        "teacher_npz": args.teacher_npz,
        "max_queries": args.max_queries,
        "samples_per_action": args.samples_per_action,
        "generation_seed": args.generation_seed,
        "controller_seeds": [
            int(value) for value in args.controller_seeds.split() if value
        ],
    }
    summary = summarize_rows(comparison["rows"], experiment=experiment)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Go/no-go summary written to {output_path}")


if __name__ == "__main__":
    main()
