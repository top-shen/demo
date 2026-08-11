"""Summarize the locked Synth-M adaptive-strength validation benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


FIXED_PATTERN = re.compile(r"^fixed_(\d+)$")
REQUIRED_CONDITIONS = {
    "original",
    "retrieval_only",
    "handcrafted",
    "learned_pair",
    "learned_score_only",
    "shuffled_pair",
}


def _read_results(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    parsed = []
    for row in rows:
        parsed.append(
            {
                "run": int(row["run"]),
                "split": row.get("split", ""),
                "cttp": float(row["cttp"]),
                "fid": float(row["fid"]),
                "jftsd": float(row["jftsd"]),
            }
        )
    return parsed


def _mean(rows, metric):
    return float(np.mean([row[metric] for row in rows]))


def _std(rows, metric):
    values = [row[metric] for row in rows]
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _copy_rate(condition_dir, runs, threshold):
    copied = 0
    total = 0
    for run in runs:
        prediction_path = condition_dir / str(run) / "rag_predictions.npz"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing validation predictions: {prediction_path}")
        with np.load(prediction_path, allow_pickle=False) as archive:
            split = str(np.asarray(archive["evaluation_split"]).item())
            if split != "valid":
                raise ValueError(f"Prediction artifact is not validation-only: {prediction_path}")
            distances = np.asarray(
                archive["generated_to_reference_distances"], dtype=np.float32
            )
        copied += int(np.sum(np.isfinite(distances) & (distances < threshold)))
        total += int(distances.size)
    return float(copied / total) if total else None


def load_conditions(root, expected_runs, copy_threshold):
    import yaml

    conditions = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        results_path = directory / "results.csv"
        if not results_path.is_file():
            continue
        rows = _read_results(results_path)
        if any(row["split"] != "valid" for row in rows):
            raise ValueError(f"Condition {directory.name} results are not validation-only")
        runs = sorted(row["run"] for row in rows)
        if runs != list(range(expected_runs)):
            raise ValueError(
                f"Condition {directory.name} has runs {runs}, expected 0..{expected_runs - 1}"
            )
        for run in runs:
            config_path = directory / str(run) / "eval_configs.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if config.get("eval", {}).get("split") != "valid":
                raise ValueError(f"Condition {directory.name} is not validation-only")
            if int(config.get("eval", {}).get("max_batches", -1)) > 0:
                raise ValueError(
                    f"Condition {directory.name} is a partial pilot, not full validation"
                )
            if directory.name in {
                "learned_pair",
                "learned_score_only",
                "shuffled_pair",
                "handcrafted",
            }:
                gate_threshold = config["eval"]["rag"]["adaptive_controller"][
                    "gate_threshold"
                ]
                if float(gate_threshold) != 0.0:
                    raise ValueError(
                        f"Unsupported gate is enabled for {directory.name}: {gate_threshold}"
                    )
        summary = {
            "runs": runs,
            "cttp_mean": _mean(rows, "cttp"),
            "cttp_std": _std(rows, "cttp"),
            "fid_mean": _mean(rows, "fid"),
            "fid_std": _std(rows, "fid"),
            "jftsd_mean": _mean(rows, "jftsd"),
            "jftsd_std": _std(rows, "jftsd"),
            "per_run": rows,
        }
        if directory.name != "original":
            summary["copy_rate"] = _copy_rate(
                directory, runs, float(copy_threshold)
            )
        match = FIXED_PATTERN.match(directory.name)
        if match:
            summary["strength"] = int(match.group(1)) / 100.0
        conditions[directory.name] = summary
    missing = REQUIRED_CONDITIONS.difference(conditions)
    if missing:
        raise ValueError(f"Missing validation conditions: {sorted(missing)}")
    if not any(FIXED_PATTERN.match(name) for name in conditions):
        raise ValueError("Validation output contains no fixed-strength sweep")
    return conditions


def _paired_wins(candidate, baseline, metric, higher_is_better):
    candidate_rows = {row["run"]: row for row in candidate["per_run"]}
    baseline_rows = {row["run"]: row for row in baseline["per_run"]}
    wins = 0
    for run in sorted(candidate_rows):
        left = candidate_rows[run][metric]
        right = baseline_rows[run][metric]
        wins += int(left > right if higher_is_better else left < right)
    return wins


def summarize(conditions, cttp_margin, copy_threshold):
    original = conditions["original"]
    fixed = {
        name: row for name, row in conditions.items() if FIXED_PATTERN.match(name)
    }
    cttp_floor = original["cttp_mean"] * (1.0 - cttp_margin)
    feasible_fixed = {
        name: row for name, row in fixed.items() if row["cttp_mean"] >= cttp_floor
    }
    if not feasible_fixed:
        raise ValueError("No fixed strength satisfies the locked CTTP non-inferiority margin")
    # Locked before reading validation results: CTTP feasibility, then minimum
    # J-FTSD; FID and higher CTTP are deterministic secondary tie-breakers.
    best_fixed_name = min(
        feasible_fixed,
        key=lambda name: (
            feasible_fixed[name]["jftsd_mean"],
            feasible_fixed[name]["fid_mean"],
            -feasible_fixed[name]["cttp_mean"],
        ),
    )
    best_fixed = feasible_fixed[best_fixed_name]
    learned = conditions["learned_pair"]
    score_only = conditions["learned_score_only"]
    shuffled = conditions["shuffled_pair"]
    handcrafted = conditions["handcrafted"]
    retrieval_only = conditions["retrieval_only"]
    run_count = len(learned["runs"])

    criteria = {
        "cttp_noninferior_to_original": learned["cttp_mean"] >= cttp_floor,
        "fid_better_than_original": learned["fid_mean"] < original["fid_mean"],
        "jftsd_better_than_original": learned["jftsd_mean"] < original["jftsd_mean"],
        "jftsd_better_than_best_fixed": learned["jftsd_mean"] < best_fixed["jftsd_mean"],
        "jftsd_better_than_handcrafted": learned["jftsd_mean"] < handcrafted["jftsd_mean"],
        "jftsd_better_than_score_only": learned["jftsd_mean"] < score_only["jftsd_mean"],
        "jftsd_better_than_shuffled": learned["jftsd_mean"] < shuffled["jftsd_mean"],
        "copy_rate_below_retrieval_only": learned["copy_rate"] < retrieval_only["copy_rate"],
        "best_fixed_direction_consistent_all_runs": _paired_wins(
            learned, best_fixed, "jftsd", higher_is_better=False
        )
        == run_count,
    }
    comparisons = {
        "cttp_floor": cttp_floor,
        "learned_minus_original_cttp": learned["cttp_mean"] - original["cttp_mean"],
        "learned_minus_original_fid": learned["fid_mean"] - original["fid_mean"],
        "learned_minus_original_jftsd": learned["jftsd_mean"] - original["jftsd_mean"],
        "learned_minus_best_fixed_jftsd": learned["jftsd_mean"] - best_fixed["jftsd_mean"],
        "learned_minus_handcrafted_jftsd": learned["jftsd_mean"] - handcrafted["jftsd_mean"],
        "learned_minus_score_only_jftsd": learned["jftsd_mean"] - score_only["jftsd_mean"],
        "learned_minus_shuffled_jftsd": learned["jftsd_mean"] - shuffled["jftsd_mean"],
        "learned_copy_rate": learned["copy_rate"],
        "retrieval_only_copy_rate": retrieval_only["copy_rate"],
        "learned_jftsd_wins_vs_best_fixed": _paired_wins(
            learned, best_fixed, "jftsd", higher_is_better=False
        ),
        "num_runs": run_count,
    }
    return {
        "schema_version": "ri-verbalts-adaptive-validation-v1",
        "evaluation_split": "valid",
        "selection_rule": {
            "cttp_noninferiority_relative_margin": cttp_margin,
            "primary": "minimum mean J-FTSD among feasible fixed strengths",
            "tie_breakers": ["minimum mean FID", "maximum mean CTTP"],
        },
        "copy_threshold": copy_threshold,
        "conditions": conditions,
        "feasible_fixed_conditions": sorted(feasible_fixed),
        "best_fixed_condition": best_fixed_name,
        "best_fixed_strength": best_fixed["strength"],
        "comparisons": comparisons,
        "criteria": criteria,
        "decision": "GO_TO_TEST" if all(criteria.values()) else "STOP_OR_REVISE",
        "failed_criteria": [name for name, passed in criteria.items() if not passed],
        "decision_scope": "dataset validation only; test split remains untouched",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", required=True)
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--expected-runs", type=int, default=3)
    parser.add_argument("--cttp-margin", type=float, default=0.01)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.teacher_manifest).read_text(encoding="utf-8"))
    copy_threshold = manifest.get("copy_constraint", {}).get("threshold")
    if copy_threshold is None:
        raise ValueError("Teacher manifest does not contain an enabled copy threshold")
    root = Path(args.validation_root)
    conditions = load_conditions(root, args.expected_runs, float(copy_threshold))
    summary = summarize(conditions, float(args.cttp_margin), float(copy_threshold))

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with Path(args.output_csv).open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "condition",
            "strength",
            "cttp_mean",
            "cttp_std",
            "fid_mean",
            "fid_std",
            "jftsd_mean",
            "jftsd_std",
            "copy_rate",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, row in sorted(conditions.items()):
            writer.writerow(
                {
                    "condition": name,
                    **{field: row.get(field) for field in fields if field != "condition"},
                }
            )
    print(json.dumps(summary, indent=2))
    print(f"Validation summary written to {output_json}")


if __name__ == "__main__":
    main()
