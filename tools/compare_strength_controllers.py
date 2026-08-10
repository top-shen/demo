"""Compare controller variants against constant and similarity-prior baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.strength_teacher import load_teacher_dataset


def gate_metrics(target, probability, threshold):
    target = np.asarray(target, dtype=np.int8)
    prediction = np.asarray(probability) >= float(threshold)
    positive = target == 1
    negative = target == 0
    tp = int(np.sum(prediction & positive))
    fp = int(np.sum(prediction & negative))
    tn = int(np.sum(~prediction & negative))
    fn = int(np.sum(~prediction & positive))
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "gate_threshold": float(threshold),
        "gate_balanced_accuracy": float((recall + specificity) / 2),
        "gate_f1": float(f1),
        "gate_precision": float(precision),
        "gate_recall": float(recall),
        "gate_specificity": float(specificity),
        "gate_tp": tp,
        "gate_fp": fp,
        "gate_tn": tn,
        "gate_fn": fn,
    }


def best_balanced_gate_threshold(target, probability):
    candidates = [
        gate_metrics(target, probability, threshold)
        for threshold in np.linspace(0.01, 0.99, 99)
    ]
    return max(
        candidates,
        key=lambda row: (
            row["gate_balanced_accuracy"],
            row["gate_f1"],
            -abs(row["gate_threshold"] - 0.5),
        ),
    )


def parse_controller(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--controller must be NAME=OUTPUT_DIR")
    name, directory = value.split("=", 1)
    if not name or not directory:
        raise argparse.ArgumentTypeError("--controller must be NAME=OUTPUT_DIR")
    return name, Path(directory)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-npz", required=True)
    parser.add_argument("--teacher-manifest", default="")
    parser.add_argument("--controller", action="append", required=True, type=parse_controller)
    parser.add_argument("--output-dir", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    teacher, _ = load_teacher_dataset(
        args.teacher_npz, args.teacher_manifest or None, for_training=True
    )
    split = np.asarray(teacher["controller_split_ids"], dtype=np.int8)
    gate = np.asarray(teacher["gate_targets"], dtype=np.float32)
    target = np.asarray(teacher["soft_strength_targets"], dtype=np.float32)
    train_positive = (split == 0) & (gate > 0.5)
    if not train_positive.any():
        raise ValueError("Relabeled Teacher has no positive controller-train actions")
    constant_strength = float(np.nanmean(target[train_positive]))

    rows = []
    for name, directory in args.controller:
        prediction_path = directory / "validation_predictions.npz"
        summary_path = directory / "training_summary.json"
        manifest_path = directory / "checkpoint_manifest.json"
        with np.load(prediction_path, allow_pickle=False) as archive:
            prediction = {field: np.array(archive[field]) for field in archive.files}
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        action = prediction["gate_targets"] > 0.5
        if not action.any():
            raise ValueError(f"Controller {name} has no positive validation actions")
        strength_target = prediction["strength_targets"][action]
        learned_mae = float(
            np.mean(np.abs(prediction["predicted_strength"][action] - strength_target))
        )
        base_mae = float(np.mean(np.abs(prediction["base_strength"][action] - strength_target)))
        constant_mae = float(np.mean(np.abs(constant_strength - strength_target)))
        probability = prediction["predicted_gate_probability"]
        threshold_metrics = best_balanced_gate_threshold(
            prediction["gate_targets"].astype(np.int8), probability
        )
        positive_probability = probability[prediction["gate_targets"] > 0.5]
        negative_probability = probability[prediction["gate_targets"] < 0.5]
        validation_metrics = summary["final_validation_metrics"]
        row = {
            "name": name,
            "feature_mode": checkpoint_manifest["feature_mode"],
            "uses_similarity_prior": bool(
                checkpoint_manifest.get("use_similarity_prior", True)
            ),
            "lambda_monotonic": float(
                json.loads((directory / "training_config.json").read_text(encoding="utf-8"))[
                    "lambda_monotonic"
                ]
            ),
            "constant_train_strength": constant_strength,
            "learned_strength_mae": learned_mae,
            "base_strength_mae": base_mae,
            "constant_strength_mae": constant_mae,
            "beats_base": bool(learned_mae < base_mae),
            "beats_constant": bool(learned_mae < constant_mae),
            "start_step_mae": float(validation_metrics["start_step_mae"]),
            "gate_auroc": validation_metrics.get("gate_auroc"),
            "gate_probability_positive_mean": float(positive_probability.mean()),
            "gate_probability_negative_mean": float(negative_probability.mean()),
            "monotonic_violation_rate": float(
                validation_metrics["monotonic_violation_rate"]
            ),
            "best_epoch": int(summary["best_epoch"]),
            **threshold_metrics,
        }
        rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "controller_variant_comparison.json").write_text(
        json.dumps({"rows": rows}, indent=2), encoding="utf-8"
    )
    with (output_dir / "controller_variant_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": rows}, indent=2))
    print(f"Comparison written to {output_dir}")


if __name__ == "__main__":
    main()
