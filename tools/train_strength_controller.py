"""Train the lightweight retrieval-adaptive strength/gate controller."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.adaptive_strength_controller import (
    CONTROLLER_VERSION,
    AdaptiveStrengthController,
    load_controller_checkpoint,
    monotonic_strength_loss,
    monotonic_violation_rate,
    quantization_aware_strength_loss,
    save_controller_checkpoint,
    similarity_confidence,
    strength_to_start_steps,
)
from retrieval.strength_teacher import load_teacher_dataset


def binary_metrics(target, probability, threshold=0.5):
    target = np.asarray(target, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    prediction = probability >= float(threshold)
    positive = target == 1
    tp = int(np.sum(prediction & positive))
    fp = int(np.sum(prediction & ~positive))
    fn = int(np.sum(~prediction & positive))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    positives = probability[positive]
    negatives = probability[~positive]
    if positives.size and negatives.size:
        order = np.argsort(probability, kind="mergesort")
        sorted_probability = probability[order]
        ranks = np.empty(probability.size, dtype=np.float64)
        start = 0
        while start < probability.size:
            end = start + 1
            while end < probability.size and sorted_probability[end] == sorted_probability[start]:
                end += 1
            average_rank = 0.5 * ((start + 1) + end)
            ranks[order[start:end]] = average_rank
            start = end
        rank_sum = ranks[positive].sum()
        auroc = float(
            (rank_sum - positives.size * (positives.size + 1) / 2)
            / (positives.size * negatives.size)
        )
    else:
        auroc = float("nan")
    return {
        "gate_auroc": auroc,
        "gate_f1": float(f1),
        "gate_precision": float(precision),
        "gate_recall": float(recall),
    }


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def evaluate_split(model, payload, indices, args, device):
    import torch
    import torch.nn.functional as F

    model.eval()
    index = np.asarray(indices, dtype=np.int64)
    output_chunks = {}
    gate_loss_sum = 0.0
    step_loss_sum = 0.0
    step_count = 0
    residual_sum = 0.0
    evaluation_batch_size = max(int(args.batch_size), 256)
    with torch.no_grad():
        for start in range(0, index.size, evaluation_batch_size):
            current = index[start : start + evaluation_batch_size]
            score = torch.as_tensor(
                payload["normalized_score_features"][current], device=device
            )
            top1 = torch.as_tensor(
                payload["top_k_similarities"][current, 0], device=device
            )
            query = torch.as_tensor(payload["query_embeddings"][current], device=device)
            reference = torch.as_tensor(
                payload["reference_embeddings"][current], device=device
            )
            gate = torch.as_tensor(payload["gate_targets"][current], device=device)
            target_strength = torch.as_tensor(
                np.nan_to_num(payload["soft_strength_targets"][current], nan=0.0),
                device=device,
            )
            output = model(score, top1, query, reference)
            gate_loss_sum += float(
                F.binary_cross_entropy_with_logits(
                    output["gate_logit"],
                    gate,
                    pos_weight=torch.tensor(args.gate_pos_weight, device=device),
                    reduction="sum",
                ).item()
            )
            valid_batch = gate > 0.5
            if bool(valid_batch.any()):
                predicted_step = output["strength"][valid_batch] * float(args.num_steps - 1)
                target_step = torch.round(
                    target_strength[valid_batch] * float(args.num_steps - 1)
                )
                step_loss_sum += float(
                    F.huber_loss(predicted_step, target_step, reduction="sum").item()
                )
                step_count += int(valid_batch.sum().item())
            residual_sum += float(output["residual"].square().sum().item())
            for name, value in output.items():
                output_chunks.setdefault(name, []).append(value.detach().cpu().numpy())

    arrays = {name: np.concatenate(values, axis=0) for name, values in output_chunks.items()}
    gate_loss_value = gate_loss_sum / max(index.size, 1)
    step_loss_value = step_loss_sum / max(step_count, 1)
    residual_loss_value = residual_sum / max(index.size, 1)
    # Exact all-pairs diagnostics are quadratic. Use a deterministic,
    # confidence-stratified subset for full-split validation; training uses
    # exact pairs inside each mini-batch.
    mono_count = min(int(index.size), 1024)
    confidence_order = np.argsort(arrays["confidence"])
    mono_positions = np.linspace(0, confidence_order.size - 1, mono_count).round().astype(np.int64)
    mono_indices = confidence_order[mono_positions]
    with torch.no_grad():
        mono_loss_tensor = monotonic_strength_loss(
            torch.as_tensor(arrays["strength"][mono_indices], device=device),
            torch.as_tensor(arrays["confidence"][mono_indices], device=device),
            args.monotonic_margin,
        )
    mono_loss_value = float(mono_loss_tensor.item())
    effective_gate_weight = 0.0 if args.disable_gate_loss else args.lambda_gate
    total_value = (
        effective_gate_weight * gate_loss_value
        + args.lambda_strength * step_loss_value
        + args.lambda_monotonic * mono_loss_value
        + args.lambda_residual * residual_loss_value
    )
    gate_array = payload["gate_targets"][index]
    target_array = payload["soft_strength_targets"][index]
    valid = gate_array > 0.5
    if np.any(valid):
        strength_mae = float(np.mean(np.abs(arrays["strength"][valid] - target_array[valid])))
        predicted_steps = strength_to_start_steps(arrays["strength"][valid], args.num_steps)
        target_steps = strength_to_start_steps(target_array[valid], args.num_steps)
        step_mae = float(np.mean(np.abs(predicted_steps - target_steps)))
    else:
        strength_mae = float("nan")
        step_mae = float("nan")
    metrics = {
        "loss": float(total_value),
        "gate_loss": float(gate_loss_value),
        "step_loss": float(step_loss_value),
        "monotonic_loss": float(mono_loss_value),
        "residual_loss": float(residual_loss_value),
        "strength_mae": strength_mae,
        "start_step_mae": step_mae,
        "monotonic_violation_rate": monotonic_violation_rate(
            arrays["strength"][mono_indices],
            arrays["confidence"][mono_indices],
            args.monotonic_margin,
        ),
        **binary_metrics(gate_array, arrays["gate_probability"], args.gate_threshold),
    }
    return metrics, arrays


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-npz", required=True)
    parser.add_argument("--teacher-manifest", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-mode", choices=["score_only", "score_plus_pair"], default="score_only")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--pair-projection-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min-strength", type=float, default=0.20)
    parser.add_argument("--max-strength", type=float, default=0.95)
    parser.add_argument("--base-gamma", type=float, default=1.0)
    parser.add_argument("--max-residual", type=float, default=0.15)
    parser.add_argument("--gate-threshold", type=float, default=0.50)
    parser.add_argument("--lambda-gate", type=float, default=1.0)
    parser.add_argument("--lambda-strength", type=float, default=1.0)
    parser.add_argument("--lambda-monotonic", type=float, default=0.2)
    parser.add_argument("--lambda-residual", type=float, default=0.01)
    parser.add_argument("--monotonic-margin", type=float, default=0.05)
    parser.add_argument("--gate-pos-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--shuffle-retrieval-features", action="store_true")
    parser.add_argument("--disable-gate-loss", action="store_true")
    parser.add_argument(
        "--direct-strength-head",
        action="store_true",
        help="Ablation: predict bounded strength directly instead of prior + residual.",
    )
    parser.add_argument(
        "--separate-task-towers",
        action="store_true",
        help="Use independent lightweight feature towers for strength and retrieval gate.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    payload, teacher_manifest = load_teacher_dataset(
        args.teacher_npz,
        args.teacher_manifest or None,
        for_training=True,
    )
    args.num_steps = int(teacher_manifest["verbalts_checkpoint"]["num_steps"])
    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    split_ids = np.asarray(payload["controller_split_ids"], dtype=np.int8)
    train_indices = np.flatnonzero(split_ids == 0)
    validation_indices = np.flatnonzero(split_ids == 1)
    if train_indices.size == 0 or validation_indices.size == 0:
        raise ValueError("Teacher file must contain controller-train and controller-validation rows")
    # Verify group isolation even if a third-party teacher file passed schema checks.
    train_samples = set(payload["query_sample_ids"][train_indices].astype(int).tolist())
    validation_samples = set(payload["query_sample_ids"][validation_indices].astype(int).tolist())
    overlap = train_samples.intersection(validation_samples)
    if overlap:
        raise ValueError(f"Controller split leaks query sample IDs: {sorted(overlap)[:5]}")

    if args.shuffle_retrieval_features:
        permutation = np.random.default_rng(args.seed).permutation(payload["query_sample_ids"].shape[0])
        payload = dict(payload)
        for field in (
            "normalized_score_features",
            "top_k_similarities",
            "query_embeddings",
            "reference_embeddings",
        ):
            payload[field] = payload[field][permutation]

    embedding_dim = int(payload["query_embeddings"].shape[1])
    model_config = {
        "embedding_dim": embedding_dim,
        "feature_mode": args.feature_mode,
        "hidden_dim": args.hidden_dim,
        "pair_projection_dim": args.pair_projection_dim,
        "dropout": args.dropout,
        "min_strength": args.min_strength,
        "max_strength": args.max_strength,
        "base_gamma": args.base_gamma,
        "max_residual": args.max_residual,
        "similarity_q05": float(teacher_manifest["similarity_q05"]),
        "similarity_q95": float(teacher_manifest["similarity_q95"]),
        "use_similarity_prior": not args.direct_strength_head,
        "separate_task_towers": args.separate_task_towers,
    }
    model = AdaptiveStrengthController(**model_config).to(device)
    if model.parameter_count > 500_000:
        raise ValueError(f"Controller has {model.parameter_count:,} parameters; limit is 500,000")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    index_tensor = torch.as_tensor(train_indices, dtype=torch.long)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(index_tensor),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_balance = {
        "train_rows": int(train_indices.size),
        "validation_rows": int(validation_indices.size),
        "train_gate_positive": int(payload["gate_targets"][train_indices].sum()),
        "train_gate_negative": int(train_indices.size - payload["gate_targets"][train_indices].sum()),
        "validation_gate_positive": int(payload["gate_targets"][validation_indices].sum()),
        "validation_gate_negative": int(validation_indices.size - payload["gate_targets"][validation_indices].sum()),
        "configured_positive_weight": args.gate_pos_weight,
    }
    print(json.dumps(class_balance, indent=2))
    manifest = {
        "controller_version": CONTROLLER_VERSION,
        "dataset_identity": teacher_manifest["dataset"],
        "retrieval_index_sha256": teacher_manifest["retrieval_index"]["sha256"],
        "embedding_dim": embedding_dim,
        "feature_mode": args.feature_mode,
        "num_steps": args.num_steps,
        "normalization": teacher_manifest["normalization"],
        "score_temperature": float(
            teacher_manifest.get("feature_definitions", {}).get("score_temperature", 1.0)
        ),
        "gate_threshold": args.gate_threshold,
        "parameter_count": model.parameter_count,
        "teacher_schema_version": teacher_manifest["schema_version"],
        "teacher_data": str(Path(args.teacher_npz)),
        "class_balance": class_balance,
        **model_config,
    }
    config_payload = vars(args).copy()
    config_payload["model"] = model_config
    (output_dir / "training_config.json").write_text(
        json.dumps(_json_safe(config_payload), indent=2), encoding="utf-8"
    )
    (output_dir / "calibration_stats.json").write_text(
        json.dumps(
            {
                "normalization": teacher_manifest["normalization"],
                "similarity_q05": teacher_manifest["similarity_q05"],
                "similarity_q95": teacher_manifest["similarity_q95"],
                "gate_threshold": args.gate_threshold,
                "class_balance": class_balance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    history = []
    best_loss = float("inf")
    patience = 0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = []
        for (batch_indices_tensor,) in loader:
            index = batch_indices_tensor.numpy()
            score = torch.as_tensor(payload["normalized_score_features"][index], device=device)
            top1 = torch.as_tensor(payload["top_k_similarities"][index, 0], device=device)
            query = torch.as_tensor(payload["query_embeddings"][index], device=device)
            reference = torch.as_tensor(payload["reference_embeddings"][index], device=device)
            gate = torch.as_tensor(payload["gate_targets"][index], device=device)
            target_strength = torch.as_tensor(
                np.nan_to_num(payload["soft_strength_targets"][index], nan=0.0), device=device
            )
            output = model(score, top1, query, reference)
            pos_weight = torch.tensor(args.gate_pos_weight, device=device)
            gate_loss = F.binary_cross_entropy_with_logits(
                output["gate_logit"], gate, pos_weight=pos_weight
            )
            step_loss = quantization_aware_strength_loss(
                output["strength"], target_strength, gate, args.num_steps
            )
            mono_loss = monotonic_strength_loss(
                output["strength"], output["confidence"], args.monotonic_margin
            )
            residual_loss = output["residual"].square().mean()
            effective_gate_weight = 0.0 if args.disable_gate_loss else args.lambda_gate
            loss = (
                effective_gate_weight * gate_loss
                + args.lambda_strength * step_loss
                + args.lambda_monotonic * mono_loss
                + args.lambda_residual * residual_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals.append(float(loss.detach().cpu()))

        train_metrics, _ = evaluate_split(model, payload, train_indices, args, device)
        validation_metrics, validation_arrays = evaluate_split(
            model, payload, validation_indices, args, device
        )
        row = {"epoch": epoch, "optimization_loss": float(np.mean(totals))}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        save_controller_checkpoint(output_dir / "last.pt", model, optimizer, epoch, manifest)
        if validation_metrics["loss"] < best_loss - args.min_delta:
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            patience = 0
            save_controller_checkpoint(output_dir / "best.pt", model, optimizer, epoch, manifest)
        else:
            patience += 1
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.5f} "
            f"val_loss={validation_metrics['loss']:.5f} "
            f"val_step_mae={validation_metrics['start_step_mae']:.3f}"
        )
        if patience >= args.patience:
            break

    with (output_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    best_model, best_manifest, _ = load_controller_checkpoint(
        output_dir / "best.pt",
        device=device,
        expected={
            "dataset_identity": teacher_manifest["dataset"],
            "embedding_dim": embedding_dim,
            "feature_mode": args.feature_mode,
            "num_steps": args.num_steps,
        },
    )
    final_metrics, arrays = evaluate_split(
        best_model, payload, validation_indices, args, device
    )
    np.savez_compressed(
        output_dir / "validation_predictions.npz",
        row_indices=validation_indices,
        query_sample_ids=payload["query_sample_ids"][validation_indices],
        gate_targets=payload["gate_targets"][validation_indices],
        strength_targets=payload["soft_strength_targets"][validation_indices],
        predicted_gate_probability=arrays["gate_probability"],
        predicted_strength=arrays["strength"],
        predicted_start_step=strength_to_start_steps(arrays["strength"], args.num_steps),
        target_start_step=strength_to_start_steps(
            np.nan_to_num(
                payload["soft_strength_targets"][validation_indices], nan=0.0
            ),
            args.num_steps,
        ),
        base_strength=arrays["base_strength"],
        predicted_residual=arrays["residual"],
    )
    residual = arrays["residual"]
    summary = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "final_validation_metrics": final_metrics,
        "residual_distribution": {
            "mean": float(residual.mean()),
            "std": float(residual.std()),
            "min": float(residual.min()),
            "max": float(residual.max()),
            "q05": float(np.quantile(residual, 0.05)),
            "q95": float(np.quantile(residual, 0.95)),
        },
        "class_balance": class_balance,
        "checkpoint_manifest": best_manifest,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    (output_dir / "checkpoint_manifest.json").write_text(
        json.dumps(_json_safe(best_manifest), indent=2), encoding="utf-8"
    )
    print(f"Best controller checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
