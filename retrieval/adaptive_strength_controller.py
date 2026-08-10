"""Learnable retrieval-adaptive diffusion strength controller.

The NumPy feature/calibration helpers intentionally remain usable without
PyTorch so data-leakage and teacher tests can run in lightweight environments.
The neural controller is imported only when PyTorch is available.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


CONTROLLER_VERSION = "ri-verbalts-adaptive-v1"
NORMALIZATION_SCHEMA = "score-zscore-controller-train-v1"
SCORE_FEATURE_NAMES = (
    "similarity_top1",
    "similarity_margin",
    "similarity_mean",
    "similarity_std",
    "similarity_entropy",
)


def compute_score_features(top_k_similarities: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute the five retrieval confidence features for every query."""
    scores = np.asarray(top_k_similarities, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("Top-K similarities must have shape [N,K] with K >= 2")
    if temperature <= 0:
        raise ValueError("score feature temperature must be positive")
    logits = scores.astype(np.float64) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1)
    return np.stack(
        [
            scores[:, 0],
            scores[:, 0] - scores[:, 1],
            scores.mean(axis=1),
            scores.std(axis=1),
            entropy.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def compute_pair_features(query_embeddings: np.ndarray, reference_embeddings: np.ndarray) -> np.ndarray:
    """Return ``[|e_q-e_r|, e_q*e_r]`` for frozen text embeddings."""
    query = np.asarray(query_embeddings, dtype=np.float32)
    reference = np.asarray(reference_embeddings, dtype=np.float32)
    if query.ndim != 2 or reference.shape != query.shape:
        raise ValueError("Query/reference embeddings must have the same [N,D] shape")
    return np.concatenate([np.abs(query - reference), query * reference], axis=1).astype(
        np.float32
    )


def fit_score_normalization(score_features: np.ndarray, train_mask=None) -> Dict:
    """Fit scalar-feature statistics using controller-train rows only."""
    values = np.asarray(score_features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(SCORE_FEATURE_NAMES):
        raise ValueError("Score features must have shape [N,5]")
    if train_mask is not None:
        mask = np.asarray(train_mask, dtype=bool)
        if mask.shape != (values.shape[0],):
            raise ValueError("train_mask has an incompatible shape")
        values = values[mask]
    if values.shape[0] == 0:
        raise ValueError("Cannot fit normalization without controller-train rows")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {
        "schema": NORMALIZATION_SCHEMA,
        "feature_names": list(SCORE_FEATURE_NAMES),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }


def normalize_score_features(score_features: np.ndarray, statistics: Dict) -> np.ndarray:
    if statistics.get("schema") != NORMALIZATION_SCHEMA:
        raise ValueError(f"Unsupported normalization schema: {statistics.get('schema')}")
    if list(statistics.get("feature_names", [])) != list(SCORE_FEATURE_NAMES):
        raise ValueError("Normalization feature definitions do not match controller features")
    values = np.asarray(score_features, dtype=np.float32)
    mean = np.asarray(statistics["mean"], dtype=np.float32)
    std = np.asarray(statistics["std"], dtype=np.float32)
    if values.shape[-1] != len(SCORE_FEATURE_NAMES) or mean.shape != (5,) or std.shape != (5,):
        raise ValueError("Invalid score normalization dimensions")
    return ((values - mean) / std).astype(np.float32)


def fit_similarity_quantiles(similarity_top1: np.ndarray, train_mask=None) -> Dict[str, float]:
    values = np.asarray(similarity_top1, dtype=np.float32).reshape(-1)
    if train_mask is not None:
        mask = np.asarray(train_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("train_mask has an incompatible shape")
        values = values[mask]
    if values.size == 0:
        raise ValueError("Cannot fit similarity quantiles without controller-train rows")
    q05, q95 = np.quantile(values, [0.05, 0.95])
    if float(q95 - q05) < 1e-6:
        q95 = q05 + 1e-6
    return {"similarity_q05": float(q05), "similarity_q95": float(q95)}


def similarity_confidence(similarity_top1, similarity_q05: float, similarity_q95: float):
    denominator = max(float(similarity_q95) - float(similarity_q05), 1e-6)
    return np.clip(
        (np.asarray(similarity_top1, dtype=np.float32) - float(similarity_q05)) / denominator,
        0.0,
        1.0,
    )


def similarity_base_strength(
    similarity_top1,
    similarity_q05: float,
    similarity_q95: float,
    min_strength: float = 0.20,
    max_strength: float = 0.95,
    gamma: float = 1.0,
):
    """Continuous prior: reliable retrieval gets lower diffusion strength.

    This implements the intended bounded decreasing mapping
    ``s_min + (s_max-s_min)*(1-c)^gamma``.  It is the only mapping consistent
    with the stated endpoints (confidence 0 -> s_max, confidence 1 -> s_min).
    """
    if not 0 <= min_strength <= max_strength <= 1:
        raise ValueError("Strength bounds must satisfy 0 <= min <= max <= 1")
    if gamma <= 0:
        raise ValueError("base_gamma must be positive")
    confidence = similarity_confidence(similarity_top1, similarity_q05, similarity_q95)
    return (
        float(min_strength)
        + (float(max_strength) - float(min_strength)) * np.power(1.0 - confidence, gamma)
    ).astype(np.float32)


def strength_to_start_steps(strength, num_steps: int) -> np.ndarray:
    if int(num_steps) < 2:
        raise ValueError("diffusion num_steps must be at least 2")
    values = np.asarray(strength, dtype=np.float32)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Strength values must be in [0,1]")
    return np.rint(values * (int(num_steps) - 1)).astype(np.int64)


def reference_embeddings_from_search(search_result: Dict, index_embeddings: np.ndarray) -> np.ndarray:
    rows = np.asarray(search_result["rows"], dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] < 1:
        raise ValueError("Search result has no Top-1 row")
    return np.asarray(index_embeddings, dtype=np.float32)[rows[:, 0]]


try:  # Keep NumPy tooling importable in environments without the training stack.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised by dependency-light scripts.
    torch = None
    nn = None
    F = None


if nn is not None:

    class AdaptiveStrengthController(nn.Module):
        """Small score/pair network with a similarity prior and two heads."""

        def __init__(
            self,
            embedding_dim: int,
            feature_mode: str = "score_only",
            hidden_dim: int = 128,
            pair_projection_dim: int = 128,
            dropout: float = 0.1,
            min_strength: float = 0.20,
            max_strength: float = 0.95,
            base_gamma: float = 1.0,
            max_residual: float = 0.15,
            similarity_q05: float = 0.0,
            similarity_q95: float = 1.0,
            use_similarity_prior: bool = True,
        ):
            super().__init__()
            if feature_mode not in {"score_only", "score_plus_pair"}:
                raise ValueError("feature_mode must be score_only or score_plus_pair")
            if not 0 <= min_strength <= max_strength <= 1:
                raise ValueError("Invalid controller strength bounds")
            if max_residual < 0:
                raise ValueError("max_residual must be nonnegative")
            self.embedding_dim = int(embedding_dim)
            self.feature_mode = feature_mode
            self.min_strength = float(min_strength)
            self.max_strength = float(max_strength)
            self.base_gamma = float(base_gamma)
            self.max_residual = float(max_residual)
            self.similarity_q05 = float(similarity_q05)
            self.similarity_q95 = float(similarity_q95)
            self.use_similarity_prior = bool(use_similarity_prior)
            if self.similarity_q95 <= self.similarity_q05:
                raise ValueError("similarity_q95 must exceed similarity_q05")

            input_dim = len(SCORE_FEATURE_NAMES)
            self.pair_projection = None
            if feature_mode == "score_plus_pair":
                self.pair_projection = nn.Sequential(
                    nn.LayerNorm(2 * self.embedding_dim),
                    nn.Linear(2 * self.embedding_dim, int(pair_projection_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                )
                input_dim += int(pair_projection_dim)
            self.trunk = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.LayerNorm(int(hidden_dim)),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )
            self.residual_head = nn.Linear(int(hidden_dim), 1)
            self.gate_head = nn.Linear(int(hidden_dim), 1)

        def _base_strength(self, similarity_top1):
            denominator = max(self.similarity_q95 - self.similarity_q05, 1e-6)
            confidence = torch.clamp(
                (similarity_top1 - self.similarity_q05) / denominator, 0.0, 1.0
            )
            base = self.min_strength + (self.max_strength - self.min_strength) * torch.pow(
                1.0 - confidence, self.base_gamma
            )
            return confidence, base

        def forward(
            self,
            score_features,
            similarity_top1,
            query_embeddings=None,
            reference_embeddings=None,
        ):
            features = score_features
            if self.feature_mode == "score_plus_pair":
                if query_embeddings is None or reference_embeddings is None:
                    raise ValueError("score_plus_pair requires query and reference embeddings")
                pair = torch.cat(
                    [
                        torch.abs(query_embeddings - reference_embeddings),
                        query_embeddings * reference_embeddings,
                    ],
                    dim=-1,
                )
                features = torch.cat([features, self.pair_projection(pair)], dim=-1)
            hidden = self.trunk(features)
            raw_strength = self.residual_head(hidden).squeeze(-1)
            gate_logit = self.gate_head(hidden).squeeze(-1)
            confidence, base = self._base_strength(similarity_top1)
            if self.use_similarity_prior:
                residual = self.max_residual * torch.tanh(raw_strength)
                strength = torch.clamp(
                    base + residual, min=self.min_strength, max=self.max_strength
                )
            else:
                strength = self.min_strength + (
                    self.max_strength - self.min_strength
                ) * torch.sigmoid(raw_strength)
                base = torch.full_like(strength, (self.min_strength + self.max_strength) / 2)
                residual = strength - base
            return {
                "confidence": confidence,
                "base_strength": base,
                "residual": residual,
                "strength": strength,
                "gate_logit": gate_logit,
                "gate_probability": torch.sigmoid(gate_logit),
            }

        @property
        def parameter_count(self) -> int:
            return sum(parameter.numel() for parameter in self.parameters())


else:

    class AdaptiveStrengthController:  # pragma: no cover - clear runtime error only.
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate AdaptiveStrengthController")


def controller_config_from_manifest(manifest: Dict) -> Dict:
    required = {
        "embedding_dim",
        "feature_mode",
        "hidden_dim",
        "pair_projection_dim",
        "dropout",
        "min_strength",
        "max_strength",
        "base_gamma",
        "max_residual",
        "similarity_q05",
        "similarity_q95",
    }
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"Controller manifest is missing fields: {sorted(missing)}")
    config = {name: manifest[name] for name in required}
    config["use_similarity_prior"] = bool(manifest.get("use_similarity_prior", True))
    return config


def validate_controller_manifest(manifest: Dict, expected: Optional[Dict] = None) -> None:
    required = {
        "controller_version",
        "dataset_identity",
        "retrieval_index_sha256",
        "embedding_dim",
        "feature_mode",
        "normalization",
        "num_steps",
    }
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"Controller manifest is missing fields: {sorted(missing)}")
    if manifest.get("controller_version") != CONTROLLER_VERSION:
        raise ValueError(f"Unsupported controller version: {manifest.get('controller_version')}")
    normalization = manifest.get("normalization", {})
    if normalization.get("schema") != NORMALIZATION_SCHEMA:
        raise ValueError("Controller normalization schema is incompatible")
    if list(normalization.get("feature_names", [])) != list(SCORE_FEATURE_NAMES):
        raise ValueError("Controller normalization features are incompatible")
    if manifest.get("feature_mode") not in {"score_only", "score_plus_pair"}:
        raise ValueError("Controller feature_mode is invalid")
    if int(manifest.get("embedding_dim", 0)) <= 0 or int(manifest.get("num_steps", 0)) < 2:
        raise ValueError("Controller embedding_dim/num_steps metadata is invalid")
    for key, value in (expected or {}).items():
        if value is not None and manifest.get(key) != value:
            raise ValueError(
                f"Controller checkpoint {key} mismatch: expected {value!r}, "
                f"found {manifest.get(key)!r}"
            )


def save_controller_checkpoint(path, model, optimizer, epoch: int, manifest: Dict) -> None:
    if torch is None:
        raise ImportError("PyTorch is required to save controller checkpoints")
    manifest = dict(manifest)
    manifest["controller_version"] = CONTROLLER_VERSION
    validate_controller_manifest(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
            "epoch": int(epoch),
            "manifest": manifest,
        },
        target,
    )


def load_controller_checkpoint(path, device="cpu", expected: Optional[Dict] = None):
    if torch is None:
        raise ImportError("PyTorch is required to load controller checkpoints")
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "manifest" not in payload or "model_state_dict" not in payload:
        raise ValueError("Invalid adaptive controller checkpoint")
    manifest = dict(payload["manifest"])
    validate_controller_manifest(manifest, expected=expected)
    model = AdaptiveStrengthController(**controller_config_from_manifest(manifest))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, manifest, payload


def quantization_aware_strength_loss(predicted_strength, target_strength, gate_target, num_steps):
    """Huber loss in discrete diffusion-step space, masked by feasible gates."""
    if torch is None:
        raise ImportError("PyTorch is required for controller losses")
    mask = gate_target > 0.5
    if not bool(mask.any()):
        return predicted_strength.sum() * 0.0
    predicted_step = predicted_strength[mask] * float(int(num_steps) - 1)
    target_step = torch.round(target_strength[mask] * float(int(num_steps) - 1))
    return F.huber_loss(predicted_step, target_step)


def monotonic_strength_loss(strength, confidence, margin: float = 0.05):
    """Penalize pairs where higher confidence receives higher strength."""
    if torch is None:
        raise ImportError("PyTorch is required for controller losses")
    confidence_difference = confidence[:, None] - confidence[None, :]
    valid = confidence_difference > float(margin)
    if not bool(valid.any()):
        return strength.sum() * 0.0
    violation = torch.relu(strength[:, None] - strength[None, :])
    return violation[valid].mean()


def monotonic_violation_rate(strength, confidence, margin: float = 0.05) -> float:
    strength = np.asarray(strength, dtype=np.float64).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    valid = confidence[:, None] - confidence[None, :] > float(margin)
    if not np.any(valid):
        return 0.0
    violations = strength[:, None] - strength[None, :] > 1e-8
    return float(violations[valid].mean())
