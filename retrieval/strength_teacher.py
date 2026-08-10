"""Teacher labels, schema validation, and semantic scoring interfaces."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


TEACHER_SCHEMA_VERSION = "ri-verbalts-strength-teacher-v1"
REQUIRED_TEACHER_FIELDS = {
    "query_sample_ids",
    "query_caption_ids",
    "reference_sample_ids",
    "reference_caption_ids",
    "query_embeddings",
    "reference_embeddings",
    "top_k_sample_ids",
    "top_k_caption_ids",
    "top_k_similarities",
    "score_features",
    "normalized_score_features",
    "candidate_strengths",
    "candidate_start_steps",
    "candidate_semantic_scores",
    "candidate_copy_distances",
    "original_semantic_scores",
    "feasible_masks",
    "soft_strength_targets",
    "teacher_ambiguity_variance",
    "gate_targets",
    "controller_split_ids",
}


class TeacherSemanticScorer(ABC):
    """Per-sample semantic scorer used by the teacher builder."""

    identity = "abstract"

    @abstractmethod
    def score(self, query_captions, generated_ts, ts_lengths):
        """Return one semantic score for every query/generated pair."""


class CTTPTeacherSemanticScorer(TeacherSemanticScorer):
    identity = "cttp"

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)

    def score(self, query_captions, generated_ts, ts_lengths):
        ts_embeddings, text_embeddings = self.embed(
            query_captions, generated_ts, ts_lengths
        )
        scores = (ts_embeddings * text_embeddings).sum(axis=-1)
        return scores.astype(np.float32)

    def embed(self, query_captions, generated_ts, ts_lengths):
        import torch

        device = self.model.device
        values = torch.as_tensor(generated_ts, dtype=torch.float32, device=device)
        lengths = torch.as_tensor(ts_lengths, dtype=torch.int32, device=device)
        with torch.no_grad():
            ts_embeddings = self.model.get_ts_coemb(values, lengths)
            text_embeddings = self.model.get_text_coemb(list(query_captions), None)
        ts_embeddings = ts_embeddings.detach().cpu().numpy().astype(np.float32)
        text_embeddings = text_embeddings.detach().cpu().numpy().astype(np.float32)
        return ts_embeddings, text_embeddings


class PrecomputedTeacherSemanticScorer(TeacherSemanticScorer):
    """Sequential adapter for externally computed per-sample scores."""

    identity = "precomputed"

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        self.cursor = 0

    def score(self, query_captions, generated_ts, ts_lengths):
        count = len(query_captions)
        end = self.cursor + count
        if end > self.scores.size:
            raise ValueError("Precomputed semantic score file is exhausted")
        result = self.scores[self.cursor:end]
        self.cursor = end
        return result.copy()


def generated_to_reference_distance(generated_ts, reference_ts, metric: str = "rmse"):
    generated = np.asarray(generated_ts, dtype=np.float32)
    reference = np.asarray(reference_ts, dtype=np.float32)
    if generated.shape != reference.shape or generated.ndim < 2:
        raise ValueError("Generated/reference arrays must have the same [N,...] shape")
    if metric != "rmse":
        raise NotImplementedError(
            f"Distance metric {metric!r} is reserved for extension; supported: rmse"
        )
    axes = tuple(range(1, generated.ndim))
    return np.sqrt(np.mean(np.square(generated - reference), axis=axes)).astype(np.float32)


def estimate_copy_threshold(
    train_ts,
    quantile: float = 0.05,
    num_pairs: int = 4096,
    seed: int = 0,
    metric: str = "rmse",
) -> Dict:
    """Estimate a low distance quantile without constructing an N^2 matrix."""
    values = np.asarray(train_ts, dtype=np.float32)
    if values.shape[0] < 2:
        raise ValueError("At least two training series are required")
    if not 0 < quantile < 1 or num_pairs < 1:
        raise ValueError("Invalid threshold-estimation settings")
    rng = np.random.default_rng(int(seed))
    left = rng.integers(0, values.shape[0], size=int(num_pairs))
    right = rng.integers(0, values.shape[0] - 1, size=int(num_pairs))
    right += right >= left
    distances = generated_to_reference_distance(values[left], values[right], metric=metric)
    return {
        "threshold": float(np.quantile(distances, quantile)),
        "quantile": float(quantile),
        "num_pairs": int(num_pairs),
        "seed": int(seed),
        "metric": metric,
        "distance_mean": float(distances.mean()),
        "distance_std": float(distances.std()),
    }


def build_teacher_targets(
    candidate_strengths,
    candidate_semantic_scores,
    original_semantic_scores,
    epsilon_sem: float,
    teacher_temperature: float,
    candidate_copy_distances=None,
    copy_threshold: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Construct the feasible set, gate, soft strength target, and ambiguity."""
    strengths = np.asarray(candidate_strengths, dtype=np.float32).reshape(-1)
    semantic = np.asarray(candidate_semantic_scores, dtype=np.float32)
    original = np.asarray(original_semantic_scores, dtype=np.float32).reshape(-1)
    if semantic.ndim != 2 or semantic.shape[1] != strengths.size:
        raise ValueError("candidate_semantic_scores must have shape [N,J]")
    if semantic.shape[0] != original.size:
        raise ValueError("Original and candidate score counts differ")
    if epsilon_sem < 0 or teacher_temperature <= 0:
        raise ValueError("epsilon_sem must be nonnegative and teacher_temperature positive")
    feasible = semantic >= (original[:, None] - float(epsilon_sem))
    if copy_threshold is not None:
        distances = np.asarray(candidate_copy_distances, dtype=np.float32)
        if distances.shape != semantic.shape:
            raise ValueError("Copy distances must match candidate semantic scores")
        feasible &= distances >= float(copy_threshold)

    gate = feasible.any(axis=1).astype(np.float32)
    targets = np.full(original.shape, np.nan, dtype=np.float32)
    ambiguity = np.full(original.shape, np.nan, dtype=np.float32)
    weights = np.zeros_like(semantic, dtype=np.float32)
    raw = np.exp(-strengths.astype(np.float64) / float(teacher_temperature))
    for row in range(semantic.shape[0]):
        mask = feasible[row]
        if not mask.any():
            continue
        row_weights = raw[mask]
        row_weights /= row_weights.sum()
        weights[row, mask] = row_weights.astype(np.float32)
        target = float(np.sum(row_weights * strengths[mask]))
        targets[row] = target
        ambiguity[row] = float(np.sum(row_weights * np.square(strengths[mask] - target)))
    return {
        "feasible_masks": feasible,
        "teacher_weights": weights,
        "soft_strength_targets": targets,
        "teacher_ambiguity_variance": ambiguity,
        "gate_targets": gate,
    }


def grouped_controller_split(
    query_sample_ids, validation_fraction: float = 0.2, seed: int = 0
) -> np.ndarray:
    """Return 0=train/1=validation while keeping captions of a sample together."""
    sample_ids = np.asarray(query_sample_ids, dtype=np.int64).reshape(-1)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    unique = np.unique(sample_ids)
    if unique.size < 2:
        raise ValueError("At least two unique sample IDs are required for a split")
    rng = np.random.default_rng(int(seed))
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    count = min(max(1, int(round(unique.size * validation_fraction))), unique.size - 1)
    validation_ids = set(shuffled[:count].tolist())
    return np.asarray([1 if int(value) in validation_ids else 0 for value in sample_ids], dtype=np.int8)


def sha256_file(path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_teacher_payload(payload: Dict[str, np.ndarray], manifest: Dict, for_training=False):
    if manifest.get("schema_version") != TEACHER_SCHEMA_VERSION:
        raise ValueError(f"Unsupported teacher schema: {manifest.get('schema_version')}")
    if manifest.get("split") != "train":
        raise ValueError("Teacher data must be constructed from split=train")
    if for_training and str(manifest.get("split", "")).lower() == "test":
        raise ValueError("Test-split teacher data cannot be used for controller training")
    manifest_required = {"schema_version", "split", "dataset", "data_file"}
    if for_training:
        manifest_required.update(
            {
                "normalization",
                "similarity_q05",
                "similarity_q95",
                "verbalts_checkpoint",
            }
        )
    missing_manifest = manifest_required.difference(manifest)
    if missing_manifest:
        raise ValueError(
            f"Teacher manifest is missing fields: {sorted(missing_manifest)}"
        )
    if for_training and "num_steps" not in manifest.get("verbalts_checkpoint", {}):
        raise ValueError("Teacher manifest verbalts_checkpoint.num_steps is required")
    if for_training:
        normalization = manifest.get("normalization", {})
        if normalization.get("schema") != "score-zscore-controller-train-v1":
            raise ValueError("Teacher normalization schema is incompatible")
        if len(normalization.get("mean", [])) != 5 or len(normalization.get("std", [])) != 5:
            raise ValueError("Teacher normalization statistics must contain five features")
    missing = REQUIRED_TEACHER_FIELDS.difference(payload)
    if missing:
        raise ValueError(f"Teacher NPZ is missing fields: {sorted(missing)}")
    count = np.asarray(payload["query_sample_ids"]).shape[0]
    row_fields = REQUIRED_TEACHER_FIELDS.difference(
        {"candidate_strengths", "candidate_start_steps"}
    )
    for field in row_fields:
        if np.asarray(payload[field]).shape[0] != count:
            raise ValueError(f"Teacher field {field} has an inconsistent row count")
    if np.any(np.asarray(payload["controller_split_ids"]) == 2):
        raise ValueError("Teacher split IDs may only encode controller train/validation")


def save_teacher_dataset(npz_path, manifest_path, payload: Dict, manifest: Dict):
    manifest = dict(manifest)
    manifest["schema_version"] = TEACHER_SCHEMA_VERSION
    manifest["split"] = "train"
    manifest["data_file"] = str(Path(npz_path))
    array_payload = {key: np.asarray(value) for key, value in payload.items()}
    validate_teacher_payload(array_payload, manifest)
    npz_target = Path(npz_path)
    manifest_target = Path(manifest_path)
    npz_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_target, **array_payload)
    manifest["build_progress"] = {
        "completed_queries": int(array_payload["query_sample_ids"].shape[0]),
        "complete": True,
    }
    manifest_target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_teacher_dataset(npz_path, manifest_path=None, for_training=False):
    npz_path = Path(npz_path)
    manifest_path = Path(manifest_path) if manifest_path else npz_path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name]) for name in archive.files}
    validate_teacher_payload(payload, manifest, for_training=for_training)
    return payload, manifest
