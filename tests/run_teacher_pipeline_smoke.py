"""Tiny fake teacher-schema pipeline smoke test (NumPy only)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.strength_teacher import (
    TeacherSemanticScorer,
    load_teacher_dataset,
    save_teacher_dataset,
)
from tools.build_strength_teacher import assemble_payload, write_sweep_summary


class FakeScorer(TeacherSemanticScorer):
    identity = "fake"

    def score(self, query_captions, generated_ts, ts_lengths):
        values = np.asarray(generated_ts, dtype=np.float32)
        return -np.square(values).mean(axis=tuple(range(1, values.ndim)))


def main():
    scorer = FakeScorer()
    assert scorer.score(["q"], np.ones((1, 4, 2)), [4]).shape == (1,)
    strengths = np.asarray([0.2, 0.5, 0.8], dtype=np.float32)
    parts = []
    for row in range(6):
        query = np.asarray([1.0, row + 1.0, 0.5], dtype=np.float32)
        query /= np.linalg.norm(query)
        reference = np.asarray([0.5, 1.0, row + 1.0], dtype=np.float32)
        reference /= np.linalg.norm(reference)
        parts.append(
            {
                "query_sample_id": np.asarray(row // 2),
                "query_caption_id": np.asarray(row % 2),
                "reference_sample_id": np.asarray((row // 2 + 1) % 3),
                "reference_caption_id": np.asarray(0),
                "query_embedding": query,
                "reference_embedding": reference,
                "top_k_sample_ids": np.asarray([4, 5, 6, 7]),
                "top_k_caption_ids": np.asarray([0, 0, 0, 0]),
                "top_k_similarities": np.asarray(
                    [0.95 - row * 0.03, 0.8, 0.7, 0.6], dtype=np.float32
                ),
                "candidate_semantic_scores": np.asarray([0.7, 0.82, 0.9]),
                "candidate_copy_distances": np.asarray([0.01, 0.1, 0.3]),
                "original_semantic_score": np.asarray(0.83),
                "retrieval_only_semantic_score": np.asarray(0.65),
                "retrieval_only_copy_distance": np.asarray(0.0),
            }
        )
    args = SimpleNamespace(
        score_temperature=1.0,
        validation_fraction=0.34,
        split_seed=9,
        epsilon_sem=0.02,
        teacher_temperature=0.1,
        disable_copy_constraint=False,
        copy_threshold=0.05,
    )
    payload, normalization, quantiles = assemble_payload(parts, strengths, 50, args)
    with tempfile.TemporaryDirectory(prefix="adaptive_teacher_smoke_") as folder:
        root = Path(folder)
        save_teacher_dataset(
            root / "teacher.npz",
            root / "teacher.json",
            payload,
            {
                "dataset": "fake",
                "normalization": normalization,
                "verbalts_checkpoint": {"num_steps": 50},
                **quantiles,
            },
        )
        loaded, manifest = load_teacher_dataset(
            root / "teacher.npz", root / "teacher.json", for_training=True
        )
        write_sweep_summary(loaded, root / "sweep.json", copy_threshold=0.05)
        assert loaded["gate_targets"].shape == (6,)
        assert loaded["normalized_score_features"].shape == (6, 5)
        assert manifest["split"] == "train"
        assert (root / "sweep.json").exists()
    print("PASS tiny fake teacher -> schema -> reload -> sweep pipeline")


if __name__ == "__main__":
    main()
