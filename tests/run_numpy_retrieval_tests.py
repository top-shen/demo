"""Dependency-light train-only retrieval checks (NumPy, no PyTorch/GPU)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval import TextRetriever, build_retrieval_index, load_retrieval_index


class FakeEmbedder:
    vectors = {
        "a0": [3, 0], "a1": [2, 1], "a2": [1, 2],
        "b0": [0, 3], "b1": [1, 3], "b2": [0, 2],
        "query-a": [1, 0], "query-mid": [1, 1], "query-low": [-1, 0],
    }

    def encode(self, texts):
        return np.asarray([self.vectors[str(text)] for text in texts], dtype=np.float32)


def main():
    with tempfile.TemporaryDirectory(prefix="ri_verbalts_numpy_") as folder:
        root = Path(folder)
        dataset = root / "Weather"
        dataset.mkdir()
        np.save(dataset / "train_text_caps.npy", np.asarray([["a0", "a1", "a2"], ["b0", "b1", "b2"]]))
        np.save(dataset / "train_ts.npy", np.arange(16, dtype=np.float32).reshape(2, 4, 2))
        np.save(dataset / "valid_text_caps.npy", np.asarray([["VALID-ONLY"]]))
        np.save(dataset / "test_text_caps.npy", np.asarray([["TEST-ONLY"]]))

        index_path = root / "index.npz"
        embedder = FakeEmbedder()
        build_retrieval_index(dataset, "Weather", index_path, embedder, "fake-longclip")
        index = load_retrieval_index(index_path)

        assert index["metadata"]["split"] == "train"
        assert "VALID-ONLY" not in index["captions"] and "TEST-ONLY" not in index["captions"]
        print("PASS train-only index and valid/test exclusion")
        assert index["sample_ids"].tolist() == [0, 0, 0, 1, 1, 1]
        assert index["caption_ids"].tolist() == [0, 1, 2, 0, 1, 2]
        print("PASS Weather multi-caption mapping")
        np.testing.assert_allclose(np.linalg.norm(index["embeddings"], axis=1), 1.0, atol=1e-6)
        print("PASS embedding L2 normalization")

        retriever = TextRetriever(index_path, embedder=embedder, seed=42)
        top1_a = retriever.retrieve(["query-a"], [8], top_k=4, selection="top1")[0]
        top1_b = retriever.retrieve(["query-a"], [8], top_k=4, selection="top1")[0]
        assert top1_a["reference_sample_id"] == top1_b["reference_sample_id"] == 0
        print("PASS deterministic Top-1")

        sampled_a = retriever.retrieve(
            ["query-mid"], [9], top_k=4, selection="sample", temperature=2.0,
            candidate_index=3,
        )[0]
        sampled_b = TextRetriever(index_path, embedder=embedder, seed=42).retrieve(
            ["query-mid"], [9], top_k=4, selection="sample", temperature=2.0,
            candidate_index=3,
        )[0]
        assert sampled_a["reference_sample_id"] == sampled_b["reference_sample_id"]
        assert sampled_a["reference_caption_id"] == sampled_b["reference_caption_id"]
        print("PASS fixed-seed Top-K sampling")

        fallback = retriever.retrieve(
            ["query-low"], [10], top_k=2, min_similarity=0.1
        )[0]
        assert fallback["fallback"] and fallback["reference_ts"] is None
        print("PASS minimum-similarity fallback")
        print("6 dependency-light retrieval checks passed")


if __name__ == "__main__":
    main()
