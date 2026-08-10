"""Text-to-text retrieval over captions from a dataset's training split.

The first RI-VerbalTS stage deliberately keeps retrieval outside training.  An
index stores one row per training caption (so Weather contributes three rows
per time series), while the referenced time series is stored once per training
sample.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


INDEX_VERSION = "ri-verbalts-text-v1"


def _as_caption_matrix(captions: np.ndarray) -> np.ndarray:
    captions = np.asarray(captions)
    if captions.ndim == 1:
        captions = captions[:, None]
    if captions.ndim != 2:
        raise ValueError(f"Expected captions with shape [samples, captions], got {captions.shape}")
    return captions


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, got {embeddings.shape}")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("The embedding model returned at least one zero vector")
    return embeddings / norms


class LongCLIPTextEmbedder:
    """Frozen LongCLIP pooled text projection used by both build and query."""

    def __init__(self, model_path: str, device: str = "cpu", batch_size: int = 64):
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModelWithProjection

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_path = str(model_path)
        self._torch = torch
        self._functional = F
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        text_config = CLIPTextConfig.from_pretrained(self.model_path, local_files_only=True)
        self.model = CLIPTextModelWithProjection.from_pretrained(
            self.model_path, config=text_config, local_files_only=True
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.max_length = int(self.model.config.max_position_embeddings)

    @classmethod
    def from_components(cls, model, tokenizer, device, batch_size: int = 64):
        """Reuse an already-loaded frozen LongCLIP (for example CTTP's copy)."""
        import torch
        import torch.nn.functional as F

        instance = cls.__new__(cls)
        instance.model_path = "shared-components"
        instance._torch = torch
        instance._functional = F
        instance.device = torch.device(device)
        instance.batch_size = int(batch_size)
        instance.tokenizer = tokenizer
        instance.model = model
        instance.model.eval()
        instance.model.requires_grad_(False)
        instance.max_length = int(model.config.max_position_embeddings)
        return instance

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [str(text) for text in texts]
        if not texts:
            return np.empty((0, int(self.model.config.projection_dim)), dtype=np.float32)

        outputs = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                tokens = self.tokenizer(
                    texts[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokens = {name: value.to(self.device) for name, value in tokens.items()}
                pooled = self.model(**tokens).text_embeds
                pooled = self._functional.normalize(pooled.float(), p=2, dim=-1)
                outputs.append(pooled.cpu())
        return self._torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)


def build_retrieval_index(
    dataset_folder: str,
    dataset_name: str,
    output_path: str,
    embedder,
    embedding_model: str,
    build_params: Optional[Dict] = None,
) -> Dict:
    """Build an index exclusively from ``train_*`` arrays.

    ``embedder`` only needs to provide ``encode(list[str]) -> np.ndarray``;
    this small interface keeps unit tests independent from LongCLIP/GPU.
    """

    dataset_folder = Path(dataset_folder)
    train_caps_path = dataset_folder / "train_text_caps.npy"
    train_ts_path = dataset_folder / "train_ts.npy"
    if not train_caps_path.exists() or not train_ts_path.exists():
        raise FileNotFoundError("A retrieval index requires train_text_caps.npy and train_ts.npy")

    caption_matrix = _as_caption_matrix(np.load(train_caps_path, allow_pickle=True))
    train_ts = np.asarray(np.load(train_ts_path), dtype=np.float32)
    if caption_matrix.shape[0] != train_ts.shape[0]:
        raise ValueError("Training captions and time series have different sample counts")

    sample_ids = np.repeat(np.arange(train_ts.shape[0], dtype=np.int64), caption_matrix.shape[1])
    caption_ids = np.tile(np.arange(caption_matrix.shape[1], dtype=np.int64), train_ts.shape[0])
    captions = caption_matrix.reshape(-1).astype(str)
    embeddings = _normalize_embeddings(embedder.encode(captions.tolist()))
    if embeddings.shape[0] != captions.shape[0]:
        raise ValueError("Embedding count does not match the number of training captions")

    metadata = {
        "index_version": INDEX_VERSION,
        "dataset_name": str(dataset_name),
        "split": "train",
        "embedding_model": str(embedding_model),
        "embedding_output": "CLIPTextModelWithProjection.text_embeds",
        "embedding_dim": int(embeddings.shape[1]),
        "l2_normalized": True,
        "num_training_samples": int(train_ts.shape[0]),
        "num_caption_records": int(captions.shape[0]),
        "captions_per_sample": int(caption_matrix.shape[1]),
        "source_dataset_folder": str(dataset_folder),
        "build_params": dict(build_params or {}),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        embeddings=embeddings,
        sample_ids=sample_ids,
        caption_ids=caption_ids,
        captions=captions,
        train_ts=train_ts,
        train_ts_sample_ids=np.arange(train_ts.shape[0], dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return metadata


def load_retrieval_index(index_path: str) -> Dict:
    with np.load(index_path, allow_pickle=False) as data:
        required = {
            "embeddings",
            "sample_ids",
            "caption_ids",
            "captions",
            "train_ts",
            "train_ts_sample_ids",
            "metadata_json",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Retrieval index is missing fields: {sorted(missing)}")
        result = {name: np.array(data[name]) for name in required if name != "metadata_json"}
        result["metadata"] = json.loads(str(data["metadata_json"].item()))

    metadata = result["metadata"]
    if metadata.get("index_version") != INDEX_VERSION:
        raise ValueError(f"Unsupported retrieval index version: {metadata.get('index_version')}")
    if metadata.get("split") != "train":
        raise ValueError("Retrieval indexes must be built from the train split only")
    embeddings = np.asarray(result["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Retrieval index embeddings are not L2-normalized")
    n_records = embeddings.shape[0]
    for field in ("sample_ids", "caption_ids", "captions"):
        if result[field].shape[0] != n_records:
            raise ValueError(f"Retrieval index field {field} has an inconsistent length")
    known_sample_ids = set(result["train_ts_sample_ids"].astype(np.int64).tolist())
    if not set(result["sample_ids"].astype(np.int64).tolist()).issubset(known_sample_ids):
        raise ValueError("At least one caption maps to a missing training time series")
    if result["train_ts"].shape[0] != result["train_ts_sample_ids"].shape[0]:
        raise ValueError("Training time-series IDs and values have inconsistent lengths")
    result["embeddings"] = embeddings
    result["captions"] = result["captions"].astype(str)
    return result


class TextRetriever:
    """Cosine Top-K retrieval with deterministic fixed-seed selection."""

    def __init__(
        self,
        index_path: str,
        embedding_model_path: Optional[str] = None,
        embedding_device: str = "cpu",
        query_batch_size: int = 64,
        embedder=None,
        seed: int = 0,
    ):
        index = load_retrieval_index(index_path)
        self.index_path = str(index_path)
        self.embeddings = index["embeddings"]
        self.sample_ids = index["sample_ids"].astype(np.int64)
        self.caption_ids = index["caption_ids"].astype(np.int64)
        self.captions = index["captions"]
        self.train_ts = index["train_ts"]
        self.train_ts_sample_ids = index["train_ts_sample_ids"].astype(np.int64)
        self.metadata = index["metadata"]
        self.seed = int(seed)
        self._sample_to_ts_row = {
            int(sample_id): row for row, sample_id in enumerate(self.train_ts_sample_ids.tolist())
        }
        if embedder is None:
            model_path = embedding_model_path or self.metadata["embedding_model"]
            embedder = LongCLIPTextEmbedder(model_path, embedding_device, query_batch_size)
        self.embedder = embedder

    def search(
        self,
        query_captions: Sequence[str],
        top_k: int,
        exclude_sample_ids: Optional[Sequence[Optional[Iterable[int]]]] = None,
    ) -> Dict:
        """Return cosine Top-K results, optionally excluding samples per query.

        Exclusions are expressed in time-series sample IDs, not caption-row IDs.
        This is important for datasets such as Weather where every series owns
        multiple captions.  The default remains backward compatible and applies
        no exclusion.
        """
        if top_k < 1:
            raise ValueError("rag_top_k must be at least 1")
        requested_top_k = int(top_k)
        query_embeddings = _normalize_embeddings(self.embedder.encode(list(query_captions)))
        if query_embeddings.shape[0] != len(query_captions):
            raise ValueError("Query embedding count does not match the number of captions")
        if query_embeddings.shape[1] != self.embeddings.shape[1]:
            raise ValueError(
                f"Query embedding dim {query_embeddings.shape[1]} does not match index dim "
                f"{self.embeddings.shape[1]}"
            )
        if exclude_sample_ids is None:
            exclude_sample_ids = [None] * len(query_captions)
        if len(exclude_sample_ids) != len(query_captions):
            raise ValueError("exclude_sample_ids must have one entry per query")

        similarities = query_embeddings @ self.embeddings.T
        ordered_rows = []
        ordered_scores = []
        for query_no, excluded in enumerate(exclude_sample_ids):
            if excluded is None:
                excluded_set = set()
            elif np.isscalar(excluded):
                excluded_set = {int(excluded)}
            else:
                excluded_set = {int(value) for value in excluded}
            eligible = ~np.isin(self.sample_ids, np.asarray(sorted(excluded_set), dtype=np.int64))
            eligible_rows = np.flatnonzero(eligible)
            if eligible_rows.size < requested_top_k:
                raise ValueError(
                    "Not enough eligible retrieval records after sample exclusion: "
                    f"query={query_no}, requested_top_k={requested_top_k}, "
                    f"eligible={eligible_rows.size}, excluded_sample_ids={sorted(excluded_set)}"
                )
            # Partition over all eligible rows. This safely expands beyond any
            # initially high-scoring captions that belonged to excluded samples.
            eligible_scores = similarities[query_no, eligible_rows]
            local = np.argpartition(-eligible_scores, requested_top_k - 1)[:requested_top_k]
            rows = eligible_rows[local]
            scores = similarities[query_no, rows]
            order = np.lexsort((rows, -scores))
            ordered_rows.append(rows[order])
            ordered_scores.append(scores[order])
        return {
            "query_embeddings": query_embeddings,
            "rows": np.stack(ordered_rows),
            "scores": np.stack(ordered_scores).astype(np.float32),
            "exclusion_enabled": any(item is not None for item in exclude_sample_ids),
        }

    def _rng(self, query_key: int, candidate_index: int) -> np.random.Generator:
        sequence = np.random.SeedSequence(
            [self.seed & 0xFFFFFFFF, int(query_key) & 0xFFFFFFFF, int(candidate_index) & 0xFFFFFFFF]
        )
        return np.random.default_rng(sequence)

    def select(
        self,
        search_result: Dict,
        query_captions: Sequence[str],
        query_sample_ids: Sequence[int],
        selection: str = "top1",
        temperature: float = 1.0,
        min_similarity: float = -1.0,
        candidate_index: int = 0,
        random_reference: bool = False,
    ) -> List[Dict]:
        if selection not in {"top1", "sample"}:
            raise ValueError("rag_selection must be 'top1' or 'sample'")
        if selection == "sample" and temperature <= 0:
            raise ValueError("rag_temperature must be positive for sampled selection")

        results: List[Dict] = []
        for query_no, (rows, scores) in enumerate(
            zip(search_result["rows"], search_result["scores"])
        ):
            query_key = int(query_sample_ids[query_no])
            fallback = bool(float(scores[0]) < float(min_similarity))
            selected_row: Optional[int] = None
            if not fallback:
                rng = self._rng(query_key, candidate_index)
                if random_reference:
                    selected_row = int(rng.integers(0, self.embeddings.shape[0]))
                elif selection == "top1":
                    selected_row = int(rows[0])
                else:
                    logits = (scores.astype(np.float64) - float(scores.max())) / float(temperature)
                    probabilities = np.exp(logits)
                    probabilities /= probabilities.sum()
                    selected_row = int(rng.choice(rows, p=probabilities))

            top_rows = [int(row) for row in rows]
            result = {
                "query_caption": str(query_captions[query_no]),
                "query_sample_id": query_key,
                "fallback": fallback,
                "top_k_sample_ids": [int(self.sample_ids[row]) for row in top_rows],
                "top_k_caption_ids": [int(self.caption_ids[row]) for row in top_rows],
                "top_k_captions": [str(self.captions[row]) for row in top_rows],
                "top_k_similarities": [float(score) for score in scores],
                "reference_ts": None,
                "reference_sample_id": None,
                "reference_caption_id": None,
                "reference_caption": None,
                # On fallback this remains the best available score; otherwise
                # it is replaced below by the selected reference's score.
                "similarity": float(scores[0]),
            }
            if selected_row is not None:
                sample_id = int(self.sample_ids[selected_row])
                ts_row = self._sample_to_ts_row[sample_id]
                similarity = float(
                    search_result["query_embeddings"][query_no] @ self.embeddings[selected_row]
                )
                result.update(
                    {
                        "reference_ts": self.train_ts[ts_row],
                        "reference_sample_id": sample_id,
                        "reference_caption_id": int(self.caption_ids[selected_row]),
                        "reference_caption": str(self.captions[selected_row]),
                        "similarity": similarity,
                    }
                )
            results.append(result)
        return results

    def retrieve(
        self,
        query_captions: Sequence[str],
        query_sample_ids: Optional[Sequence[int]] = None,
        top_k: int = 1,
        selection: str = "top1",
        temperature: float = 1.0,
        min_similarity: float = -1.0,
        candidate_index: int = 0,
        random_reference: bool = False,
        exclude_sample_ids: Optional[Sequence[Optional[Iterable[int]]]] = None,
    ) -> List[Dict]:
        if query_sample_ids is None:
            query_sample_ids = list(range(len(query_captions)))
        search_result = self.search(
            query_captions, top_k, exclude_sample_ids=exclude_sample_ids
        )
        return self.select(
            search_result,
            query_captions,
            query_sample_ids,
            selection=selection,
            temperature=temperature,
            min_similarity=min_similarity,
            candidate_index=candidate_index,
            random_reference=random_reference,
        )

    def get_reference_ts(self, sample_id: int) -> np.ndarray:
        """Return a stored training series by stable sample ID."""
        sample_id = int(sample_id)
        if sample_id not in self._sample_to_ts_row:
            raise KeyError(f"Unknown retrieval sample ID: {sample_id}")
        return self.train_ts[self._sample_to_ts_row[sample_id]]
