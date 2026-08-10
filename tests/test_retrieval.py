import numpy as np
import torch

from models.conditional_generator import ConditionalGenerator
from retrieval.text_retriever import TextRetriever, build_retrieval_index, load_retrieval_index
from samplers.ddpm import DDPMSampler


class MappingEmbedder:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, texts):
        return np.asarray([self.mapping[str(text)] for text in texts], dtype=np.float32)


def write_dataset(root, train_caps, train_ts):
    root.mkdir()
    np.save(root / "train_text_caps.npy", np.asarray(train_caps))
    np.save(root / "train_ts.npy", np.asarray(train_ts, dtype=np.float32))
    np.save(root / "valid_text_caps.npy", np.asarray([["VALID-ONLY"]]))
    np.save(root / "valid_ts.npy", np.zeros((1,) + np.asarray(train_ts).shape[1:]))
    np.save(root / "test_text_caps.npy", np.asarray([["TEST-ONLY"]]))
    np.save(root / "test_ts.npy", np.zeros((1,) + np.asarray(train_ts).shape[1:]))


def build_small_index(tmp_path, weather=False):
    dataset = tmp_path / "dataset"
    if weather:
        captions = [["w00", "w01", "w02"], ["w10", "w11", "w12"]]
        mapping = {
            "w00": [1, 0], "w01": [0.9, 0.1], "w02": [0.8, 0.2],
            "w10": [0, 1], "w11": [0.1, 0.9], "w12": [0.2, 0.8],
            "query-a": [1, 0], "query-mid": [0.7, 0.7], "query-low": [-1, 0],
        }
    else:
        captions = [["train-a"], ["train-b"], ["train-c"]]
        mapping = {
            "train-a": [3, 0], "train-b": [0, 2], "train-c": [1, 1],
            "query-a": [1, 0], "query-mid": [0.7, 0.7], "query-low": [-1, 0],
        }
    train_ts = np.arange(len(captions) * 8, dtype=np.float32).reshape(len(captions), 4, 2)
    write_dataset(dataset, captions, train_ts)
    index_path = tmp_path / "index.npz"
    embedder = MappingEmbedder(mapping)
    build_retrieval_index(
        str(dataset), "Weather" if weather else "synth-m", str(index_path),
        embedder, "fake-longclip", {"test": True},
    )
    return index_path, embedder


def test_index_is_train_only_and_excludes_valid_test(tmp_path):
    index_path, _ = build_small_index(tmp_path)
    index = load_retrieval_index(index_path)
    assert index["metadata"]["split"] == "train"
    assert set(index["captions"]) == {"train-a", "train-b", "train-c"}
    assert "VALID-ONLY" not in index["captions"]
    assert "TEST-ONLY" not in index["captions"]
    assert index["train_ts"].shape[0] == 3


def test_weather_captions_map_to_the_same_time_series(tmp_path):
    index_path, _ = build_small_index(tmp_path, weather=True)
    index = load_retrieval_index(index_path)
    assert index["sample_ids"].tolist() == [0, 0, 0, 1, 1, 1]
    assert index["caption_ids"].tolist() == [0, 1, 2, 0, 1, 2]
    assert index["train_ts"].shape[0] == 2


def test_index_embeddings_are_l2_normalized(tmp_path):
    index_path, _ = build_small_index(tmp_path)
    embeddings = load_retrieval_index(index_path)["embeddings"]
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6)


def test_top1_is_deterministic(tmp_path):
    index_path, embedder = build_small_index(tmp_path)
    retriever = TextRetriever(index_path, embedder=embedder, seed=11)
    first = retriever.retrieve(["query-a"], [17], top_k=3, selection="top1")[0]
    second = retriever.retrieve(["query-a"], [17], top_k=3, selection="top1")[0]
    assert first["reference_sample_id"] == second["reference_sample_id"] == 0
    assert first["reference_caption"] == "train-a"


def test_topk_sampling_is_reproducible_for_fixed_seed(tmp_path):
    index_path, embedder = build_small_index(tmp_path)
    kwargs = dict(
        query_captions=["query-mid"], query_sample_ids=[23], top_k=3,
        selection="sample", temperature=2.0, candidate_index=4,
    )
    first = TextRetriever(index_path, embedder=embedder, seed=99).retrieve(**kwargs)[0]
    second = TextRetriever(index_path, embedder=embedder, seed=99).retrieve(**kwargs)[0]
    assert first["reference_sample_id"] == second["reference_sample_id"]
    assert first["reference_caption_id"] == second["reference_caption_id"]


def test_min_similarity_triggers_fallback(tmp_path):
    index_path, embedder = build_small_index(tmp_path)
    result = TextRetriever(index_path, embedder=embedder, seed=1).retrieve(
        ["query-low"], [2], top_k=2, min_similarity=0.1
    )[0]
    assert result["fallback"] is True
    assert result["reference_ts"] is None
    assert result["reference_sample_id"] is None


def test_synth_leave_one_series_out_excludes_same_sample(tmp_path):
    index_path, embedder = build_small_index(tmp_path)
    result = TextRetriever(index_path, embedder=embedder).retrieve(
        ["train-a"],
        [0],
        top_k=2,
        exclude_sample_ids=[[0]],
    )[0]
    assert result["reference_sample_id"] != 0
    assert 0 not in result["top_k_sample_ids"]


def test_weather_leave_one_series_out_excludes_all_three_captions(tmp_path):
    index_path, embedder = build_small_index(tmp_path, weather=True)
    result = TextRetriever(index_path, embedder=embedder).retrieve(
        ["w00"],
        [0],
        top_k=3,
        exclude_sample_ids=[[0]],
    )[0]
    assert result["top_k_sample_ids"] == [1, 1, 1]
    assert set(result["top_k_caption_ids"]) == {0, 1, 2}


def test_exclusion_reports_topk_shortage(tmp_path):
    index_path, embedder = build_small_index(tmp_path, weather=True)
    retriever = TextRetriever(index_path, embedder=embedder)
    try:
        retriever.search(["w00"], 4, exclude_sample_ids=[[0]])
    except ValueError as error:
        assert "eligible=3" in str(error)
    else:
        raise AssertionError("Expected a clear Top-K eligibility error")


def test_reference_tensor_shape_dtype_and_device():
    like = torch.zeros(2, 2, 4, dtype=torch.float64)
    result = [
        {"reference_ts": np.ones((4, 2), dtype=np.float32)},
        {"reference_ts": np.full((4, 2), 2, dtype=np.float32)},
    ]
    references, success = ConditionalGenerator._prepare_reference_tensor(result, like)
    assert references.shape == like.shape
    assert references.dtype == like.dtype
    assert references.device == like.device
    assert success.tolist() == [True, True]


def test_forward_noise_matches_ddpm_formula():
    sampler = DDPMSampler(5, 0.0001, 0.2, "linear", "cpu")
    x0 = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    noise = torch.full_like(x0, 0.25)
    t = torch.tensor([0, 3], dtype=torch.long)
    actual = sampler.forward(x0, t, noise)
    expected = sampler.alpha_bar_sqrt[t] * x0 + sampler.one_minus_alpha_bar_sqrt[t] * noise
    torch.testing.assert_close(actual, expected)


class DummyEncoder(torch.nn.Module):
    def forward(self, captions):
        return torch.zeros(len(captions), 1, 1)


class DummyProjector(torch.nn.Module):
    def forward(self, embedding, t):
        return embedding


class DummySampler:
    def __init__(self):
        self.reverse_steps = []

    def forward(self, x0, t, noise):
        return x0 + noise * 0.1

    def reverse(self, x, pred_noise, t, noise, is_determin=False):
        self.reverse_steps.append(int(t[0]))
        return x


class DummyDiffusion(torch.nn.Module):
    def __init__(self, num_steps=4):
        super().__init__()
        self.num_steps = num_steps
        self.ddpm = DummySampler()
        self.ddim = DummySampler()
        self.predict_steps = []
        self.first_x = None

    def predict_noise(self, x, tp, attr_emb, t):
        if self.first_x is None:
            self.first_x = x.clone()
        self.predict_steps.append(int(t[0]))
        return torch.zeros_like(x), {}


class FakeRetriever:
    def __init__(self, reference_ts):
        self.reference_ts = reference_ts
        self.search_calls = 0

    def search(self, captions, top_k):
        self.search_calls += 1
        return {"unused": True}

    def select(self, search_result, captions, sample_ids, **kwargs):
        return [
            {
                "query_caption": captions[0], "query_sample_id": sample_ids[0],
                "fallback": False, "top_k_sample_ids": [7], "top_k_caption_ids": [0],
                "top_k_captions": ["reference"], "top_k_similarities": [0.9],
                "reference_ts": self.reference_ts, "reference_sample_id": 7,
                "reference_caption_id": 0, "reference_caption": "reference",
                "similarity": 0.9,
            }
        ]


def make_generator(rag_enabled=False, mode="diffusion", start_step=2):
    model = ConditionalGenerator.__new__(ConditionalGenerator)
    torch.nn.Module.__init__(model)
    model.device = "cpu"
    model.cond_configs = {
        "cond_modal": "simple_text",
        "text": {"text_projector": "var_scale_diffstep_multi"},
    }
    model.attr_en = DummyEncoder()
    model.cond_projector = DummyProjector()
    model.generator = DummyDiffusion()
    model.rag_retriever = FakeRetriever(np.arange(8, dtype=np.float32).reshape(4, 2))
    model.rag_config = {
        "enabled": rag_enabled, "top_k": 1, "selection": "top1",
        "temperature": 1.0, "min_similarity": -1.0, "seed": 5,
        "start_step": start_step, "strength": 0.5, "mode": mode,
        "diverse_reference": False,
    }
    model.last_retrieval_trace = []
    model.last_reference_ids = []
    return model


def generation_batch():
    return {
        "ts": torch.zeros(1, 4, 2),
        "tp": torch.arange(4).reshape(1, 4),
        "cap": ["query"],
        "sample_id": torch.tensor([3]),
        "caption_id": torch.tensor([0]),
    }


def test_rag_disabled_uses_original_gaussian_full_loop():
    model = make_generator(rag_enabled=False)
    batch = generation_batch()
    # generate_text permutes [B,L,V] to the non-contiguous [B,V,L] model
    # layout. randn_like preserves that layout, so the expected tensor must be
    # sampled from an equivalent template rather than contiguous torch.randn.
    model_layout = batch["ts"].permute(0, 2, 1)
    torch.manual_seed(123)
    expected_initial = torch.randn_like(model_layout)
    torch.manual_seed(123)
    model.generate_text(batch, n_samples=1, sampler="ddim")
    assert model.rag_retriever.search_calls == 0
    assert model.generator.predict_steps == [3, 2, 1, 0]
    torch.testing.assert_close(model.generator.first_x, expected_initial)
    assert model.last_retrieval_trace[0]["fallback_reason"] == "original_rag_disabled"


def test_rag_reverse_loop_starts_at_configured_step():
    model = make_generator(rag_enabled=True, start_step=2)
    model.generate_text(generation_batch(), n_samples=1, sampler="ddim")
    assert model.generator.predict_steps == [2, 1, 0]
    assert model.generator.ddim.reverse_steps == [2, 1, 0]


def test_retrieval_only_does_not_call_reverse_diffusion():
    model = make_generator(rag_enabled=True, mode="retrieval_only")
    output = model.generate_text(generation_batch(), n_samples=1, sampler="ddim")
    assert model.generator.predict_steps == []
    assert model.generator.ddim.reverse_steps == []
    expected = torch.arange(8, dtype=torch.float32).reshape(4, 2).T
    torch.testing.assert_close(output[0, 0], expected)
