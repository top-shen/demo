import json

import numpy as np
import torch

from models.conditional_generator import ConditionalGenerator
from retrieval.adaptive_strength_controller import (
    CONTROLLER_VERSION,
    NORMALIZATION_SCHEMA,
    SCORE_FEATURE_NAMES,
    AdaptiveStrengthController,
    load_controller_checkpoint,
    monotonic_strength_loss,
    quantization_aware_strength_loss,
    save_controller_checkpoint,
)
from test_retrieval import make_generator


def make_controller(feature_mode="score_only"):
    torch.manual_seed(3)
    return AdaptiveStrengthController(
        embedding_dim=4,
        feature_mode=feature_mode,
        hidden_dim=16,
        pair_projection_dim=8,
        dropout=0.0,
        min_strength=0.2,
        max_strength=0.95,
        base_gamma=1.0,
        max_residual=0.15,
        similarity_q05=0.1,
        similarity_q95=0.9,
    )


def checkpoint_manifest(feature_mode="score_only"):
    return {
        "controller_version": CONTROLLER_VERSION,
        "dataset_identity": "tiny",
        "retrieval_index_sha256": "tiny-index-hash",
        "embedding_dim": 4,
        "feature_mode": feature_mode,
        "num_steps": 4,
        "normalization": {
            "schema": NORMALIZATION_SCHEMA,
            "feature_names": list(SCORE_FEATURE_NAMES),
            "mean": [0.0] * 5,
            "std": [1.0] * 5,
        },
        "gate_threshold": 0.5,
        "hidden_dim": 16,
        "pair_projection_dim": 8,
        "dropout": 0.0,
        "min_strength": 0.2,
        "max_strength": 0.95,
        "base_gamma": 1.0,
        "max_residual": 0.15,
        "similarity_q05": 0.1,
        "similarity_q95": 0.9,
    }


def test_bounded_residual_and_strength_bounds():
    model = make_controller()
    with torch.no_grad():
        model.residual_head.weight.zero_()
        model.residual_head.bias.fill_(100.0)
    output = model(torch.zeros(3, 5), torch.tensor([0.0, 0.5, 1.0]))
    assert torch.all(output["residual"] <= 0.15 + 1e-7)
    assert torch.all(output["residual"] >= -0.15 - 1e-7)
    assert torch.all(output["strength"] >= 0.2)
    assert torch.all(output["strength"] <= 0.95)


def test_pair_mode_shape_and_batch_determinism():
    model = make_controller("score_plus_pair").eval()
    score = torch.randn(6, 5)
    similarity = torch.linspace(0.1, 0.9, 6)
    query = torch.randn(6, 4)
    reference = torch.randn(6, 4)
    with torch.no_grad():
        complete = model(score, similarity, query, reference)["strength"]
        split = torch.cat(
            [
                model(score[:2], similarity[:2], query[:2], reference[:2])["strength"],
                model(score[2:], similarity[2:], query[2:], reference[2:])["strength"],
            ]
        )
    torch.testing.assert_close(complete, split)


def test_gate_rejection_is_true_gaussian_full_path():
    model = make_generator(rag_enabled=True)
    model.rag_controller = torch.nn.Linear(1, 1)
    model.rag_controller_manifest = checkpoint_manifest()
    model.rag_config["adaptive_controller"] = {"enabled": True}
    result = {
        "reference_ts": np.ones((4, 2), dtype=np.float32),
        "controller_decision": {
            "predicted_start_step": 1,
            "gate_accept": False,
            "predicted_strength": 0.3,
            "gate_probability": 0.1,
        },
    }
    template = torch.zeros(1, 2, 4)
    torch.manual_seed(7)
    expected = torch.randn_like(template)
    torch.manual_seed(7)
    initial, steps = model._initialize_generation_state(template, [result])
    torch.testing.assert_close(initial, expected)
    assert steps.tolist() == [3]


def test_batch_different_adaptive_start_steps():
    model = make_generator(rag_enabled=True)
    model.rag_controller = torch.nn.Linear(1, 1)
    model.rag_controller_manifest = checkpoint_manifest()
    model.rag_config["adaptive_controller"] = {"enabled": True}
    results = [
        {
            "reference_ts": np.ones((4, 2), dtype=np.float32),
            "controller_decision": {
                "predicted_start_step": 1,
                "gate_accept": True,
                "predicted_strength": 0.3,
                "gate_probability": 0.9,
            },
        },
        {
            "reference_ts": np.ones((4, 2), dtype=np.float32),
            "controller_decision": {
                "predicted_start_step": 3,
                "gate_accept": True,
                "predicted_strength": 0.9,
                "gate_probability": 0.9,
            },
        },
    ]
    _, steps = model._initialize_generation_state(torch.zeros(2, 2, 4), results)
    assert steps.tolist() == [1, 3]


def test_quantization_aware_loss_masks_gate_rejects():
    predicted = torch.tensor([0.2, 0.9], requires_grad=True)
    target = torch.tensor([0.4, 0.1])
    gate = torch.tensor([1.0, 0.0])
    loss = quantization_aware_strength_loss(predicted, target, gate, num_steps=11)
    expected = torch.nn.functional.huber_loss(torch.tensor([2.0]), torch.tensor([4.0]))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert predicted.grad[1].item() == 0.0


def test_monotonic_loss():
    ordered = monotonic_strength_loss(
        torch.tensor([0.9, 0.6, 0.2]), torch.tensor([0.1, 0.5, 0.9]), margin=0.1
    )
    violated = monotonic_strength_loss(
        torch.tensor([0.2, 0.6, 0.9]), torch.tensor([0.1, 0.5, 0.9]), margin=0.1
    )
    assert ordered.item() == 0.0
    assert violated.item() > 0.0


def test_checkpoint_roundtrip_and_mismatch(tmp_path):
    model = make_controller().eval()
    path = tmp_path / "controller.pt"
    manifest = checkpoint_manifest()
    save_controller_checkpoint(path, model, None, 2, manifest)
    loaded, loaded_manifest, _ = load_controller_checkpoint(
        path,
        expected={
            "dataset_identity": "tiny",
            "embedding_dim": 4,
            "feature_mode": "score_only",
            "num_steps": 4,
        },
    )
    score = torch.randn(3, 5)
    similarity = torch.tensor([0.2, 0.5, 0.8])
    with torch.no_grad():
        torch.testing.assert_close(
            model(score, similarity)["strength"], loaded(score, similarity)["strength"]
        )
    assert loaded_manifest["controller_version"] == CONTROLLER_VERSION
    try:
        load_controller_checkpoint(path, expected={"dataset_identity": "wrong"})
    except ValueError as error:
        assert "dataset_identity mismatch" in str(error)
    else:
        raise AssertionError("Dataset mismatch must fail fast")


def test_adaptive_conflicting_explicit_step_fails_fast():
    model = make_generator(rag_enabled=True, start_step=2)
    try:
        model.configure_retrieval(
            model.rag_retriever,
            {
                **model.rag_config,
                "top_k": 4,
                "selection": "top1",
                "adaptive_controller": {"enabled": True},
            },
            controller=make_controller(),
            controller_manifest=checkpoint_manifest(),
        )
    except ValueError as error:
        assert "start_step" in str(error)
    else:
        raise AssertionError("Explicit start step/controller conflict must fail")


def test_trace_matches_adaptive_execution_step():
    model = make_generator(rag_enabled=True, start_step=-1)
    model.rag_controller = make_controller()
    model.rag_controller_manifest = checkpoint_manifest()
    model.rag_config["top_k"] = 4
    model.rag_config["adaptive_controller"] = {
        "enabled": True,
        "checkpoint_path": "tiny.pt",
        "feature_mode": "score_only",
        "gate_threshold": 0.5,
    }
    result = {
        "query_sample_id": 4,
        "query_caption": "q",
        "fallback": False,
        "reference_ts": np.ones((4, 2), dtype=np.float32),
        "reference_sample_id": 8,
        "reference_caption_id": 0,
        "reference_caption": "r",
        "similarity": 0.9,
        "top_k_sample_ids": [8, 9, 10, 11],
        "top_k_caption_ids": [0, 0, 0, 0],
        "top_k_similarities": [0.9, 0.8, 0.7, 0.6],
        "controller_decision": {
            "similarity_top1": 0.9,
            "similarity_top2": 0.8,
            "similarity_margin": 0.1,
            "similarity_mean": 0.75,
            "similarity_std": 0.11,
            "similarity_entropy": 1.3,
            "base_strength": 0.2,
            "predicted_residual": 0.1,
            "predicted_strength": 0.3,
            "predicted_start_step": 1,
            "gate_probability": 0.9,
            "gate_threshold": 0.5,
            "gate_accept": True,
        },
    }
    batch = {"caption_id": torch.tensor([0])}
    model._record_retrieval_trace([result], batch, 0)
    trace = model.last_retrieval_trace[0]
    assert trace["predicted_start_step"] == trace["start_step"] == 1
    assert trace["fallback_reason"] is None
