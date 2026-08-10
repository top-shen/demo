import numpy as np
import tempfile
from pathlib import Path
from types import SimpleNamespace

from retrieval.adaptive_strength_controller import (
    NORMALIZATION_SCHEMA,
    compute_pair_features,
    compute_score_features,
    fit_score_normalization,
    fit_similarity_quantiles,
    monotonic_violation_rate,
    normalize_score_features,
    similarity_base_strength,
    similarity_confidence,
    strength_to_start_steps,
)
from retrieval.strength_teacher import (
    build_teacher_targets,
    generated_to_reference_distance,
    grouped_controller_split,
    load_teacher_dataset,
    save_teacher_dataset,
    validate_teacher_payload,
)
from tools.build_strength_teacher import _standardize_single_ts, prepare_generation_configs


def test_score_features_and_entropy():
    scores = np.asarray([[0.9, 0.7, 0.2, 0.0]], dtype=np.float32)
    features = compute_score_features(scores, temperature=1.0)
    assert features.shape == (1, 5)
    np.testing.assert_allclose(features[0, :4], [0.9, 0.2, 0.45, scores.std()], atol=1e-6)
    probabilities = np.exp(scores[0] - scores[0].max())
    probabilities /= probabilities.sum()
    expected_entropy = -(probabilities * np.log(probabilities)).sum()
    np.testing.assert_allclose(features[0, 4], expected_entropy, atol=1e-6)


def test_pair_feature_shape_and_values():
    query = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    reference = np.asarray([[0.0, 3.0], [2.0, 1.0]], dtype=np.float32)
    pair = compute_pair_features(query, reference)
    assert pair.shape == (2, 4)
    np.testing.assert_allclose(pair[0], [1.0, 1.0, 0.0, 6.0])


def test_controller_train_only_normalization_and_quantiles():
    raw = np.arange(30, dtype=np.float32).reshape(6, 5)
    train = np.asarray([True, True, True, False, False, False])
    stats = fit_score_normalization(raw, train)
    assert stats["schema"] == NORMALIZATION_SCHEMA
    normalized = normalize_score_features(raw, stats)
    np.testing.assert_allclose(normalized[train].mean(axis=0), 0.0, atol=1e-6)
    quantiles = fit_similarity_quantiles(raw[:, 0], train)
    assert quantiles["similarity_q05"] < quantiles["similarity_q95"] < raw[3, 0]


def test_similarity_confidence_and_base_strength_are_bounded_monotonic():
    similarities = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    confidence = similarity_confidence(similarities, 0.1, 0.9)
    strength = similarity_base_strength(similarities, 0.1, 0.9, 0.2, 0.95, 1.0)
    assert np.all((confidence >= 0) & (confidence <= 1))
    assert np.all((strength >= 0.2) & (strength <= 0.95))
    assert np.all(np.diff(strength) <= 1e-7)
    np.testing.assert_allclose([strength[0], strength[-1]], [0.95, 0.2])


def test_strength_to_start_step_quantization():
    actual = strength_to_start_steps([0.0, 0.2, 0.5, 0.95, 1.0], 50)
    np.testing.assert_array_equal(actual, [0, 10, 24, 47, 49])


def test_teacher_feasible_set_soft_target_and_weights():
    result = build_teacher_targets(
        [0.2, 0.5, 0.8],
        [[0.80, 0.85, 0.90], [0.1, 0.2, 0.3]],
        [0.86, 0.8],
        epsilon_sem=0.02,
        teacher_temperature=0.1,
    )
    np.testing.assert_array_equal(result["feasible_masks"][0], [False, True, True])
    assert result["gate_targets"].tolist() == [1.0, 0.0]
    np.testing.assert_allclose(result["teacher_weights"][0].sum(), 1.0)
    assert 0.5 <= result["soft_strength_targets"][0] <= 0.8
    assert np.isnan(result["soft_strength_targets"][1])
    assert np.isnan(result["teacher_ambiguity_variance"][1])


def test_copy_constraint_removes_copying_candidate():
    result = build_teacher_targets(
        [0.2, 0.5],
        [[0.9, 0.89]],
        [0.9],
        epsilon_sem=0.02,
        teacher_temperature=0.1,
        candidate_copy_distances=[[0.01, 0.2]],
        copy_threshold=0.05,
    )
    np.testing.assert_array_equal(result["feasible_masks"], [[False, True]])
    np.testing.assert_allclose(result["soft_strength_targets"], [0.5])


def test_dimension_normalized_rmse():
    generated = np.asarray([[[0.0], [2.0]]], dtype=np.float32)
    reference = np.asarray([[[0.0], [0.0]]], dtype=np.float32)
    np.testing.assert_allclose(
        generated_to_reference_distance(generated, reference), [np.sqrt(2.0)]
    )


def test_grouped_split_never_leaks_captions():
    sample_ids = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2])
    split = grouped_controller_split(sample_ids, validation_fraction=0.34, seed=4)
    for sample_id in np.unique(sample_ids):
        assert np.unique(split[sample_ids == sample_id]).size == 1


def test_monotonic_violation_rate():
    confidence = [0.1, 0.5, 0.9]
    assert monotonic_violation_rate([0.9, 0.6, 0.2], confidence) == 0.0
    assert monotonic_violation_rate([0.2, 0.6, 0.9], confidence) == 1.0


def test_test_split_teacher_is_rejected():
    try:
        validate_teacher_payload({}, {"schema_version": "ri-verbalts-strength-teacher-v1", "split": "test"}, for_training=True)
    except ValueError as error:
        assert "split=train" in str(error) or "Test-split" in str(error)
    else:
        raise AssertionError("Test-split teacher data must be rejected")


def test_teacher_schema_roundtrip():
    count, grid, embedding_dim, top_k = 2, 2, 3, 4
    payload = {
        "query_sample_ids": [0, 1],
        "query_caption_ids": [0, 0],
        "reference_sample_ids": [1, 0],
        "reference_caption_ids": [0, 0],
        "query_embeddings": np.zeros((count, embedding_dim)),
        "reference_embeddings": np.zeros((count, embedding_dim)),
        "top_k_sample_ids": np.zeros((count, top_k), dtype=np.int64),
        "top_k_caption_ids": np.zeros((count, top_k), dtype=np.int64),
        "top_k_similarities": np.zeros((count, top_k)),
        "score_features": np.zeros((count, 5)),
        "normalized_score_features": np.zeros((count, 5)),
        "candidate_strengths": [0.2, 0.8],
        "candidate_start_steps": [1, 4],
        "candidate_semantic_scores": np.zeros((count, grid)),
        "candidate_copy_distances": np.ones((count, grid)),
        "original_semantic_scores": np.zeros(count),
        "feasible_masks": np.ones((count, grid), dtype=bool),
        "soft_strength_targets": [0.2, 0.8],
        "teacher_ambiguity_variance": [0.0, 0.0],
        "gate_targets": [1.0, 1.0],
        "controller_split_ids": [0, 1],
    }
    with tempfile.TemporaryDirectory(prefix="adaptive_teacher_schema_") as folder:
        npz_path = Path(folder) / "teacher.npz"
        manifest_path = Path(folder) / "teacher.json"
        save_teacher_dataset(
            npz_path,
            manifest_path,
            payload,
            {
                "dataset": "tiny",
                "normalization": {
                    "schema": NORMALIZATION_SCHEMA,
                    "mean": [0.0] * 5,
                    "std": [1.0] * 5,
                },
                "similarity_q05": 0.1,
                "similarity_q95": 0.9,
                "verbalts_checkpoint": {"num_steps": 5},
            },
        )
        loaded, manifest = load_teacher_dataset(npz_path, manifest_path, for_training=True)
        np.testing.assert_array_equal(loaded["query_sample_ids"], [0, 1])
        assert manifest["split"] == "train"


def test_teacher_builder_injects_run_time_text_defaults():
    diff = {"diffusion": {"multipatch_num": 2, "L_patch_len": 2}}
    cond = {"text": {"num_stages": 3}}
    args = SimpleNamespace(
        device="cuda:0",
        cond_modal="simple_text",
        text_output_type="all",
        text_pos_emb="none",
        base_patch=4,
        multipatch_num=3,
        patch_length=3,
        diff_stage_num=3,
    )
    prepared_diff, prepared_cond = prepare_generation_configs(diff, cond, args)
    assert prepared_diff["generator_pretrain_path"] == ""
    assert prepared_diff["diffusion"]["base_patch"] == 4
    assert prepared_diff["diffusion"]["multipatch_num"] == 3
    assert prepared_diff["diffusion"]["L_patch_len"] == 3
    assert prepared_cond["cond_modal"] == "simple_text"
    assert prepared_cond["text"]["pos_emb"] == "none"
    assert prepared_cond["text"]["output_type"] == "all"


def test_teacher_builder_preserves_single_series_layout():
    multivariate = np.zeros((128, 2), dtype=np.float32)
    univariate = np.zeros(128, dtype=np.float32)
    assert _standardize_single_ts(multivariate).shape == (128, 2)
    assert _standardize_single_ts(univariate).shape == (128, 1)
    generated = np.zeros((1, 128, 2), dtype=np.float32)
    reference = _standardize_single_ts(multivariate)[None, ...]
    assert generated_to_reference_distance(generated, reference).shape == (1,)
