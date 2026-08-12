"""Side-effect-free metric helpers shared by evaluation and Oracle analysis."""

from __future__ import annotations

import numpy as np


def _sqrtm_product(first_cov, second_cov):
    """Return sqrt(first_cov @ second_cov), preferring the released SciPy path."""
    try:
        from scipy import linalg

        value, _ = linalg.sqrtm(first_cov.dot(second_cov), disp=False)
        return value
    except ImportError:
        # The eigenvalues of AB and sqrt(A) B sqrt(A) agree for positive
        # semidefinite covariance matrices.  This fallback keeps tiny tests
        # dependency-light; released experiments use SciPy.
        first_cov = (first_cov + first_cov.T) / 2
        second_cov = (second_cov + second_cov.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(first_cov)
        first_sqrt = (eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))).dot(
            eigenvectors.T
        )
        middle = first_sqrt.dot(second_cov).dot(first_sqrt)
        middle = (middle + middle.T) / 2
        middle_values = np.linalg.eigvalsh(middle)
        # Only the trace is consumed by calculate_frechet_distance.
        trace_sqrt = float(np.sqrt(np.clip(middle_values, 0, None)).sum())
        dimension = first_cov.shape[0]
        return np.eye(dimension, dtype=np.float64) * (trace_sqrt / dimension)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Match the FID/J-FTSD calculation used by :class:`BaseEvaluator`."""
    mu1 = np.atleast_1d(np.asarray(mu1, dtype=np.float64))
    mu2 = np.atleast_1d(np.asarray(mu2, dtype=np.float64))
    sigma1 = np.atleast_2d(np.asarray(sigma1, dtype=np.float64))
    sigma2 = np.atleast_2d(np.asarray(sigma2, dtype=np.float64))
    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    difference = mu1 - mu2
    covariance_mean = _sqrtm_product(sigma1, sigma2)
    if not np.isfinite(covariance_mean).all():
        print(
            "fid calculation produces singular product; "
            f"adding {eps} to diagonal of cov estimates"
        )
        offset = np.eye(sigma1.shape[0]) * float(eps)
        covariance_mean = _sqrtm_product(sigma1 + offset, sigma2 + offset)
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    value = (
        difference.dot(difference)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2 * np.trace(covariance_mean)
    )
    return float(value)


def metrics_from_embeddings(
    ts_embeddings,
    text_embeddings,
    training_ts_mean,
    training_ts_cov,
    training_joint_mean,
    training_joint_cov,
):
    """Compute CTTP, FID and J-FTSD for one complete recomposed set.

    This function deliberately accepts only the selected set's embeddings.  It
    has no interface for action-level aggregate FID/J-FTSD, preventing their
    invalid weighted averaging in Oracle analysis.
    """
    ts_embeddings = np.asarray(ts_embeddings, dtype=np.float64)
    text_embeddings = np.asarray(text_embeddings, dtype=np.float64)
    if ts_embeddings.ndim != 2 or text_embeddings.shape != ts_embeddings.shape:
        raise ValueError("CTTP time-series/text embeddings must have equal [N,D] shape")
    if ts_embeddings.shape[0] < 2:
        raise ValueError("Set-level FID/J-FTSD requires at least two samples")
    joint_embeddings = np.concatenate([ts_embeddings, text_embeddings], axis=1)
    ts_mean = ts_embeddings.mean(axis=0)
    ts_cov = np.cov(ts_embeddings, rowvar=False)
    joint_mean = joint_embeddings.mean(axis=0)
    joint_cov = np.cov(joint_embeddings, rowvar=False)
    return {
        "cttp": float(np.mean(np.sum(ts_embeddings * text_embeddings, axis=1))),
        "fid": calculate_frechet_distance(
            training_ts_mean, training_ts_cov, ts_mean, ts_cov
        ),
        "jftsd": calculate_frechet_distance(
            training_joint_mean, training_joint_cov, joint_mean, joint_cov
        ),
    }
