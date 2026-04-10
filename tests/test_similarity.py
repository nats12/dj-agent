"""Tests for track similarity."""

from pathlib import Path

import numpy as np

from dj_agent.similarity import compute_feature_vector, cosine_similarity, find_similar


def test_cosine_similarity_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_zero_vector():
    a = np.zeros(3)
    b = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, b) == 0.0


def test_compute_feature_vector(sample_sine_wav: Path):
    vec = compute_feature_vector(sample_sine_wav)
    assert vec.ndim == 1
    assert len(vec) > 50  # Should be ~62 dimensions
    assert np.all(np.isfinite(vec))


def test_find_similar():
    vecs = {
        "a": np.array([1.0, 0.0, 0.0]),
        "b": np.array([0.9, 0.1, 0.0]),  # very similar to a
        "c": np.array([0.0, 0.0, 1.0]),  # orthogonal to a
    }
    target = np.array([1.0, 0.0, 0.0])
    results = find_similar(target, vecs, top_k=2, exclude_id="a")
    assert results[0][0] == "b"  # most similar
    assert results[0][1] > results[1][1]  # b is more similar than c
