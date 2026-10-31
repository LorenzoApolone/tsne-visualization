import numpy as np
import pytest
from tsne_exact import (
    compute_squared_pairwise_distances,
    compute_conditional_probabilities_and_entropy,
    compute_joint_probabilities,
    tsne,
)

@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    return np.random.randn(20, 5)

def test_squared_pairwise_distances(synthetic_data):
    X = synthetic_data
    D = compute_squared_pairwise_distances(X)
    
    # 1. Output must be square (n_samples, n_samples)
    assert D.shape == (X.shape[0], X.shape[0])
    # 2. Self-distances must be zero
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-7)
    # 3. Must be symmetric
    np.testing.assert_allclose(D, D.T, atol=1e-7)
    # 4. Distances must be non-negative
    assert np.all(D >= 0.0)

def test_conditional_probabilities_and_entropy():
    # Known distance layout
    d_row = np.array([1.0, 4.0, 9.0])
    beta = 1.0
    H, P = compute_conditional_probabilities_and_entropy(d_row, beta)

    # Probabilities must sum to 1
    assert np.isclose(np.sum(P), 1.0)
    # Probabilities must decrease monotonically with squared distance
    assert P[0] > P[1] > P[2]
    # Shannon entropy must be strictly positive
    assert H > 0.0

def test_joint_probabilities_properties(synthetic_data):
    X = synthetic_data
    n = X.shape[0]
    P = compute_joint_probabilities(X, perplexity=5.0)

    # 1. Proper matrix dimensions
    assert P.shape == (n, n)
    # 2. Symmetric affinities: P == P.T
    np.testing.assert_allclose(P, P.T, atol=1e-6)
    # 3. Total joint distribution sums to 1
    assert np.isclose(np.sum(P), 1.0, atol=1e-4)

def test_tsne_execution(synthetic_data):
    X = synthetic_data
    no_dims = 2
    max_iter = 150  # Lowered iterations for faster test run
    
    Y = tsne(X, no_dims=no_dims, perplexity=5.0, max_iter=max_iter)

    # 1. Target coordinate shape
    assert Y.shape == (X.shape[0], no_dims)
    # 2. No numerical explosions (NaN or Inf)
    assert not np.isnan(Y).any()
    assert not np.isinf(Y).any()
    # 3. Embedding must be zero-centered
    np.testing.assert_allclose(np.mean(Y, axis=0), 0.0, atol=1e-3)