"""
tsne_exact.py

Implementation of exact t-Distributed Stochastic Neighbor Embedding (t-SNE)
based on the original paper:
"Visualizing Data using t-SNE" by Laurens van der Maaten and Geoffrey Hinton (2008).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def compute_squared_pairwise_distances(X: np.ndarray) -> np.ndarray:
    """
    Computes the matrix of squared pairwise Euclidean distances ||x_i - x_j||^2.
    Uses the algebraic expansion: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * <x_i, x_j>.

    Parameters:
        X (np.ndarray): High-dimensional input data matrix of shape (n_samples, n_features).

    Returns:
        np.ndarray: Squared distance matrix of shape (n_samples, n_samples).
    """
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2.0 * np.dot(X, X.T), sum_X).T, sum_X)
    return np.maximum(D, 0.0)


def compute_conditional_probabilities_and_entropy(D_row: np.ndarray, beta: float):
    """
    Calculates conditional probabilities p_{j|i} and Shannon entropy H(P_i)
    for a given precision parameter beta = 1 / (2 * sigma_i^2).

    Parameters:
        D_row (np.ndarray): Squared distances between point i and all other points (excluding itself).
        beta (float): Gaussian precision value.

    Returns:
        tuple[float, np.ndarray]: (Entropy H, Conditional probability distribution P_i).
    """
    # Exponentiate negative squared distances scaled by precision
    P = np.exp(-D_row * beta)
    sumP = np.sum(P)

    if sumP == 0.0:
        H = 0.0
        P = np.zeros_like(D_row)
    else:
        # H = log(sum(P)) + beta * sum(D * P) / sum(P)
        H = np.log(sumP) + beta * np.sum(D_row * P) / sumP
        P = P / sumP

    return H, P


def compute_joint_probabilities(X: np.ndarray, perplexity: float = 30.0, tol: float = 1e-5, max_iter: int = 50) -> np.ndarray:
    """
    Computes high-dimensional symmetric joint probabilities p_ij by performing
    binary search to tune sigma_i (precision beta_i) such that the entropy matches
    log2(perplexity).

    Parameters:
        X (np.ndarray): Input data matrix of shape (n_samples, n_features).
        perplexity (float): Target perplexity (effective number of local neighbors).
        tol (float): Tolerance for binary search convergence.
        max_iter (int): Maximum binary search steps per sample.

    Returns:
        np.ndarray: Symmetric joint probability matrix P of shape (n_samples, n_samples).
    """
    n = X.shape[0]
    D = compute_squared_pairwise_distances(X)
    P = np.zeros((n, n), dtype=np.float64)
    beta = np.ones(n, dtype=np.float64)
    target_H = np.log(perplexity)

    for i in range(n):
        # Exclude self-distance (d_ii = 0)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        D_i = D[i, mask]

        betamin = -np.inf
        betamax = np.inf
        H, thisP = compute_conditional_probabilities_and_entropy(D_i, beta[i])
        Hdiff = H - target_H

        tries = 0
        while np.abs(Hdiff) > tol and tries < max_iter:
            if Hdiff > 0:
                betamin = beta[i]
                beta[i] = beta[i] * 2.0 if betamax == np.inf else (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i]
                beta[i] = beta[i] / 2.0 if betamin == -np.inf else (beta[i] + betamin) / 2.0

            H, thisP = compute_conditional_probabilities_and_entropy(D_i, beta[i])
            Hdiff = H - target_H
            tries += 1

        P[i, mask] = thisP

    # Symmetrize conditional probabilities: p_ij = (p_{j|i} + p_{i|j}) / (2 * n)
    P = P + P.T
    P = P / np.sum(P)
    return np.maximum(P, 1e-12)


def tsne(X: np.ndarray, no_dims: int = 2, perplexity: float = 30.0,
         max_iter: int = 500, eta: float = 200.0,
         initial_momentum: float = 0.5, final_momentum: float = 0.8) -> np.ndarray:
    """
    Executes the exact t-SNE algorithm to map high-dimensional data X to low-dimensional embedding Y.

    Parameters:
        X (np.ndarray): High-dimensional input data of shape (n_samples, n_features).
        no_dims (int): Dimensionality of the target embedded space (typically 2).
        perplexity (float): Target perplexity parameter.
        max_iter (int): Total number of gradient descent iterations.
        eta (float): Learning rate (gradient descent step size).
        initial_momentum (float): Momentum term during early optimization.
        final_momentum (float): Momentum term after early optimization phase.

    Returns:
        np.ndarray: Low-dimensional coordinates Y of shape (n_samples, no_dims).
    """
    n, _ = X.shape

    # 1. Compute high-dimensional joint probabilities P
    P = compute_joint_probabilities(X, perplexity=perplexity)

    # Apply early exaggeration to force distinct clusters to form tight groups
    early_exaggeration_factor = 4.0
    P = P * early_exaggeration_factor

    # 2. Initialize low-dimensional coordinates Y with small Gaussian noise
    Y = np.random.randn(n, no_dims) * 1e-4
    dY = np.zeros((n, no_dims), dtype=np.float64)
    iY = np.zeros((n, no_dims), dtype=np.float64)

    for iter_idx in range(max_iter):
        # Compute low-dimensional squared Euclidean distances
        sum_Y = np.sum(np.square(Y), axis=1)
        num = -2.0 * np.dot(Y, Y.T)
        # Compute Student-t kernel: q_ij ~ (1 + ||y_i - y_j||^2)^(-1)
        num = 1.0 / (1.0 + np.add(np.add(num, sum_Y).T, sum_Y))
        np.fill_diagonal(num, 0.0)

        # Normalize to obtain joint distribution Q
        sum_num = np.sum(num)
        Q = num / sum_num
        Q = np.maximum(Q, 1e-12)

        # 3. Compute analytical gradient dC/dY_i
        # PQ = (p_ij - q_ij) * (1 + ||y_i - y_j||^2)^(-1)
        PQ = (P - Q) * num
        for i in range(n):
            dY[i, :] = 4.0 * np.sum((Y[i, :] - Y) * PQ[i, :, np.newaxis], axis=0)

        # 4. Perform gradient descent update with momentum
        momentum = initial_momentum if iter_idx < 250 else final_momentum
        iY = momentum * iY - eta * dY
        Y = Y + iY

        # Zero-center embedding to prevent drifting
        Y = Y - np.mean(Y, axis=0)

        # Disable early exaggeration after iteration 100
        if iter_idx == 100:
            P = P / early_exaggeration_factor

        # Log Kullback-Leibler divergence periodically
        if (iter_idx + 1) % 100 == 0:
            cost = np.sum(P * np.log(P / Q))
            print(f"Iteration {iter_idx + 1:4d} | KL Divergence: {cost:.4f}")

    return Y


if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic 50-dimensional dataset with 3 clusters
    n_samples = 300
    n_features = 50
    centers = 3
    X, labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=42)

    print(f"Running t-SNE on {n_samples} samples with {n_features} features...")
    Y = tsne(X, no_dims=2, perplexity=30.0, max_iter=500)

    # Plot the 2D projected embedding
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap="viridis", s=40, edgecolors="k", alpha=0.85)
    plt.title("2D Projection via Exact t-SNE (van der Maaten & Hinton, 2008)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.colorbar(scatter, label="Ground Truth Cluster")
    plt.tight_layout()
    plt.savefig("tsne_projection_demo.png", dpi=150)
    print("Execution complete. Projection figure saved as 'tsne_projection_demo.png'.")
