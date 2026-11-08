"""
tsne_exact.py

Implementation of exact t-Distributed Stochastic Neighbor Embedding (t-SNE)
based on the original paper:
"Visualizing Data using t-SNE" by Laurens van der Maaten and Geoffrey Hinton (2008).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits, load_iris


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
         max_iter: int = 500, eta: float = 100.0,
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
    P = compute_joint_probabilities(X, perplexity=perplexity)
    
    # force clusters to tighten early on
    early_exaggeration_factor = 4.0
    P = P * early_exaggeration_factor
    
    Y = np.random.randn(n, no_dims) * 1e-4
    dY = np.zeros((n, no_dims), dtype=np.float64)
    iY = np.zeros((n, no_dims), dtype=np.float64)
    gains = np.ones((n, no_dims), dtype=np.float64)
    
    for iter_idx in range(max_iter):
        # pairwise affinities in low-dim space (Student-t)
        sum_Y = np.sum(np.square(Y), axis=1)
        num = -2.0 * np.dot(Y, Y.T)
        num = 1.0 / (1.0 + np.add(np.add(num, sum_Y).T, sum_Y))
        np.fill_diagonal(num, 0.0)
        
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        
        # gradient dC/dY
        PQ = (P - Q) * num
        for i in range(n):
            dY[i, :] = 4.0 * np.sum((Y[i, :] - Y) * PQ[i, :, np.newaxis], axis=0)
        
        momentum = initial_momentum if iter_idx < 250 else final_momentum
        
        # adaptive learning rate (Jacobs, 1988)
        # boost gain if gradient and momentum have same sign, dampen otherwise
        gains = (gains + 0.2) * ((np.sign(dY) != np.sign(iY)) * 1.0) + \
                (gains * 0.8) * ((np.sign(dY) == np.sign(iY)) * 1.0)
        gains[gains < 0.01] = 0.01
        
        # apply update
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        
        # zero-center to prevent drift
        Y = Y - np.mean(Y, axis=0)
        
        # stop early exaggeration
        if iter_idx == 50:
            P = P / early_exaggeration_factor
            
        if (iter_idx + 1) % 100 == 0:
            cost = np.sum(P * np.log(P / Q))
            print(f"Iteration {iter_idx + 1:4d} | KL Divergence: {cost:.4f}")
            
    return Y


if __name__ == "__main__":
    np.random.seed(42)

    # Use a reproducible subset of the real handwritten-digits dataset.
    X, labels = load_digits(return_X_y=True)
    sample_indices = np.random.choice(X.shape[0], size=500, replace=False)
    X = X[sample_indices]
    labels = labels[sample_indices]

    # Reduce the 64 pixel features to 30 before running exact t-SNE.
    X_pca = PCA(n_components=30, random_state=42).fit_transform(X)
    print(f"Running t-SNE on {len(X)} Digits samples after PCA ({X.shape[1]} -> {X_pca.shape[1]} features)...")
    Y = tsne(X_pca, no_dims=2, perplexity=30.0, max_iter=500)

    # Plot the 2D projected embedding
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap="viridis", s=40, edgecolors="k", alpha=0.85)
    plt.title("Digits dataset: 2D projection via exact t-SNE")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.colorbar(scatter, label="Ground Truth Cluster")
    plt.tight_layout()
    plt.savefig("tsne_projection_demo.png", dpi=150)
    print("Execution complete. Projection saved as 'tsne_projection_demo.png'.")

    # Compare t-SNE with the known species labels in the Iris dataset.
    iris = load_iris()
    iris_reference = PCA(n_components=2, random_state=42).fit_transform(iris.data)
    print(f"Running t-SNE on {len(iris.data)} Iris samples ({iris.data.shape[1]} features)...")
    iris_embedding = tsne(iris.data, no_dims=2, perplexity=30.0, max_iter=500)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, coordinates, title in zip(
        axes,
        (iris_reference, iris_embedding),
        ("PCA reference (known species)", "Exact t-SNE embedding"),
    ):
        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=iris.target,
            cmap="viridis",
            vmin=0,
            vmax=2,
            s=40,
            edgecolors="k",
            alpha=0.85,
        )
        ax.set_title(title)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(True, linestyle="--", alpha=0.5)

    colorbar = fig.colorbar(
        scatter,
        ax=axes,
        ticks=np.arange(len(iris.target_names)),
        label="Known Iris species",
    )
    colorbar.ax.set_yticklabels(iris.target_names)
    fig.suptitle("Iris: known labels compared with t-SNE")
    fig.savefig("iris_tsne_comparison.png", dpi=150)
    print("Iris comparison saved as 'iris_tsne_comparison.png'.")
