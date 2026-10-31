# Exact t-SNE (van der Maaten & Hinton, 2008)

A vectorized, modular NumPy implementation of the **Exact t-Distributed Stochastic Neighbor Embedding (t-SNE)** algorithm, built according to the original paper:
> *Laurens van der Maaten & Geoffrey Hinton (2008). "Visualizing Data using t-SNE", Journal of Machine Learning Research, 9: 2579–2605.*

---

## 📌 Features

- **Pairwise Squared Distance Matrix:** Algebraic vectorization via $\lVert x_i - x_j \rVert^2 = \lVert x_i \rVert^2 + \lVert x_j \rVert^2 - 2 \langle x_i, x_j \rangle$.
- **Binary Search for Perplexity:** Adaptive calculation of local Gaussian precisions ($\beta_i = \frac{1}{2\sigma_i^2}$) matching target Shannon entropy.
- **Symmetric Joint Distribution:** $p_{ij} = \frac{p_{j\vert{}i} + p_{i\vert{}j}}{2n}$ with zero self-affinity diagonal.
- **Early Exaggeration:** Scales initial probabilities $P$ by $4\times$ during the first 100 iterations to encourage well-separated cluster formation.
- **Student-t Kernel (1 DOF):** Solves the crowding problem in the low-dimensional embedding space.
- **Momentum-Driven Gradient Descent:** Adaptive momentum switching (0.5 initially, 0.8 post-exaggeration) paired with zero-centering to eliminate coordinate drift.

---

## 🛠 Requirements & Installation

Python 3.8+ is recommended. Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy matplotlib scikit-learn pytest
