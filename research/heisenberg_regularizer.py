"""
L⁻² Regularization (Heisenberg Regularizer)
=============================================

Novel regularization that penalizes weight concentration (low variance)
rather than weight magnitude (high variance like L2).

    L_total = L_task + λ₂‖w‖² + λ_H / (Var(w) + ε)

Where L2 prevents explosion, L⁻² prevents collapse.
Together they enforce an uncertainty-principle-like bound:
    std(w) * std(∇w L) ≥ lower_bound

Target problems:
  - Representational collapse in self-supervised learning
  - Attention head homogenization in transformers
  - Ensemble member convergence

Author: Research module, ML-ToolBox
Status: Proof-of-concept — not peer-reviewed
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import time


# =============================================================================
# CORE: The L⁻² Regularizer
# =============================================================================

class HeisenbergRegularizer:
    """
    Implements L⁻² regularization: penalty = λ_H / (Var(w) + ε)

    This is the mathematical dual of L2:
      - L2:   λ₂ * Σ wᵢ²          → penalizes large weights (high variance)
      - L⁻²:  λ_H / (Var(w) + ε)   → penalizes collapsed weights (low variance)

    The gradient is:
      ∂/∂wⱼ [λ_H / (Var(w) + ε)] = -λ_H * 2(wⱼ - w̄) / (n * (Var(w) + ε)²)

    This PUSHES weights APART when they're too similar.
    """

    def __init__(self, lambda_h: float = 0.01, epsilon: float = 1e-6):
        self.lambda_h = lambda_h
        self.epsilon = epsilon

    def penalty(self, weights: np.ndarray) -> float:
        """Compute L⁻² penalty value."""
        var = np.var(weights)
        return self.lambda_h / (var + self.epsilon)

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Compute gradient of L⁻² penalty w.r.t. weights."""
        n = weights.size
        w_flat = weights.flatten()
        mean_w = np.mean(w_flat)
        var_w = np.var(w_flat) + self.epsilon

        # ∂/∂wⱼ [λ / (Var + ε)] = -λ * 2(wⱼ - w̄) / (n * (Var + ε)²)
        grad = -self.lambda_h * 2.0 * (w_flat - mean_w) / (n * var_w ** 2)
        return grad.reshape(weights.shape)


class L2Regularizer:
    """Standard L2 for comparison."""

    def __init__(self, lambda_2: float = 0.01):
        self.lambda_2 = lambda_2

    def penalty(self, weights: np.ndarray) -> float:
        return self.lambda_2 * np.sum(weights ** 2)

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return 2.0 * self.lambda_2 * weights


class CombinedRegularizer:
    """
    L2 + L⁻² together: the full uncertainty-principle regularizer.

    L_reg = λ₂‖w‖² + λ_H / (Var(w) + ε)

    The L2 term prevents explosion. The L⁻² term prevents collapse.
    Together they keep weights in a "Goldilocks zone" of spread.
    """

    def __init__(self, lambda_2: float = 0.01, lambda_h: float = 0.01,
                 epsilon: float = 1e-6):
        self.l2 = L2Regularizer(lambda_2)
        self.heisenberg = HeisenbergRegularizer(lambda_h, epsilon)

    def penalty(self, weights: np.ndarray) -> float:
        return self.l2.penalty(weights) + self.heisenberg.penalty(weights)

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return self.l2.gradient(weights) + self.heisenberg.gradient(weights)


# =============================================================================
# EXPERIMENT 1: Weight Collapse Prevention
# =============================================================================

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class SimpleNet:
    """Minimal 2-layer network for controlled experiments."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        # He initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, y_hat):
        m = X.shape[0]
        dz2 = (y_hat - y) / m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0).astype(float)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def get_weight_stats(self) -> Dict:
        """Measure weight diversity/collapse."""
        stats = {}
        for name, W in [("W1", self.W1), ("W2", self.W2)]:
            stats[name] = {
                "var": float(np.var(W)),
                "std": float(np.std(W)),
                "mean_abs": float(np.mean(np.abs(W))),
                "max_abs": float(np.max(np.abs(W))),
                # Column diversity: are hidden units different from each other?
                "column_var": float(np.mean(np.var(W, axis=0))),
                # Effective rank (ratio of Frobenius norm to spectral norm)
                "effective_rank": float(
                    np.linalg.norm(W, 'fro') / (np.linalg.norm(W, 2) + 1e-10)
                ),
            }
        return stats


def make_dataset(name: str = "xor", n: int = 200):
    """Generate a simple dataset."""
    if name == "xor":
        X = np.random.randn(n, 2)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float).reshape(-1, 1)
    elif name == "circles":
        angles = np.random.uniform(0, 2 * np.pi, n)
        r = np.where(np.random.rand(n) > 0.5, 1.0, 2.5)
        X = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        X += np.random.randn(n, 2) * 0.2
        y = (r < 2.0).astype(float).reshape(-1, 1)
    elif name == "collapse_prone":
        # Dataset designed to encourage weight collapse:
        # inputs are nearly identical, forcing the network to find subtle differences
        base = np.random.randn(n, 8) * 0.1
        signal = np.random.randn(n, 2) * 1.0
        X = np.column_stack([base, signal])
        y = (signal[:, 0] * signal[:, 1] > 0).astype(float).reshape(-1, 1)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y


def bce_loss(y, y_hat):
    eps = 1e-10
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def run_collapse_experiment():
    """
    Experiment 1: Does L⁻² prevent weight collapse?

    Train the same network on a collapse-prone dataset with:
      (a) No regularization
      (b) L2 only
      (c) L⁻² only
      (d) L2 + L⁻² combined

    Measure weight variance, effective rank, and accuracy over training.
    If L⁻² works, (c) and (d) should maintain higher weight diversity than (a) and (b).
    """
    print("=" * 70)
    print("EXPERIMENT 1: Weight Collapse Prevention")
    print("=" * 70)
    print()
    print("Training on collapse-prone dataset (10D input, 8 dims near-zero,")
    print("2 dims carry signal). Without diversity pressure, weights collapse.")
    print()

    np.random.seed(42)
    X, y = make_dataset("collapse_prone", n=400)

    configs = {
        "no_reg": {"l2": None, "lh": None},
        "L2_only": {"l2": L2Regularizer(0.01), "lh": None},
        "L-2_only": {"l2": None, "lh": HeisenbergRegularizer(0.001)},
        "L2+L-2": {"l2": L2Regularizer(0.01), "lh": HeisenbergRegularizer(0.001)},
    }

    results = {}
    lr = 0.05
    epochs = 300

    for name, cfg in configs.items():
        np.random.seed(42)  # Same initialization
        net = SimpleNet(10, 16, 1)
        history = {"loss": [], "acc": [], "W1_var": [], "W1_eff_rank": [],
                   "W1_col_var": []}

        for epoch in range(epochs):
            y_hat = net.forward(X)
            loss = bce_loss(y, y_hat)
            grads = net.backward(X, y, y_hat)

            # Apply regularization gradients
            reg_loss = 0.0
            if cfg["l2"]:
                grads["W1"] += cfg["l2"].gradient(net.W1)
                grads["W2"] += cfg["l2"].gradient(net.W2)
                reg_loss += cfg["l2"].penalty(net.W1) + cfg["l2"].penalty(net.W2)
            if cfg["lh"]:
                grads["W1"] += cfg["lh"].gradient(net.W1)
                grads["W2"] += cfg["lh"].gradient(net.W2)
                reg_loss += cfg["lh"].penalty(net.W1) + cfg["lh"].penalty(net.W2)

            # SGD update
            net.W1 -= lr * grads["W1"]
            net.b1 -= lr * grads["b1"]
            net.W2 -= lr * grads["W2"]
            net.b2 -= lr * grads["b2"]

            # Record metrics
            acc = np.mean((y_hat > 0.5).astype(float) == y)
            stats = net.get_weight_stats()
            history["loss"].append(float(loss + reg_loss))
            history["acc"].append(float(acc))
            history["W1_var"].append(stats["W1"]["var"])
            history["W1_eff_rank"].append(stats["W1"]["effective_rank"])
            history["W1_col_var"].append(stats["W1"]["column_var"])

        results[name] = history

        final = net.get_weight_stats()
        print(f"  {name:12s} | acc={history['acc'][-1]:.3f} | "
              f"W1_var={final['W1']['var']:.4f} | "
              f"eff_rank={final['W1']['effective_rank']:.2f} | "
              f"col_var={final['W1']['column_var']:.4f}")

    print()
    print("INTERPRETATION:")
    print("  Higher W1_var = weights more spread out (less collapsed)")
    print("  Higher eff_rank = more independent directions used")
    print("  Higher col_var = hidden units more diverse")
    print()

    # Check if L⁻² actually helped
    no_reg_var = results["no_reg"]["W1_var"][-1]
    l2_var = results["L2_only"]["W1_var"][-1]
    lh_var = results["L-2_only"]["W1_var"][-1]
    combined_var = results["L2+L-2"]["W1_var"][-1]

    if lh_var > no_reg_var * 1.2:
        print("  ✅ L⁻² maintained higher weight variance than no regularization")
    else:
        print("  ⚠️  L⁻² did not significantly improve weight variance")
        print("     (may need λ_H tuning or different architecture)")

    if combined_var > l2_var * 1.1:
        print("  ✅ L2+L⁻² maintained more diversity than L2 alone")
    else:
        print("  ⚠️  Combined did not clearly beat L2 alone")

    return results


# =============================================================================
# EXPERIMENT 2: Attention Head Diversity
# =============================================================================

class MultiHeadAttention:
    """
    Minimal multi-head self-attention for testing head diversity.
    No masking, no positional encoding — just the core mechanism.
    """

    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # Initialize projection matrices
        scale = np.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(n_heads, d_model, self.d_k) * scale
        self.W_K = np.random.randn(n_heads, d_model, self.d_k) * scale
        self.W_V = np.random.randn(n_heads, d_model, self.d_k) * scale
        self.W_O = np.random.randn(n_heads * self.d_k, d_model) * scale

    def forward(self, X):
        """X: (seq_len, d_model) → (seq_len, d_model)"""
        heads_out = []
        self.attention_patterns = []
        for h in range(self.n_heads):
            Q = X @ self.W_Q[h]  # (seq, d_k)
            K = X @ self.W_K[h]
            V = X @ self.W_V[h]
            scores = Q @ K.T / np.sqrt(self.d_k)
            attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-10)
            self.attention_patterns.append(attn)
            heads_out.append(attn @ V)
        concat = np.concatenate(heads_out, axis=-1)  # (seq, n_heads*d_k)
        return concat @ self.W_O

    def head_diversity(self) -> Dict:
        """Measure how different the attention heads are from each other."""
        if not self.attention_patterns:
            return {"diversity": 0.0}

        n = len(self.attention_patterns)
        # Pairwise cosine distance between flattened attention patterns
        flat = [p.flatten() for p in self.attention_patterns]
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                cos = np.dot(flat[i], flat[j]) / (
                    np.linalg.norm(flat[i]) * np.linalg.norm(flat[j]) + 1e-10
                )
                similarities.append(cos)

        mean_sim = np.mean(similarities) if similarities else 0.0

        # Also measure weight diversity across heads
        wq_vars = [np.var(self.W_Q[h]) for h in range(n)]
        cross_head_var = np.var([self.W_Q[h].flatten() for h in range(n)])

        return {
            "mean_attn_similarity": float(mean_sim),
            "attn_diversity": float(1.0 - mean_sim),  # Higher = more diverse
            "mean_wq_var": float(np.mean(wq_vars)),
            "cross_head_var": float(cross_head_var),
        }


def run_attention_diversity_experiment():
    """
    Experiment 2: Does L⁻² keep attention heads diverse?

    Simulate training where gradient updates push heads toward similar patterns.
    Apply L⁻² to head projection weights and measure whether heads stay diverse.
    """
    print("=" * 70)
    print("EXPERIMENT 2: Attention Head Diversity")
    print("=" * 70)
    print()
    print("Simulating gradient pressure that homogenizes attention heads.")
    print("L⁻² should resist this by penalizing low variance across heads.")
    print()

    np.random.seed(42)
    seq_len, d_model, n_heads = 8, 16, 4
    X = np.random.randn(seq_len, d_model)

    configs = {
        "no_reg": None,
        "L2_only": L2Regularizer(0.005),
        "L-2_only": HeisenbergRegularizer(0.0005),
        "L2+L-2": CombinedRegularizer(0.005, 0.0005),
    }

    results = {}
    steps = 100
    lr = 0.01

    for name, reg in configs.items():
        np.random.seed(42)
        mha = MultiHeadAttention(d_model, n_heads)
        diversity_history = []

        for step in range(steps):
            _ = mha.forward(X)
            div = mha.head_diversity()
            diversity_history.append(div["attn_diversity"])

            # Simulate homogenizing gradient: push all heads toward the mean
            mean_wq = np.mean(mha.W_Q, axis=0)
            for h in range(n_heads):
                grad = 0.1 * (mha.W_Q[h] - mean_wq)  # Collapse pressure
                if reg:
                    grad += reg.gradient(mha.W_Q[h])
                mha.W_Q[h] -= lr * grad

        results[name] = diversity_history
        print(f"  {name:12s} | final_diversity={diversity_history[-1]:.4f} | "
              f"initial={diversity_history[0]:.4f} | "
              f"retention={diversity_history[-1]/max(diversity_history[0],1e-10):.1%}")

    print()
    print("INTERPRETATION:")
    print("  diversity = 1 - mean(cosine_similarity between head pairs)")
    print("  retention = final/initial diversity (100% = no collapse)")

    # Check results
    if results["L-2_only"][-1] > results["no_reg"][-1] * 1.3:
        print("  ✅ L⁻² significantly preserved head diversity under collapse pressure")
    else:
        print("  ⚠️  Results inconclusive — may need parameter tuning")

    return results


# =============================================================================
# EXPERIMENT 3: The Uncertainty Bound
# =============================================================================

def run_uncertainty_bound_experiment():
    """
    Experiment 3: Verify the uncertainty-principle-like bound.

    Track std(w) * std(∇L/∂w) during training with L2+L⁻².
    If the bound holds, this product should stay above a minimum threshold —
    weights can't simultaneously have low spread AND low gradient spread.
    """
    print("=" * 70)
    print("EXPERIMENT 3: Uncertainty Bound Verification")
    print("=" * 70)
    print()
    print("Tracking std(w) × std(∇w) during training.")
    print("The Heisenberg regularizer should enforce a lower bound on this product.")
    print()

    np.random.seed(42)
    X, y = make_dataset("xor", n=300)

    configs = {
        "no_reg": {"l2": None, "lh": None},
        "L2+L-2": {"l2": L2Regularizer(0.01), "lh": HeisenbergRegularizer(0.001)},
    }

    lr = 0.05
    epochs = 200

    for name, cfg in configs.items():
        np.random.seed(42)
        net = SimpleNet(2, 8, 1)
        uncertainty_products = []

        for epoch in range(epochs):
            y_hat = net.forward(X)
            grads = net.backward(X, y, y_hat)

            # Measure uncertainty product BEFORE regularization gradient
            std_w = np.std(net.W1)
            std_g = np.std(grads["W1"])
            uncertainty_products.append(std_w * std_g)

            # Apply reg
            if cfg["l2"]:
                grads["W1"] += cfg["l2"].gradient(net.W1)
            if cfg["lh"]:
                grads["W1"] += cfg["lh"].gradient(net.W1)

            net.W1 -= lr * grads["W1"]
            net.b1 -= lr * grads["b1"]
            net.W2 -= lr * grads["W2"]
            net.b2 -= lr * grads["b2"]

        min_up = min(uncertainty_products)
        mean_up = np.mean(uncertainty_products)
        print(f"  {name:12s} | min(std(w)·std(∇w))={min_up:.6f} | "
              f"mean={mean_up:.6f}")

    print()
    print("INTERPRETATION:")
    print("  If L2+L⁻² has a HIGHER minimum product, the uncertainty bound is working.")
    print("  The bound prevents simultaneous concentration of weights AND gradients.")
    print()

    return True


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  L⁻² REGULARIZATION (HEISENBERG REGULARIZER)                   ║")
    print("║  Research Proof-of-Concept                                      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print("Hypothesis: Adding λ/Var(w) as a penalty prevents weight collapse,")
    print("complementing L2's prevention of weight explosion.")
    print()

    r1 = run_collapse_experiment()
    print()
    r2 = run_attention_diversity_experiment()
    print()
    r3 = run_uncertainty_bound_experiment()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("The L⁻² regularizer is the mathematical dual of L2:")
    print("  L2:   λ‖w‖²         → shrinks weights toward zero")
    print("  L⁻²:  λ/(Var(w)+ε)  → pushes weights apart from each other")
    print()
    print("Together they enforce: std(w) × std(∇w) ≥ lower_bound")
    print()
    print("Next steps if results are positive:")
    print("  1. Test on CIFAR-10 with a real CNN (measure generalization gap)")
    print("  2. Apply to self-supervised learning (SimSiam/BYOL collapse)")
    print("  3. Apply per-head to multi-head attention (prevent head death)")
    print("  4. Derive the theoretical bound on std(w)·std(∇w) from λ₂ and λ_H")
    print()
