"""Demos for ML Theory Lab. Illustrates theoretical concepts with computational examples."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np


def theory_vc_dimension():
    """VC dimension illustration."""
    try:
        out = "VC Dimension Examples\n\n"
        out += "Linear classifier in R^d: VC = d + 1\n"
        out += "  R^1 (threshold): VC = 2\n"
        out += "  R^2 (line): VC = 3\n"
        out += "  R^10 (hyperplane): VC = 11\n\n"
        out += "Axis-aligned rectangles in R^2: VC = 4\n"
        out += "Circles in R^2: VC = 3\n\n"
        out += "Shattering: A hypothesis class H shatters S if\n"
        out += "  for every labeling of S, ∃h∈H that achieves it.\n"
        out += "VC(H) = max |S| such that H shatters S"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def theory_pac_bounds():
    """PAC learning sample complexity."""
    try:
        # Sample complexity for finite hypothesis class
        def sample_complexity(H_size, epsilon, delta):
            # m ≥ (1/ε) * (ln|H| + ln(1/δ))
            return int(np.ceil((1 / epsilon) * (np.log(H_size) + np.log(1 / delta))))
        
        out = "PAC Learning Sample Complexity\n\n"
        out += "For finite H: m ≥ (1/ε)(ln|H| + ln(1/δ))\n\n"
        out += "Examples (ε=0.1, δ=0.05):\n"
        for H_size in [10, 100, 1000, 10000]:
            m = sample_complexity(H_size, 0.1, 0.05)
            out += f"  |H| = {H_size:>5}: m ≥ {m}\n"
        out += "\nMore hypotheses → more samples needed"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def theory_generalization_bound():
    """VC generalization bound calculation."""
    try:
        def vc_bound(m, vc_dim, delta):
            # L(h) ≤ L_S(h) + sqrt((vc_dim * (ln(2m/vc_dim) + 1) + ln(4/delta)) / m)
            if m <= vc_dim:
                return float('inf')
            term = (vc_dim * (np.log(2 * m / vc_dim) + 1) + np.log(4 / delta)) / m
            return np.sqrt(term)
        
        out = "VC Generalization Bound\n\n"
        out += "L(h) ≤ L̂(h) + √[(VC·(ln(2m/VC)+1) + ln(4/δ))/m]\n\n"
        out += "Bound values (VC=10, δ=0.05):\n"
        for m in [100, 500, 1000, 5000, 10000]:
            bound = vc_bound(m, 10, 0.05)
            out += f"  m = {m:>5}: gap ≤ {bound:.4f}\n"
        out += "\nMore data → tighter generalization bound"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def theory_rademacher():
    """Rademacher complexity estimation."""
    try:
        np.random.seed(42)
        # Empirical Rademacher complexity for linear functions
        # R(H) = E[sup_h (1/m) Σ σ_i h(z_i)]
        m = 100
        d = 5
        X = np.random.randn(m, d)
        
        n_trials = 1000
        sup_values = []
        for _ in range(n_trials):
            sigma = np.random.choice([-1, 1], m)  # Rademacher variables
            # For linear h(x) = w·x with ||w||≤1, sup = ||Σ σ_i x_i||
            weighted_sum = np.sum(sigma[:, None] * X, axis=0)
            sup_h = np.linalg.norm(weighted_sum) / m
            sup_values.append(sup_h)
        
        rademacher = np.mean(sup_values)
        out = "Rademacher Complexity (empirical)\n\n"
        out += "H = linear functions with ||w||≤1\n"
        out += f"Data: {m} samples, {d} features\n\n"
        out += f"Estimated R̂(H) = {rademacher:.4f}\n\n"
        out += "Interpretation: Measures how well H can fit random noise\n"
        out += "Lower Rademacher → better generalization"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def theory_bias_variance():
    """Bias-variance tradeoff demonstration."""
    try:
        np.random.seed(42)
        # Generate data from true function f(x) = sin(x)
        def true_f(x):
            return np.sin(x)
        
        n_datasets = 50
        n_samples = 30
        x_test = np.array([2.5])
        
        # Fit polynomial models of different degrees
        results = {}
        for degree in [1, 3, 10]:
            predictions = []
            for _ in range(n_datasets):
                x = np.random.uniform(0, 5, n_samples)
                y = true_f(x) + np.random.normal(0, 0.2, n_samples)
                coeffs = np.polyfit(x, y, degree)
                pred = np.polyval(coeffs, x_test)[0]
                predictions.append(pred)
            
            bias = (np.mean(predictions) - true_f(x_test)[0]) ** 2
            variance = np.var(predictions)
            results[degree] = {"bias²": bias, "variance": variance, "total": bias + variance}
        
        out = "Bias-Variance Tradeoff\n\n"
        out += "True function: sin(x), test at x=2.5\n\n"
        out += f"{'Degree':<8} {'Bias²':<10} {'Variance':<10} {'Total':<10}\n"
        out += "-" * 38 + "\n"
        for deg, r in results.items():
            out += f"{deg:<8} {r['bias²']:<10.4f} {r['variance']:<10.4f} {r['total']:<10.4f}\n"
        out += "\nLow degree: high bias, low variance\n"
        out += "High degree: low bias, high variance"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


DEMO_HANDLERS = {
    "theory_vc": theory_vc_dimension,
    "theory_pac": theory_pac_bounds,
    "theory_gen_bound": theory_generalization_bound,
    "theory_rademacher": theory_rademacher,
    "theory_bias_var": theory_bias_variance,
}


def run_demo(demo_id: str):
    if demo_id in DEMO_HANDLERS:
        try:
            return DEMO_HANDLERS[demo_id]()
        except Exception as e:
            return {"ok": False, "output": "", "error": str(e)}
    return {"ok": False, "output": "", "error": f"No demo: {demo_id}"}
