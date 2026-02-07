"""
Enrich ML Theory Lab curriculum from 11 → 28+ items.
Covers Understanding Machine Learning (Shalev-Shwartz & Ben-David):
PAC learning, VC dimension, Rademacher complexity, stability, boosting theory, online learning.
"""
import json
from pathlib import Path

def generate_curriculum_items():
    """Generate comprehensive ML theory curriculum."""
    
    items = []
    
    # ============================================================================
    # PAC LEARNING FOUNDATIONS (6 items)
    # ============================================================================
    
    items.append({
        "id": "theory_formal_model",
        "book_id": "pac",
        "level": "basics",
        "title": "Formal Model of Learning",
        "learn": "Domain X, label set Y, hypothesis class H, loss function ℓ. True risk L_D(h) = E[ℓ(h(x),y)]. Empirical risk L_S(h) = (1/m)∑ℓ(h(x_i),y_i). Goal: minimize true risk using finite sample.",
        "try_code": "# Formal framework: (X, Y, H, ℓ, D)\n# Example: Binary classification\nX = 'R^d'  # Domain\nY = {0, 1}  # Binary labels\nH = 'Linear classifiers'  # Hypothesis class\nloss = '0-1 loss'  # ℓ(ŷ, y) = 1[ŷ ≠ y]\nD = 'Unknown distribution'\n# True risk: L_D(h) = P_{(x,y)~D}[h(x) ≠ y]\n# Empirical risk: L_S(h) = (1/m)∑_{i=1}^m 1[h(x_i) ≠ y_i]",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "theory_erm",
        "book_id": "pac",
        "level": "basics",
        "title": "Empirical Risk Minimization (ERM)",
        "learn": "ERM: choose h ∈ H that minimizes empirical risk L_S(h). Assumption: H contains good hypothesis. Inductive bias = choice of H. Overfitting risk: small L_S but large L_D.",
        "try_code": "# ERM paradigm\ndef ERM(H, S):\n    # S = training set [(x_1,y_1), ..., (x_m,y_m)]\n    h_erm = None\n    min_loss = float('inf')\n    \n    for h in H:\n        empirical_loss = sum(loss(h(x), y) for x, y in S) / len(S)\n        if empirical_loss < min_loss:\n            min_loss = empirical_loss\n            h_erm = h\n    \n    return h_erm\n\n# Key question: When does L_S(h) approximate L_D(h)?",
        "try_demo": None,
        "prerequisites": ["theory_formal_model"]
    })
    
    items.append({
        "id": "theory_pac",
        "book_id": "pac",
        "level": "intermediate",
        "title": "Probably Approximately Correct (PAC)",
        "learn": "H is (ε,δ)-PAC learnable if ∃ algorithm A, sample complexity m_H(ε,δ) s.t. ∀D, with prob ≥1-δ over S~D^m (m≥m_H), L_D(A(S)) ≤ min_{h∈H} L_D(h) + ε. Sample complexity: how many examples for (ε,δ)-guarantee.",
        "try_code": "# PAC learning example: Finite H\n# Sample complexity for finite H (realizable case):\n# m_H(ε, δ) ≤ ceil(ln(|H|/δ) / ε)\n\nimport math\n\ndef pac_sample_complexity_finite(H_size, epsilon, delta):\n    return math.ceil(math.log(H_size / delta) / epsilon)\n\n# Example: |H| = 1000, ε = 0.1, δ = 0.05\nH_size = 1000\nepsilon = 0.1\ndelta = 0.05\nm = pac_sample_complexity_finite(H_size, epsilon, delta)\nprint(f'PAC sample complexity: {m} examples')\n# Need ~100 examples to be 0.1-accurate with 95% confidence",
        "try_demo": "theory_pac",
        "prerequisites": ["theory_erm"]
    })
    
    items.append({
        "id": "theory_realizable",
        "book_id": "pac",
        "level": "intermediate",
        "title": "Realizable vs Agnostic PAC",
        "learn": "Realizable: ∃h*∈H with L_D(h*)=0. Agnostic: no assumption, learn best in class. Agnostic harder: need more samples, L_D(h) ≤ min_{h'∈H}L_D(h') + ε. Real-world usually agnostic.",
        "try_code": "# Realizable PAC: assumes perfect hypothesis exists\n# m_H(ε, δ) = O(log(|H|/δ) / ε)\n\n# Agnostic PAC: no realizability assumption\n# m_H(ε, δ) = O(log(|H|/δ) / ε²)  # Note: ε² instead of ε!\n\nimport math\n\ndef sample_complexity_comparison(H_size, epsilon, delta):\n    realizable = math.log(H_size / delta) / epsilon\n    agnostic = math.log(H_size / delta) / (epsilon ** 2)\n    \n    return realizable, agnostic\n\nreal, agn = sample_complexity_comparison(1000, 0.1, 0.05)\nprint(f'Realizable: {real:.0f}, Agnostic: {agn:.0f}')\n# Agnostic needs O(1/ε²) vs realizable O(1/ε)",
        "try_demo": None,
        "prerequisites": ["theory_pac"]
    })
    
    items.append({
        "id": "theory_no_free_lunch",
        "book_id": "pac",
        "level": "intermediate",
        "title": "No Free Lunch Theorem",
        "learn": "No universal learner: for any algorithm A, ∃ distribution D where A fails. Inductive bias (restricting H) is necessary. Different H's suit different problems. Tradeoff: expressiveness vs sample complexity.",
        "try_code": "# No Free Lunch intuition:\n# If H is too rich (e.g., all functions X→Y),\n# then cannot learn from finite samples\n\n# Example: Consider all possible functions on domain X\n# For |X| = n, |Y| = 2, there are 2^n functions\n# Cannot distinguish from finite sample which is correct\n\n# Solution: Restrict H based on prior knowledge\n# - Linear classifiers\n# - Decision trees of bounded depth\n# - Neural nets with specific architecture\n\n# Key insight: Learning = inductive bias + data",
        "try_demo": None,
        "prerequisites": ["theory_realizable"]
    })
    
    items.append({
        "id": "theory_agnostic",
        "book_id": "pac",
        "level": "advanced",
        "title": "Agnostic PAC Learning",
        "learn": "No realizable assumption: best in class has error L*=min_{h∈H}L_D(h). Learner finds h with L_D(h)≤L*+ε with prob≥1-δ. Requires m=Ω(VC(H)/ε² + log(1/δ)/ε²) samples. Fundamental regime for practical ML.",
        "try_code": "# Agnostic PAC sample complexity bound:\n# m ≥ C · (VC(H)/ε² + log(1/δ)/ε²)\n# where C is universal constant\n\nimport math\n\ndef agnostic_pac_bound(vc_dim, epsilon, delta, C=32):\n    # Simplified bound from Shalev-Shwartz & Ben-David\n    term1 = vc_dim / (epsilon ** 2)\n    term2 = math.log(1 / delta) / (epsilon ** 2)\n    return int(math.ceil(C * (term1 + term2)))\n\n# Example: VC = 10, ε = 0.1, δ = 0.05\nvc = 10\neps = 0.1\ndelta = 0.05\nm = agnostic_pac_bound(vc, eps, delta)\nprint(f'Agnostic PAC needs ≥{m} samples')\n\n# Key: Sample complexity depends on VC dimension!",
        "try_demo": "theory_pac",
        "prerequisites": ["theory_realizable"]
    })
    
    # ============================================================================
    # VC DIMENSION & GENERALIZATION (7 items)
    # ============================================================================
    
    items.append({
        "id": "theory_shattering",
        "book_id": "vc",
        "level": "basics",
        "title": "Shattering and VC Dimension",
        "learn": "H shatters set C if H realizes all 2^|C| labelings. VC(H) = max size of set H can shatter. Measures expressiveness. Examples: Linear classifiers in R^d have VC(H)=d+1. Intervals on R have VC=2.",
        "try_code": "# VC dimension examples:\n\n# 1. Intervals [a,b] on real line\n# Can shatter 2 points? Yes: [a,b] with a < x1 < x2 < b\n# Can shatter 3 points? No: cannot get x1=+, x2=-, x3=+\n# VC = 2\n\n# 2. Linear classifiers in R^2: w·x + b ≥ 0\n# Can shatter 3 points (not collinear)? Yes!\n# Can shatter 4 points? No (at least one inside triangle of others)\n# VC = 3 (d+1 for R^d)\n\n# 3. Axis-aligned rectangles in R^2\n# VC = 4 (can shatter 4 points in specific configuration)\n\n# Key: VC finite ⟹ learnable (Fundamental Theorem)",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "theory_vc",
        "book_id": "vc",
        "level": "intermediate",
        "title": "VC Dimension Theory",
        "learn": "VC(H)=d means: (1) ∃ size-d set H shatters, (2) no size-(d+1) set shattered. Key results: VC finite ⟺ PAC learnable. Sample complexity m=O(VC/ε² + log(1/δ)/ε²). Rich class ⟹ large VC ⟹ need more data.",
        "try_code": "# VC dimension bounds sample complexity\n\nimport math\n\ndef vc_sample_bound(vc_dim, epsilon, delta):\n    # Upper bound from Vapnik-Chervonenkis\n    # m ≥ O(VC(H)/ε + log(1/δ)/ε)\n    return int(math.ceil(\n        (8 / epsilon**2) * (vc_dim * math.log(13 / epsilon) + math.log(4 / delta))\n    ))\n\n# Example comparisons:\nfor vc in [1, 10, 100, 1000]:\n    m = vc_sample_bound(vc, 0.1, 0.05)\n    print(f'VC={vc:4d} → need ≥{m:6d} samples')\n\n# Output shows: Higher VC ⟹ more samples needed\n# This is the bias-complexity tradeoff!",
        "try_demo": "theory_vc",
        "prerequisites": ["theory_shattering"]
    })
    
    items.append({
        "id": "theory_fundamental_thm",
        "book_id": "vc",
        "level": "advanced",
        "title": "Fundamental Theorem of Learning",
        "learn": "H is (agnostic) PAC learnable ⟺ VC(H) is finite. Sample complexity m_H(ε,δ)=Θ(VC(H)/ε² + log(1/δ)/ε²). Characterizes learnability completely. One of deepest results in ML theory.",
        "try_code": "# Fundamental Theorem of Statistical Learning:\n# \n# For binary classification with 0-1 loss:\n# \n# PAC Learnable ⟺ VC(H) < ∞\n# \n# Moreover:\n# \n# Lower bound: m ≥ Ω(VC(H)/ε + log(1/δ)/ε)\n# Upper bound: m ≤ O((VC(H) + log(1/δ))/ε²)\n# \n# Implications:\n# 1. VC dimension fully characterizes learnability\n# 2. Cannot learn with infinite VC dimension\n# 3. Sample complexity polynomial in VC, 1/ε, log(1/δ)\n\n# Example: Neural networks\n# Perceptron in R^d: VC = d+1 (finite → learnable)\n# All Boolean functions: VC = 2^n (infinite → not learnable)",
        "try_demo": None,
        "prerequisites": ["theory_vc"]
    })
    
    items.append({
        "id": "theory_gen",
        "book_id": "vc",
        "level": "advanced",
        "title": "Generalization Bounds (VC-based)",
        "learn": "With prob ≥1-δ: L_D(h) ≤ L_S(h) + O(√(VC(H)/m) + √(log(1/δ)/m)). Uniform convergence over H. Bounds gap between empirical and true risk. Validates ERM principle.",
        "try_code": "import math\n\ndef vc_generalization_bound(vc_dim, m, delta):\n    # Generalization bound: with prob ≥ 1-δ,\n    # L_D(h) ≤ L_S(h) + ε_gen\n    # where ε_gen is:\n    \n    term1 = (4 * vc_dim) / m\n    term2 = (2 * math.log(2 / delta)) / m\n    epsilon_gen = math.sqrt(term1 + term2)\n    \n    return epsilon_gen\n\n# Example: How does bound shrink with more data?\nvc = 10\ndelta = 0.05\n\nfor m in [100, 1000, 10000, 100000]:\n    bound = vc_generalization_bound(vc, m, delta)\n    print(f'm={m:6d}: generalization gap ≤ {bound:.4f}')\n\n# Output shows: gap ~ O(1/√m)\n# Need 4x data to halve the gap!",
        "try_demo": "theory_gen_bound",
        "prerequisites": ["theory_fundamental_thm"]
    })
    
    items.append({
        "id": "theory_vc_computation",
        "book_id": "vc",
        "level": "advanced",
        "title": "Computing VC Dimension",
        "learn": "Examples: Linear classifiers R^d: VC=d+1. Decision trees depth k: VC=O(2^k). Neural nets: VC≥W (W=weights). Unions/intersections: VC(H₁∪H₂)≤VC(H₁)+VC(H₂)+1. Polynomial growth ⟹ learnable.",
        "try_code": "# VC dimension examples and computations:\n\n# 1. Linear threshold in R^d: VC = d+1\nlinear_vc = lambda d: d + 1\n\n# 2. Decision tree depth k: VC ≈ O(nodes * log(nodes))\n# For complete binary tree: nodes = 2^(k+1) - 1\nimport math\ntree_vc = lambda depth: int((2**(depth+1) - 1) * math.log2(2**(depth+1)))\n\n# 3. Neural network with W weights: VC ≥ W\n# Upper bound: VC ≤ O(W² log W)\nnn_vc_lower = lambda W: W\nnn_vc_upper = lambda W: int(W**2 * math.log2(max(W, 2)))\n\n# Examples:\nprint(f'Linear R^10: VC = {linear_vc(10)}')\nprint(f'Tree depth 5: VC ≈ {tree_vc(5)}')\nprint(f'NN 100 weights: {nn_vc_lower(100)} ≤ VC ≤ {nn_vc_upper(100)}')\n\n# 4. Composition rules:\n# VC(H1 ∪ H2) ≤ VC(H1) + VC(H2) + 1\n# VC(H1 ∩ H2) ≤ VC(H1) + VC(H2)\nunion_vc_bound = lambda vc1, vc2: vc1 + vc2 + 1",
        "try_demo": None,
        "prerequisites": ["theory_vc"]
    })
    
    items.append({
        "id": "theory_structural_risk",
        "book_id": "vc",
        "level": "advanced",
        "title": "Structural Risk Minimization",
        "learn": "SRM: Choose H from nested sequence H₁⊆H₂⊆... trading bias vs complexity. Minimize L_S(h) + penalty(VC(H_i)). Occam's Razor principle. Used in model selection, regularization. Theory behind cross-validation.",
        "try_code": "# Structural Risk Minimization (SRM)\n# Select from nested hypothesis classes\n\nimport math\n\ndef srm_bound(empirical_risk, vc_dim, m, delta_i):\n    # Bound for class H_i with VC dimension vc_dim\n    complexity_penalty = math.sqrt(\n        (vc_dim * math.log(m / vc_dim) + math.log(1 / delta_i)) / m\n    )\n    return empirical_risk + complexity_penalty\n\n# Example: Choose between models with different complexity\nm = 1000  # sample size\nmodels = [\n    ('Linear', 0.15, 5),       # (name, emp_risk, VC)\n    ('Poly degree 3', 0.10, 20),\n    ('Poly degree 5', 0.08, 56),\n    ('Deep NN', 0.05, 500)\n]\n\nfor i, (name, emp_risk, vc) in enumerate(models):\n    delta_i = 0.05 / len(models)  # Bonferroni correction\n    bound = srm_bound(emp_risk, vc, m, delta_i)\n    print(f'{name:15s}: emp={emp_risk:.3f}, VC={vc:3d}, bound={bound:.3f}')\n\n# SRM selects model minimizing bound (not just empirical risk!)",
        "try_demo": None,
        "prerequisites": ["theory_gen"]
    })
    
    items.append({
        "id": "theory_bias_complexity",
        "book_id": "vc",
        "level": "advanced",
        "title": "Bias-Complexity Tradeoff",
        "learn": "True error = approximation error (bias) + estimation error (complexity). Simple H: low estimation error (few params), high bias (limited expressiveness). Rich H: low bias, high estimation error. Optimal H balances both.",
        "try_code": "# Bias-Complexity Tradeoff Decomposition\n# \n# L_D(h_S) = approximation + estimation + optimization\n# \n# 1. Approximation error (bias):\n#    L* = min_{h∈H} L_D(h)\n#    How well can H represent true function?\n# \n# 2. Estimation error (complexity):\n#    L_D(h_S) - L*\n#    Gap due to finite sample S\n#    Grows with VC(H), shrinks with m\n# \n# 3. Optimization error:\n#    If cannot find exact ERM (NP-hard)\n\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n# Simplified illustration\ndef bias_complexity_curve(vc_range, m=1000):\n    bias = [10 / (vc + 1) for vc in vc_range]  # Decreases with complexity\n    complexity = [np.sqrt(vc / m) for vc in vc_range]  # Increases with VC\n    total = [b + c for b, c in zip(bias, complexity)]\n    return bias, complexity, total\n\nvc_range = range(1, 100)\nbias, complexity, total = bias_complexity_curve(vc_range)\n\n# Optimal VC minimizes total error\noptimal_vc = vc_range[np.argmin(total)]\nprint(f'Optimal VC dimension: {optimal_vc}')\nprint(f'At optimum: bias={bias[optimal_vc-1]:.3f}, complexity={complexity[optimal_vc-1]:.3f}')",
        "try_demo": "theory_bias_var",
        "prerequisites": ["theory_structural_risk"]
    })
    
    # ============================================================================
    # RADEMACHER COMPLEXITY & STABILITY (6 items)
    # ============================================================================
    
    items.append({
        "id": "theory_rademacher_def",
        "book_id": "stability",
        "level": "advanced",
        "title": "Rademacher Complexity Definition",
        "learn": "R_m(H) = E_σ[sup_{h∈H} (1/m)∑σᵢh(zᵢ)] where σᵢ∈{±1} uniform. Measures how well H correlates with random noise. Finer than VC: data-dependent. Generalization: E[L_D(h)] ≤ L_S(h) + 2R_m(H) + O(√(log(1/δ)/m)).",
        "try_code": "# Rademacher complexity intuition:\n# If H can fit random labels σ, it's too complex\n\nimport numpy as np\n\ndef empirical_rademacher(H, S):\n    # Compute empirical Rademacher complexity\n    # H: list of hypothesis functions\n    # S: sample [(x_1, y_1), ..., (x_m, y_m)]\n    \n    m = len(S)\n    n_trials = 1000\n    max_correlations = []\n    \n    for _ in range(n_trials):\n        # Random signs\n        sigma = np.random.choice([-1, 1], size=m)\n        \n        # Find hypothesis with max correlation to noise\n        max_corr = max(\n            np.mean([sigma[i] * h(S[i][0]) for i in range(m)])\n            for h in H\n        )\n        max_correlations.append(max_corr)\n    \n    return np.mean(max_correlations)\n\n# Example: Linear functions on unit ball\n# R(H) ≈ O(√(d/m)) for linear H in R^d\n\n# Key: Lower Rademacher → better generalization",
        "try_demo": "theory_rademacher",
        "prerequisites": []
    })
    
    items.append({
        "id": "theory_rademacher",
        "book_id": "stability",
        "level": "expert",
        "title": "Rademacher Complexity Bounds",
        "learn": "R_m(H)=E_σ[sup_{h∈H}(1/m)∑σᵢh(zᵢ)]. Bounds: With prob≥1-δ, L_D(h)≤L_S(h)+2R_m(H)+O(√(log(1/δ)/m)). Tighter than VC for some classes. Examples: Linear functions: R_m≈O(√(d/m)), Neural nets: R_m depends on norms.",
        "try_code": "import math\n\ndef rademacher_gen_bound(emp_loss, rad_complexity, m, delta):\n    # Generalization bound via Rademacher complexity\n    # With prob ≥ 1-δ: L_D(h) ≤ L_S(h) + 2R_m + 3√(log(2/δ)/(2m))\n    \n    penalty = 2 * rad_complexity + 3 * math.sqrt(math.log(2/delta) / (2*m))\n    return emp_loss + penalty\n\n# Example: Linear classifiers in R^d\n# R_m ≈ (B*W)/√m where B=data norm, W=weight norm\n\ndef linear_rademacher(d, B, W, m):\n    return (B * W * math.sqrt(d)) / math.sqrt(m)\n\nd = 100  # dimensions\nB = 1.0  # data bounded in unit ball\nW = 1.0  # weight norm\nm = 1000  # samples\n\nrad = linear_rademacher(d, B, W, m)\nemp_loss = 0.1\nbound = rademacher_gen_bound(emp_loss, rad, m, 0.05)\n\nprint(f'Rademacher complexity: {rad:.4f}')\nprint(f'Generalization bound: {bound:.4f}')\n\n# Note: Bound tightens with more data (R_m ~ 1/√m)",
        "try_demo": "theory_rademacher",
        "prerequisites": ["theory_rademacher_def"]
    })
    
    items.append({
        "id": "theory_stability_def",
        "book_id": "stability",
        "level": "advanced",
        "title": "Algorithmic Stability",
        "learn": "Algorithm A is β-uniformly stable if |ℓ(A(S),z) - ℓ(A(S^i),z)| ≤ β ∀z, where S^i differs from S in one example. Stable algorithms generalize: E[L_D(A(S))] ≤ L_S(A(S)) + β + O(√(log(1/δ)/m)).",
        "try_code": "# Stability: Changing one training example changes loss by ≤ β\n\ndef check_stability(algorithm, S, test_point):\n    # Train on full dataset\n    h = algorithm(S)\n    loss_full = compute_loss(h, test_point)\n    \n    # Train on datasets with one example removed\n    max_diff = 0\n    for i in range(len(S)):\n        S_minus_i = S[:i] + S[i+1:]  # Remove i-th example\n        h_minus_i = algorithm(S_minus_i)\n        loss_minus_i = compute_loss(h_minus_i, test_point)\n        \n        diff = abs(loss_full - loss_minus_i)\n        max_diff = max(max_diff, diff)\n    \n    return max_diff  # This is β\n\n# Examples:\n# 1. SGD with step size η: β = O(η)  (stable if η small)\n# 2. SVM with regularization λ: β = O(1/(λm))  (stable)\n# 3. Memorization (nearest neighbor): β = ∞  (unstable!)",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "theory_stability",
        "book_id": "stability",
        "level": "advanced",
        "title": "Stability and Generalization",
        "learn": "Uniform stability: changing one sample changes loss by ≤β. Stable algorithms generalize well even with large VC. Examples: Ridge regression (β=O(1/(λm))), SGD with small steps. Alternative to VC/Rademacher.",
        "try_code": "import math\n\ndef stability_generalization_bound(beta, m, delta):\n    # If algorithm is β-uniformly stable, then with prob ≥ 1-δ:\n    # L_D(h) ≤ L_S(h) + β + (4m*β + 1) * √(log(1/δ)/(2m))\n    \n    term1 = beta\n    term2 = (4 * m * beta + 1) * math.sqrt(math.log(1/delta) / (2*m))\n    return term1 + term2\n\n# Example: Ridge regression with regularization λ\ndef ridge_stability(lambda_param, m, L=1.0):\n    # β = L² / (2 * λ * m)  for L-Lipschitz loss\n    return (L ** 2) / (2 * lambda_param * m)\n\nm = 1000\nlambda_param = 0.01\n\nbeta = ridge_stability(lambda_param, m)\ngen_gap = stability_generalization_bound(beta, m, 0.05)\n\nprint(f'Stability parameter β: {beta:.6f}')\nprint(f'Generalization gap bound: {gen_gap:.4f}')\n\n# Key insight: Regularization → stability → generalization\n# This explains why λ ↑ improves test performance (up to a point)",
        "try_demo": None,
        "prerequisites": ["theory_stability_def"]
    })
    
    items.append({
        "id": "theory_sgd_stability",
        "book_id": "stability",
        "level": "expert",
        "title": "SGD and Stability Analysis",
        "learn": "SGD with step size η: uniform stability β=O(ηT/m) where T=iterations. Small η + early stopping ⟹ stable ⟹ generalize. Implicit regularization: SGD prefers flat minima. Explains why deep learning generalizes despite overparameterization.",
        "try_code": "import math\n\ndef sgd_stability_bound(eta, T, m, L=1.0, smoothness=1.0):\n    # SGD uniform stability (Hardt et al., 2016)\n    # β ≤ (2 * L² * η * T) / m  for L-Lipschitz, smooth loss\n    \n    beta = (2 * L**2 * eta * T) / m\n    return beta\n\ndef sgd_generalization(beta, m, delta):\n    # Generalization via stability\n    return beta + math.sqrt(math.log(1/delta) / (2*m))\n\n# Example: Training neural network\nm = 10000  # training set size\nT = 1000   # SGD iterations\neta = 0.01 # learning rate\n\nbeta = sgd_stability_bound(eta, T, m)\ngen_gap = sgd_generalization(beta, m, 0.05)\n\nprint(f'SGD stability: β = {beta:.6f}')\nprint(f'Expected generalization gap: {gen_gap:.4f}')\n\n# Key insights:\n# 1. Smaller η → more stable → better generalization\n# 2. Early stopping (smaller T) → more stable\n# 3. More data (larger m) → more stable\n# 4. This explains implicit regularization of SGD!",
        "try_demo": None,
        "prerequisites": ["theory_stability"]
    })
    
    items.append({
        "id": "theory_compression",
        "book_id": "stability",
        "level": "expert",
        "title": "Compression Bounds",
        "learn": "If hypothesis h can be represented using k examples from S, then generalization gap = O(k/m). Compression ⟹ generalization. Examples: SVM support vectors, decision tree leaves. Alternative view of Occam's Razor.",
        "try_code": "import math\n\ndef compression_bound(k, m, delta):\n    # If hypothesis uses k examples (compression size),\n    # then with prob ≥ 1-δ:\n    # L_D(h) ≤ L_S(h) + O(√(k log(m/k) / m) + √(log(1/δ) / m))\n    \n    if k >= m:\n        return float('inf')  # No compression\n    \n    term1 = math.sqrt((k * math.log(m / k)) / m)\n    term2 = math.sqrt(math.log(1 / delta) / m)\n    \n    return 2 * term1 + term2\n\n# Example: SVM with support vectors\nm = 1000  # training size\nfor k in [10, 50, 100, 500]:\n    bound = compression_bound(k, m, 0.05)\n    print(f'k={k:3d} support vectors: gen gap ≤ {bound:.4f}')\n\n# Insights:\n# 1. Fewer support vectors → better bound\n# 2. Large margin → fewer SVs → generalization\n# 3. This justifies margin-based learning!",
        "try_demo": None,
        "prerequisites": ["theory_sgd_stability"]
    })
    
    # ============================================================================
    # ADVANCED TOPICS (9 items)
    # ============================================================================
    
    items.append({
        "id": "theory_boosting",
        "book_id": "bounds",
        "level": "advanced",
        "title": "Boosting Theory",
        "learn": "AdaBoost combines weak learners (>50% accuracy) into strong learner. Training error decays exponentially: ε_t ≤ exp(-2t∑γ_t²). Generalization via margins: large margin ⟹ good generalization despite no explicit regularization.",
        "try_code": "import numpy as np\n\n# AdaBoost training error bound\ndef adaboost_training_error(T, gamma_min):\n    # If each weak learner has edge γ_t ≥ γ_min > 0,\n    # then training error ≤ exp(-2 * T * γ_min²)\n    return np.exp(-2 * T * gamma_min**2)\n\n# Example: weak learners with 55% accuracy (γ=0.05)\ngamma = 0.05\nfor T in [10, 50, 100, 500]:\n    err = adaboost_training_error(T, gamma)\n    print(f'T={T:3d} rounds: training error ≤ {err:.6f}')\n\n# Generalization: margin theory (Schapire et al., 1998)\ndef margin_bound(m, d, margin_theta, delta):\n    # With prob ≥ 1-δ: L_D(h) ≤ P[margin(h,z)<θ] + O(√((d log²(m/d))/(θ²m)) + √(log(1/δ)/m))\n    # d = VC dimension of base learners\n    import math\n    term1 = math.sqrt((d * math.log(m/d)**2) / (margin_theta**2 * m))\n    term2 = math.sqrt(math.log(1/delta) / m)\n    return term1 + term2\n\n# Key: Large margin → good generalization (even if many weak learners!)",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "theory_online_learning",
        "book_id": "bounds",
        "level": "advanced",
        "title": "Online Learning and Regret",
        "learn": "Online: see x_t, predict ŷ_t, observe y_t, incur loss. Regret: R_T = ∑ℓ(ŷ_t,y_t) - min_{h∈H}∑ℓ(h(x_t),y_t). Goal: sublinear regret R_T = o(T). Multiplicative weights, gradient descent achieve R_T = O(√T).",
        "try_code": "import numpy as np\n\n# Online Gradient Descent for convex loss\nclass OnlineGD:\n    def __init__(self, d, eta):\n        self.w = np.zeros(d)\n        self.eta = eta  # step size\n        self.losses = []\n    \n    def predict(self, x):\n        return np.dot(self.w, x)\n    \n    def update(self, x, y, loss_grad):\n        # loss_grad = gradient of loss at current w\n        self.w = self.w - self.eta * loss_grad\n        \n        # Project onto bounded set if needed\n        norm = np.linalg.norm(self.w)\n        if norm > 1.0:\n            self.w = self.w / norm\n    \n    def compute_regret(self, optimal_loss):\n        total_loss = sum(self.losses)\n        return total_loss - optimal_loss\n\n# Regret bound: R_T ≤ (D²/(2η)) + (ηT*L²)/2\n# where D = diameter of domain, L = Lipschitz constant\n# Optimal η = D/(L√T) gives R_T = O(DL√T)\n\nimport math\ndef ogd_regret_bound(T, D=1.0, L=1.0):\n    eta_opt = D / (L * math.sqrt(T))\n    regret = D * L * math.sqrt(T)\n    return regret, eta_opt\n\nfor T in [100, 1000, 10000]:\n    regret, eta = ogd_regret_bound(T)\n    print(f'T={T:5d}: Regret ≤ {regret:.2f}, use η={eta:.4f}')",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "theory_kernel_theory",
        "book_id": "bounds",
        "level": "advanced",
        "title": "Kernel Methods and Representer Theorem",
        "learn": "Kernel k(x,x')=⟨φ(x),φ(x')⟩ allows learning in high/infinite dim. Representer theorem: solution h*=∑α_i k(x_i,·). Generalization via kernel alignment, margin bounds. Examples: RBF, polynomial kernels.",
        "try_code": "import numpy as np\n\n# Representer Theorem: Solution lies in span of training data\n# For regularized risk: min_h ∑ℓ(h(x_i),y_i) + λ||h||²_H\n# Solution: h*(x) = ∑_{i=1}^m α_i k(x_i, x)\n\n# Kernel Ridge Regression\nclass KernelRidgeRegression:\n    def __init__(self, kernel, lambda_reg):\n        self.kernel = kernel\n        self.lambda_reg = lambda_reg\n        self.alpha = None\n        self.X_train = None\n    \n    def fit(self, X, y):\n        m = len(X)\n        # Compute kernel matrix K[i,j] = k(x_i, x_j)\n        K = np.array([[self.kernel(X[i], X[j]) for j in range(m)] \n                      for i in range(m)])\n        \n        # Solve: α = (K + λI)^{-1} y\n        self.alpha = np.linalg.solve(K + self.lambda_reg * np.eye(m), y)\n        self.X_train = X\n    \n    def predict(self, x):\n        # h(x) = ∑ α_i k(x_i, x)\n        return sum(self.alpha[i] * self.kernel(self.X_train[i], x) \n                  for i in range(len(self.X_train)))\n\n# Example kernels\ndef rbf_kernel(x1, x2, gamma=1.0):\n    return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)\n\ndef poly_kernel(x1, x2, degree=2):\n    return (1 + np.dot(x1, x2))**degree\n\n# Generalization: depends on kernel matrix eigenvalues\n# Effective dimension d_eff = tr(K) / λ_max\n# Better kernels → better generalization",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "theory_neural_net_theory",
        "book_id": "bounds",
        "level": "expert",
        "title": "Neural Network Generalization",
        "learn": "Overparameterized NNs (W≫m) can fit noise yet generalize. Explanations: (1) Implicit regularization of SGD prefers flat minima, (2) Norm-based bounds: generalization via ||W||, (3) Compression via pruning. Active research area.",
        "try_code": "import math\n\n# Norm-based generalization for neural networks\ndef nn_norm_bound(weights_product, margin, depth, m, delta):\n    # Bartlett-Mendelson bound (1999, 2017)\n    # For depth-L network with weight matrices W_1,...,W_L\n    # Bound depends on product of Frobenius norms:\n    # \n    # L_D(h) ≤ L_S(h) + O((∏||W_i||_F)/(margin * √m)) + O(√(log(1/δ)/m))\n    \n    term1 = (weights_product * math.sqrt(depth)) / (margin * math.sqrt(m))\n    term2 = math.sqrt(math.log(1/delta) / m)\n    \n    return term1 + term2\n\n# Example: 3-layer network\nW1_norm = 2.0\nW2_norm = 2.0\nW3_norm = 2.0\nproduct = W1_norm * W2_norm * W3_norm\nmargin = 0.1\ndepth = 3\nm = 10000\n\nbound = nn_norm_bound(product, margin, depth, m, 0.05)\nprint(f'NN generalization bound: {bound:.4f}')\n\n# Key insights:\n# 1. Smaller weight norms → better bound\n# 2. This justifies weight decay / L2 regularization!\n# 3. Larger margin → better bound (like SVM)\n# 4. Bound is independent of width (overparameterization OK!)\n\n# Modern view: SGD finds low-norm solutions (implicit regularization)",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "theory_double_descent",
        "book_id": "bounds",
        "level": "expert",
        "title": "Double Descent Phenomenon",
        "learn": "Risk curve non-monotonic: decreases, peaks at interpolation threshold (W=m), then decreases again. Classical regime (W<m): bias-variance tradeoff. Modern regime (W>m): implicit regularization. Challenges traditional theory.",
        "try_code": "# Double Descent: Risk as function of model complexity\n# \n# Classical U-curve (textbook):\n# Risk = Bias² + Variance\n# - Underfitting (low complexity): high bias, low variance\n# - Overfitting (high complexity): low bias, high variance\n# - Optimal in between\n# \n# Double Descent (modern ML):\n# - Classical regime: U-curve as expected\n# - Interpolation threshold (W ≈ m): PEAK (worst generalization!)\n# - Overparameterized regime (W >> m): Risk decreases again!\n# \n# Three regimes:\n# 1. Underparameterized (W << m): Classical bias-variance\n# 2. Critically parameterized (W ≈ m): Interpolates with high norm\n# 3. Overparameterized (W >> m): Interpolates with low norm (implicit reg)\n\nimport numpy as np\nimport matplotlib.pyplot as plt\n\ndef double_descent_curve(complexity_range, m=100):\n    risk = []\n    for W in complexity_range:\n        if W < m * 0.8:\n            # Underparameterized: U-curve\n            r = 1.0 / W + (W / m) * 0.1\n        elif W < m * 1.2:\n            # Interpolation threshold: PEAK\n            r = 2.0 + 5 * abs(W - m) / m\n        else:\n            # Overparameterized: decreasing\n            r = 1.0 + 10.0 / (W - m)\n        risk.append(r)\n    return risk\n\n# Explanation: At threshold, unique interpolating solution has high norm\n# Beyond threshold, many solutions → SGD finds low-norm one",
        "try_demo": None,
        "prerequisites": ["theory_neural_net_theory"]
    })
    
    items.append({
        "id": "theory_pac_bayes",
        "book_id": "bounds",
        "level": "expert",
        "title": "PAC-Bayesian Theory",
        "learn": "Bayesian learning theory: prior P over H, posterior Q. PAC-Bayes bound: KL(Q||P) ≤ (1/λ)∑ℓ(h,z) + O(log(m/δ)/λm). Tighter for stochastic predictors. Applications: deep learning generalization, compression.",
        "try_code": "import math\n\ndef pac_bayes_bound(empirical_loss, kl_divergence, m, delta, lambda_param=1.0):\n    # PAC-Bayes bound (McAllester, 1999):\n    # With prob ≥ 1-δ over S ~ D^m:\n    # \n    # E_{h~Q}[L_D(h)] ≤ E_{h~Q}[L_S(h)] + √((KL(Q||P) + log(2√m/δ))/(2m))\n    # \n    # Where:\n    # - P: prior over hypotheses (before seeing data)\n    # - Q: posterior (after seeing data)\n    # - KL(Q||P): how much Q deviates from P\n    \n    kl_term = kl_divergence + math.log(2 * math.sqrt(m) / delta)\n    penalty = math.sqrt(kl_term / (2 * m))\n    \n    return empirical_loss + penalty\n\n# Example: Gaussian posterior over weights\ndef kl_gaussian(mu_post, sigma_post, mu_prior, sigma_prior, d):\n    # KL between two Gaussians\n    kl = 0.5 * d * (\n        math.log(sigma_prior**2 / sigma_post**2) +\n        (sigma_post**2 + (mu_post - mu_prior)**2) / sigma_prior**2 - 1\n    )\n    return kl\n\nd = 100  # dimensions\nmu_post = 0.1  # learned mean\nsigma_post = 0.5  # learned std\nmu_prior = 0.0  # prior mean\nsigma_prior = 1.0  # prior std\nm = 1000  # samples\n\nkl = kl_gaussian(mu_post, sigma_post, mu_prior, sigma_prior, d)\nemp_loss = 0.15\nbound = pac_bayes_bound(emp_loss, kl, m, 0.05)\n\nprint(f'KL(Q||P) = {kl:.2f}')\nprint(f'PAC-Bayes bound: {bound:.4f}')\n\n# Key: Posterior close to prior → small KL → better bound",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "theory_label_complexity",
        "book_id": "bounds",
        "level": "advanced",
        "title": "Label Complexity and Active Learning",
        "learn": "Active learning: learner chooses which examples to label. Can achieve exp(VC) reduction in label complexity vs passive. Query strategies: uncertainty sampling, QBC. Theory: disagreement coefficient bounds improvement.",
        "try_code": "# Active Learning: Label Complexity\n# \n# Passive learning: Random i.i.d. samples\n# Label complexity: O(VC(H)/ε² + log(1/δ)/ε²)\n# \n# Active learning: Learner chooses which x to query\n# Label complexity: O(θ * VC(H) * log(1/ε) + log(1/δ)/ε)\n# where θ = disagreement coefficient\n# \n# Can be EXPONENTIALLY better for some H!\n\nimport math\n\ndef label_complexity_passive(vc, epsilon, delta):\n    return (vc / epsilon**2) + (math.log(1/delta) / epsilon**2)\n\ndef label_complexity_active(vc, epsilon, delta, theta):\n    # θ = disagreement coefficient (problem-dependent)\n    # θ = 1 for linear separators in 1D\n    # θ = O(d) for linear in R^d\n    return theta * vc * math.log(1/epsilon) + math.log(1/delta) / epsilon\n\nvc = 10\nepsilon = 0.1\ndelta = 0.05\n\npassive = label_complexity_passive(vc, epsilon, delta)\nactive_best = label_complexity_active(vc, epsilon, delta, theta=1)\nactive_worst = label_complexity_active(vc, epsilon, delta, theta=vc)\n\nprint(f'Passive: {passive:.0f} labels')\nprint(f'Active (best case θ=1): {active_best:.0f} labels')\nprint(f'Active (worst case θ=VC): {active_worst:.0f} labels')\nprint(f'\\nSpeedup: {passive/active_best:.1f}x (best case)')",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "theory_privacy",
        "book_id": "bounds",
        "level": "expert",
        "title": "Differential Privacy and Learning",
        "learn": "(ε,δ)-differential privacy: Pr[A(D)]≤e^ε Pr[A(D')]+δ for neighboring datasets. Private learning possible via output perturbation or gradient noise. Cost: O(1/εn) sample complexity overhead. Trade privacy for accuracy.",
        "try_code": "import numpy as np\n\n# Differential Privacy: (ε, δ)-DP\n# Algorithm A is (ε, δ)-differentially private if\n# for all neighboring datasets D, D' (differ in one record):\n# \n# Pr[A(D) ∈ S] ≤ e^ε · Pr[A(D') ∈ S] + δ\n# \n# for all sets S\n\n# Example: Private mean estimation\ndef private_mean(data, epsilon, sensitivity=1.0):\n    # True mean\n    mean = np.mean(data)\n    \n    # Laplace noise for ε-DP\n    # Scale = sensitivity / ε\n    noise_scale = sensitivity / epsilon\n    noise = np.random.laplace(0, noise_scale)\n    \n    return mean + noise\n\n# Example: Private gradient descent\ndef dp_sgd_step(gradient, epsilon, C=1.0):\n    # Clip gradient to bound sensitivity\n    norm = np.linalg.norm(gradient)\n    if norm > C:\n        gradient = gradient * (C / norm)\n    \n    # Add noise\n    noise_scale = C / epsilon\n    noise = np.random.normal(0, noise_scale, size=gradient.shape)\n    \n    return gradient + noise\n\n# Privacy-Accuracy tradeoff:\n# Tighter privacy (smaller ε) → more noise → worse accuracy\n# Need more samples to compensate\n\ndef dp_sample_complexity(vc, epsilon_acc, delta_acc, epsilon_priv):\n    # Private PAC learning sample complexity\n    # m = O((VC + log(1/δ))/ε² + VC/(ε_priv·ε))\n    return (vc / epsilon_acc**2) + (vc / (epsilon_priv * epsilon_acc))\n\nprint('Sample complexity for private learning:')\nfor eps_priv in [10, 1, 0.1]:\n    m = dp_sample_complexity(10, 0.1, 0.05, eps_priv)\n    print(f'ε_priv={eps_priv:4.1f}: m ≈ {m:.0f}')",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "theory_multiclass",
        "book_id": "bounds",
        "level": "advanced",
        "title": "Multiclass Learning Theory",
        "learn": "K classes: Natarajan dimension extends VC. Sample complexity m=O(d·log(K)/ε²) where d=Natarajan dim. Error-correcting codes, one-vs-all, tree-based. Generalization via margin for output code.",
        "try_code": "import math\n\n# Multiclass learning theory\n# \n# Binary: VC dimension d → m = O(d/ε²)\n# Multiclass (K classes): Natarajan dimension d_K → m = O(d_K·log(K)/ε²)\n# \n# Natarajan dimension: Generalization of VC to multiclass\n\ndef natarajan_sample_bound(d_nat, K, epsilon, delta):\n    # Sample complexity for K-class learning\n    # m ≥ C · (d_nat · log(K) / ε² + log(1/δ) / ε²)\n    \n    C = 32  # universal constant\n    term1 = (d_nat * math.log(K)) / (epsilon ** 2)\n    term2 = math.log(1 / delta) / (epsilon ** 2)\n    \n    return int(math.ceil(C * (term1 + term2)))\n\n# Example: multiclass linear classifiers\n# Binary linear in R^d: VC = d+1\n# K-class linear in R^d: Natarajan ≈ (d+1) · K\n\nd = 100\nfor K in [2, 5, 10, 100]:\n    d_nat = (d + 1) * K  # rough approximation\n    m = natarajan_sample_bound(d_nat, K, 0.1, 0.05)\n    print(f'K={K:3d} classes: need ≥{m:7d} samples')\n\n# Reduction strategies:\n# 1. One-vs-all: K binary problems\n# 2. All-pairs: K(K-1)/2 binary problems\n# 3. Error-correcting output codes: log(K) binary\n# 4. Tree-based: log(K) depth",
        "try_demo": None,
        "prerequisites": []
    })
    
    return items


def main():
    """Generate and save enriched curriculum."""
    items = generate_curriculum_items()
    
    # Save to cache
    cache_dir = Path(__file__).parent.parent / '.cache'
    cache_dir.mkdir(exist_ok=True)
    
    output_file = cache_dir / 'ml_theory_enriched.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=2)
    
    # Statistics
    by_level = {}
    by_book = {}
    for item in items:
        by_level[item['level']] = by_level.get(item['level'], 0) + 1
        by_book[item['book_id']] = by_book.get(item['book_id'], 0) + 1
    
    print(f"\n{'='*70}")
    print(f"✅ Generated {len(items)} curriculum items for ML Theory Lab")
    print(f"{'='*70}")
    print(f"\nBy Level:")
    for level in ['basics', 'intermediate', 'advanced', 'expert']:
        print(f"  {level}: {by_level.get(level, 0)}")
    
    print(f"\nBy Topic:")
    for book_id, count in by_book.items():
        print(f"  {book_id}: {count}")
    
    print(f"\nSaved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
