"""
Enrich Cross-Domain Lab - Interdisciplinary ML Concepts
========================================================

Generates curriculum items for cross-domain concepts in ML.
Covers quantum computing, information theory, statistical mechanics, 
linguistics, game theory, and self-organization.
"""

from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parents[1]


def generate_cross_domain_curriculum():
    """Generate cross-domain curriculum items."""
    
    items = [
        # INFORMATION THEORY (4 items)
        {"id": "entropy", "book_id": "info_theory", "level": "basics", "title": "Entropy & Information",
         "learn": "H(X) = -Σ p(x) log p(x). Measures uncertainty/information content. 0 bits for deterministic, max for uniform. Foundation for KL divergence, mutual information.",
         "try_code": "import numpy as np\nfrom scipy.stats import entropy\n\n# Entropy of distributions\np1 = np.array([0.5, 0.5])  # Max entropy for binary\np2 = np.array([0.9, 0.1])  # Low entropy\np3 = np.array([1.0, 0.0])  # Zero entropy\n\nprint(f'Uniform: {entropy(p1, base=2):.3f} bits')\nprint(f'Skewed: {entropy(p2, base=2):.3f} bits')\nprint(f'Deterministic: {entropy(p3, base=2):.3f} bits')",
         "try_demo": "cross_entropy"},
        
        {"id": "kl_divergence", "book_id": "info_theory", "level": "intermediate", "title": "KL Divergence",
         "learn": "KL(P||Q) = Σ p(x) log(p(x)/q(x)). Measures how P differs from Q. Non-symmetric. Used in VAE loss, policy optimization. KL(P||Q) ≥ 0, equals 0 iff P=Q.",
         "try_code": "import numpy as np\nfrom scipy.stats import entropy\n\np = np.array([0.4, 0.6])\nq = np.array([0.5, 0.5])\n\nkl_pq = entropy(p, q)\nkl_qp = entropy(q, p)\n\nprint(f'KL(P||Q): {kl_pq:.4f}')\nprint(f'KL(Q||P): {kl_qp:.4f}')\nprint(f'Asymmetric: {kl_pq != kl_qp}')",
         "try_demo": "cross_kl"},
        
        {"id": "mutual_information", "book_id": "info_theory", "level": "intermediate", "title": "Mutual Information",
         "learn": "I(X;Y) = H(X) - H(X|Y) = H(X) + H(Y) - H(X,Y). Measures shared information. Symmetric. Used for feature selection, ICA. I(X;Y) ≥ 0, equals 0 iff independent.",
         "try_code": "from sklearn.feature_selection import mutual_info_classif\nimport numpy as np\n\nX = np.random.randn(1000, 5)\ny = (X[:, 0] + X[:, 1] > 0).astype(int)  # y depends on X[:, 0] and X[:, 1]\n\nmi = mutual_info_classif(X, y, random_state=42)\nprint('Mutual information with target:')\nfor i, mi_val in enumerate(mi):\n    print(f'  Feature {i}: {mi_val:.3f}')",
         "try_demo": "cross_mi"},
        
        {"id": "rate_distortion", "book_id": "info_theory", "level": "advanced", "title": "Rate-Distortion Theory",
         "learn": "Tradeoff between compression rate R and distortion D. R(D) = min I(X;Y) s.t. E[d(X,Y)] ≤ D. Connects information theory and compression. Relevant to autoencoders, quantization.",
         "try_code": "import numpy as np\n\n# Rate-distortion concept: compress data with acceptable loss\n# Example: k-means is a rate-distortion compressor\nfrom sklearn.cluster import KMeans\n\nX = np.random.randn(1000, 10)\n\n# Different compression rates (k)\nfor k in [2, 5, 10, 20]:\n    kmeans = KMeans(n_clusters=k, random_state=42)\n    labels = kmeans.fit_predict(X)\n    X_compressed = kmeans.cluster_centers_[labels]\n    distortion = np.mean((X - X_compressed)**2)\n    rate = np.log2(k)  # bits per sample\n    print(f'k={k:2d}: Rate={rate:.2f} bits, Distortion={distortion:.4f}')",
         "try_demo": None},
        
        # STATISTICAL MECHANICS (3 items)
        {"id": "sa", "book_id": "stat_mech", "level": "intermediate", "title": "Simulated Annealing",
         "learn": "Optimization inspired by cooling: temperature schedule, acceptance probability. From textbook_concepts.statistical_mechanics.",
         "try_code": "from ml_toolbox.textbook_concepts.statistical_mechanics import SimulatedAnnealing",
         "try_demo": "cross_sa"},
        
        {"id": "boltzmann", "book_id": "stat_mech", "level": "intermediate", "title": "Boltzmann Distribution",
         "learn": "P(state) ∝ exp(-E(state)/kT). Low energy states more probable. Temperature controls exploration. Used in Boltzmann machines, simulated annealing.",
         "try_code": "import numpy as np\nimport matplotlib.pyplot as plt\n\n# Boltzmann distribution\nenergies = np.linspace(0, 10, 100)\n\nfor T in [0.5, 1.0, 2.0, 5.0]:\n    probs = np.exp(-energies / T)\n    probs /= probs.sum()\n    plt.plot(energies, probs, label=f'T={T}')\n\nplt.xlabel('Energy')\nplt.ylabel('Probability')\nplt.legend()\nplt.title('Boltzmann Distribution at Different Temperatures')\n# plt.show()\nprint('Higher T → more uniform, lower T → concentrate on low energy')",
         "try_demo": "cross_boltzmann"},
        
        {"id": "ising_model", "book_id": "stat_mech", "level": "advanced", "title": "Ising Model & Spin Systems",
         "learn": "Lattice of binary spins with pairwise interactions. E = -Σ J_ij s_i s_j. Phase transitions at critical temperature. Related to Hopfield networks, Boltzmann machines.",
         "try_code": "import numpy as np\n\ndef ising_energy(spins, J):\n    \"\"\"Energy of Ising model.\"\"\"\n    energy = 0\n    n = len(spins)\n    for i in range(n):\n        for j in range(i+1, n):\n            energy -= J[i, j] * spins[i] * spins[j]\n    return energy\n\n# Small example\nn = 5\nspins = np.random.choice([-1, 1], n)\nJ = np.random.randn(n, n)\nJ = (J + J.T) / 2  # Symmetric\n\nprint(f'Spins: {spins}')\nprint(f'Energy: {ising_energy(spins, J):.3f}')",
         "try_demo": None},
        
        # QUANTUM-INSPIRED ML (3 items)
        {"id": "qm_basics", "book_id": "quantum", "level": "intermediate", "title": "Quantum-inspired ML",
         "learn": "Quantum mechanics concepts applied to ML: superposition, measurement. From textbook_concepts.quantum_mechanics.",
         "try_code": "from ml_toolbox.textbook_concepts.quantum_mechanics import ...",
         "try_demo": None},
        
        {"id": "quantum_annealing", "book_id": "quantum", "level": "advanced", "title": "Quantum Annealing",
         "learn": "Use quantum tunneling for optimization. Hamiltonian H(t) = (1-t)H_0 + tH_problem. Evolve from easy to hard problem. D-Wave systems. Related to adiabatic quantum computing.",
         "try_code": "# Quantum annealing concept (classical simulation)\nimport numpy as np\n\n# QUBO (Quadratic Unconstrained Binary Optimization)\n# min x^T Q x, x ∈ {0,1}^n\n\n# Example: find assignment minimizing quadratic form\nQ = np.array([\n    [2, -1, 0],\n    [-1, 2, -1],\n    [0, -1, 2]\n])\n\n# Brute force for small problem\nbest_x = None\nbest_energy = float('inf')\n\nfor i in range(2**3):\n    x = np.array([int(b) for b in format(i, '03b')])\n    energy = x @ Q @ x\n    if energy < best_energy:\n        best_energy = energy\n        best_x = x\n\nprint(f'Optimal x: {best_x}, Energy: {best_energy}')",
         "try_demo": None},
        
        {"id": "quantum_ml", "book_id": "quantum", "level": "expert", "title": "Quantum Machine Learning",
         "learn": "Quantum algorithms for ML: quantum PCA, quantum SVM, variational quantum eigensolver (VQE). Potential exponential speedups. NISQ era: noisy intermediate-scale quantum.",
         "try_code": "# Conceptual: quantum circuit for ML\n# In practice, use Qiskit, Cirq, PennyLane\n\nprint('Quantum ML Pipeline:')\nprint('  1. Encode classical data into quantum states')\nprint('  2. Apply parameterized quantum circuit (ansatz)')\nprint('  3. Measure to get classical output')\nprint('  4. Optimize parameters via classical gradient descent')\nprint()\nprint('Examples:')\nprint('  - VQE: find ground state energy')\nprint('  - QAOA: combinatorial optimization')\nprint('  - Quantum kernel methods: feature space in Hilbert space')",
         "try_demo": None},
        
        # LINGUISTICS & NLP (3 items)
        {"id": "ling_parser", "book_id": "linguistics", "level": "basics", "title": "Syntactic Parsing & Grammar",
         "learn": "Simple syntactic parser and grammar-based feature extraction for text. From textbook_concepts.linguistics.",
         "try_code": "from ml_toolbox.textbook_concepts.linguistics import SimpleSyntacticParser, GrammarBasedFeatureExtractor",
         "try_demo": "cross_ling"},
        
        {"id": "zipf_law", "book_id": "linguistics", "level": "basics", "title": "Zipf's Law & Power Laws",
         "learn": "Word frequency ∝ 1/rank. f(r) = C/r^α. Empirical law in natural language. Power laws appear in web links, citations, city sizes. Heavy-tailed distributions.",
         "try_code": "import numpy as np\nimport matplotlib.pyplot as plt\nfrom collections import Counter\n\n# Simulate Zipf distribution\ntext = 'the ' * 1000 + 'of ' * 500 + 'and ' * 333 + 'to ' * 250 + 'a ' * 200\nwords = text.split()\nfreq = Counter(words)\n\n# Rank by frequency\nranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)\nranks = np.arange(1, len(ranked) + 1)\nfreqs = np.array([f for _, f in ranked])\n\n# Zipf: log(freq) = -α log(rank) + log(C)\nplt.loglog(ranks, freqs, 'o')\nplt.xlabel('Rank')\nplt.ylabel('Frequency')\nplt.title(\"Zipf's Law\")\n# plt.show()\nprint('Power law: frequency ∝ 1/rank^α')",
         "try_demo": "cross_zipf"},
        
        {"id": "chomsky_hierarchy", "book_id": "linguistics", "level": "intermediate", "title": "Chomsky Hierarchy",
         "learn": "4 levels of formal grammars: Type 0 (unrestricted), Type 1 (context-sensitive), Type 2 (context-free), Type 3 (regular). CFGs parse programming languages. RNNs are Turing complete (Type 0).",
         "try_code": "# Context-Free Grammar example\nimport nltk\nfrom nltk import CFG\n\ngrammar = CFG.fromstring(\"\"\"\n  S -> NP VP\n  NP -> Det N | Det N PP\n  VP -> V NP | V NP PP\n  PP -> P NP\n  Det -> 'the' | 'a'\n  N -> 'cat' | 'dog' | 'park'\n  V -> 'chased' | 'saw'\n  P -> 'in' | 'with'\n\"\"\")\n\nfrom nltk.parse import ChartParser\nparser = ChartParser(grammar)\n\nsentence = 'the cat chased the dog'.split()\nfor tree in parser.parse(sentence):\n    print(tree)\n    break",
         "try_demo": None},
        
        # GAME THEORY (3 items)
        {"id": "nash_equilibrium", "book_id": "game_theory", "level": "intermediate", "title": "Nash Equilibrium",
         "learn": "Strategy profile where no player benefits from unilateral deviation. Every finite game has ≥1 Nash equilibrium (mixed or pure). Used in multi-agent RL, GANs.",
         "try_code": "import numpy as np\nfrom scipy.optimize import linprog\n\n# Prisoner's Dilemma payoff matrix\n# (Cooperate, Cooperate): (3, 3)\n# (Cooperate, Defect): (0, 5)\n# (Defect, Cooperate): (5, 0)\n# (Defect, Defect): (1, 1)\n\n# Pure strategy Nash equilibrium: (Defect, Defect)\nprint('Prisoner\\'s Dilemma:')\nprint('  Nash Equilibrium: Both Defect (1, 1)')\nprint('  Pareto Optimal: Both Cooperate (3, 3)')\nprint('  → Tension between individual rationality and collective benefit')",
         "try_demo": "cross_nash"},
        
        {"id": "minimax", "book_id": "game_theory", "level": "intermediate", "title": "Minimax & Zero-Sum Games",
         "learn": "Minimax: max_a min_b u(a,b). Player maximizes worst-case outcome. Zero-sum: u_1 + u_2 = 0. Chess, Go use minimax with alpha-beta pruning. Value = minimax value.",
         "try_code": "import numpy as np\n\ndef minimax(depth, is_maximizing, alpha, beta, game_state, eval_fn):\n    \"\"\"Minimax with alpha-beta pruning.\"\"\"\n    if depth == 0:\n        return eval_fn(game_state)\n    \n    if is_maximizing:\n        max_eval = -float('inf')\n        for child in get_children(game_state):\n            eval_score = minimax(depth-1, False, alpha, beta, child, eval_fn)\n            max_eval = max(max_eval, eval_score)\n            alpha = max(alpha, eval_score)\n            if beta <= alpha:\n                break  # Prune\n        return max_eval\n    else:\n        min_eval = float('inf')\n        for child in get_children(game_state):\n            eval_score = minimax(depth-1, True, alpha, beta, child, eval_fn)\n            min_eval = min(min_eval, eval_score)\n            beta = min(beta, eval_score)\n            if beta <= alpha:\n                break  # Prune\n        return min_eval\n\nprint('Minimax searches game tree to find optimal move')\nprint('Alpha-beta pruning reduces search space')",
         "try_demo": None},
        
        {"id": "mechanism_design", "book_id": "game_theory", "level": "advanced", "title": "Mechanism Design & Auctions",
         "learn": "Reverse game theory: design game rules to achieve desired outcome. VCG mechanism: truthful bidding is dominant strategy. Used in ad auctions, spectrum allocation.",
         "try_code": "# Vickrey (2nd-price) auction concept\nimport numpy as np\n\n# Bidders' true valuations\nvaluations = np.array([100, 80, 70, 60])\n\n# In Vickrey auction, bidding true valuation is optimal\nbids = valuations.copy()  # Truthful bidding\n\nwinner_idx = np.argmax(bids)\nprice = np.partition(bids, -2)[-2]  # 2nd highest bid\n\nprint(f'Winner: Bidder {winner_idx} (valuation={valuations[winner_idx]})')\nprint(f'Price paid: {price} (2nd highest bid)')\nprint(f'Utility: {valuations[winner_idx] - price}')\nprint('\\nTruthful bidding is dominant strategy in Vickrey auction')",
         "try_demo": None},
        
        # SELF-ORGANIZATION (3 items)
        {"id": "som", "book_id": "self_org", "level": "intermediate", "title": "Self-Organizing Map (SOM)",
         "learn": "Unsupervised topology-preserving maps. From textbook_concepts.self_organization.",
         "try_code": "from ml_toolbox.textbook_concepts.self_organization import SelfOrganizingMap",
         "try_demo": None},
        
        {"id": "dissipative", "book_id": "self_org", "level": "advanced", "title": "Dissipative Structures",
         "learn": "Far-from-equilibrium dynamics. From textbook_concepts.self_organization.",
         "try_code": "from ml_toolbox.textbook_concepts.self_organization import DissipativeStructure",
         "try_demo": None},
        
        {"id": "emergent_behavior", "book_id": "self_org", "level": "advanced", "title": "Emergent Behavior & Complexity",
         "learn": "Complex global behavior from simple local rules. Conway's Game of Life, flocking (Boids), cellular automata. Relevant to swarm intelligence, multi-agent systems.",
         "try_code": "import numpy as np\n\n# Conway's Game of Life rules\ndef game_of_life_step(grid):\n    \"\"\"One step of Game of Life.\"\"\"\n    rows, cols = grid.shape\n    new_grid = np.zeros_like(grid)\n    \n    for i in range(rows):\n        for j in range(cols):\n            # Count live neighbors\n            neighbors = 0\n            for di in [-1, 0, 1]:\n                for dj in [-1, 0, 1]:\n                    if di == 0 and dj == 0:\n                        continue\n                    ni, nj = (i+di) % rows, (j+dj) % cols\n                    neighbors += grid[ni, nj]\n            \n            # Apply rules\n            if grid[i, j] == 1:  # Live cell\n                new_grid[i, j] = 1 if neighbors in [2, 3] else 0\n            else:  # Dead cell\n                new_grid[i, j] = 1 if neighbors == 3 else 0\n    \n    return new_grid\n\n# Initialize random grid\ngrid = np.random.choice([0, 1], size=(10, 10), p=[0.7, 0.3])\nprint('Game of Life: complex patterns emerge from simple rules')",
         "try_demo": "cross_gol"},
    ]
    
    return items


def main():
    print("=" * 70)
    print("CROSS-DOMAIN LAB ENRICHMENT")
    print("=" * 70)
    print()
    
    items = generate_cross_domain_curriculum()
    
    print(f"✅ Generated {len(items)} curriculum items")
    
    # Distribution
    from collections import Counter
    level_counts = Counter(item['level'] for item in items)
    print(f"\nBy level:")
    for level in ['basics', 'intermediate', 'advanced', 'expert']:
        count = level_counts.get(level, 0)
        print(f"  {level:15s}: {count}")
    
    # Save
    output_dir = REPO_ROOT / "learning_apps" / ".cache"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "cross_domain_enriched.json"
    output_file.write_text(json.dumps(items, indent=2))
    
    print(f"\n✅ Saved to: {output_file}")
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"New items: {len(items)}")
    print(f"Target achieved: 19+ items covering cross-domain ML concepts")
    print(f"\nTopics:")
    print(f"  • Information Theory (entropy, KL divergence, mutual information, rate-distortion)")
    print(f"  • Statistical Mechanics (simulated annealing, Boltzmann, Ising model)")
    print(f"  • Quantum-Inspired ML (quantum annealing, quantum ML)")
    print(f"  • Linguistics (parsing, Zipf's law, Chomsky hierarchy)")
    print(f"  • Game Theory (Nash equilibrium, minimax, mechanism design)")
    print(f"  • Self-Organization (SOM, dissipative structures, emergent behavior)")


if __name__ == "__main__":
    main()
