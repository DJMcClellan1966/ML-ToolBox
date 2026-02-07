"""
Curriculum: Cross-domain "unusual" concepts — information theory, stat mech, quantum, linguistics, game theory, self-organization.
From ml_toolbox.textbook_concepts and interdisciplinary ML.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "info_theory", "name": "Information Theory", "short": "Info Theory", "color": "#3b82f6"},
    {"id": "stat_mech", "name": "Statistical Mechanics", "short": "Stat. Mech.", "color": "#0ea5e9"},
    {"id": "quantum", "name": "Quantum Computing", "short": "Quantum", "color": "#8b5cf6"},
    {"id": "linguistics", "name": "Linguistics & NLP", "short": "Linguistics", "color": "#10b981"},
    {"id": "game_theory", "name": "Game Theory", "short": "Game Theory", "color": "#f59e0b"},
    {"id": "self_org", "name": "Self-Organization", "short": "Self-Org", "color": "#ec4899"},
]

CURRICULUM: List[Dict[str, Any]] = [
    # INFORMATION THEORY (4 items)
    {"id": "entropy", "book_id": "info_theory", "level": "basics", "title": "Entropy & Information",
     "learn": "H(X) = -Σ p(x) log p(x). Measures uncertainty/information content. 0 bits for deterministic, max for uniform. Foundation for KL divergence, mutual information.",
     "try_code": "import numpy as np\nfrom scipy.stats import entropy\n\np1 = np.array([0.5, 0.5])  # Max entropy\np2 = np.array([0.9, 0.1])  # Low entropy\nprint(f'Uniform: {entropy(p1, base=2):.3f} bits')\nprint(f'Skewed: {entropy(p2, base=2):.3f} bits')",
     "try_demo": "cross_entropy"},
    {"id": "kl_divergence", "book_id": "info_theory", "level": "intermediate", "title": "KL Divergence",
     "learn": "KL(P||Q) = Σ p(x) log(p(x)/q(x)). Measures how P differs from Q. Non-symmetric. Used in VAE loss, policy optimization.",
     "try_code": "import numpy as np\nfrom scipy.stats import entropy\n\np, q = np.array([0.4, 0.6]), np.array([0.5, 0.5])\nprint(f'KL(P||Q): {entropy(p, q):.4f}')\nprint(f'KL(Q||P): {entropy(q, p):.4f}')",
     "try_demo": "cross_kl"},
    {"id": "mutual_information", "book_id": "info_theory", "level": "intermediate", "title": "Mutual Information",
     "learn": "I(X;Y) = H(X) - H(X|Y). Measures shared information. Used for feature selection, ICA. I(X;Y) ≥ 0, equals 0 iff independent.",
     "try_code": "from sklearn.feature_selection import mutual_info_classif\nimport numpy as np\n\nX = np.random.randn(1000, 5)\ny = (X[:, 0] + X[:, 1] > 0).astype(int)\nmi = mutual_info_classif(X, y)\nprint('MI with target:', mi)",
     "try_demo": "cross_mi"},
    {"id": "rate_distortion", "book_id": "info_theory", "level": "advanced", "title": "Rate-Distortion Theory",
     "learn": "Tradeoff between compression rate R and distortion D. Connects information theory and compression. Relevant to autoencoders, quantization.",
     "try_code": "from sklearn.cluster import KMeans\nimport numpy as np\n\nX = np.random.randn(1000, 10)\nfor k in [2, 5, 10, 20]:\n    kmeans = KMeans(n_clusters=k)\n    labels = kmeans.fit_predict(X)\n    X_comp = kmeans.cluster_centers_[labels]\n    dist = np.mean((X - X_comp)**2)\n    print(f'k={k}: Rate={np.log2(k):.1f} bits, Distortion={dist:.3f}')",
     "try_demo": None},
    
    # STATISTICAL MECHANICS (3 items)
    {"id": "sa", "book_id": "stat_mech", "level": "intermediate", "title": "Simulated Annealing",
     "learn": "Optimization inspired by cooling: temperature schedule, acceptance probability.",
     "try_code": "from ml_toolbox.textbook_concepts.statistical_mechanics import SimulatedAnnealing",
     "try_demo": "cross_sa"},
    {"id": "boltzmann", "book_id": "stat_mech", "level": "intermediate", "title": "Boltzmann Distribution",
     "learn": "P(state) ∝ exp(-E(state)/kT). Low energy states more probable. Used in Boltzmann machines, simulated annealing.",
     "try_code": "import numpy as np\n\nenergies = np.linspace(0, 10, 100)\nfor T in [0.5, 2.0, 5.0]:\n    probs = np.exp(-energies / T)\n    probs /= probs.sum()\n    regime = 'low' if T < 1 else 'high'\n    print(f'T={T}: concentrates at {regime} energy')",
     "try_demo": "cross_boltzmann"},
    {"id": "ising_model", "book_id": "stat_mech", "level": "advanced", "title": "Ising Model & Spin Systems",
     "learn": "Lattice of binary spins with pairwise interactions. E = -Σ J_ij s_i s_j. Related to Hopfield networks, Boltzmann machines.",
     "try_code": "import numpy as np\n\nn = 5\nspins = np.random.choice([-1, 1], n)\nJ = np.random.randn(n, n); J = (J + J.T) / 2\nenergy = -sum(J[i,j] * spins[i] * spins[j] for i in range(n) for j in range(i+1,n))\nprint(f'Spins: {spins}, Energy: {energy:.2f}')",
     "try_demo": None},
    
    # QUANTUM-INSPIRED ML (3 items)
    {"id": "qm_basics", "book_id": "quantum", "level": "intermediate", "title": "Quantum-inspired ML",
     "learn": "Quantum mechanics concepts applied to ML: superposition, measurement.",
     "try_code": "from ml_toolbox.textbook_concepts.quantum_mechanics import ...",
     "try_demo": None},
    {"id": "quantum_annealing", "book_id": "quantum", "level": "advanced", "title": "Quantum Annealing",
     "learn": "Use quantum tunneling for optimization. Hamiltonian H(t) = (1-t)H_0 + tH_problem. D-Wave systems.",
     "try_code": "import numpy as np\n\n# QUBO example\nQ = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])\nbest_x, best_e = None, float('inf')\nfor i in range(2**3):\n    x = np.array([int(b) for b in format(i, '03b')])\n    e = x @ Q @ x\n    if e < best_e: best_x, best_e = x, e\nprint(f'Optimal: {best_x}, Energy: {best_e}')",
     "try_demo": None},
    {"id": "quantum_ml", "book_id": "quantum", "level": "expert", "title": "Quantum Machine Learning",
     "learn": "Quantum algorithms for ML: quantum PCA, quantum SVM, VQE. Potential exponential speedups. NISQ era devices.",
     "try_code": "print('Quantum ML: encode data → quantum circuit → measure → optimize')\nprint('Examples: VQE, QAOA, quantum kernels')",
     "try_demo": None},
    
    # LINGUISTICS (3 items)
    {"id": "ling_parser", "book_id": "linguistics", "level": "basics", "title": "Syntactic Parsing & Grammar",
     "learn": "Syntactic parser and grammar-based feature extraction for text.",
     "try_code": "from ml_toolbox.textbook_concepts.linguistics import SimpleSyntacticParser",
     "try_demo": "cross_ling"},
    {"id": "zipf_law", "book_id": "linguistics", "level": "basics", "title": "Zipf's Law & Power Laws",
     "learn": "Word frequency ∝ 1/rank. f(r) = C/r^α. Empirical law in natural language. Heavy-tailed distributions.",
     "try_code": "from collections import Counter\nimport numpy as np\n\ntext = 'the ' * 1000 + 'of ' * 500 + 'and ' * 333\nfreq = Counter(text.split())\nranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)\nfor i, (word, count) in enumerate(ranked[:5], 1):\n    print(f'Rank {i}: {word} ({count})')",
     "try_demo": "cross_zipf"},
    {"id": "chomsky_hierarchy", "book_id": "linguistics", "level": "intermediate", "title": "Chomsky Hierarchy",
     "learn": "4 levels of formal grammars: Type 0 (unrestricted), Type 1 (context-sensitive), Type 2 (context-free), Type 3 (regular).",
     "try_code": "print('Chomsky Hierarchy:')\nprint('  Type 3: Regular (FSA)')\nprint('  Type 2: Context-Free (CFG, parsers)')\nprint('  Type 1: Context-Sensitive')\nprint('  Type 0: Unrestricted (Turing-complete)')",
     "try_demo": None},
    
    # GAME THEORY (3 items)
    {"id": "nash_equilibrium", "book_id": "game_theory", "level": "intermediate", "title": "Nash Equilibrium",
     "learn": "Strategy profile where no player benefits from unilateral deviation. Used in multi-agent RL, GANs.",
     "try_code": "print('Prisoner\\'s Dilemma:')\nprint('  Nash Equilibrium: Both Defect (1, 1)')\nprint('  Pareto Optimal: Both Cooperate (3, 3)')\nprint('  Tension between individual and collective rationality')",
     "try_demo": "cross_nash"},
    {"id": "minimax", "book_id": "game_theory", "level": "intermediate", "title": "Minimax & Zero-Sum Games",
     "learn": "Minimax: max_a min_b u(a,b). Player maximizes worst-case. Chess, Go use minimax with alpha-beta pruning.",
     "try_code": "print('Minimax Algorithm:')\nprint('  1. Generate game tree')\nprint('  2. Evaluate leaf nodes')\nprint('  3. Backpropagate: max at player nodes, min at opponent')\nprint('  4. Alpha-beta pruning reduces search')",
     "try_demo": None},
    {"id": "mechanism_design", "book_id": "game_theory", "level": "advanced", "title": "Mechanism Design & Auctions",
     "learn": "Reverse game theory: design game rules to achieve desired outcome. VCG mechanism. Used in ad auctions.",
     "try_code": "import numpy as np\n\n# Vickrey auction\nvaluations = np.array([100, 80, 70, 60])\nbids = valuations  # Truthful\nwinner = np.argmax(bids)\nprice = np.partition(bids, -2)[-2]\nprint(f'Winner pays 2nd price: {price}')",
     "try_demo": None},
    
    # SELF-ORGANIZATION (3 items)
    {"id": "som", "book_id": "self_org", "level": "intermediate", "title": "Self-Organizing Map (SOM)",
     "learn": "Unsupervised topology-preserving maps.",
     "try_code": "from ml_toolbox.textbook_concepts.self_organization import SelfOrganizingMap",
     "try_demo": None},
    {"id": "dissipative", "book_id": "self_org", "level": "advanced", "title": "Dissipative Structures",
     "learn": "Far-from-equilibrium dynamics.",
     "try_code": "from ml_toolbox.textbook_concepts.self_organization import DissipativeStructure",
     "try_demo": None},
    {"id": "emergent_behavior", "book_id": "self_org", "level": "advanced", "title": "Emergent Behavior & Complexity",
     "learn": "Complex global behavior from simple local rules. Conway's Game of Life, flocking, cellular automata.",
     "try_code": "print('Emergent Systems:')\nprint('  Game of Life: 4 rules → complex patterns')\nprint('  Boids: 3 rules (separation, alignment, cohesion) → flocking')\nprint('  Swarm intelligence: simple agents → collective behavior')",
     "try_demo": "cross_gol"},
]


def get_curriculum(): return list(CURRICULUM)
def get_books(): return list(BOOKS)
def get_levels(): return list(LEVELS)
def get_by_book(book_id: str): return [c for c in CURRICULUM if c["book_id"] == book_id]
def get_by_level(level: str): return [c for c in CURRICULUM if c["level"] == level]
def get_item(item_id: str):
    for c in CURRICULUM:
        if c["id"] == item_id: return c
    return None
