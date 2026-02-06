"""
Curriculum: Understanding Machine Learning (Shalev-Shwartz & Ben-David).
PAC learning, VC dimension, generalization, stability, Rademacher complexity. Theory-focused; no code required for concepts.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "pac", "name": "PAC Learning", "short": "PAC", "color": "#2563eb"},
    {"id": "vc", "name": "VC Dimension & Generalization", "short": "VC & Gen", "color": "#059669"},
    {"id": "stability", "name": "Stability & Rademacher", "short": "Stability", "color": "#7c3aed"},
    {"id": "bounds", "name": "Generalization Bounds", "short": "Bounds", "color": "#dc2626"},
]

CURRICULUM: List[Dict[str, Any]] = [
    {"id": "theory_pac", "book_id": "pac", "level": "intermediate", "title": "Probably Approximately Correct (PAC)",
     "learn": "Concept: (ε, δ)-PAC learnable if with prob ≥ 1-δ, err(h) ≤ ε. Sample complexity: how many examples needed.",
     "try_code": "# Theory: no code; see Shalev-Shwartz & Ben-David Ch. 3",
     "try_demo": None},
    {"id": "theory_agnostic", "book_id": "pac", "level": "advanced", "title": "Agnostic PAC Learning",
     "learn": "No realizable assumption: best in class has err L*; learner finds h with err ≤ L* + ε with high probability.",
     "try_code": "# Theory: sample complexity bounds",
     "try_demo": None},
    {"id": "theory_vc", "book_id": "vc", "level": "intermediate", "title": "VC Dimension",
     "learn": "VC(H) = max size of set shattered by H. Shattering: H realizes all 2^m labelings. Key for generalization bounds.",
     "try_code": "# VC dimension of linear classifiers in R^d is d+1",
     "try_demo": None},
    {"id": "theory_gen", "book_id": "vc", "level": "advanced", "title": "Generalization Bounds (VC)",
     "learn": "With prob ≥ 1-δ: L(h) ≤ L_S(h) + O(sqrt(VC(H)/m) + sqrt(log(1/δ)/m)). Trade-off complexity vs data.",
     "try_code": "# Bounds from SSBD Ch. 6",
     "try_demo": None},
    {"id": "theory_stability", "book_id": "stability", "level": "advanced", "title": "Stability of Learning Algorithms",
     "learn": "Uniform stability: changing one sample changes loss by ≤ β. Stable algorithms generalize (SSBD Ch. 13).",
     "try_code": "# e.g. SGD with small step size is stable",
     "try_demo": None},
    {"id": "theory_rademacher", "book_id": "stability", "level": "expert", "title": "Rademacher Complexity",
     "learn": "R(H) = E_σ[sup_h (1/m) sum σ_i h(z_i)]. Bounds generalization via R(H) + sqrt(log(1/δ)/m).",
     "try_code": "# Rademacher complexity of linear classes",
     "try_demo": None},
    {"id": "theory_bias_complexity", "book_id": "bounds", "level": "advanced", "title": "Bias-Complexity Tradeoff",
     "learn": "True error = approximation error (bias) + estimation error (complexity). Rich vs simple hypothesis classes.",
     "try_code": "# Structural risk minimization",
     "try_demo": None},
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
