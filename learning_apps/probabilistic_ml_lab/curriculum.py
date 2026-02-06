"""
Curriculum: Machine Learning — A Probabilistic Perspective (Murphy).
Graphical models, EM, variational inference, Bayesian learning. Uses ml_toolbox.textbook_concepts.probabilistic_ml.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "graphical", "name": "Graphical Models", "short": "Graphical", "color": "#2563eb"},
    {"id": "em", "name": "EM Algorithm", "short": "EM", "color": "#059669"},
    {"id": "variational", "name": "Variational Inference", "short": "VI", "color": "#7c3aed"},
    {"id": "bayesian", "name": "Bayesian Learning", "short": "Bayesian", "color": "#dc2626"},
]

CURRICULUM: List[Dict[str, Any]] = [
    {"id": "murphy_graphical", "book_id": "graphical", "level": "intermediate", "title": "Bayesian Networks & DAGs",
     "learn": "Directed graphical models: nodes = RVs, edges = dependencies. Factorization P(X) = prod P(X_i | pa(X_i)).",
     "try_code": "from ml_toolbox.textbook_concepts.probabilistic_ml import GraphicalModels",
     "try_demo": None},
    {"id": "murphy_em", "book_id": "em", "level": "intermediate", "title": "Expectation-Maximization",
     "learn": "E-step: compute posterior over latent Z given params. M-step: maximize expected complete-data log-likelihood. GMM, HMM.",
     "try_code": "from ml_toolbox.textbook_concepts.probabilistic_ml import EMAlgorithm\nimport numpy as np\nX = np.random.randn(200, 2)\nem = EMAlgorithm(n_components=2); em.fit(X)\nprint('means:', em.means_)",
     "try_demo": None},
    {"id": "murphy_em_convergence", "book_id": "em", "level": "advanced", "title": "EM Convergence & Lower Bound",
     "learn": "EM maximizes a lower bound on log p(X). Monotonic increase; convergence to local optimum.",
     "try_code": "from ml_toolbox.textbook_concepts.probabilistic_ml import EMAlgorithm",
     "try_demo": None},
    {"id": "murphy_vi", "book_id": "variational", "level": "advanced", "title": "Variational Inference",
     "learn": "Approximate posterior q(z) ≈ p(z|x). ELBO = E_q[log p(x,z)] - E_q[log q(z)]. Mean-field, coordinate ascent.",
     "try_code": "from ml_toolbox.textbook_concepts.probabilistic_ml import VariationalInference",
     "try_demo": None},
    {"id": "murphy_bayesian", "book_id": "bayesian", "level": "intermediate", "title": "Bayesian Learning",
     "learn": "Prior p(θ), likelihood p(D|θ), posterior p(θ|D). Predictive distribution p(x_new|D). Conjugate priors.",
     "try_code": "from ml_toolbox.textbook_concepts.probabilistic_ml import BayesianLearning",
     "try_demo": None},
    {"id": "murphy_predictive", "book_id": "bayesian", "level": "advanced", "title": "Bayesian Predictive Distribution",
     "learn": "p(x_new|D) = ∫ p(x_new|θ) p(θ|D) dθ. Uncertainty in predictions from posterior over parameters.",
     "try_code": "from ml_toolbox.textbook_concepts.probabilistic_ml import BayesianLearning",
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
