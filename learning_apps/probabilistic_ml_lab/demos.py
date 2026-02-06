"""Demos for Probabilistic ML Lab. Uses ml_toolbox.textbook_concepts.probabilistic_ml."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np


def prob_em_gmm():
    """EM Algorithm for Gaussian Mixture demo."""
    try:
        from ml_toolbox.textbook_concepts.probabilistic_ml import EMAlgorithm
        np.random.seed(42)
        # Generate data from 2 Gaussians
        X1 = np.random.randn(100, 2) + np.array([2, 2])
        X2 = np.random.randn(100, 2) + np.array([-2, -2])
        X = np.vstack([X1, X2])
        
        em = EMAlgorithm(n_components=2)
        em.fit(X)
        out = "EM Algorithm (Gaussian Mixture, 2 components)\n"
        out += f"Data: 200 samples from 2 Gaussians\n"
        out += f"Learned means:\n  {em.means_[0].round(2).tolist()}\n  {em.means_[1].round(2).tolist()}\n"
        out += f"True centers: [2,2] and [-2,-2]"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def prob_bayesian_update():
    """Bayesian learning demo (coin flip)."""
    try:
        # Beta-Bernoulli conjugate model
        np.random.seed(42)
        # Prior: Beta(1, 1) = uniform
        alpha, beta = 1, 1
        # Observe 7 heads out of 10 flips
        heads, tails = 7, 3
        # Posterior: Beta(alpha + heads, beta + tails)
        alpha_post = alpha + heads
        beta_post = beta + tails
        posterior_mean = alpha_post / (alpha_post + beta_post)
        
        out = "Bayesian Learning: Beta-Bernoulli\n"
        out += f"Prior: Beta(1, 1) (uniform)\n"
        out += f"Data: 7 heads, 3 tails\n"
        out += f"Posterior: Beta({alpha_post}, {beta_post})\n"
        out += f"Posterior mean (p_head): {posterior_mean:.3f}\n"
        out += "True probability: 0.7 (unknown to model)"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def prob_variational():
    """Simple variational inference concept demo."""
    try:
        # Illustrate ELBO computation for a simple model
        np.random.seed(42)
        # True model: p(x|z) = N(z, 1), p(z) = N(0, 1)
        # Observed x = 2
        x_obs = 2.0
        # Variational approx: q(z) = N(mu, sigma^2)
        mu_q = 1.5  # Variational mean
        sigma_q = 0.5  # Variational std
        
        # ELBO = E_q[log p(x|z)] - KL(q || prior)
        # For illustration, compute approximate ELBO
        n_samples = 1000
        z_samples = np.random.normal(mu_q, sigma_q, n_samples)
        log_likelihood = -0.5 * (x_obs - z_samples) ** 2  # log N(x|z,1)
        kl_term = 0.5 * (mu_q ** 2 + sigma_q ** 2 - 1 - 2 * np.log(sigma_q))
        elbo = np.mean(log_likelihood) - kl_term
        
        out = "Variational Inference (ELBO)\n"
        out += f"Model: p(x|z) = N(z, 1), p(z) = N(0, 1)\n"
        out += f"Observed x = {x_obs}\n"
        out += f"Variational q(z) = N({mu_q}, {sigma_q}²)\n"
        out += f"Estimated ELBO: {elbo:.3f}\n"
        out += "Goal: Maximize ELBO to approximate posterior"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def prob_graphical():
    """Graphical model structure demo."""
    try:
        from ml_toolbox.textbook_concepts.probabilistic_ml import GraphicalModels
        # Create simple Bayesian network
        gm = GraphicalModels()
        gm.add_node("Rain", parents=[])
        gm.add_node("Sprinkler", parents=["Rain"])
        gm.add_node("WetGrass", parents=["Rain", "Sprinkler"])
        
        out = "Bayesian Network Structure\n"
        out += "Nodes: Rain → Sprinkler → WetGrass\n"
        out += "       Rain → WetGrass\n"
        out += f"Graph structure: {gm.get_structure()}\n"
        out += "Factorization: P(R)·P(S|R)·P(W|R,S)"
        return {"ok": True, "output": out}
    except Exception as e:
        # Fallback if GraphicalModels not available
        out = "Bayesian Network Structure (conceptual)\n"
        out += "Nodes: Rain → Sprinkler → WetGrass\n"
        out += "       Rain → WetGrass\n"
        out += "Factorization: P(R)·P(S|R)·P(W|R,S)\n"
        out += "Directed Acyclic Graph (DAG) encodes dependencies"
        return {"ok": True, "output": out}


DEMO_HANDLERS = {
    "prob_em": prob_em_gmm,
    "prob_bayesian": prob_bayesian_update,
    "prob_vi": prob_variational,
    "prob_graphical": prob_graphical,
}


def run_demo(demo_id: str):
    if demo_id in DEMO_HANDLERS:
        try:
            return DEMO_HANDLERS[demo_id]()
        except Exception as e:
            return {"ok": False, "output": "", "error": str(e)}
    return {"ok": False, "output": "", "error": f"No demo: {demo_id}"}
