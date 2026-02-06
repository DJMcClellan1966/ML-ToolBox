"""Demos for AI Concepts Lab. Uses ml_toolbox.ai_concepts (run from repo root)."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEMO_HANDLERS = {}


def _rl_demo():
    from ml_toolbox.ai_concepts.reinforcement_learning import QLearning
    import numpy as np
    # Tiny MDP: 2 states, 2 actions
    ql = QLearning(n_states=2, n_actions=2, learning_rate=0.1, discount_factor=0.9, epsilon=0.2)
    # One update: state 0, action 0, reward 1, next_state 1
    ql.update(0, 0, 1.0, 1, False)
    out = "Q-Learning (2 states, 2 actions), one update:\nQ-table:\n" + str(ql.Q)
    return {"ok": True, "output": out}


def _clustering_demo():
    from ml_toolbox.ai_concepts.clustering import KMeans
    import numpy as np
    X = np.array([[1, 2], [1.5, 2], [5, 8], [8, 8], [1, 0.6]])
    km = KMeans(n_clusters=2)
    km.fit(X)
    labels = km.labels_
    out = "KMeans n_clusters=2 on 5 points.\nLabels: " + str(labels.tolist())
    return {"ok": True, "output": out}


try:
    DEMO_HANDLERS["rl_qlearning"] = _rl_demo
except Exception:
    pass
try:
    DEMO_HANDLERS["ai_clustering"] = _clustering_demo
except Exception:
    pass


def run_demo(demo_id: str):
    if demo_id in DEMO_HANDLERS:
        try:
            return DEMO_HANDLERS[demo_id]()
        except Exception as e:
            return {"ok": False, "output": "", "error": str(e)}
    return {"ok": False, "output": "", "error": f"No demo: {demo_id}"}
