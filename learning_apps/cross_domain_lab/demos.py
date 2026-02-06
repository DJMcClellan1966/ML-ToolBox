"""Demos for Cross-Domain Lab. Uses ml_toolbox.textbook_concepts (run from repo root)."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _sa_demo():
    from ml_toolbox.textbook_concepts.statistical_mechanics import SimulatedAnnealing
    import numpy as np
    def obj(x): return float(np.sum(x ** 2))
    sa = SimulatedAnnealing(objective_function=obj, initial_solution=np.array([5.0, 5.0]),
                           initial_temperature=10.0, cooling_rate=0.95, max_iterations=100)
    result = sa.optimize()
    out = "SimulatedAnnealing minimize x^2+y^2 from (5,5).\nBest: " + str(result["best_solution"]) + ", energy: " + str(result["best_energy"])
    return {"ok": True, "output": out}


def _ling_demo():
    from ml_toolbox.textbook_concepts.linguistics import SimpleSyntacticParser
    parser = SimpleSyntacticParser()
    tags = parser.extract_pos_tags("The model is learning.")
    out = "SimpleSyntacticParser on 'The model is learning.'\nPOS tags: " + str(tags)
    return {"ok": True, "output": out}


DEMO_HANDLERS = {"cross_sa": _sa_demo, "cross_ling": _ling_demo}


def run_demo(demo_id: str):
    if demo_id in DEMO_HANDLERS:
        try:
            return DEMO_HANDLERS[demo_id]()
        except Exception as e:
            return {"ok": False, "output": "", "error": str(e)}
    return {"ok": False, "output": "", "error": f"No demo: {demo_id}"}
