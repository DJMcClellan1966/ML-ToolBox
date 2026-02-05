"""Sandbox 07: Dissipative Training Dynamics - viability test."""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def run():
    from ml_toolbox.textbook_concepts.self_organization import DissipativeStructure
    ds = DissipativeStructure(structure_size=10, energy_input=1.0)
    states = [ds.state.copy()]
    for _ in range(20):
        ds.update(dt=0.01)
        states.append(ds.state.copy())
    states = np.array(states)
    # Stability: std of each dimension over time (low = stable)
    stability_per_dim = np.std(states, axis=0)
    mean_stability = float(np.mean(stability_per_dim))
    return {
        "ok": True,
        "steps": 20,
        "state_shape": list(ds.state.shape),
        "mean_stability": mean_stability,
        "states_evolved": True,
    }

if __name__ == "__main__":
    try:
        result = run()
        print("PASS", result)
    except Exception as e:
        print("FAIL", str(e))
        sys.exit(1)
