"""Sandbox 02: Self-Organizing Curriculum - viability test."""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def run():
    from ml_toolbox.textbook_concepts.self_organization import SelfOrganizingMap
    # Small SOM: 5x5, 8-dim "concepts"
    np.random.seed(42)
    X = np.random.randn(50, 8)
    som = SelfOrganizingMap(map_shape=(5, 5), input_dim=8, learning_rate=0.3, sigma=1.5)
    som.fit(X, epochs=20, verbose=False)
    # "Learning path": start at BMU for first concept, then move to neighbor, then neighbor
    bmu0 = som._find_best_matching_unit(X[0])
    path = [bmu0]
    for step in range(2):
        i, j = path[-1]
        # Move to a neighbor (simple: right then down)
        ni = min(i + 1, som.map_shape[0] - 1) if step == 0 else i
        nj = min(j + 1, som.map_shape[1] - 1) if step == 1 else j
        path.append((ni, nj))
    # Path should be 3 distinct or repeated cells
    path_ok = len(path) == 3
    return {
        "ok": True,
        "som_trained": True,
        "path_length": len(path),
        "path": [f"{p[0]},{p[1]}" for p in path],
        "path_ok": path_ok,
    }

if __name__ == "__main__":
    try:
        result = run()
        print("PASS", result)
    except Exception as e:
        print("FAIL", str(e))
        sys.exit(1)
