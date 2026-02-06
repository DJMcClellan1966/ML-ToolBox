"""Demos for Python Practice Lab. Uses reed_zelle_patterns at repo root."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _decomp_ml():
    from reed_zelle_patterns import ProblemDecomposition
    out = ProblemDecomposition.decompose_ml_problem("classify images")
    return {"ok": True, "output": "decompose_ml_problem('classify images'):\n" + str(out)}


def _div_conquer():
    from reed_zelle_patterns import AlgorithmPatterns
    total = AlgorithmPatterns.divide_and_conquer([1, 2, 3, 4, 5], lambda a, b: (a or 0) + (b or 0))
    return {"ok": True, "output": "divide_and_conquer([1,2,3,4,5], sum): " + str(total)}


DEMO_HANDLERS = {"decomp_ml": _decomp_ml, "div_conquer": _div_conquer}


def run_demo(demo_id: str):
    if demo_id in DEMO_HANDLERS:
        try:
            return DEMO_HANDLERS[demo_id]()
        except Exception as e:
            return {"ok": False, "output": "", "error": str(e)}
    return {"ok": False, "output": "", "error": f"No demo: {demo_id}"}
