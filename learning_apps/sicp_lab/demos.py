"""Demos for SICP Lab. Uses sicp_methods.py at repo root."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def sicp_map():
    from sicp_methods import FunctionalMLPipeline
    out = FunctionalMLPipeline.map_ml(lambda x: x * 2, [1, 2, 3, 4, 5])
    return {"ok": True, "output": f"map(x*2, [1..5]) = {out}"}


def sicp_compose():
    from sicp_methods import FunctionalMLPipeline
    f = FunctionalMLPipeline.compose(lambda x: x + 1, lambda x: x * 2)
    result = f(3)
    return {"ok": True, "output": f"compose(inc, double)(3) = {result}"}


def sicp_stream():
    from sicp_methods import Stream
    s = Stream.integers(0, 1).take(8)
    return {"ok": True, "output": f"Stream.integers(0,1).take(8) = {s}"}


def sicp_pair():
    from sicp_methods import DataAbstraction
    p = DataAbstraction.Pair.cons(1, DataAbstraction.Pair.cons(2, DataAbstraction.Pair.cons(3, None)))
    lst = DataAbstraction.Pair.to_python_list(p)
    return {"ok": True, "output": f"cons list (1,2,3) -> {lst}"}


def sicp_symbolic():
    from sicp_methods import SymbolicComputation
    e = SymbolicComputation.Expression.make_expression("+", 10, 32)
    val = e.evaluate()
    return {"ok": True, "output": f"(+ 10 32) = {val}"}


DEMO_HANDLERS = {"sicp_map": sicp_map, "sicp_compose": sicp_compose, "sicp_stream": sicp_stream, "sicp_pair": sicp_pair, "sicp_symbolic": sicp_symbolic}


def run_demo(demo_id: str):
    if demo_id not in DEMO_HANDLERS:
        return {"ok": False, "output": "", "error": f"Unknown demo: {demo_id}"}
    try:
        return DEMO_HANDLERS[demo_id]()
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}
