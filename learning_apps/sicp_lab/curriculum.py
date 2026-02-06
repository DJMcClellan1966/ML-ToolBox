"""
Curriculum: SICP (Structure and Interpretation of Computer Programs) — Abelson, Sussman, Sussman.
From sicp_methods.py at repo root: functional pipeline, streams, data abstraction, symbolic computation.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "functional", "name": "Functional ML Pipeline", "short": "Functional", "color": "#2563eb"},
    {"id": "streams", "name": "Streams", "short": "Streams", "color": "#059669"},
    {"id": "data_abstraction", "name": "Data Abstraction", "short": "Data Abstraction", "color": "#7c3aed"},
    {"id": "symbolic", "name": "Symbolic Computation", "short": "Symbolic", "color": "#dc2626"},
]

CURRICULUM: List[Dict[str, Any]] = [
    {"id": "sicp_map_filter", "book_id": "functional", "level": "basics", "title": "Map, Filter, Reduce",
     "learn": "Higher-order functions: map (transform), filter (select), reduce (aggregate). Foundation of functional ML pipelines.",
     "try_code": "from sicp_methods import FunctionalMLPipeline\nout = FunctionalMLPipeline.map_ml(lambda x: x*2, [1,2,3])\nprint(out)",
     "try_demo": "sicp_map"},
    {"id": "sicp_compose_pipe", "book_id": "functional", "level": "intermediate", "title": "Compose and Pipe",
     "learn": "Compose: (f ∘ g)(x) = f(g(x)). Pipe: pass data through a chain of functions. Enables declarative pipelines.",
     "try_code": "from sicp_methods import FunctionalMLPipeline\nf = FunctionalMLPipeline.compose(lambda x: x+1, lambda x: x*2)\nprint(f(3))",
     "try_demo": "sicp_compose"},
    {"id": "sicp_fold", "book_id": "functional", "level": "intermediate", "title": "Fold Left / Right",
     "learn": "Fold (reduce) with explicit initial value. Left fold: process from left; right fold: from right. Essential for accumulators.",
     "try_code": "from sicp_methods import FunctionalMLPipeline\nprint(FunctionalMLPipeline.fold_left(lambda a,b: a+b, 0, [1,2,3,4]))",
     "try_demo": None},
    {"id": "sicp_stream", "book_id": "streams", "level": "intermediate", "title": "Lazy Streams",
     "learn": "Stream: (first, rest) with lazy computation. Infinite sequences without storing all elements. Stream.from_list, take(n).",
     "try_code": "from sicp_methods import Stream\ns = Stream.integers(0, 1).take(5)\nprint(s)",
     "try_demo": "sicp_stream"},
    {"id": "sicp_stream_map", "book_id": "streams", "level": "advanced", "title": "Stream Map and Filter",
     "learn": "Lazy map and filter on streams. Build pipelines over infinite or large sequences.",
     "try_code": "from sicp_methods import Stream\ns = Stream.range_stream(0, 10).map(lambda x: x*2).take(5)\nprint(s)",
     "try_demo": None},
    {"id": "sicp_pair", "book_id": "data_abstraction", "level": "basics", "title": "Pairs and Lists (cons, car, cdr)",
     "learn": "SICP pairs: cons(first, rest), car(pair), cdr(pair). Build lists from pairs. from_python_list / to_python_list.",
     "try_code": "from sicp_methods import DataAbstraction\np = DataAbstraction.Pair.cons(1, DataAbstraction.Pair.cons(2, None))\nprint(DataAbstraction.Pair.to_python_list(p))",
     "try_demo": "sicp_pair"},
    {"id": "sicp_tree", "book_id": "data_abstraction", "level": "intermediate", "title": "Tree Abstraction",
     "learn": "make_tree(value, left, right), value(tree), left(tree), right(tree). Binary tree as data abstraction.",
     "try_code": "from sicp_methods import DataAbstraction\nt = DataAbstraction.Tree.make_tree(2, DataAbstraction.Tree.make_tree(1, None, None), DataAbstraction.Tree.make_tree(3, None, None))",
     "try_demo": None},
    {"id": "sicp_symbolic", "book_id": "symbolic", "level": "advanced", "title": "Symbolic Expressions",
     "learn": "Expression = (operator, operands). make_expression, evaluate(env). Symbolic computation for expressions.",
     "try_code": "from sicp_methods import SymbolicComputation\ne = SymbolicComputation.Expression.make_expression('+', 1, 2)\nprint(e.evaluate())",
     "try_demo": "sicp_symbolic"},
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
