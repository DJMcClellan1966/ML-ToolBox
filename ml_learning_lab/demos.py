"""
Safe runnable demos for the learning app: Knuth, foundational, textbook concepts.
Each function returns {"ok": bool, "output": str, "error": str | None}.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def knuth_lcg():
    from knuth_algorithms import KnuthRandom
    kr = KnuthRandom(42)
    nums = kr.linear_congruential_generator(10)
    return {"ok": True, "output": str(nums)}


def knuth_shuffle():
    from knuth_algorithms import KnuthRandom
    kr = KnuthRandom(42)
    arr = [1, 2, 3, 4, 5]
    out = kr.fisher_yates_shuffle(arr)
    return {"ok": True, "output": f"Shuffled {arr} -> {out}"}


def knuth_sample():
    from knuth_algorithms import KnuthRandom
    kr = KnuthRandom(42)
    out = kr.random_sample_without_replacement(list(range(20)), 5)
    return {"ok": True, "output": f"Sample of 5 from 0..19: {out}"}


def knuth_heapsort():
    from knuth_algorithms import KnuthSorting
    arr = [3, 1, 4, 1, 5, 9, 2, 6]
    out = KnuthSorting.heapsort(arr)
    return {"ok": True, "output": f"Heapsort {arr} -> {out}"}


def knuth_quicksort():
    from knuth_algorithms import KnuthSorting
    arr = [3, 1, 4, 1, 5, 9, 2, 6]
    out = KnuthSorting.quicksort_median_of_three(arr)
    return {"ok": True, "output": f"Quicksort {arr} -> {out}"}


def knuth_binary_search():
    from knuth_algorithms import KnuthSearching
    arr = [1, 3, 5, 7, 9]
    idx = KnuthSearching.binary_search(arr, 5)
    return {"ok": True, "output": f"binary_search([1,3,5,7,9], 5) -> index {idx}"}


def knuth_combinations():
    from knuth_algorithms import KnuthCombinatorial
    perms = list(KnuthCombinatorial.generate_permutations_lexicographic([1, 2, 3]))
    return {"ok": True, "output": f"Permutations of [1,2,3]: {perms}"}


def bentley_kadane():
    from foundational_algorithms import BentleyAlgorithms
    arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    result = BentleyAlgorithms.maximum_subarray_kadane(arr)
    return {"ok": True, "output": f"max_subarray({arr}) -> {result}"}


def entropy_demo():
    try:
        from ml_toolbox.textbook_concepts.information_theory import Entropy
        import numpy as np
        p = np.array([0.5, 0.5])
        h = Entropy.entropy(p)
        return {"ok": True, "output": f"Entropy([0.5, 0.5]) = {h}"}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def channel_capacity():
    try:
        from ml_toolbox.textbook_concepts.communication_theory import channel_capacity
        c = channel_capacity(10.0, 1.0)
        return {"ok": True, "output": f"channel_capacity(S=10, N=1) = {c}"}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def explain_bias_variance():
    try:
        from ML_Compass.explainers import explain_concept
        out = explain_concept("bias_variance")
        if out.get("ok") and out.get("views"):
            lines = [f"{k}: {v[:80]}..." if len(v) > 80 else f"{k}: {v}" for k, v in out["views"].items()]
            return {"ok": True, "output": "\n".join(lines)}
        return {"ok": True, "output": str(out)}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def error_correcting():
    try:
        from ml_toolbox.textbook_concepts.communication_theory import ErrorCorrectingPredictions
        import numpy as np
        ec = ErrorCorrectingPredictions(redundancy_factor=3)
        preds = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
        corrected = ec.correct_predictions(preds, method="majority_vote")
        return {"ok": True, "output": f"majority_vote on 3 models -> {corrected.tolist()}"}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def skiena_backtrack():
    try:
        from foundational_algorithms import SkienaAlgorithms
        items = [("a", 12, 4), ("b", 10, 3), ("c", 8, 2)]
        def value_func(x):
            return x[1]
        def constraint_func(selected, item):
            total = sum(x[2] for x in selected) + item[2]
            return total <= 6
        def selection_func(remaining):
            return max(remaining, key=value_func) if remaining else None
        result = SkienaAlgorithms.greedy_approximation(items, value_func, constraint_func, selection_func)
        return {"ok": True, "output": f"Greedy (value, weight<=6): {result}"}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def knuth_machine_run(steps: list, params: dict):
    from knuth_algorithms import KnuthRandom
    kr = KnuthRandom(params.get("seed", 42))
    log = []
    data = None
    try:
        if "lcg" in steps:
            n = int(params.get("lcg_n", 10))
            nums = kr.linear_congruential_generator(n)
            data = nums
            log.append(f"LCG({n}): {nums}")
        if "shuffle" in steps:
            arr = data if data is not None else list(range(10))
            data = kr.fisher_yates_shuffle(arr)
            log.append(f"Shuffle: {data}")
        if "sample" in steps:
            pop = data if data is not None else list(range(20))
            k = min(int(params.get("sample_k", 5)), len(pop))
            data = kr.random_sample_without_replacement(pop, k)
            log.append(f"Sample(k={k}): {data}")
        return {"ok": True, "output": "\n".join(log), "steps": steps}
    except Exception as e:
        return {"ok": False, "output": "\n".join(log) if log else "", "error": str(e)}


DEMO_HANDLERS = {
    "knuth_lcg": knuth_lcg,
    "knuth_shuffle": knuth_shuffle,
    "knuth_sample": knuth_sample,
    "knuth_heapsort": knuth_heapsort,
    "knuth_quicksort": knuth_quicksort,
    "knuth_binary_search": knuth_binary_search,
    "knuth_combinations": knuth_combinations,
    "bentley_kadane": bentley_kadane,
    "entropy_demo": entropy_demo,
    "channel_capacity": channel_capacity,
    "explain_bias_variance": explain_bias_variance,
    "error_correcting": error_correcting,
    "skiena_backtrack": skiena_backtrack,
}


def run_demo(demo_id: str) -> dict:
    if demo_id not in DEMO_HANDLERS:
        return {"ok": False, "output": "", "error": f"Unknown demo: {demo_id}"}
    try:
        return DEMO_HANDLERS[demo_id]()
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}
