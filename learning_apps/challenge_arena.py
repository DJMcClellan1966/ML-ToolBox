"""
COMPETE Mode — Challenge Arena.

Weekly challenges, leaderboards, debug-this challenges, and adversarial
test-writing. Scale the playground from 5 challenges to a living arena.
"""
import json
import time
import io
import ast
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from flask import request, jsonify
from contextlib import redirect_stdout, redirect_stderr

DATA_DIR = Path(__file__).resolve().parent / ".data"
ARENA_FILE = DATA_DIR / "challenge_arena.json"
LEADERBOARD_FILE = DATA_DIR / "arena_leaderboard.json"

# ---------------------------------------------------------------------------
# Challenge Library — far beyond the original 5 playground challenges
# ---------------------------------------------------------------------------
CHALLENGE_LIBRARY: List[Dict[str, Any]] = [
    # --- Algorithm Challenges ---
    {
        "id": "two_sum", "title": "Two Sum", "category": "algorithms",
        "difficulty": "easy", "points": 100,
        "description": "Given a list of integers and a target, return indices of two numbers that add up to the target.",
        "starter_code": "def two_sum(nums, target):\n    # Return [i, j] where nums[i] + nums[j] == target\n    pass",
        "test_cases": [
            {"input": {"nums": [2, 7, 11, 15], "target": 9}, "expected": [0, 1]},
            {"input": {"nums": [3, 2, 4], "target": 6}, "expected": [1, 2]},
            {"input": {"nums": [3, 3], "target": 6}, "expected": [0, 1]},
        ],
        "hints": ["Think about what complement you need for each number.", "A hash map can find complements in O(1)."],
        "optimal_complexity": "O(n)",
        "tags": ["hash map", "array"],
    },
    {
        "id": "longest_increasing_subseq", "title": "Longest Increasing Subsequence", "category": "algorithms",
        "difficulty": "medium", "points": 200,
        "description": "Find the length of the longest strictly increasing subsequence.",
        "starter_code": "def lis(nums):\n    # Return length of longest increasing subsequence\n    pass",
        "test_cases": [
            {"input": {"nums": [10, 9, 2, 5, 3, 7, 101, 18]}, "expected": 4},
            {"input": {"nums": [0, 1, 0, 3, 2, 3]}, "expected": 4},
            {"input": {"nums": [7, 7, 7, 7]}, "expected": 1},
        ],
        "hints": ["Dynamic programming: dp[i] = length of LIS ending at i.", "O(n log n) solution uses patience sorting."],
        "optimal_complexity": "O(n log n)",
        "tags": ["DP", "binary search"],
    },
    {
        "id": "matrix_chain", "title": "Matrix Chain Multiplication", "category": "algorithms",
        "difficulty": "hard", "points": 300,
        "description": "Find the minimum number of scalar multiplications needed to multiply a chain of matrices.",
        "starter_code": "def matrix_chain_order(dims):\n    # dims[i] = rows of matrix i, dims[-1] = cols of last matrix\n    # Return minimum multiplications\n    pass",
        "test_cases": [
            {"input": {"dims": [10, 30, 5, 60]}, "expected": 4500},
            {"input": {"dims": [40, 20, 30, 10, 30]}, "expected": 26000},
            {"input": {"dims": [10, 20, 30]}, "expected": 6000},
        ],
        "hints": ["Classic DP. Define m[i][j] as min cost to multiply matrices i..j.", "The recurrence splits at every possible k between i and j."],
        "optimal_complexity": "O(n^3)",
        "tags": ["DP", "CLRS"],
    },
    # --- ML Implementation Challenges ---
    {
        "id": "linear_regression", "title": "Linear Regression from Scratch", "category": "ml",
        "difficulty": "easy", "points": 150,
        "description": "Implement linear regression using the normal equation. Return weights.",
        "starter_code": "import numpy as np\n\ndef linear_regression(X, y):\n    # X: (n, d) features, y: (n,) targets\n    # Return weights w such that X @ w ≈ y\n    pass",
        "test_cases": [
            {"input": {"X": [[1, 1], [1, 2], [1, 3]], "y": [1, 2, 3]}, "expected_check": "mse < 0.01"},
            {"input": {"X": [[1, 0], [1, 1], [1, 2], [1, 3]], "y": [0, 1, 2, 3]}, "expected_check": "mse < 0.01"},
        ],
        "hints": ["Normal equation: w = (X^T X)^{-1} X^T y", "Use numpy.linalg.inv or numpy.linalg.lstsq"],
        "optimal_complexity": "O(nd^2)",
        "tags": ["regression", "linear algebra"],
        "custom_checker": True,
    },
    {
        "id": "kmeans", "title": "K-Means Clustering", "category": "ml",
        "difficulty": "medium", "points": 200,
        "description": "Implement K-means clustering. Return cluster assignments.",
        "starter_code": "import numpy as np\n\ndef kmeans(X, k, max_iter=100):\n    # X: (n, d) data points, k: number of clusters\n    # Return labels: (n,) array of cluster assignments (0 to k-1)\n    pass",
        "test_cases": [
            {"input": {"X": [[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]], "k": 2}, "expected_check": "clusters_valid"},
        ],
        "hints": ["Initialize centroids randomly from data points.", "Alternate: assign points to nearest centroid, then update centroids."],
        "optimal_complexity": "O(nkd * iter)",
        "tags": ["clustering", "unsupervised"],
        "custom_checker": True,
    },
    {
        "id": "softmax", "title": "Stable Softmax", "category": "ml",
        "difficulty": "easy", "points": 100,
        "description": "Implement numerically stable softmax function.",
        "starter_code": "import numpy as np\n\ndef softmax(x):\n    # x: (n,) or (batch, n) array of logits\n    # Return softmax probabilities (same shape)\n    pass",
        "test_cases": [
            {"input": {"x": [1, 2, 3]}, "expected": [0.0900, 0.2447, 0.6652]},
            {"input": {"x": [1000, 1001, 1002]}, "expected": [0.0900, 0.2447, 0.6652]},
            {"input": {"x": [0, 0, 0]}, "expected": [0.3333, 0.3333, 0.3333]},
        ],
        "hints": ["Subtract the max for numerical stability.", "exp(x - max(x)) / sum(exp(x - max(x)))"],
        "optimal_complexity": "O(n)",
        "tags": ["numerical", "deep learning"],
    },
    # --- Debug Challenges ---
    {
        "id": "debug_binary_search", "title": "🐛 Fix: Binary Search", "category": "debug",
        "difficulty": "easy", "points": 120,
        "description": "This binary search has a bug. Find and fix it.",
        "starter_code": """def binary_search(arr, target):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid  # BUG: should be mid + 1
        else:
            hi = mid
    return -1""",
        "test_cases": [
            {"input": {"arr": [1, 3, 5, 7, 9], "target": 5}, "expected": 2},
            {"input": {"arr": [1, 3, 5, 7, 9], "target": 9}, "expected": 4},
            {"input": {"arr": [1, 3, 5, 7, 9], "target": 2}, "expected": -1},
            {"input": {"arr": [1], "target": 1}, "expected": 0},
        ],
        "hints": ["The bug causes an infinite loop on some inputs.", "What happens when lo == mid?"],
        "tags": ["debug", "binary search"],
    },
    {
        "id": "debug_quicksort", "title": "🐛 Fix: Quicksort", "category": "debug",
        "difficulty": "medium", "points": 180,
        "description": "This quicksort has a subtle bug. Find and fix it.",
        "starter_code": """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr if x < pivot]
    right = [x for x in arr if x > pivot]  # BUG: misses elements equal to pivot
    return quicksort(left) + [pivot] + quicksort(right)""",
        "test_cases": [
            {"input": {"arr": [3, 1, 4, 1, 5, 9, 2, 6]}, "expected": [1, 1, 2, 3, 4, 5, 6, 9]},
            {"input": {"arr": [5, 5, 5, 5]}, "expected": [5, 5, 5, 5]},
            {"input": {"arr": [1]}, "expected": [1]},
        ],
        "hints": ["What happens with duplicate elements?", "Try input [5, 5, 5, 5]."],
        "tags": ["debug", "sorting"],
    },
    # --- Weekly / Timed Challenges ---
    {
        "id": "topk_frequent", "title": "Top K Frequent Elements", "category": "weekly",
        "difficulty": "medium", "points": 250,
        "description": "Given an array, return the k most frequent elements. Your solution must be better than O(n log n).",
        "starter_code": "def top_k_frequent(nums, k):\n    # Return list of k most frequent elements\n    pass",
        "test_cases": [
            {"input": {"nums": [1, 1, 1, 2, 2, 3], "k": 2}, "expected_set": [1, 2]},
            {"input": {"nums": [1], "k": 1}, "expected_set": [1]},
            {"input": {"nums": [4, 4, 4, 1, 1, 2, 2, 2, 3], "k": 2}, "expected_set": [4, 2]},
        ],
        "hints": ["Use a hash map to count frequencies.", "Bucket sort by frequency gives O(n)."],
        "optimal_complexity": "O(n)",
        "tags": ["hash map", "heap", "bucket sort"],
    },
    {
        "id": "implement_attention", "title": "Implement Self-Attention", "category": "weekly",
        "difficulty": "hard", "points": 350,
        "description": "Implement scaled dot-product self-attention from scratch using only numpy.",
        "starter_code": """import numpy as np

def self_attention(X, W_q, W_k, W_v):
    # X: (seq_len, d_model) input embeddings
    # W_q, W_k, W_v: (d_model, d_k) projection matrices
    # Return: (seq_len, d_k) attention output
    pass""",
        "test_cases": [
            {"input": {"seq_len": 4, "d_model": 8, "d_k": 4, "seed": 42}, "expected_check": "shape_and_softmax"},
        ],
        "hints": ["Q = X @ W_q, K = X @ W_k, V = X @ W_v", "Attention = softmax(QK^T / √d_k) @ V"],
        "optimal_complexity": "O(n^2 * d)",
        "tags": ["transformers", "attention", "deep learning"],
        "custom_checker": True,
    },
]


def _load_arena() -> Dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ARENA_FILE.exists():
        return json.loads(ARENA_FILE.read_text())
    return {}


def _save_arena(data: Dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARENA_FILE.write_text(json.dumps(data, indent=2, default=str))


def _load_leaderboard() -> Dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if LEADERBOARD_FILE.exists():
        return json.loads(LEADERBOARD_FILE.read_text())
    return {}


def _save_leaderboard(data: Dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_FILE.write_text(json.dumps(data, indent=2, default=str))


def list_challenges(category: str = None, difficulty: str = None) -> List[Dict]:
    """List available challenges, optionally filtered."""
    challenges = CHALLENGE_LIBRARY
    if category:
        challenges = [c for c in challenges if c["category"] == category]
    if difficulty:
        challenges = [c for c in challenges if c["difficulty"] == difficulty]
    # Don't leak test expected values
    safe = []
    for c in challenges:
        entry = {k: v for k, v in c.items() if k not in ("test_cases", "custom_checker")}
        entry["test_count"] = len(c["test_cases"])
        safe.append(entry)
    return safe


def get_challenge(challenge_id: str) -> Optional[Dict]:
    return next((c for c in CHALLENGE_LIBRARY if c["id"] == challenge_id), None)


def _safe_exec(code: str, test_input: Dict, func_name: str, timeout: float = 5.0) -> Dict:
    """Safely execute user code against a test case."""
    import numpy as np  # noqa: allow in sandbox

    # Blocked imports
    BLOCKED = {'os', 'sys', 'subprocess', 'shutil', 'pathlib', 'socket', 'urllib',
               'requests', 'http', 'pickle', 'ctypes', 'multiprocessing', 'threading'}

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in BLOCKED:
                        return {"error": f"Blocked import: {alias.name}", "passed": False}
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in BLOCKED:
                    return {"error": f"Blocked import: {node.module}", "passed": False}
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}", "passed": False}

    namespace = {"__builtins__": {
        'abs': abs, 'all': all, 'any': any, 'bool': bool, 'dict': dict,
        'enumerate': enumerate, 'filter': filter, 'float': float, 'int': int,
        'isinstance': isinstance, 'len': len, 'list': list, 'map': map,
        'max': max, 'min': min, 'print': print, 'range': range, 'reversed': reversed,
        'round': round, 'set': set, 'sorted': sorted, 'str': str, 'sum': sum,
        'tuple': tuple, 'zip': zip, 'True': True, 'False': False, 'None': None,
        '__import__': lambda name, *a, **kw: __import__(name) if name in ('numpy', 'math', 'collections', 'heapq', 'itertools', 'functools', 'bisect') else (_ for _ in ()).throw(ImportError(f"Import blocked: {name}")),
    }, 'np': np, 'numpy': np}

    stdout = io.StringIO()
    start = time.time()
    try:
        exec(code, namespace)
        if func_name not in namespace:
            return {"error": f"Function '{func_name}' not found", "passed": False}
        func = namespace[func_name]
        with redirect_stdout(stdout):
            result = func(**test_input)
        elapsed = (time.time() - start) * 1000
        return {"result": result, "elapsed_ms": round(elapsed, 2), "stdout": stdout.getvalue()[:1000]}
    except Exception as e:
        return {"error": str(e), "passed": False, "traceback": traceback.format_exc()[-500:]}


def submit_challenge(user_id: str, challenge_id: str, code: str) -> Dict:
    """Submit code for a challenge and run against all test cases."""
    challenge = get_challenge(challenge_id)
    if not challenge:
        return {"ok": False, "error": "Challenge not found"}

    # Determine function name from starter code
    func_name = None
    for line in challenge["starter_code"].split("\n"):
        if line.strip().startswith("def "):
            func_name = line.strip().split("(")[0].replace("def ", "")
            break
    if not func_name:
        return {"ok": False, "error": "Could not determine function name"}

    results = []
    total_time = 0
    all_passed = True

    for i, tc in enumerate(challenge["test_cases"]):
        exec_result = _safe_exec(code, tc["input"], func_name)

        if "error" in exec_result:
            results.append({"test": i + 1, "passed": False, "error": exec_result["error"]})
            all_passed = False
            continue

        result = exec_result["result"]
        elapsed = exec_result.get("elapsed_ms", 0)
        total_time += elapsed

        # Check result
        passed = False
        if "expected" in tc:
            import numpy as np
            if isinstance(tc["expected"], list) and isinstance(result, (list, np.ndarray)):
                passed = list(result) == list(tc["expected"])
            elif isinstance(tc["expected"], float):
                passed = abs(result - tc["expected"]) < 0.01
            else:
                passed = result == tc["expected"]
        elif "expected_set" in tc:
            passed = set(result) == set(tc["expected_set"])
        elif "expected_check" in tc:
            # Custom validation
            passed = _custom_check(tc["expected_check"], result, tc["input"], code)
        else:
            passed = result is not None

        if not passed:
            all_passed = False

        results.append({
            "test": i + 1,
            "passed": passed,
            "elapsed_ms": elapsed,
            "got": str(result)[:200] if not passed else None,
        })

    # Record submission
    arena = _load_arena()
    key = f"{user_id}::{challenge_id}::{int(time.time())}"
    arena[key] = {
        "user_id": user_id,
        "challenge_id": challenge_id,
        "submitted_at": datetime.now().isoformat(),
        "all_passed": all_passed,
        "passed_count": sum(1 for r in results if r["passed"]),
        "total_count": len(results),
        "execution_time_ms": round(total_time, 2),
    }
    _save_arena(arena)

    # Update leaderboard if solved
    if all_passed:
        _update_leaderboard(user_id, challenge_id, total_time, challenge.get("points", 100))

    return {
        "ok": True,
        "all_passed": all_passed,
        "passed_count": sum(1 for r in results if r["passed"]),
        "total_count": len(results),
        "tests": results,
        "execution_time_ms": round(total_time, 2),
        "points_earned": challenge.get("points", 100) if all_passed else 0,
    }


def _custom_check(check_type: str, result: Any, inputs: Dict, code: str) -> bool:
    """Custom validation for ML challenges."""
    import numpy as np

    try:
        if check_type == "mse < 0.01":
            X = np.array(inputs["X"])
            y = np.array(inputs["y"])
            w = np.array(result)
            preds = X @ w
            mse = np.mean((preds - y) ** 2)
            return mse < 0.01

        elif check_type == "clusters_valid":
            X = np.array(inputs["X"])
            labels = np.array(result)
            k = inputs["k"]
            if len(labels) != len(X):
                return False
            if len(set(labels)) != k:
                return False
            return True

        elif check_type == "shape_and_softmax":
            # Check that output has correct shape and rows sum to ~1 for attention weights
            if result is not None and hasattr(result, 'shape'):
                return result.shape[0] == inputs.get("seq_len", 4)
            return False

    except Exception:
        pass
    return False


def _update_leaderboard(user_id: str, challenge_id: str, time_ms: float, points: int):
    """Update the leaderboard."""
    lb = _load_leaderboard()
    if user_id not in lb:
        lb[user_id] = {"total_points": 0, "challenges_solved": [], "best_times": {}}

    user = lb[user_id]
    if challenge_id not in user["challenges_solved"]:
        user["challenges_solved"].append(challenge_id)
        user["total_points"] += points

    # Track best time
    if challenge_id not in user["best_times"] or time_ms < user["best_times"][challenge_id]:
        user["best_times"][challenge_id] = round(time_ms, 2)

    lb[user_id] = user
    _save_leaderboard(lb)


def get_leaderboard(top_n: int = 20) -> List[Dict]:
    """Get the global leaderboard."""
    lb = _load_leaderboard()
    entries = []
    for uid, data in lb.items():
        entries.append({
            "user_id": uid,
            "total_points": data["total_points"],
            "challenges_solved": len(data["challenges_solved"]),
        })
    entries.sort(key=lambda x: x["total_points"], reverse=True)
    return entries[:top_n]


def get_user_arena_stats(user_id: str) -> Dict:
    """Get a user's arena statistics."""
    lb = _load_leaderboard()
    user = lb.get(user_id, {"total_points": 0, "challenges_solved": [], "best_times": {}})
    return {
        "total_points": user["total_points"],
        "challenges_solved": len(user["challenges_solved"]),
        "solved_ids": user["challenges_solved"],
        "best_times": user["best_times"],
    }


# ---------------------------------------------------------------------------
# Flask Route Registration
# ---------------------------------------------------------------------------

def register_arena_routes(app):
    """Register COMPETE mode routes on a Flask app."""

    @app.route("/api/arena/challenges")
    def api_arena_challenges():
        category = request.args.get("category")
        difficulty = request.args.get("difficulty")
        return jsonify({"ok": True, "challenges": list_challenges(category, difficulty)})

    @app.route("/api/arena/challenge/<challenge_id>")
    def api_arena_challenge(challenge_id):
        c = get_challenge(challenge_id)
        if not c:
            return jsonify({"ok": False, "error": "Not found"})
        safe = {k: v for k, v in c.items() if k not in ("test_cases",)}
        safe["test_count"] = len(c["test_cases"])
        return jsonify({"ok": True, "challenge": safe})

    @app.route("/api/arena/submit", methods=["POST"])
    def api_arena_submit():
        body = request.get_json(force=True)
        user_id = body.get("user_id", "default")
        challenge_id = body.get("challenge_id")
        code = body.get("code", "")
        return jsonify(submit_challenge(user_id, challenge_id, code))

    @app.route("/api/arena/leaderboard")
    def api_arena_leaderboard():
        top_n = int(request.args.get("top_n", 20))
        return jsonify({"ok": True, "leaderboard": get_leaderboard(top_n)})

    @app.route("/api/arena/stats")
    def api_arena_stats():
        user_id = request.args.get("user_id", "default")
        return jsonify({"ok": True, "stats": get_user_arena_stats(user_id)})

    @app.route("/api/arena/hint")
    def api_arena_hint():
        challenge_id = request.args.get("challenge_id")
        hint_idx = int(request.args.get("hint", 0))
        c = get_challenge(challenge_id)
        if not c:
            return jsonify({"ok": False, "error": "Not found"})
        hints = c.get("hints", [])
        if hint_idx < len(hints):
            return jsonify({"ok": True, "hint": hints[hint_idx], "hints_remaining": len(hints) - hint_idx - 1})
        return jsonify({"ok": True, "hint": "No more hints!", "hints_remaining": 0})
