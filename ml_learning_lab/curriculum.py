"""
Curriculum: books, levels (basics → expert), learn + try-it content.
Maps Knuth, Skiena, Bentley, Sedgewick, textbook_concepts, information/communication theory, etc.
"""
from typing import Dict, Any, List

# Level order: basics → intermediate → advanced → expert
LEVELS = ["basics", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "knuth", "name": "Knuth (TAOCP)", "short": "TAOCP Vol 1–4", "color": "#2563eb"},
    {"id": "skiena_bentley", "name": "Skiena & Bentley", "short": "Algorithm Design & Pearls", "color": "#059669"},
    {"id": "sedgewick", "name": "Sedgewick", "short": "Algorithms 4e", "color": "#7c3aed"},
    {"id": "textbook_concepts", "name": "Textbook Concepts", "short": "Bishop, Goodfellow, Russell & Norvig", "color": "#dc2626"},
    {"id": "information_theory", "name": "Information & Communication", "short": "Shannon, channel capacity", "color": "#ea580c"},
    {"id": "algorithm_design", "name": "Algorithm Design Patterns", "short": "Templates & problem–solution mapping", "color": "#ca8a04"},
]

# Full curriculum: each item has id, book_id, level, title, learn (markdown-like text), try_code (snippet), try_demo (backend demo key)
CURRICULUM: List[Dict[str, Any]] = [
    # ---- KNUTH ----
    {
        "id": "knuth_lcg",
        "book_id": "knuth",
        "level": "basics",
        "title": "Linear Congruential Generator (Vol. 2)",
        "learn": "Knuth's Algorithm A: x_{n+1} = (a*x_n + c) mod m. Fast, reproducible RNG. Used in ML for seeds, bootstrap, cross-validation splits.",
        "try_code": "from knuth_algorithms import KnuthRandom\nkr = KnuthRandom(42)\nnums = kr.linear_congruential_generator(10)\nprint(nums)",
        "try_demo": "knuth_lcg",
    },
    {
        "id": "knuth_fisher_yates",
        "book_id": "knuth",
        "level": "basics",
        "title": "Fisher-Yates Shuffle (Vol. 2)",
        "learn": "Algorithm P: unbiased shuffle in O(n). Correct way to shuffle data for train/test splits. Never use naive 'sort by random'.",
        "try_code": "from knuth_algorithms import KnuthRandom\nkr = KnuthRandom(42)\nprint(kr.fisher_yates_shuffle([1,2,3,4,5]))",
        "try_demo": "knuth_shuffle",
    },
    {
        "id": "knuth_sample",
        "book_id": "knuth",
        "level": "intermediate",
        "title": "Random Sample Without Replacement (Vol. 2)",
        "learn": "Efficient k-from-n sampling using partial Fisher-Yates or reservoir sampling. Essential for bootstrap and subsampling.",
        "try_code": "from knuth_algorithms import KnuthRandom\nkr = KnuthRandom(42)\nprint(kr.random_sample_without_replacement(list(range(20)), 5))",
        "try_demo": "knuth_sample",
    },
    {
        "id": "knuth_heapsort",
        "book_id": "knuth",
        "level": "intermediate",
        "title": "Heapsort (Vol. 3)",
        "learn": "Algorithm H: O(n log n) worst-case, in-place. No extra array. Useful when memory is tight or you need guaranteed complexity.",
        "try_code": "from knuth_algorithms import KnuthSorting\nprint(KnuthSorting.heapsort([3,1,4,1,5,9,2,6]))",
        "try_demo": "knuth_heapsort",
    },
    {
        "id": "knuth_quicksort",
        "book_id": "knuth",
        "level": "intermediate",
        "title": "Quicksort median-of-three (Vol. 3)",
        "learn": "Partition by median of first, middle, last; reduces worst-case. Knuth's analysis: average comparisons ~ 2n ln n.",
        "try_code": "from knuth_algorithms import KnuthSorting\nprint(KnuthSorting.quicksort_median_of_three([3,1,4,1,5,9,2,6]))",
        "try_demo": "knuth_quicksort",
    },
    {
        "id": "knuth_binary_search",
        "book_id": "knuth",
        "level": "basics",
        "title": "Binary Search (Vol. 3)",
        "learn": "Algorithm B: find key in sorted array in O(log n). Foundation for ordered search and many divide-and-conquer algorithms.",
        "try_code": "from knuth_algorithms import KnuthSearching\narr = [1,3,5,7,9]\nprint(KnuthSearching.binary_search(arr, 5))",
        "try_demo": "knuth_binary_search",
    },
    {
        "id": "knuth_combinatorics",
        "book_id": "knuth",
        "level": "advanced",
        "title": "Combinatorial Generation (Vol. 4)",
        "learn": "Lexicographic permutations, combinations, Gray codes. Used in hyperparameter grids, feature subsets, and enumeration.",
        "try_code": "from knuth_algorithms import KnuthCombinatorial\nprint(list(KnuthCombinatorics.lexicographic_permutations([1,2,3]))[:6])",
        "try_demo": "knuth_combinations",
    },
    # ---- SKIENA / BENTLEY ----
    {
        "id": "skiena_backtracking",
        "book_id": "skiena_bentley",
        "level": "intermediate",
        "title": "Backtracking Framework (Skiena)",
        "learn": "Systematic search with prune: try choices, recurse, undo. Feature subset selection, constraint satisfaction.",
        "try_code": "from foundational_algorithms import SkienaAlgorithms\n# N-queens style backtracking demo",
        "try_demo": "skiena_backtrack",
    },
    {
        "id": "bentley_kadane",
        "book_id": "skiena_bentley",
        "level": "intermediate",
        "title": "Maximum Subarray (Bentley)",
        "learn": "Kadane's algorithm: O(n) max contiguous sum. Pattern for 1D dynamic programming and sequence problems.",
        "try_code": "from foundational_algorithms import BentleyAlgorithms\nprint(BentleyAlgorithms.maximum_subarray_kadane([-2,1,-3,4,-1,2,1,-5,4]))",
        "try_demo": "bentley_kadane",
    },
    {
        "id": "algo_greedy",
        "book_id": "algorithm_design",
        "level": "basics",
        "title": "Greedy Template",
        "learn": "Make locally best choice; prove it yields global optimum. Scheduling, Huffman, many ML heuristics.",
        "try_code": "from algorithm_design_patterns import AlgorithmDesignPatterns",
        "try_demo": None,
    },
    {
        "id": "algo_dp",
        "book_id": "algorithm_design",
        "level": "intermediate",
        "title": "Dynamic Programming Template",
        "learn": "Overlapping subproblems + optimal substructure. Memoize or tabulate. Edit distance, knapsack, sequence alignment.",
        "try_code": "from algorithm_design_patterns import AlgorithmDesignPatterns",
        "try_demo": None,
    },
    # ---- SEDGEWICK ----
    {
        "id": "sedgewick_redblack",
        "book_id": "sedgewick",
        "level": "advanced",
        "title": "Red-Black Tree (Sedgewick)",
        "learn": "Self-balancing BST: O(log n) insert/lookup. Ordered maps and sets; useful for sorted feature indices.",
        "try_code": "from foundational_algorithms import SedgewickDataStructures\nrbt = SedgewickDataStructures.RedBlackTree()\nfor k in [3,1,4,1,5]: rbt.insert(k)\n# in-order traversal",
        "try_demo": None,
    },
    # ---- TEXTBOOK CONCEPTS ----
    {
        "id": "info_entropy",
        "book_id": "information_theory",
        "level": "basics",
        "title": "Shannon Entropy",
        "learn": "H(X) = -sum p(x) log p(x). Uncertainty in bits. Information gain in trees, cross-entropy loss in classification.",
        "try_code": "from ml_toolbox.textbook_concepts.information_theory import Entropy\nimport numpy as np\np = np.array([0.5, 0.5])\nprint(Entropy.entropy(p))",
        "try_demo": "entropy_demo",
    },
    {
        "id": "comm_capacity",
        "book_id": "information_theory",
        "level": "intermediate",
        "title": "Channel Capacity (Shannon)",
        "learn": "C = B log2(1 + S/N). Bounds reliable transmission. Theory-as-channel: ensemble redundancy improves reliable decisions.",
        "try_code": "from ml_toolbox.textbook_concepts.communication_theory import channel_capacity\nprint(channel_capacity(10.0, 1.0))",
        "try_demo": "channel_capacity",
    },
    {
        "id": "bishop_bias_variance",
        "book_id": "textbook_concepts",
        "level": "basics",
        "title": "Bias-Variance Tradeoff (Bishop)",
        "learn": "Error = bias + variance + noise. Simple model: high bias; complex model: high variance. Regularization and ensemble balance them.",
        "try_code": "from ML_Compass.explainers import explain_concept\nprint(explain_concept('bias_variance'))",
        "try_demo": "explain_bias_variance",
    },
    {
        "id": "practical_cv",
        "book_id": "textbook_concepts",
        "level": "basics",
        "title": "Cross-Validation (Practical ML)",
        "learn": "Train on k-1 folds, validate on 1; rotate. Unbiased estimate of generalization. Use Pipeline to avoid leakage.",
        "try_code": "from ml_toolbox.textbook_concepts.practical_ml import CrossValidation",
        "try_demo": None,
    },
    {
        "id": "statmech_annealing",
        "book_id": "textbook_concepts",
        "level": "advanced",
        "title": "Simulated Annealing (Statistical Mechanics)",
        "learn": "Optimize by cooling: accept worse moves with probability exp(-dE/T). T schedule matters. Global optimization.",
        "try_code": "from ml_toolbox.textbook_concepts.statistical_mechanics import SimulatedAnnealing",
        "try_demo": None,
    },
    {
        "id": "comm_error_correcting",
        "book_id": "information_theory",
        "level": "advanced",
        "title": "Error-Correcting Predictions (Communication Theory)",
        "learn": "Encode predictions with redundancy; decode by majority or code. Ensemble as error-correcting code.",
        "try_code": "from ml_toolbox.textbook_concepts.communication_theory import ErrorCorrectingPredictions",
        "try_demo": "error_correcting",
    },
]


def get_curriculum() -> List[Dict[str, Any]]:
    return list(CURRICULUM)


def get_books() -> List[Dict[str, Any]]:
    return list(BOOKS)


def get_levels() -> List[str]:
    return list(LEVELS)


def get_by_book(book_id: str) -> List[Dict[str, Any]]:
    return [c for c in CURRICULUM if c["book_id"] == book_id]


def get_by_level(level: str) -> List[Dict[str, Any]]:
    return [c for c in CURRICULUM if c["level"] == level]


def get_item(item_id: str) -> Dict[str, Any] | None:
    for c in CURRICULUM:
        if c["id"] == item_id:
            return c
    return None
