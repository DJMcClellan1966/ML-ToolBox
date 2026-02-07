"""
Intelligent Demo Engine - Beyond Incredible Learning Experience

Transforms basic demos into rich, educational experiences with:
1. PRE-EXPLANATION: What you're about to see, why it matters, real-world applications
2. STEP-BY-STEP: Animated breakdown of algorithm execution with state visualization
3. POST-EXPLANATION: Key takeaways, complexity analysis, when to use
4. RELATED CONCEPTS: Links to corpus, further reading
5. PRACTICE: Follow-up challenges and Socratic questions
6. VISUALIZATION DATA: D3.js-ready data structures for interactive graphics

Uses corpus RAG for rich explanations and LLM for dynamic Q&A.
"""
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import ML_Compass for explanations
try:
    from ML_Compass.explainers import explain_concept
    from ML_Compass.socratic import debate_and_question
    from ML_Compass.theory_corpus import get_corpus
    from ML_Compass.oracle import suggest as oracle_suggest
    COMPASS_AVAILABLE = True
except:
    COMPASS_AVAILABLE = False

# Import LLM integration
try:
    import ollama
    OLLAMA_AVAILABLE = True
except:
    OLLAMA_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False


# =============================================================================
# DEMO KNOWLEDGE BASE - Rich metadata for each demo
# =============================================================================

DEMO_KNOWLEDGE = {
    # CLRS Algorithms Lab
    "clrs_obst": {
        "title": "Optimal Binary Search Tree",
        "category": "Dynamic Programming",
        "book": "CLRS Chapter 15",
        "complexity": {"time": "O(n³)", "space": "O(n²)"},
        "difficulty": "advanced",
        "prerequisites": ["binary_search_tree", "dynamic_programming", "recurrence_relations"],
        
        "pre_explanation": """
## 🎯 What You're About to See

**Optimal Binary Search Tree (OBST)** solves a fascinating problem: given keys with known access frequencies, 
how do we arrange them in a BST to minimize expected search cost?

### 💡 Why This Matters
- **Database indexing**: B-trees in databases use similar principles
- **Compiler optimization**: Symbol tables optimize for frequent lookups
- **Huffman coding**: Related to optimal prefix codes in compression
- **Search engines**: Query optimization based on term frequency

### 🧠 Key Insight
Unlike a balanced BST (which minimizes worst-case), OBST minimizes *expected* cost. 
Frequently accessed keys should be near the root, even if that makes the tree unbalanced!

### 📊 The Dynamic Programming Approach
We build a table where `cost[i][j]` = optimal cost for keys i to j.
The recurrence: `cost[i][j] = min over r { cost[i][r-1] + cost[r+1][j] + sum(freq[i..j]) }`
""",
        
        "algorithm_steps": [
            {"step": 1, "title": "Initialize Base Cases", 
             "description": "Single keys have cost = their frequency. Empty ranges have cost = 0.",
             "state": "cost[i][i] = freq[i] for all i"},
            {"step": 2, "title": "Fill DP Table Diagonally",
             "description": "Consider chains of increasing length. For each chain, try every key as root.",
             "state": "Process length=2, then 3, then ... up to n"},
            {"step": 3, "title": "Find Optimal Root for Each Subproblem",
             "description": "For keys i..j, try each key k as root. Cost = left subtree + right subtree + sum of frequencies.",
             "state": "root[i][j] = argmin_k { cost[i][k-1] + cost[k+1][j] + W(i,j) }"},
            {"step": 4, "title": "Construct Optimal Tree",
             "description": "Backtrack through root table to build the actual tree structure.",
             "state": "root[0][n-1] gives the root of optimal tree"}
        ],
        
        "post_explanation": """
## 📈 What Just Happened

The algorithm found the tree structure that minimizes **expected search cost**, 
weighted by how often each key is accessed.

### ⏱️ Complexity Analysis
- **Time**: O(n³) — For each of O(n²) subproblems, we try O(n) roots
- **Space**: O(n²) — Two 2D tables: cost and root

### 🔍 Key Observations
1. This is **not** the same as building a balanced tree
2. Frequently accessed keys "bubble up" toward the root
3. The DP exhibits **optimal substructure**: optimal trees have optimal subtrees

### 🌍 Real-World Applications
| Application | How OBST Helps |
|------------|----------------|
| Compiler symbol tables | Frequently used variables accessed faster |
| Database indexes | Common queries hit higher levels |
| Spell checkers | Common words found quicker |
| Autocomplete | Popular suggestions prioritized |
""",
        
        "related_concepts": ["dynamic_programming", "binary_search_tree", "huffman_coding", "greedy_algorithms"],
        
        "practice_challenges": [
            {"type": "modify", "prompt": "What if search frequencies change over time? How would you adapt?"},
            {"type": "extend", "prompt": "Extend to handle unsuccessful searches (keys between existing keys)"},
            {"type": "compare", "prompt": "Compare the structure to a balanced AVL tree. When is OBST better?"}
        ],
        
        "visualization_type": "tree",
        "visualization_config": {
            "animateConstruction": True,
            "showCostTable": True,
            "highlightOptimalPath": True
        }
    },
    
    "clrs_lis": {
        "title": "Longest Increasing Subsequence",
        "category": "Dynamic Programming",
        "book": "CLRS Chapter 15",
        "complexity": {"time": "O(n²) naive, O(n log n) optimal", "space": "O(n)"},
        "difficulty": "intermediate",
        "prerequisites": ["arrays", "dynamic_programming", "binary_search"],
        
        "pre_explanation": """
## 🎯 What You're About to See

**Longest Increasing Subsequence (LIS)** finds the longest subsequence where each element 
is strictly greater than the previous. Elements don't need to be consecutive!

### 💡 Why This Matters
- **Version control**: Finding longest common sequence of compatible changes
- **Stock analysis**: Longest streak of price increases (not necessarily consecutive)
- **Genomics**: Comparing DNA sequences for evolutionary patterns
- **Text processing**: Sentence alignment, diff algorithms

### 🧠 Key Insight
For each position i, we ask: "What's the longest increasing subsequence ending at i?"
The answer depends on all previous positions j where arr[j] < arr[i].

### 📊 The Approach
**Naive DP**: For each i, look at all j < i. If arr[j] < arr[i], consider extending.
**Optimal**: Use binary search on the "smallest tail" array for O(n log n)!

### 📝 Example
Array: [10, 22, 9, 33, 21, 50, 41, 60]
LIS: [10, 22, 33, 50, 60] — length 5
Notice: 9, 21, 41 are skipped to maintain increasing order
""",
        
        "algorithm_steps": [
            {"step": 1, "title": "Initialize DP Array",
             "description": "dp[i] = length of LIS ending at index i. Initialize all to 1 (each element is its own subsequence).",
             "state": "dp = [1, 1, 1, 1, 1, 1, 1, 1]"},
            {"step": 2, "title": "Compare Each Pair",
             "description": "For each i, check all j < i. If arr[j] < arr[i], we can extend the subsequence.",
             "state": "If arr[j] < arr[i]: dp[i] = max(dp[i], dp[j] + 1)"},
            {"step": 3, "title": "Track Predecessors",
             "description": "To reconstruct the sequence, remember which j gave the best dp[i].",
             "state": "parent[i] = j that gave max dp[i]"},
            {"step": 4, "title": "Backtrack to Build Sequence",
             "description": "Start from the index with maximum dp value, follow parent pointers.",
             "state": "Result: the actual subsequence, not just length"}
        ],
        
        "post_explanation": """
## 📈 What Just Happened

We found the longest subsequence where each element strictly increases, 
demonstrating a classic DP pattern: **extend from all valid predecessors**.

### ⏱️ Complexity Analysis
- **Naive**: O(n²) time — two nested loops
- **Optimal (patience sorting)**: O(n log n) — binary search on tails array
- **Space**: O(n) for dp array

### 🔍 Key Observations
1. **Subsequence ≠ Subarray**: Elements need not be consecutive!
2. **Multiple valid LIS**: There can be many subsequences of the same max length
3. **DP recurrence**: dp[i] = 1 + max(dp[j] for all j < i where arr[j] < arr[i])

### 🚀 Optimization: Patience Sorting
Instead of O(n²), maintain an array of "smallest tails":
- For each element, binary search to find where it fits
- Either extend the longest pile or replace a larger tail
- This gives O(n log n)!

### 🌍 Applications
- **Git merge**: Finding compatible change sequences
- **Scheduling**: Longest chain of non-overlapping tasks
- **Patience card game**: Origin of the O(n log n) algorithm!
""",
        
        "related_concepts": ["dynamic_programming", "binary_search", "patience_sorting", "longest_common_subsequence"],
        
        "visualization_type": "array_sequence",
        "visualization_config": {
            "highlightIndices": True,
            "animateSelection": True,
            "showDPArray": True
        }
    },
    
    "clrs_coin": {
        "title": "Coin Change Problem",
        "category": "Dynamic Programming",
        "book": "CLRS Chapter 15",
        "complexity": {"time": "O(n × amount)", "space": "O(amount)"},
        "difficulty": "intermediate",
        "prerequisites": ["dynamic_programming", "greedy_algorithms"],
        
        "pre_explanation": """
## 🎯 What You're About to See

**Coin Change** finds the minimum number of coins needed to make a given amount.
A classic example where greedy fails but DP succeeds!

### 💡 Why This Matters
- **ATM machines**: Dispensing minimum bills
- **Making change**: Real cashier problem
- **Resource allocation**: Minimal resources to meet a quota
- **Knapsack variants**: Foundation for many optimization problems

### 🧠 Key Insight: Why Greedy Fails
Greedy (always take the largest coin) fails for some denominations!
- Coins: [1, 3, 4], Amount: 6
- Greedy: 4 + 1 + 1 = 3 coins ❌
- Optimal: 3 + 3 = 2 coins ✅

### 📊 The DP Approach
For each amount from 1 to target:
- Try each coin that doesn't exceed current amount
- dp[amount] = 1 + min(dp[amount - coin] for each valid coin)
""",
        
        "algorithm_steps": [
            {"step": 1, "title": "Initialize DP Array",
             "description": "dp[0] = 0 (zero coins for amount 0). All others start at infinity.",
             "state": "dp = [0, ∞, ∞, ∞, ∞, ∞, ∞]"},
            {"step": 2, "title": "Process Amount 1",
             "description": "Only coin 1 works. dp[1] = dp[0] + 1 = 1",
             "state": "dp = [0, 1, ∞, ∞, ∞, ∞, ∞]"},
            {"step": 3, "title": "Process Each Amount",
             "description": "For each amount, try all coins. Take minimum.",
             "state": "dp[i] = min(dp[i], dp[i-coin] + 1) for each valid coin"},
            {"step": 4, "title": "Track Coin Choices",
             "description": "Remember which coin gave the minimum to reconstruct solution.",
             "state": "For amount 6: we used coin 3 twice"}
        ],
        
        "post_explanation": """
## 📈 What Just Happened

DP found the true minimum (2 coins: 3+3) where greedy would have failed (4+1+1 = 3 coins).

### ⏱️ Complexity Analysis
- **Time**: O(amount × n) where n = number of coin types
- **Space**: O(amount) for the DP array (can be optimized from O(amount × n))

### 🔍 Why Greedy Fails
Greedy works for canonical coin systems (like US: 1, 5, 10, 25).
But arbitrary denominations can fool it! The DP explores ALL combinations.

### 🎯 Two Variants
1. **Min coins** (this demo): Minimize count
2. **Count ways**: How many combinations make the amount?

### 🌍 Real Applications
- **Currency design**: Choosing denominations that make greedy work
- **Postage stamps**: Frobenius number (largest unmakeable amount)
- **Resource scheduling**: Minimum resources for a task
""",
        
        "related_concepts": ["dynamic_programming", "greedy_algorithms", "knapsack_problem"],
        
        "visualization_type": "dp_table",
        "visualization_config": {
            "showBacktrack": True,
            "highlightOptimalPath": True
        }
    },
    
    # Deep Learning concepts
    "backprop": {
        "title": "Backpropagation",
        "category": "Deep Learning",
        "book": "Deep Learning (Goodfellow) Chapter 6",
        "complexity": {"time": "O(n) per layer", "space": "O(n) for gradients"},
        "difficulty": "intermediate",
        "prerequisites": ["chain_rule", "gradient_descent", "neural_networks"],
        
        "pre_explanation": """
## 🎯 What You're About to See

**Backpropagation** is the algorithm that makes deep learning possible.
It efficiently computes gradients for all weights in a neural network.

### 💡 Why This Matters
- **Every neural network** uses backprop for training
- **Automatic differentiation** frameworks (PyTorch, TensorFlow) implement this
- **Understanding backprop** demystifies "how neural networks learn"

### 🧠 Key Insight: The Chain Rule
A neural network is a composition of functions: f(g(h(x))).
The chain rule lets us compute ∂loss/∂w for any weight w by multiplying local gradients.

### 📊 Forward then Backward
1. **Forward pass**: Compute activations layer by layer
2. **Backward pass**: Compute gradients from output to input

The magic: each layer only needs its local gradient and the gradient from above!
""",
        
        "algorithm_steps": [
            {"step": 1, "title": "Forward Pass",
             "description": "Compute each layer's output, storing activations for later.",
             "state": "a[l] = σ(W[l] · a[l-1] + b[l])"},
            {"step": 2, "title": "Compute Output Loss",
             "description": "Compare prediction to target. Compute loss gradient.",
             "state": "∂L/∂a[L] = prediction - target (for MSE)"},
            {"step": 3, "title": "Backward Through Each Layer",
             "description": "Apply chain rule: multiply incoming gradient by local gradient.",
             "state": "∂L/∂W[l] = ∂L/∂a[l] · ∂a[l]/∂W[l]"},
            {"step": 4, "title": "Update Weights",
             "description": "Apply gradient descent using computed gradients.",
             "state": "W[l] -= learning_rate × ∂L/∂W[l]"}
        ],
        
        "post_explanation": """
## 📈 What Just Happened

Backprop computed exact gradients for every weight in O(n) time,
where n is the number of weights. This is the same order as a forward pass!

### ⏱️ Complexity Analysis
- **Time**: O(n) per sample — remarkably efficient!
- **Space**: O(n) — must store activations for backward pass

### 🔍 Key Observations
1. **Not gradient descent**: Backprop computes gradients; GD uses them
2. **Differentiable everything**: Only works for differentiable operations
3. **Vanishing/exploding gradients**: Deep networks multiply many terms

### 🧮 The Math in One Line
∂L/∂W[l] = ∂L/∂a[L] × ∂a[L]/∂a[L-1] × ... × ∂a[l+1]/∂a[l] × ∂a[l]/∂W[l]
""",
        
        "related_concepts": ["chain_rule", "gradient_descent", "computational_graph", "autodiff"],
        
        "visualization_type": "neural_network",
        "visualization_config": {
            "showGradientFlow": True,
            "animateBackprop": True,
            "highlightActivations": True
        }
    },
    
    # Reinforcement Learning
    "rl_qlearning": {
        "title": "Q-Learning",
        "category": "Reinforcement Learning",
        "book": "Sutton & Barto Chapter 6",
        "complexity": {"time": "O(episodes × steps)", "space": "O(S × A)"},
        "difficulty": "intermediate",
        "prerequisites": ["markov_decision_process", "bellman_equation", "temporal_difference"],
        
        "pre_explanation": """
## 🎯 What You're About to See

**Q-Learning** is the foundation of modern RL. It learns the value of taking 
each action in each state — without needing a model of the environment!

### 💡 Why This Matters
- **Model-free**: Learns from experience, not a perfect world model
- **Off-policy**: Can learn optimal policy while following exploratory policy  
- **Foundation for DQN**: Deep Q-Networks extend this to neural networks

### 🧠 Key Insight: The Q-Function
Q(s, a) = expected total reward starting from state s, taking action a, then acting optimally.
If we know Q, the optimal policy is: pick the action with highest Q!

### 📊 The Update Rule
After taking action a in state s, getting reward r, and landing in s':
```
Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]
```
This is the **Bellman error** — the difference between expected and actual.
""",
        
        "algorithm_steps": [
            {"step": 1, "title": "Initialize Q-Table",
             "description": "Start with Q(s,a) = 0 for all state-action pairs.",
             "state": "Q = zeros(n_states, n_actions)"},
            {"step": 2, "title": "Explore with ε-Greedy",
             "description": "With probability ε, take random action. Otherwise, take argmax_a Q(s, a).",
             "state": "Balances exploration and exploitation"},
            {"step": 3, "title": "Observe Transition",
             "description": "Take action a, observe reward r and next state s'.",
             "state": "(s, a, r, s', done)"},
            {"step": 4, "title": "Update Q-Value",
             "description": "Apply TD update: Q(s,a) += α(r + γ max Q(s') - Q(s,a)).",
             "state": "TD error = r + γ max_a' Q(s',a') - Q(s,a)"},
            {"step": 5, "title": "Repeat Until Convergence",
             "description": "Q-values converge to optimal Q* with enough exploration.",
             "state": "Optimal policy: π*(s) = argmax_a Q*(s,a)"}
        ],
        
        "post_explanation": """
## 📈 What Just Happened

Q-Learning discovered the optimal action for each state by learning from experience.
Notice how Q-values increase as states get closer to the goal.

### ⏱️ Complexity Analysis
- **Space**: O(|S| × |A|) — one entry per state-action pair
- **Time**: Depends on exploration; tabular Q converges asymptotically

### 🔍 Key Observations
1. **Off-policy**: We update toward max Q(s'), regardless of what action we actually took
2. **Exploration matters**: Without enough ε, we might miss the optimal path
3. **Discount factor γ**: Balances immediate vs future rewards

### 🚀 Extensions
- **DQN**: Replace Q-table with neural network for large state spaces
- **Double Q-Learning**: Reduces overestimation bias
- **Dueling DQN**: Separate value and advantage streams

### 🌍 Applications
- Game playing (Atari, Go, Chess)
- Robot control
- Recommendation systems
- Resource management
""",
        
        "related_concepts": ["bellman_equation", "temporal_difference", "epsilon_greedy", "deep_q_network"],
        
        "practice_challenges": [
            {"type": "modify", "prompt": "What happens if you set γ = 0? How does the policy change?"},
            {"type": "extend", "prompt": "Implement Double Q-Learning to reduce overestimation."},
            {"type": "analyze", "prompt": "Why does the agent sometimes take suboptimal paths during training?"}
        ],
        
        "visualization_type": "grid_world",
        "visualization_config": {
            "showQValues": True,
            "animateAgent": True,
            "showPolicy": True
        }
    },
    
    "rl_value_iteration": {
        "title": "Value Iteration",
        "category": "Reinforcement Learning",
        "book": "Sutton & Barto Chapter 4",
        "complexity": {"time": "O(|S|² × |A|) per iteration", "space": "O(|S|)"},
        "difficulty": "intermediate",
        "prerequisites": ["markov_decision_process", "bellman_equation"],
        
        "pre_explanation": """
## 🎯 What You're About to See

**Value Iteration** computes the optimal value function directly,
then extracts the optimal policy from it.

### 💡 Why This Matters
- **Guaranteed optimal**: Converges to V* for finite MDPs
- **Dynamic programming**: Uses Bellman optimality equation
- **Foundation for policy iteration**: Shows the core idea

### 🧠 Key Insight: Bellman Optimality
V*(s) = max_a Σ_s' P(s'|s,a) [R(s,a,s') + γ V*(s')]

In words: the optimal value of a state is the best action's expected return.

### 📊 The Algorithm
1. Initialize V(s) = 0 for all states
2. Repeat until converged:
   - For each state, update V(s) = max_a [R + γ Σ V(s')]
3. Extract policy: π(s) = argmax_a [expected value]
""",
        
        "algorithm_steps": [
            {"step": 1, "title": "Initialize Values",
             "description": "Set V(s) = 0 for all states (or terminal state values).",
             "state": "V = [0, 0, 0, 0]"},
            {"step": 2, "title": "Bellman Backup",
             "description": "For each state, compute max over actions of expected return.",
             "state": "V(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V(s')]"},
            {"step": 3, "title": "Check Convergence",
             "description": "If max change < threshold, stop. Otherwise repeat.",
             "state": "δ = max|V_new - V_old|"},
            {"step": 4, "title": "Extract Policy",
             "description": "For each state, pick action with highest expected value.",
             "state": "π(s) = argmax_a [R + γ Σ P(s'|s,a) V(s')]"}
        ],
        
        "post_explanation": """
## 📈 What Just Happened

Value iteration found the optimal value for each state.
States closer to the goal have higher values.

### ⏱️ Complexity Analysis
- **Per iteration**: O(|S|² × |A|) — for each state, evaluate each action
- **Iterations**: Typically O(1/(1-γ) log(1/ε)) for ε-convergence

### 🔍 Key Observations
1. **Model-based**: Requires knowing P(s'|s,a) and R
2. **Synchronous updates**: Update all states each iteration
3. **Contraction**: Bellman backup is a γ-contraction → guaranteed convergence

### 🆚 vs Policy Iteration
- Value iteration: Many small improvements to V
- Policy iteration: Full policy evaluation, then improve

### 🌍 Real Applications
- Robotics path planning
- Inventory management
- Network routing optimization
""",
        
        "related_concepts": ["bellman_equation", "policy_iteration", "dynamic_programming", "contraction_mapping"],
        
        "visualization_type": "value_heatmap",
        "visualization_config": {
            "showIteration": True,
            "animateConvergence": True
        }
    },
    
    # Support Vector Machines
    "esl_svm": {
        "title": "Support Vector Machine",
        "category": "Machine Learning",
        "book": "ESL Chapter 12",
        "complexity": {"time": "O(n² to n³)", "space": "O(n²)"},
        "difficulty": "intermediate",
        "prerequisites": ["linear_classification", "margin", "kernel_trick"],
        
        "pre_explanation": """
## 🎯 What You're About to See

**Support Vector Machine (SVM)** finds the hyperplane that maximizes 
the margin between classes — the "widest street" separating the data.

### 💡 Why This Matters
- **Theoretically principled**: Maximizing margin minimizes generalization error
- **Kernel trick**: Handle non-linear boundaries in high-dimensional space
- **Sparse solution**: Only support vectors matter

### 🧠 Key Insight: Maximum Margin
Among all hyperplanes that separate the classes, SVM picks the one 
with the largest distance to the nearest points (support vectors).

Margin = 2 / ||w|| where w·x + b = 0 is the hyperplane.

### 📊 The Optimization
Minimize ||w||² subject to y_i(w·x_i + b) ≥ 1 for all i.

Dual form introduces kernel K(x_i, x_j) = φ(x_i)·φ(x_j) — no explicit feature mapping!
""",
        
        "algorithm_steps": [
            {"step": 1, "title": "Choose Kernel",
             "description": "Linear (dot product), RBF (Gaussian), Polynomial, etc.",
             "state": "K(x_i, x_j) = exp(-γ||x_i - x_j||²) for RBF"},
            {"step": 2, "title": "Solve Dual QP",
             "description": "Find Lagrange multipliers α_i that maximize the dual objective.",
             "state": "max Σα_i - ½ΣΣ α_i α_j y_i y_j K(x_i, x_j)"},
            {"step": 3, "title": "Identify Support Vectors",
             "description": "Points with α_i > 0 are support vectors — they define the boundary.",
             "state": "Typically a small fraction of training points"},
            {"step": 4, "title": "Compute Decision Function",
             "description": "Classify new points using support vectors and kernel.",
             "state": "f(x) = sign(Σ α_i y_i K(x_i, x) + b)"}
        ],
        
        "post_explanation": """
## 📈 What Just Happened

SVM found the maximum-margin hyperplane (in kernel space) that separates the classes.

### ⏱️ Complexity Analysis
- **Training**: O(n² to n³) depending on solver (SMO is efficient)
- **Prediction**: O(n_sv × d) where n_sv = number of support vectors

### 🔍 Key Observations
1. **Soft margin (C)**: Allows misclassifications; C controls trade-off
2. **RBF kernel**: Creates smooth, flexible decision boundaries
3. **Sparsity**: Only support vectors are needed for prediction

### 🎛️ Hyperparameters
- **C**: Regularization — high C = fit training data harder
- **γ (RBF)**: Inverse bandwidth — high γ = tighter fit around points
- **Kernel choice**: Linear for high-D sparse, RBF for general nonlinear

### 🌍 Applications
- Text classification (with linear kernel)
- Image recognition
- Bioinformatics (protein classification)
- Anomaly detection (one-class SVM)
""",
        
        "related_concepts": ["kernel_methods", "margin_maximization", "quadratic_programming", "regularization"],
        
        "practice_challenges": [
            {"type": "modify", "prompt": "Try different values of C. What happens with very high C?"},
            {"type": "extend", "prompt": "Compare linear vs RBF kernel on this data."},
            {"type": "analyze", "prompt": "Which points are support vectors? What makes them special?"}
        ],
        
        "visualization_type": "decision_boundary",
        "visualization_config": {
            "showSupportVectors": True,
            "showMargin": True,
            "animateKernel": False
        }
    },
}


# =============================================================================
# STEP-BY-STEP EXECUTION ENGINE
# =============================================================================

class StepByStepEngine:
    """
    Executes algorithms step-by-step with state snapshots for visualization.
    """
    
    @staticmethod
    def trace_optimal_bst(keys: List[str], freq: List[float]) -> List[Dict]:
        """Trace OBST algorithm with intermediate states."""
        n = len(keys)
        cost = [[0] * n for _ in range(n)]
        root = [[0] * n for _ in range(n)]
        
        steps = []
        
        # Base case
        for i in range(n):
            cost[i][i] = freq[i]
            root[i][i] = i
        steps.append({
            "step": 1,
            "description": "Initialize single-key costs",
            "cost_table": [row[:] for row in cost],
            "root_table": [row[:] for row in root],
            "highlight": [(i, i) for i in range(n)]
        })
        
        # Fill for increasing lengths
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                cost[i][j] = float('inf')
                total_freq = sum(freq[i:j+1])
                
                for k in range(i, j + 1):
                    left = cost[i][k-1] if k > i else 0
                    right = cost[k+1][j] if k < j else 0
                    c = left + right + total_freq
                    if c < cost[i][j]:
                        cost[i][j] = c
                        root[i][j] = k
                
                steps.append({
                    "step": len(steps) + 1,
                    "description": f"Compute cost[{i}][{j}] for keys {keys[i]}..{keys[j]}",
                    "cost_table": [row[:] for row in cost],
                    "root_table": [row[:] for row in root],
                    "highlight": [(i, j)],
                    "optimal_root": keys[root[i][j]],
                    "total_freq": total_freq
                })
        
        return steps
    
    @staticmethod
    def trace_lis(arr: List[int]) -> List[Dict]:
        """Trace LIS algorithm with intermediate states."""
        n = len(arr)
        dp = [1] * n
        parent = [-1] * n
        
        steps = []
        steps.append({
            "step": 1,
            "description": "Initialize all dp values to 1",
            "array": arr,
            "dp": dp[:],
            "parent": parent[:],
            "current_lis": []
        })
        
        for i in range(1, n):
            for j in range(i):
                if arr[j] < arr[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
            
            # Reconstruct current best
            max_idx = dp.index(max(dp))
            current_lis = []
            idx = max_idx
            while idx != -1:
                current_lis.append(arr[idx])
                idx = parent[idx]
            current_lis.reverse()
            
            steps.append({
                "step": i + 1,
                "description": f"Process element {arr[i]} at index {i}",
                "array": arr,
                "dp": dp[:],
                "parent": parent[:],
                "current_index": i,
                "current_lis": current_lis,
                "comparing_with": [(j, arr[j]) for j in range(i) if arr[j] < arr[i]]
            })
        
        return steps
    
    @staticmethod
    def trace_coin_change(coins: List[int], amount: int) -> List[Dict]:
        """Trace coin change algorithm with intermediate states."""
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        parent = [-1] * (amount + 1)
        
        steps = []
        steps.append({
            "step": 1,
            "description": "Initialize dp[0] = 0, others = infinity",
            "dp": [0 if x == 0 else "∞" for x in range(amount + 1)],
            "coins_used": []
        })
        
        for amt in range(1, amount + 1):
            for coin in coins:
                if coin <= amt and dp[amt - coin] + 1 < dp[amt]:
                    dp[amt] = dp[amt - coin] + 1
                    parent[amt] = coin
            
            # Reconstruct coins used
            coins_used = []
            a = amt
            while a > 0 and parent[a] != -1:
                coins_used.append(parent[a])
                a -= parent[a]
            
            steps.append({
                "step": amt + 1,
                "description": f"Compute minimum coins for amount {amt}",
                "dp": [x if x != float('inf') else "∞" for x in dp[:amt+1]],
                "current_amount": amt,
                "coins_tried": [c for c in coins if c <= amt],
                "coins_used": coins_used,
                "min_coins": dp[amt] if dp[amt] != float('inf') else "impossible"
            })
        
        return steps
    
    @staticmethod
    def trace_qlearning(n_states: int = 6, n_actions: int = 2, 
                        n_episodes: int = 10, gamma: float = 0.9,
                        alpha: float = 0.1, epsilon: float = 0.1) -> List[Dict]:
        """Trace Q-Learning algorithm with intermediate states."""
        import random
        
        # Simple grid world: states 0-5, goal at 5
        Q = [[0.0] * n_actions for _ in range(n_states)]
        
        # Simple transition: action 0 = left, action 1 = right
        def step(state, action):
            if action == 1:  # right
                next_state = min(state + 1, n_states - 1)
            else:  # left
                next_state = max(state - 1, 0)
            reward = 10.0 if next_state == n_states - 1 else -0.1
            done = next_state == n_states - 1
            return next_state, reward, done
        
        steps = []
        steps.append({
            "step": 1,
            "description": "Initialize Q-table to zeros",
            "Q": [row[:] for row in Q],
            "episode": 0,
            "state": None,
            "action": None,
            "reward": None
        })
        
        total_step = 2
        for episode in range(n_episodes):
            state = 0
            for _ in range(20):  # max steps per episode
                # Epsilon-greedy
                if random.random() < epsilon:
                    action = random.randint(0, n_actions - 1)
                else:
                    action = Q[state].index(max(Q[state]))
                
                next_state, reward, done = step(state, action)
                
                # Q-update
                td_target = reward + gamma * max(Q[next_state]) if not done else reward
                td_error = td_target - Q[state][action]
                Q[state][action] += alpha * td_error
                
                steps.append({
                    "step": total_step,
                    "description": f"Episode {episode+1}: s={state}, a={'right' if action else 'left'}, r={reward:.1f}, s'={next_state}",
                    "Q": [row[:] for row in Q],
                    "episode": episode + 1,
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                    "reward": reward,
                    "td_error": round(td_error, 3),
                    "done": done
                })
                total_step += 1
                
                if done:
                    break
                state = next_state
        
        return steps
    
    @staticmethod
    def trace_value_iteration(n_states: int = 4, gamma: float = 0.9,
                               threshold: float = 0.01) -> List[Dict]:
        """Trace Value Iteration algorithm."""
        # Simple MDP: states 0-3, actions left/right
        # Goal at state 3 with reward 1
        V = [0.0] * n_states
        
        # Transition probabilities and rewards
        def get_next(s, a):
            if a == 1:  # right
                s_next = min(s + 1, n_states - 1)
            else:  # left
                s_next = max(s - 1, 0)
            r = 1.0 if s_next == n_states - 1 else 0.0
            return s_next, r
        
        steps = []
        steps.append({
            "step": 1,
            "description": "Initialize V(s) = 0 for all states",
            "V": V[:],
            "iteration": 0,
            "delta": None,
            "policy": ["-"] * n_states
        })
        
        iteration = 0
        while True:
            iteration += 1
            delta = 0
            V_new = V[:]
            policy = []
            
            for s in range(n_states):
                if s == n_states - 1:  # terminal
                    policy.append("goal")
                    continue
                
                # Try all actions
                values = []
                for a in range(2):
                    s_next, r = get_next(s, a)
                    values.append(r + gamma * V[s_next])
                
                V_new[s] = max(values)
                policy.append("→" if values[1] > values[0] else "←")
                delta = max(delta, abs(V_new[s] - V[s]))
            
            V = V_new
            
            steps.append({
                "step": iteration + 1,
                "description": f"Iteration {iteration}: Bellman backup for all states",
                "V": V[:],
                "iteration": iteration,
                "delta": round(delta, 4),
                "policy": policy,
                "converged": delta < threshold
            })
            
            if delta < threshold:
                break
        
        return steps


# =============================================================================
# D3.JS VISUALIZATION GENERATORS
# =============================================================================

class VisualizationGenerator:
    """Generate D3.js-ready data structures for interactive visualizations."""
    
    @staticmethod
    def generate_tree_data(keys: List[str], root_table: List[List[int]], 
                           i: int = 0, j: int = None) -> Optional[Dict]:
        """Convert OBST root table to D3 tree structure."""
        if j is None:
            j = len(keys) - 1
        if i > j:
            return None
        
        r = root_table[i][j]
        return {
            "name": keys[r],
            "children": [
                child for child in [
                    VisualizationGenerator.generate_tree_data(keys, root_table, i, r-1),
                    VisualizationGenerator.generate_tree_data(keys, root_table, r+1, j)
                ] if child is not None
            ]
        }
    
    @staticmethod
    def generate_array_viz(arr: List, highlights: List[int] = None,
                          labels: Dict[int, str] = None) -> Dict:
        """Generate array visualization data."""
        return {
            "type": "array",
            "data": [
                {
                    "index": i,
                    "value": v,
                    "highlight": i in (highlights or []),
                    "label": (labels or {}).get(i, "")
                }
                for i, v in enumerate(arr)
            ]
        }
    
    @staticmethod
    def generate_dp_table(table: List[List], row_labels: List = None,
                         col_labels: List = None, highlights: List[tuple] = None) -> Dict:
        """Generate 2D DP table visualization data."""
        return {
            "type": "dp_table",
            "rows": len(table),
            "cols": len(table[0]) if table else 0,
            "row_labels": row_labels,
            "col_labels": col_labels,
            "data": [
                {
                    "row": i,
                    "col": j,
                    "value": table[i][j],
                    "highlight": (i, j) in (highlights or [])
                }
                for i in range(len(table))
                for j in range(len(table[i]))
            ]
        }
    
    @staticmethod
    def generate_neural_network(layers: List[int], activations: List[List[float]] = None,
                               gradients: List[List[float]] = None) -> Dict:
        """Generate neural network visualization data."""
        nodes = []
        links = []
        node_id = 0
        
        for layer_idx, size in enumerate(layers):
            for neuron_idx in range(size):
                nodes.append({
                    "id": node_id,
                    "layer": layer_idx,
                    "neuron": neuron_idx,
                    "activation": activations[layer_idx][neuron_idx] if activations else 0,
                    "gradient": gradients[layer_idx][neuron_idx] if gradients else 0
                })
                
                # Connect to previous layer
                if layer_idx > 0:
                    prev_start = sum(layers[:layer_idx-1])
                    for prev_neuron in range(layers[layer_idx-1]):
                        links.append({
                            "source": prev_start + prev_neuron,
                            "target": node_id
                        })
                
                node_id += 1
        
        return {"type": "neural_network", "nodes": nodes, "links": links}


# =============================================================================
# LLM-POWERED EXPLANATIONS
# =============================================================================

class LLMExplainer:
    """Generate dynamic explanations using LLM."""
    
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.available = OLLAMA_AVAILABLE or OPENAI_AVAILABLE
    
    def explain(self, demo_id: str, context: Dict, question: str = None) -> str:
        """Generate an explanation for the demo result."""
        if not self.available:
            return self._fallback_explain(demo_id, context)
        
        knowledge = DEMO_KNOWLEDGE.get(demo_id, {})
        prompt = f"""You are an expert computer science tutor. 
        
The student just ran a demo of: {knowledge.get('title', demo_id)}
Category: {knowledge.get('category', 'Algorithms')}
Book: {knowledge.get('book', 'Unknown')}

Demo output: {context.get('output', '')}

{f'Student question: {question}' if question else 'Provide a clear, educational explanation of what happened and why.'}

Be concise but insightful. Use analogies when helpful. Highlight the key insight."""
        
        try:
            if OLLAMA_AVAILABLE:
                response = ollama.chat(model=self.model, messages=[
                    {"role": "system", "content": "You are a helpful CS tutor. Be concise and educational."},
                    {"role": "user", "content": prompt}
                ])
                return response['message']['content']
        except:
            pass
        
        return self._fallback_explain(demo_id, context)
    
    def _fallback_explain(self, demo_id: str, context: Dict) -> str:
        """Template-based fallback when LLM unavailable."""
        knowledge = DEMO_KNOWLEDGE.get(demo_id, {})
        return knowledge.get('post_explanation', 'Demo completed. See the output above.')
    
    def generate_socratic_question(self, demo_id: str, context: Dict) -> str:
        """Generate a thought-provoking follow-up question."""
        if COMPASS_AVAILABLE:
            try:
                knowledge = DEMO_KNOWLEDGE.get(demo_id, {})
                result = debate_and_question(f"I just learned about {knowledge.get('title', demo_id)}")
                return result.get('question', '')
            except:
                pass
        
        # Fallback questions
        knowledge = DEMO_KNOWLEDGE.get(demo_id, {})
        challenges = knowledge.get('practice_challenges', [])
        if challenges:
            return challenges[0].get('prompt', 'What would happen if the input changed?')
        return "What would happen if you modified the input?"


# =============================================================================
# CORPUS RAG INTEGRATION
# =============================================================================

class CorpusIntegration:
    """Deep integration with the knowledge corpus via RAG."""
    
    def __init__(self):
        self.corpus_available = COMPASS_AVAILABLE
        self.corpus = []
        if COMPASS_AVAILABLE:
            try:
                self.corpus = get_corpus()
            except:
                pass
    
    def get_related_concepts(self, demo_id: str) -> List[Dict]:
        """Retrieve related concepts from corpus."""
        knowledge = DEMO_KNOWLEDGE.get(demo_id, {})
        related = knowledge.get('related_concepts', [])
        
        results = []
        for concept in related:
            if COMPASS_AVAILABLE:
                try:
                    explanation = explain_concept(concept)
                    if explanation.get('ok'):
                        results.append({
                            "concept": concept,
                            "views": explanation.get('views', {}),
                            "available_views": list(explanation.get('views', {}).keys())
                        })
                except:
                    results.append({"concept": concept, "views": {}, "available_views": []})
            else:
                results.append({"concept": concept, "views": {}, "available_views": []})
        
        return results
    
    def search_corpus(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search corpus for relevant chunks."""
        if not self.corpus:
            return []
        
        # Simple keyword search (quantum_kernel would do semantic)
        results = []
        query_lower = query.lower()
        for chunk in self.corpus:
            content = chunk.get('content', '').lower()
            if any(word in content for word in query_lower.split()):
                results.append(chunk)
                if len(results) >= top_k:
                    break
        
        return results


# =============================================================================
# MAIN INTELLIGENT DEMO FUNCTION
# =============================================================================

def run_intelligent_demo(demo_id: str, params: Dict = None) -> Dict:
    """
    Run a demo with full intelligent context:
    - Pre-explanation
    - Step-by-step execution with state
    - Visualization data
    - Post-explanation
    - Related concepts from corpus
    - Practice challenges
    - Socratic follow-up question
    """
    knowledge = DEMO_KNOWLEDGE.get(demo_id, {})
    
    if not knowledge:
        return {
            "ok": False,
            "error": f"No knowledge base entry for demo: {demo_id}",
            "suggestion": "Run the basic demo or add this demo to DEMO_KNOWLEDGE"
        }
    
    result = {
        "ok": True,
        "demo_id": demo_id,
        "title": knowledge.get("title", demo_id),
        "category": knowledge.get("category", ""),
        "book": knowledge.get("book", ""),
        "difficulty": knowledge.get("difficulty", "intermediate"),
        "complexity": knowledge.get("complexity", {}),
        "prerequisites": knowledge.get("prerequisites", []),
        
        # Educational content
        "pre_explanation": knowledge.get("pre_explanation", ""),
        "algorithm_steps": knowledge.get("algorithm_steps", []),
        "post_explanation": knowledge.get("post_explanation", ""),
        
        # Execution trace (to be filled by running demo)
        "execution_trace": [],
        "demo_output": "",
        
        # Visualization
        "visualization": {
            "type": knowledge.get("visualization_type", "none"),
            "config": knowledge.get("visualization_config", {}),
            "data": None
        },
        
        # Related learning
        "related_concepts": [],
        "practice_challenges": knowledge.get("practice_challenges", []),
        "socratic_question": "",
        
        # Metadata
        "timestamp": datetime.now().isoformat()
    }
    
    # Run the actual demo and get trace
    try:
        if demo_id == "clrs_obst":
            keys = params.get("keys", ["A", "B", "C"]) if params else ["A", "B", "C"]
            freq = params.get("freq", [0.5, 0.3, 0.2]) if params else [0.5, 0.3, 0.2]
            result["execution_trace"] = StepByStepEngine.trace_optimal_bst(keys, freq)
            final = result["execution_trace"][-1] if result["execution_trace"] else {}
            result["demo_output"] = f"Optimal BST cost = {final.get('cost_table', [[]])[0][-1] if final.get('cost_table') else 'N/A'}"
            
            # Generate tree visualization
            if final.get('root_table'):
                result["visualization"]["data"] = VisualizationGenerator.generate_tree_data(
                    keys, final['root_table']
                )
        
        elif demo_id == "clrs_lis":
            arr = params.get("arr", [10, 22, 9, 33, 21, 50, 41, 60]) if params else [10, 22, 9, 33, 21, 50, 41, 60]
            result["execution_trace"] = StepByStepEngine.trace_lis(arr)
            final = result["execution_trace"][-1] if result["execution_trace"] else {}
            result["demo_output"] = f"LIS = {final.get('current_lis', [])}, length = {max(final.get('dp', [1]))}"
            
            # Generate array visualization
            result["visualization"]["data"] = VisualizationGenerator.generate_array_viz(
                arr, highlights=final.get('current_lis_indices', [])
            )
        
        elif demo_id == "clrs_coin":
            coins = params.get("coins", [1, 3, 4]) if params else [1, 3, 4]
            amount = params.get("amount", 6) if params else 6
            result["execution_trace"] = StepByStepEngine.trace_coin_change(coins, amount)
            final = result["execution_trace"][-1] if result["execution_trace"] else {}
            result["demo_output"] = f"Min coins = {final.get('min_coins', 'N/A')}, coins used = {final.get('coins_used', [])}"
        
    except Exception as e:
        result["demo_output"] = f"Error during trace: {str(e)}"
        result["execution_trace"] = []
    
    # Get related concepts from corpus
    corpus_integration = CorpusIntegration()
    result["related_concepts"] = corpus_integration.get_related_concepts(demo_id)
    
    # Generate Socratic follow-up
    llm_explainer = LLMExplainer()
    result["socratic_question"] = llm_explainer.generate_socratic_question(demo_id, result)
    
    return result


def get_demo_knowledge(demo_id: str) -> Dict:
    """Get knowledge base entry for a demo."""
    return DEMO_KNOWLEDGE.get(demo_id, {})


def list_intelligent_demos() -> List[Dict]:
    """List all demos with intelligent support."""
    return [
        {
            "id": demo_id,
            "title": info.get("title", demo_id),
            "category": info.get("category", ""),
            "difficulty": info.get("difficulty", "intermediate"),
            "has_visualization": info.get("visualization_type") != "none"
        }
        for demo_id, info in DEMO_KNOWLEDGE.items()
    ]


# =============================================================================
# FLASK ROUTE REGISTRATION
# =============================================================================

def register_intelligent_demo_routes(app):
    """Register intelligent demo API routes."""
    from flask import jsonify, request
    
    @app.route("/api/intelligent-demo/<demo_id>", methods=["GET", "POST"])
    def api_intelligent_demo(demo_id):
        """Run an intelligent demo with full educational context."""
        params = request.get_json(silent=True) if request.method == "POST" else {}
        result = run_intelligent_demo(demo_id, params)
        return jsonify(result)
    
    @app.route("/api/intelligent-demo/<demo_id>/knowledge")
    def api_demo_knowledge(demo_id):
        """Get knowledge base entry for a demo."""
        knowledge = get_demo_knowledge(demo_id)
        if not knowledge:
            return jsonify({"ok": False, "error": f"Unknown demo: {demo_id}"}), 404
        return jsonify({"ok": True, **knowledge})
    
    @app.route("/api/intelligent-demos")
    def api_list_intelligent_demos():
        """List all demos with intelligent support."""
        return jsonify({"ok": True, "demos": list_intelligent_demos()})
    
    @app.route("/api/intelligent-demo/<demo_id>/ask", methods=["POST"])
    def api_demo_ask(demo_id):
        """Ask a follow-up question about a demo."""
        data = request.get_json(silent=True) or {}
        question = data.get("question", "")
        context = data.get("context", {})
        
        llm_explainer = LLMExplainer()
        answer = llm_explainer.explain(demo_id, context, question)
        
        return jsonify({
            "ok": True,
            "demo_id": demo_id,
            "question": question,
            "answer": answer
        })
    
    @app.route("/api/corpus/search")
    def api_corpus_search():
        """Search the knowledge corpus."""
        query = request.args.get("q", "")
        top_k = int(request.args.get("top_k", 5))
        
        corpus_integration = CorpusIntegration()
        results = corpus_integration.search_corpus(query, top_k)
        
        return jsonify({
            "ok": True,
            "query": query,
            "results": results
        })
    
    @app.route("/api/concept/<concept>")
    def api_explain_concept(concept):
        """Get multi-view explanation of a concept."""
        if COMPASS_AVAILABLE:
            try:
                result = explain_concept(concept)
                return jsonify(result)
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": False, "error": "ML_Compass not available"}), 503
