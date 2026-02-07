"""
Curriculum: AI concepts — Russell & Norvig style (game theory, search, RL, probabilistic reasoning).
From ml_toolbox.ai_concepts.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "game_theory", "name": "Game Theory", "short": "Game Theory", "color": "#2563eb"},
    {"id": "search_planning", "name": "Search & Planning", "short": "Search & Planning", "color": "#059669"},
    {"id": "reinforcement", "name": "Reinforcement Learning", "short": "RL", "color": "#7c3aed"},
    {"id": "probabilistic", "name": "Probabilistic Reasoning", "short": "Prob. Reasoning", "color": "#dc2626"},
]

CURRICULUM: List[Dict[str, Any]] = [
    # ============================================================================
    # GAME THEORY (6 items)
    # ============================================================================
    {"id": "gt_intro_games", "book_id": "game_theory", "level": "basics", "title": "Introduction to Games",
     "learn": "Games defined by players, actions, payoffs. Normal form (matrix) vs extensive form (tree). Zero-sum vs cooperative. Strategic thinking models rational agents.",
     "try_code": "from ml_toolbox.ai_concepts.game_theory import Game\ngame = Game(players=2, payoff_matrix=[[(3,3), (0,5)], [(5,0), (1,1)]])\nprint(f'Payoffs: {game.payoff_matrix}')",
     "try_demo": None, "prerequisites": []},
    
    {"id": "gt_dominant_strategies", "book_id": "game_theory", "level": "basics", "title": "Dominant Strategies",
     "learn": "A strategy dominates another if it yields better payoffs regardless of opponents' choices. Dominant strategy equilibrium: all players play dominant strategies. Prisoner's Dilemma has dominant defection.",
     "try_code": "payoffs_cooperate = [3, 0]; payoffs_defect = [5, 1]\nis_dominant = all(d > c for d, c in zip(payoffs_defect, payoffs_cooperate))\nprint(f'Defect dominates: {is_dominant}')",
     "try_demo": None, "prerequisites": ["gt_intro_games"]},
    
    {"id": "gt_nash", "book_id": "game_theory", "level": "intermediate", "title": "Nash Equilibrium",
     "learn": "Strategy profile where no player gains by unilaterally deviating. Pure Nash: deterministic strategies. Mixed Nash: probabilistic strategies. Every finite game has at least one Nash equilibrium (possibly mixed).",
     "try_code": "from ml_toolbox.ai_concepts.game_theory import find_nash_equilibrium\ngame = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]\nnash = find_nash_equilibrium(game)\nprint(f'Mixed Nash: {nash}')",
     "try_demo": None, "prerequisites": ["gt_dominant_strategies"]},
    
    {"id": "gt_cooperative", "book_id": "game_theory", "level": "advanced", "title": "Cooperative Games & Coalitions",
     "learn": "Players form binding coalitions. Characteristic function v(S) assigns value to each coalition S. Core: stable allocations where no coalition can do better alone. Shapley value: fair division based on marginal contributions.",
     "try_code": "from ml_toolbox.ai_concepts.cooperative_games import shapley_value\nv = {(): 0, (1,): 0, (2,): 0, (3,): 0, (1,2): 90, (1,3): 80, (2,3): 70, (1,2,3): 120}\nshapley = shapley_value(v, n_players=3)\nprint(f'Fair division: {shapley}')",
     "try_demo": None, "prerequisites": ["gt_nash"]},
    
    {"id": "gt_mechanism_design", "book_id": "game_theory", "level": "advanced", "title": "Mechanism Design",
     "learn": "Inverse game theory: design rules to achieve desired outcomes. Auctions (Vickrey: truthful bidding), voting systems. Incentive compatibility: truth-telling is optimal. Used in ad auctions, resource allocation.",
     "try_code": "bids = [10, 15, 12, 8]\nwinner = max(range(len(bids)), key=lambda i: bids[i])\nprice = sorted(bids, reverse=True)[1]\nprint(f'Winner: {winner}, pays: {price}')",
     "try_demo": None, "prerequisites": ["gt_nash"]},
    
    {"id": "gt_evolutionary", "book_id": "game_theory", "level": "expert", "title": "Evolutionary Game Theory",
     "learn": "Strategies evolve via reproduction/selection. Evolutionary stable strategy (ESS): resists invasion by mutants. Replicator dynamics model population change. Applications: biology, social dynamics, ML (evolutionary algorithms).",
     "try_code": "import numpy as np\ndef replicator(x, payoff_matrix, dt=0.01):\n    fitness = payoff_matrix @ x; avg_fitness = x @ fitness\n    return x + dt * x * (fitness - avg_fitness)\npayoff = np.array([[0, 2], [1, 1]])\nx = np.array([0.5, 0.5])\nfor _ in range(100): x = replicator(x, payoff)\nprint(f'ESS population: {x}')",
     "try_demo": None, "prerequisites": ["gt_cooperative"]},
    
    # ============================================================================
    # SEARCH & PLANNING (8 items)
    # ============================================================================
    {"id": "search_bfs_dfs", "book_id": "search_planning", "level": "basics", "title": "BFS and DFS",
     "learn": "State space search in graphs. BFS: queue, finds shortest path, complete. DFS: stack, memory efficient, may not terminate. Fringe: nodes to explore. Complexity: O(b^d) where b=branching, d=depth.",
     "try_code": "from collections import deque\ndef bfs(graph, start, goal):\n    queue = deque([(start, [start])]); visited = {start}\n    while queue:\n        node, path = queue.popleft()\n        if node == goal: return path\n        for neighbor in graph[node]:\n            if neighbor not in visited:\n                visited.add(neighbor); queue.append((neighbor, path + [neighbor]))\ngraph = {'A': ['B','C'], 'B': ['D'], 'C': ['D'], 'D': []}\nprint(bfs(graph, 'A', 'D'))",
     "try_demo": None, "prerequisites": []},
    
    {"id": "search_uniform_cost", "book_id": "search_planning", "level": "basics", "title": "Uniform Cost Search",
     "learn": "Dijkstra's algorithm for weighted graphs. Priority queue orders by path cost g(n). Optimal if costs non-negative. Expands cheapest node first. Guarantees shortest path in weighted graphs.",
     "try_code": "import heapq\ndef uniform_cost_search(graph, start, goal):\n    pq = [(0, start, [start])]; visited = set()\n    while pq:\n        cost, node, path = heapq.heappop(pq)\n        if node == goal: return cost, path\n        if node in visited: continue\n        visited.add(node)\n        for neighbor, edge_cost in graph[node]:\n            heapq.heappush(pq, (cost + edge_cost, neighbor, path + [neighbor]))\ngraph = {'A': [('B',1), ('C',4)], 'B': [('D',2)], 'C': [('D',1)], 'D': []}\nprint(uniform_cost_search(graph, 'A', 'D'))",
     "try_demo": None, "prerequisites": ["search_bfs_dfs"]},
    
    {"id": "search_astar", "book_id": "search_planning", "level": "intermediate", "title": "A* Search",
     "learn": "Best-first search with f(n) = g(n) + h(n). g(n): cost so far. h(n): heuristic estimate to goal. Optimal if h is admissible (never overestimates). Optimally efficient: expands fewest nodes among optimal algorithms.",
     "try_code": "import heapq\ndef astar(graph, start, goal, h):\n    pq = [(h[start], 0, start, [start])]; visited = set()\n    while pq:\n        f, g, node, path = heapq.heappop(pq)\n        if node == goal: return g, path\n        if node in visited: continue\n        visited.add(node)\n        for neighbor, cost in graph[node]:\n            g_new = g + cost; f_new = g_new + h[neighbor]\n            heapq.heappush(pq, (f_new, g_new, neighbor, path + [neighbor]))\nh = {'A': 3, 'B': 2, 'C': 2, 'D': 0}\ngraph = {'A': [('B',1), ('C',4)], 'B': [('D',2)], 'C': [('D',1)], 'D': []}\nprint(astar(graph, 'A', 'D', h))",
     "try_demo": None, "prerequisites": ["search_uniform_cost"]},
    
    {"id": "search_heuristics", "book_id": "search_planning", "level": "intermediate", "title": "Heuristic Design",
     "learn": "Admissible: h(n) ≤ true cost (ensures optimality). Consistent: h(n) ≤ c(n,n') + h(n') (triangle inequality, implies admissibility). Dominance: h2 dominates h1 if h2(n) ≥ h1(n) for all n. Better heuristics = fewer nodes expanded.",
     "try_code": "def manhattan_distance(state, goal):\n    dist = 0\n    for i in range(9):\n        if state[i] != 0:\n            goal_pos = goal.index(state[i])\n            dist += abs(i//3 - goal_pos//3) + abs(i%3 - goal_pos%3)\n    return dist\nstate = [1,2,3,4,0,5,6,7,8]; goal = [1,2,3,4,5,6,7,8,0]\nprint(f'Manhattan: {manhattan_distance(state, goal)}')",
     "try_demo": None, "prerequisites": ["search_astar"]},
    
    {"id": "search_adversarial", "book_id": "search_planning", "level": "intermediate", "title": "Adversarial Search (Minimax)",
     "learn": "Two-player zero-sum games. Minimax: maximize minimum guaranteed payoff. Ply: half-move. Depth-limited search with evaluation function. Alpha-beta pruning: prune branches that can't affect outcome. O(b^(d/2)) vs O(b^d).",
     "try_code": "def minimax(node, depth, is_maximizing, alpha=-float('inf'), beta=float('inf')):\n    if depth == 0 or is_terminal(node): return evaluate(node)\n    if is_maximizing:\n        max_eval = -float('inf')\n        for child in get_children(node):\n            eval = minimax(child, depth-1, False, alpha, beta)\n            max_eval = max(max_eval, eval); alpha = max(alpha, eval)\n            if beta <= alpha: break\n        return max_eval\n    else:\n        min_eval = float('inf')\n        for child in get_children(node):\n            eval = minimax(child, depth-1, True, alpha, beta)\n            min_eval = min(min_eval, eval); beta = min(beta, eval)\n            if beta <= alpha: break\n        return min_eval",
     "try_demo": None, "prerequisites": ["search_astar"]},
    
    {"id": "planning_strips", "book_id": "search_planning", "level": "advanced", "title": "STRIPS Planning",
     "learn": "Classical planning with actions: preconditions, add/delete effects. State space = set of propositions. Forward search (progression) or backward (regression). Domain-independent heuristics: relaxed problem (ignore delete effects).",
     "try_code": "class Action:\n    def __init__(self, name, precond, add_effects, del_effects):\n        self.name = name; self.precond = precond; self.add = add_effects; self.delete = del_effects\n    def is_applicable(self, state): return self.precond.issubset(state)\n    def apply(self, state): return (state - self.delete) | self.add\nmove_a_to_b = Action('move(A, table, B)', precond={'clear(A)', 'clear(B)', 'on(A, table)'}, add_effects={'on(A, B)'}, del_effects={'clear(B)', 'on(A, table)'})\nstate = {'clear(A)', 'clear(B)', 'on(A, table)', 'on(B, table)'}\nif move_a_to_b.is_applicable(state): print(move_a_to_b.apply(state))",
     "try_demo": None, "prerequisites": ["search_astar"]},
    
    {"id": "planning_pddl", "book_id": "search_planning", "level": "advanced", "title": "PDDL & Graphplan",
     "learn": "PDDL: standardized planning language. Graphplan: build planning graph with proposition/action layers. Mutual exclusion (mutex) constraints. Extract solution plan by backtracking. Polynomial heuristic for many domains.",
     "try_code": "class PlanningGraph:\n    def __init__(self, init_state, actions):\n        self.layers = [init_state]; self.actions = actions\n    def expand(self):\n        prev_props = self.layers[-1]\n        applicable = [a for a in self.actions if a.precond.issubset(prev_props)]\n        next_props = prev_props.copy()\n        for action in applicable: next_props |= action.add\n        self.layers.append(next_props)\n        return next_props\n    def is_goal_reachable(self, goal): return goal.issubset(self.layers[-1])",
     "try_demo": None, "prerequisites": ["planning_strips"]},
    
    {"id": "planning_hierarchical", "book_id": "search_planning", "level": "expert", "title": "Hierarchical Task Networks",
     "learn": "HTN planning: decompose high-level tasks into subtasks. Methods: ways to achieve tasks. Total-order vs partial-order plans. Real-world applications: robotics, games. Handles complex domains better than classical planning.",
     "try_code": "class HTNTask:\n    def __init__(self, name, methods=None, primitive_action=None):\n        self.name = name; self.methods = methods or []; self.primitive = primitive_action\n    def decompose(self, method_idx): return self.methods[method_idx]\ndef htn_search(task, state, depth=0):\n    if task.primitive: return [task.primitive] if task.primitive.applicable(state) else None\n    for method in task.methods:\n        plan = []; current_state = state\n        for subtask in method:\n            subplan = htn_search(subtask, current_state, depth+1)\n            if subplan is None: break\n            plan.extend(subplan); current_state = apply_effects(current_state, subplan)\n        else: return plan\n    return None",
     "try_demo": None, "prerequisites": ["planning_pddl"]},
    
    # ============================================================================
    # REINFORCEMENT LEARNING (6 items)
    # ============================================================================
    {"id": "rl_mdp", "book_id": "reinforcement", "level": "basics", "title": "Markov Decision Processes",
     "learn": "MDP: states S, actions A, transition P(s'|s,a), rewards R(s,a,s'), discount γ. Markov property: future independent of past given present. Policy π: state → action. Goal: maximize expected return E[∑γ^t r_t].",
     "try_code": "import numpy as np\nclass GridWorld:\n    def __init__(self, size=5, goal=(4,4)): self.size = size; self.goal = goal; self.state = (0, 0)\n    def step(self, action):\n        moves = [(-1,0), (0,1), (1,0), (0,-1)]\n        next_state = tuple(np.clip(np.add(self.state, moves[action]), 0, self.size-1))\n        reward = 10 if next_state == self.goal else -0.1\n        self.state = next_state; done = next_state == self.goal\n        return next_state, reward, done\nenv = GridWorld()\nstate, reward, done = env.step(action=1)\nprint(f'State: {state}, Reward: {reward}')",
     "try_demo": None, "prerequisites": []},
    
    {"id": "rl_value", "book_id": "reinforcement", "level": "intermediate", "title": "Value Functions & Bellman Equations",
     "learn": "V^π(s): expected return from state s under policy π. Q^π(s,a): expected return from action a in state s. Bellman equations: V^π(s) = ∑P(s'|s,π(s))[R + γV^π(s')]. Optimal: V*(s) = max_a Q*(s,a).",
     "try_code": "import numpy as np\ndef value_iteration(mdp, gamma=0.9, theta=1e-6):\n    V = np.zeros(mdp.n_states)\n    while True:\n        delta = 0\n        for s in range(mdp.n_states):\n            v = V[s]\n            V[s] = max([sum([p * (r + gamma * V[s1]) for p, s1, r in mdp.transitions(s, a)]) for a in range(mdp.n_actions)])\n            delta = max(delta, abs(v - V[s]))\n        if delta < theta: break\n    return V\ndef extract_policy(mdp, V, gamma=0.9):\n    policy = np.zeros(mdp.n_states, dtype=int)\n    for s in range(mdp.n_states):\n        q_values = [sum([p * (r + gamma * V[s1]) for p, s1, r in mdp.transitions(s, a)]) for a in range(mdp.n_actions)]\n        policy[s] = np.argmax(q_values)\n    return policy",
     "try_demo": "rl_value_iter", "prerequisites": ["rl_mdp"]},
    
    {"id": "rl_qlearning", "book_id": "reinforcement", "level": "intermediate", "title": "Q-Learning",
     "learn": "Off-policy TD control. Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]. Converges to Q* under conditions: visit all (s,a) infinitely, decaying learning rate. Model-free: learns from experience without transition model.",
     "try_code": "import numpy as np\nclass QLearning:\n    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):\n        self.Q = np.zeros((n_states, n_actions)); self.alpha = alpha; self.gamma = gamma; self.epsilon = epsilon\n    def select_action(self, state):\n        if np.random.random() < self.epsilon: return np.random.randint(self.Q.shape[1])\n        return np.argmax(self.Q[state])\n    def update(self, state, action, reward, next_state):\n        target = reward + self.gamma * np.max(self.Q[next_state])\n        self.Q[state, action] += self.alpha * (target - self.Q[state, action])\nagent = QLearning(n_states=25, n_actions=4)",
     "try_demo": "rl_qlearning", "prerequisites": ["rl_value"]},
    
    {"id": "rl_policy_gradient", "book_id": "reinforcement", "level": "advanced", "title": "Policy Gradient Methods",
     "learn": "Directly optimize policy π_θ(a|s). REINFORCE: ∇J(θ) = E[∇logπ_θ(a|s) Q^π(s,a)]. Monte Carlo estimate with baseline to reduce variance. Actor-Critic: learn V(s) as baseline. Advantage: A(s,a) = Q(s,a) - V(s).",
     "try_code": "import torch\nimport torch.nn as nn\nclass PolicyNetwork(nn.Module):\n    def __init__(self, state_dim, action_dim):\n        super().__init__()\n        self.fc = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, action_dim), nn.Softmax(dim=-1))\n    def forward(self, state): return self.fc(state)\npolicy = PolicyNetwork(state_dim=4, action_dim=2)\nprobs = policy(torch.randn(4))\naction = torch.multinomial(probs, 1).item()\nprint(f'Action probabilities: {probs}, Selected: {action}')",
     "try_demo": None, "prerequisites": ["rl_qlearning"]},
    
    {"id": "rl_dqn", "book_id": "reinforcement", "level": "advanced", "title": "Deep Q-Networks (DQN)",
     "learn": "Q-learning with neural network function approximation. Experience replay: break correlations. Target network: stabilize learning. Loss: (r + γ max_a' Q_target(s',a') - Q(s,a))². Applications: Atari games, robotics.",
     "try_code": "import torch.nn as nn\nfrom collections import deque\nimport random\nclass DQN(nn.Module):\n    def __init__(self, state_dim, action_dim):\n        super().__init__()\n        self.net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_dim))\n    def forward(self, x): return self.net(x)\nclass ReplayBuffer:\n    def __init__(self, capacity=10000): self.buffer = deque(maxlen=capacity)\n    def add(self, state, action, reward, next_state, done): self.buffer.append((state, action, reward, next_state, done))\n    def sample(self, batch_size): return random.sample(self.buffer, batch_size)",
     "try_demo": None, "prerequisites": ["rl_qlearning"]},
    
    {"id": "rl_mcts", "book_id": "reinforcement", "level": "expert", "title": "Monte Carlo Tree Search",
     "learn": "Combines tree search with Monte Carlo sampling. Four phases: Selection (UCT), Expansion, Simulation (rollout), Backpropagation. UCT: balance exploration/exploitation. AlphaGo combines MCTS with deep learning. Applications: games, planning.",
     "try_code": "import math\nimport random\nclass MCTSNode:\n    def __init__(self, state, parent=None):\n        self.state = state; self.parent = parent; self.children = []; self.visits = 0; self.value = 0\n    def uct_select(self, c=1.41):\n        return max(self.children, key=lambda n: n.value/n.visits + c * math.sqrt(math.log(self.visits) / n.visits))\n    def expand(self, actions):\n        for action in actions:\n            next_state = self.state.apply_action(action)\n            child = MCTSNode(next_state, parent=self)\n            self.children.append(child)\n    def backpropagate(self, value):\n        self.visits += 1; self.value += value\n        if self.parent: self.parent.backpropagate(value)",
     "try_demo": None, "prerequisites": ["rl_policy_gradient", "search_adversarial"]},
    
    # ============================================================================
    # PROBABILISTIC REASONING (8 items)
    # ============================================================================
    {"id": "prob_bayes", "book_id": "probabilistic", "level": "basics", "title": "Bayesian Reasoning",
     "learn": "Bayes' rule: P(H|E) = P(E|H)P(H)/P(E). Prior P(H), likelihood P(E|H), posterior P(H|E). Reasoning under uncertainty. Applications: diagnosis, spam filtering, robotics.",
     "try_code": "def bayes_rule(prior, likelihood, evidence): return (likelihood * prior) / evidence\nP_disease = 0.01; P_pos_given_disease = 0.95; P_pos_given_healthy = 0.05\nP_pos = P_pos_given_disease * P_disease + P_pos_given_healthy * (1 - P_disease)\nP_disease_given_pos = bayes_rule(P_disease, P_pos_given_disease, P_pos)\nprint(f'P(disease|positive test) = {P_disease_given_pos:.3f}')",
     "try_demo": None, "prerequisites": []},
    
    {"id": "prob_independence", "book_id": "probabilistic", "level": "basics", "title": "Independence & Conditional Independence",
     "learn": "Independence: P(X,Y) = P(X)P(Y). Conditional independence: P(X,Y|Z) = P(X|Z)P(Y|Z). Key for compact representations. Naive Bayes assumes features independent given class. Enables efficient inference.",
     "try_code": "import numpy as np\nclass NaiveBayes:\n    def fit(self, X, y):\n        self.classes = np.unique(y)\n        self.class_prior = {c: (y == c).mean() for c in self.classes}\n        self.feature_probs = {}\n        for c in self.classes:\n            X_c = X[y == c]\n            self.feature_probs[c] = {'mean': X_c.mean(axis=0), 'std': X_c.std(axis=0) + 1e-6}\n    def predict(self, X):\n        predictions = []\n        for x in X:\n            posteriors = {}\n            for c in self.classes:\n                prior = np.log(self.class_prior[c])\n                likelihood = -0.5 * np.sum(((x - self.feature_probs[c]['mean']) / self.feature_probs[c]['std'])**2)\n                posteriors[c] = prior + likelihood\n            predictions.append(max(posteriors, key=posteriors.get))\n        return predictions",
     "try_demo": None, "prerequisites": ["prob_bayes"]},
    
    {"id": "prob_reasoning", "book_id": "probabilistic", "level": "intermediate", "title": "Bayesian Networks",
     "learn": "Directed acyclic graph (DAG) encoding conditional independencies. Nodes: random variables. Edges: dependencies. Joint: P(X1,...,Xn) = ∏P(Xi|Parents(Xi)). Compact representation. Inference via belief propagation, variable elimination.",
     "try_code": "class BayesianNetwork:\n    def __init__(self): self.nodes = {}; self.edges = {}\n    def add_node(self, name, parents, cpt):\n        self.nodes[name] = {'parents': parents, 'cpt': cpt}\n        for parent in parents:\n            if parent not in self.edges: self.edges[parent] = []\n            self.edges[parent].append(name)\nbn = BayesianNetwork()\nbn.add_node('Burglary', [], {'T': 0.001})\nbn.add_node('Earthquake', [], {'T': 0.002})\nbn.add_node('Alarm', ['Burglary', 'Earthquake'], {(True, True): 0.95, (True, False): 0.94, (False, True): 0.29, (False, False): 0.001})\nbn.add_node('Call', ['Alarm'], {True: 0.90, False: 0.05})",
     "try_demo": None, "prerequisites": ["prob_independence"]},
    
    {"id": "prob_hmm", "book_id": "probabilistic", "level": "intermediate", "title": "Hidden Markov Models",
     "learn": "Temporal probabilistic model. Hidden states X_t, observations Y_t. Transition P(X_t|X_{t-1}), emission P(Y_t|X_t). Forward algorithm: filtering P(X_t|y_1:t). Viterbi: most likely sequence. Baum-Welch: parameter learning.",
     "try_code": "import numpy as np\nclass HMM:\n    def __init__(self, n_states, n_obs):\n        self.n_states = n_states; self.n_obs = n_obs\n        self.transition = np.random.rand(n_states, n_states); self.transition /= self.transition.sum(axis=1, keepdims=True)\n        self.emission = np.random.rand(n_states, n_obs); self.emission /= self.emission.sum(axis=1, keepdims=True)\n        self.initial = np.ones(n_states) / n_states\n    def forward(self, observations):\n        T = len(observations); alpha = np.zeros((T, self.n_states))\n        alpha[0] = self.initial * self.emission[:, observations[0]]; alpha[0] /= alpha[0].sum()\n        for t in range(1, T):\n            alpha[t] = (alpha[t-1] @ self.transition) * self.emission[:, observations[t]]; alpha[t] /= alpha[t].sum()\n        return alpha\nhmm = HMM(n_states=2, n_obs=3)\nprint(hmm.forward([0, 1, 2, 1, 0]))",
     "try_demo": None, "prerequisites": ["prob_reasoning"]},
    
    {"id": "prob_particle_filter", "book_id": "probabilistic", "level": "advanced", "title": "Particle Filters",
     "learn": "Sequential Monte Carlo for non-linear/non-Gaussian systems. Represent belief as particles (samples). Propagate via motion model, weight by observation likelihood, resample. Effective for robot localization, tracking.",
     "try_code": "import numpy as np\nclass ParticleFilter:\n    def __init__(self, n_particles, state_dim):\n        self.n_particles = n_particles\n        self.particles = np.random.randn(n_particles, state_dim)\n        self.weights = np.ones(n_particles) / n_particles\n    def predict(self, motion_model, control):\n        for i in range(self.n_particles):\n            self.particles[i] = motion_model(self.particles[i], control)\n            self.particles[i] += np.random.randn(*self.particles[i].shape) * 0.1\n    def update(self, observation, sensor_model):\n        for i in range(self.n_particles):\n            self.weights[i] = sensor_model(self.particles[i], observation)\n        self.weights /= self.weights.sum()\n    def estimate(self): return np.average(self.particles, weights=self.weights, axis=0)\npf = ParticleFilter(n_particles=1000, state_dim=3)",
     "try_demo": None, "prerequisites": ["prob_hmm"]},
    
    {"id": "prob_em", "book_id": "probabilistic", "level": "advanced", "title": "Expectation-Maximization",
     "learn": "Unsupervised learning for latent variable models. E-step: compute expected sufficient statistics given current parameters. M-step: optimize parameters. Guaranteed to increase likelihood. Applications: GMM, HMM training, missing data.",
     "try_code": "import numpy as np\nfrom scipy.stats import multivariate_normal\nclass GMM:\n    def __init__(self, n_components, n_features):\n        self.K = n_components\n        self.pi = np.ones(n_components) / n_components\n        self.mu = np.random.randn(n_components, n_features)\n        self.sigma = np.array([np.eye(n_features) for _ in range(n_components)])\n    def e_step(self, X):\n        N = len(X); gamma = np.zeros((N, self.K))\n        for k in range(self.K): gamma[:, k] = self.pi[k] * multivariate_normal.pdf(X, mean=self.mu[k], cov=self.sigma[k])\n        gamma /= gamma.sum(axis=1, keepdims=True)\n        return gamma\n    def m_step(self, X, gamma):\n        N_k = gamma.sum(axis=0)\n        self.pi = N_k / len(X)\n        self.mu = (gamma.T @ X) / N_k[:, np.newaxis]\n        for k in range(self.K):\n            diff = X - self.mu[k]\n            self.sigma[k] = (gamma[:, k] * diff.T @ diff) / N_k[k]\ngmm = GMM(n_components=3, n_features=2)",
     "try_demo": None, "prerequisites": ["prob_reasoning"]},
    
    {"id": "prob_dbns", "book_id": "probabilistic", "level": "advanced", "title": "Dynamic Bayesian Networks",
     "learn": "Temporal extension of Bayesian networks. Template for time-slice connections. HMMs are simple DBNs. Kalman filters: linear-Gaussian DBNs. Inference: forward filtering, smoothing (forward-backward), MPE (Viterbi).",
     "try_code": "class DBN:\n    def __init__(self, intra_slice, inter_slice):\n        self.intra = intra_slice; self.inter = inter_slice\n    def forward_filter(self, observations, T):\n        beliefs = []; belief = self.prior()\n        for t in range(T):\n            predicted = self.predict(belief)\n            belief = self.update(predicted, observations[t])\n            beliefs.append(belief)\n        return beliefs\n    def smooth(self, observations):\n        forward = self.forward_filter(observations, len(observations))\n        backward = []\n        return forward\n# Example: Robot SLAM with pose + landmarks",
     "try_demo": None, "prerequisites": ["prob_hmm", "prob_reasoning"]},
    
    {"id": "clustering_ml", "book_id": "probabilistic", "level": "basics", "title": "Clustering Algorithms",
     "learn": "Unsupervised grouping. K-means: iterative centroid assignment. Hierarchical: agglomerative (bottom-up) or divisive (top-down). DBSCAN: density-based. Evaluation: silhouette score, within-cluster variance.",
     "try_code": "import numpy as np\nclass KMeans:\n    def __init__(self, n_clusters=3, max_iter=100): self.K = n_clusters; self.max_iter = max_iter\n    def fit(self, X):\n        idx = np.random.choice(len(X), self.K, replace=False); self.centroids = X[idx]\n        for _ in range(self.max_iter):\n            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)\n            labels = np.argmin(distances, axis=1)\n            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.K)])\n            if np.allclose(new_centroids, self.centroids): break\n            self.centroids = new_centroids\n        self.labels_ = labels\n        return self\n    def predict(self, X):\n        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)\n        return np.argmin(distances, axis=1)\nkmeans = KMeans(n_clusters=3)\nX = np.random.randn(100, 2)\nkmeans.fit(X)\nprint(f'Centroids: {kmeans.centroids}')",
     "try_demo": "ai_clustering", "prerequisites": ["prob_independence"]},
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
