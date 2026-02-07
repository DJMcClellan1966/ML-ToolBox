"""
Enrich AI Concepts Lab curriculum from 10 → 30+ items.
Covers Russell & Norvig AI: game theory, search/planning, RL, probabilistic reasoning.
"""
import json
from pathlib import Path

def generate_curriculum_items():
    """Generate comprehensive AI concepts curriculum."""
    
    items = []
    
    # ============================================================================
    # GAME THEORY (6 items)
    # ============================================================================
    
    items.append({
        "id": "gt_intro_games",
        "book_id": "game_theory",
        "level": "basics",
        "title": "Introduction to Games",
        "learn": "Games defined by players, actions, payoffs. Normal form (matrix) vs extensive form (tree). Zero-sum vs cooperative. Strategic thinking models rational agents.",
        "try_code": """from ml_toolbox.ai_concepts.game_theory import Game
# Prisoner's Dilemma payoff matrix
game = Game(players=2, payoff_matrix=[
    [(3,3), (0,5)],  # Cooperate/Cooperate, Cooperate/Defect
    [(5,0), (1,1)]   # Defect/Cooperate, Defect/Defect
])
print(f"Payoffs: {game.payoff_matrix}")""",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "gt_dominant_strategies",
        "book_id": "game_theory",
        "level": "basics",
        "title": "Dominant Strategies",
        "learn": "A strategy dominates another if it yields better payoffs regardless of opponents' choices. Dominant strategy equilibrium: all players play dominant strategies. Prisoner's Dilemma has dominant defection.",
        "try_code": """# Check for dominant strategy in Prisoner's Dilemma
# Defect dominates Cooperate: 5>3 and 1>0
payoffs_cooperate = [3, 0]  # vs C, vs D
payoffs_defect = [5, 1]     # vs C, vs D
is_dominant = all(d > c for d, c in zip(payoffs_defect, payoffs_cooperate))
print(f"Defect dominates: {is_dominant}")  # True""",
        "try_demo": None,
        "prerequisites": ["gt_intro_games"]
    })
    
    items.append({
        "id": "gt_nash",
        "book_id": "game_theory",
        "level": "intermediate",
        "title": "Nash Equilibrium",
        "learn": "Strategy profile where no player gains by unilaterally deviating. Pure Nash: deterministic strategies. Mixed Nash: probabilistic strategies. Every finite game has at least one Nash equilibrium (possibly mixed).",
        "try_code": """from ml_toolbox.ai_concepts.game_theory import find_nash_equilibrium
# Rock-Paper-Scissors has mixed Nash: (1/3, 1/3, 1/3)
game = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]  # RPS payoffs
nash = find_nash_equilibrium(game)
print(f"Mixed Nash: {nash}")  # Equal probability""",
        "try_demo": None,
        "prerequisites": ["gt_dominant_strategies"]
    })
    
    items.append({
        "id": "gt_cooperative",
        "book_id": "game_theory",
        "level": "advanced",
        "title": "Cooperative Games & Coalitions",
        "learn": "Players form binding coalitions. Characteristic function v(S) assigns value to each coalition S. Core: stable allocations where no coalition can do better alone. Shapley value: fair division based on marginal contributions.",
        "try_code": """from ml_toolbox.ai_concepts.cooperative_games import shapley_value
# 3-player coalition game
v = {(): 0, (1,): 0, (2,): 0, (3,): 0, 
     (1,2): 90, (1,3): 80, (2,3): 70, (1,2,3): 120}
shapley = shapley_value(v, n_players=3)
print(f"Fair division: {shapley}")  # Marginal contribution average""",
        "try_demo": None,
        "prerequisites": ["gt_nash"]
    })
    
    items.append({
        "id": "gt_mechanism_design",
        "book_id": "game_theory",
        "level": "advanced",
        "title": "Mechanism Design",
        "learn": "Inverse game theory: design rules to achieve desired outcomes. Auctions (Vickrey: truthful bidding), voting systems. Incentive compatibility: truth-telling is optimal. Used in ad auctions, resource allocation.",
        "try_code": """# Vickrey auction: second-price sealed-bid
bids = [10, 15, 12, 8]  # Bidders' valuations
winner = max(range(len(bids)), key=lambda i: bids[i])
price = sorted(bids, reverse=True)[1]  # Second highest
print(f"Winner: {winner}, pays: {price}")  # Truthful bidding dominant""",
        "try_demo": None,
        "prerequisites": ["gt_nash"]
    })
    
    items.append({
        "id": "gt_evolutionary",
        "book_id": "game_theory",
        "level": "expert",
        "title": "Evolutionary Game Theory",
        "learn": "Strategies evolve via reproduction/selection. Evolutionary stable strategy (ESS): resists invasion by mutants. Replicator dynamics model population change. Applications: biology, social dynamics, ML (evolutionary algorithms).",
        "try_code": """import numpy as np
# Replicator dynamics for Hawk-Dove game
def replicator(x, payoff_matrix, dt=0.01):
    fitness = payoff_matrix @ x
    avg_fitness = x @ fitness
    return x + dt * x * (fitness - avg_fitness)

payoff = np.array([[0, 2], [1, 1]])  # Hawk vs Dove
x = np.array([0.5, 0.5])  # Initial population
for _ in range(100): x = replicator(x, payoff)
print(f"ESS population: {x}")  # Converges to stable mix""",
        "try_demo": None,
        "prerequisites": ["gt_cooperative"]
    })
    
    # ============================================================================
    # SEARCH & PLANNING (8 items)
    # ============================================================================
    
    items.append({
        "id": "search_bfs_dfs",
        "book_id": "search_planning",
        "level": "basics",
        "title": "BFS and DFS",
        "learn": "State space search in graphs. BFS: queue, finds shortest path, complete. DFS: stack, memory efficient, may not terminate. Fringe: nodes to explore. Complexity: O(b^d) where b=branching, d=depth.",
        "try_code": """from collections import deque
# BFS shortest path
def bfs(graph, start, goal):
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        if node == goal: return path
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
graph = {'A': ['B','C'], 'B': ['D'], 'C': ['D'], 'D': []}
print(bfs(graph, 'A', 'D'))  # ['A', 'B', 'D']""",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "search_uniform_cost",
        "book_id": "search_planning",
        "level": "basics",
        "title": "Uniform Cost Search",
        "learn": "Dijkstra's algorithm for weighted graphs. Priority queue orders by path cost g(n). Optimal if costs non-negative. Expands cheapest node first. Guarantees shortest path in weighted graphs.",
        "try_code": """import heapq
def uniform_cost_search(graph, start, goal):
    pq = [(0, start, [start])]  # (cost, node, path)
    visited = set()
    while pq:
        cost, node, path = heapq.heappop(pq)
        if node == goal: return cost, path
        if node in visited: continue
        visited.add(node)
        for neighbor, edge_cost in graph[node]:
            heapq.heappush(pq, (cost + edge_cost, neighbor, path + [neighbor]))
graph = {'A': [('B',1), ('C',4)], 'B': [('D',2)], 'C': [('D',1)], 'D': []}
print(uniform_cost_search(graph, 'A', 'D'))  # (3, ['A','B','D'])""",
        "try_demo": None,
        "prerequisites": ["search_bfs_dfs"]
    })
    
    items.append({
        "id": "search_astar",
        "book_id": "search_planning",
        "level": "intermediate",
        "title": "A* Search",
        "learn": "Best-first search with f(n) = g(n) + h(n). g(n): cost so far. h(n): heuristic estimate to goal. Optimal if h is admissible (never overestimates). Optimally efficient: expands fewest nodes among optimal algorithms.",
        "try_code": """import heapq
def astar(graph, start, goal, h):
    pq = [(h[start], 0, start, [start])]  # (f, g, node, path)
    visited = set()
    while pq:
        f, g, node, path = heapq.heappop(pq)
        if node == goal: return g, path
        if node in visited: continue
        visited.add(node)
        for neighbor, cost in graph[node]:
            g_new = g + cost
            f_new = g_new + h[neighbor]
            heapq.heappush(pq, (f_new, g_new, neighbor, path + [neighbor]))
# Example: Manhattan distance heuristic
h = {'A': 3, 'B': 2, 'C': 2, 'D': 0}
graph = {'A': [('B',1), ('C',4)], 'B': [('D',2)], 'C': [('D',1)], 'D': []}
print(astar(graph, 'A', 'D', h))  # Optimal path with heuristic""",
        "try_demo": None,
        "prerequisites": ["search_uniform_cost"]
    })
    
    items.append({
        "id": "search_heuristics",
        "book_id": "search_planning",
        "level": "intermediate",
        "title": "Heuristic Design",
        "learn": "Admissible: h(n) ≤ true cost (ensures optimality). Consistent: h(n) ≤ c(n,n') + h(n') (triangle inequality, implies admissibility). Dominance: h2 dominates h1 if h2(n) ≥ h1(n) for all n. Better heuristics = fewer nodes expanded.",
        "try_code": """# 8-puzzle heuristics
def manhattan_distance(state, goal):
    dist = 0
    for i in range(9):
        if state[i] != 0:
            goal_pos = goal.index(state[i])
            dist += abs(i//3 - goal_pos//3) + abs(i%3 - goal_pos%3)
    return dist

def misplaced_tiles(state, goal):
    return sum(1 for i in range(9) if state[i] != 0 and state[i] != goal[i])

state = [1,2,3,4,0,5,6,7,8]
goal = [1,2,3,4,5,6,7,8,0]
print(f"Manhattan: {manhattan_distance(state, goal)}")  # 2
print(f"Misplaced: {misplaced_tiles(state, goal)}")    # 2
# Manhattan dominates misplaced (more informed)""",
        "try_demo": None,
        "prerequisites": ["search_astar"]
    })
    
    items.append({
        "id": "search_adversarial",
        "book_id": "search_planning",
        "level": "intermediate",
        "title": "Adversarial Search (Minimax)",
        "learn": "Two-player zero-sum games. Minimax: maximize minimum guaranteed payoff. Ply: half-move. Depth-limited search with evaluation function. Alpha-beta pruning: prune branches that can't affect outcome. O(b^(d/2)) vs O(b^d).",
        "try_code": """def minimax(node, depth, is_maximizing, alpha=-float('inf'), beta=float('inf')):
    if depth == 0 or is_terminal(node):
        return evaluate(node)
    
    if is_maximizing:
        max_eval = -float('inf')
        for child in get_children(node):
            eval = minimax(child, depth-1, False, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha: break  # Beta cutoff
        return max_eval
    else:
        min_eval = float('inf')
        for child in get_children(node):
            eval = minimax(child, depth-1, True, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha: break  # Alpha cutoff
        return min_eval

# Usage: best_move = max(moves, key=lambda m: minimax(m, depth=4, True))""",
        "try_demo": None,
        "prerequisites": ["search_astar"]
    })
    
    items.append({
        "id": "planning_strips",
        "book_id": "search_planning",
        "level": "advanced",
        "title": "STRIPS Planning",
        "learn": "Classical planning with actions: preconditions, add/delete effects. State space = set of propositions. Forward search (progression) or backward (regression). Domain-independent heuristics: relaxed problem (ignore delete effects).",
        "try_code": """# STRIPS action representation
class Action:
    def __init__(self, name, precond, add_effects, del_effects):
        self.name = name
        self.precond = precond
        self.add = add_effects
        self.delete = del_effects
    
    def is_applicable(self, state):
        return self.precond.issubset(state)
    
    def apply(self, state):
        return (state - self.delete) | self.add

# Example: Blocks World
move_a_to_b = Action('move(A, table, B)',
    precond={'clear(A)', 'clear(B)', 'on(A, table)'},
    add_effects={'on(A, B)'},
    del_effects={'clear(B)', 'on(A, table)'})

state = {'clear(A)', 'clear(B)', 'on(A, table)', 'on(B, table)'}
if move_a_to_b.is_applicable(state):
    new_state = move_a_to_b.apply(state)
    print(f"New state: {new_state}")""",
        "try_demo": None,
        "prerequisites": ["search_astar"]
    })
    
    items.append({
        "id": "planning_pddl",
        "book_id": "search_planning",
        "level": "advanced",
        "title": "PDDL & Graphplan",
        "learn": "PDDL: standardized planning language. Graphplan: build planning graph with proposition/action layers. Mutual exclusion (mutex) constraints. Extract solution plan by backtracking. Polynomial heuristic for many domains.",
        "try_code": """# Simplified Graphplan structure
class PlanningGraph:
    def __init__(self, init_state, actions):
        self.layers = [init_state]  # Proposition layers
        self.actions = actions
    
    def expand(self):
        # Add action layer: applicable actions
        prev_props = self.layers[-1]
        applicable = [a for a in self.actions if a.precond.issubset(prev_props)]
        
        # Add next proposition layer: effects
        next_props = prev_props.copy()
        for action in applicable:
            next_props |= action.add
        
        self.layers.append(next_props)
        return next_props
    
    def is_goal_reachable(self, goal):
        return goal.issubset(self.layers[-1])

# Build graph until goal appears or fixed point""",
        "try_demo": None,
        "prerequisites": ["planning_strips"]
    })
    
    items.append({
        "id": "planning_hierarchical",
        "book_id": "search_planning",
        "level": "expert",
        "title": "Hierarchical Task Networks",
        "learn": "HTN planning: decompose high-level tasks into subtasks. Methods: ways to achieve tasks. Total-order vs partial-order plans. Real-world applications: robotics, games. Handles complex domains better than classical planning.",
        "try_code": """# HTN decomposition example
class HTNTask:
    def __init__(self, name, methods=None, primitive_action=None):
        self.name = name
        self.methods = methods or []  # Decomposition methods
        self.primitive = primitive_action
    
    def decompose(self, method_idx):
        return self.methods[method_idx]  # Returns subtask list

# Example: Get-Coffee task
get_coffee = HTNTask('get_coffee', methods=[
    ['go_to_cafe', 'order_coffee', 'pay', 'wait', 'take_coffee'],
    ['make_coffee_at_home']  # Alternative method
])

def htn_search(task, state, depth=0):
    if task.primitive:
        return [task.primitive] if task.primitive.applicable(state) else None
    for method in task.methods:
        plan = []
        current_state = state
        for subtask in method:
            subplan = htn_search(subtask, current_state, depth+1)
            if subplan is None: break
            plan.extend(subplan)
            current_state = apply_effects(current_state, subplan)
        else:
            return plan  # Success
    return None  # No method worked""",
        "try_demo": None,
        "prerequisites": ["planning_pddl"]
    })
    
    # ============================================================================
    # REINFORCEMENT LEARNING (6 items)
    # ============================================================================
    
    items.append({
        "id": "rl_mdp",
        "book_id": "reinforcement",
        "level": "basics",
        "title": "Markov Decision Processes",
        "learn": "MDP: states S, actions A, transition P(s'|s,a), rewards R(s,a,s'), discount γ. Markov property: future independent of past given present. Policy π: state → action. Goal: maximize expected return E[∑γ^t r_t].",
        "try_code": """import numpy as np
# Simple GridWorld MDP
class GridWorld:
    def __init__(self, size=5, goal=(4,4)):
        self.size = size
        self.goal = goal
        self.state = (0, 0)
    
    def step(self, action):  # 0:up, 1:right, 2:down, 3:left
        moves = [(-1,0), (0,1), (1,0), (0,-1)]
        next_state = tuple(np.clip(np.add(self.state, moves[action]), 0, self.size-1))
        reward = 10 if next_state == self.goal else -0.1
        self.state = next_state
        done = next_state == self.goal
        return next_state, reward, done

env = GridWorld()
state, reward, done = env.step(action=1)  # Move right
print(f"State: {state}, Reward: {reward}")""",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "rl_value",
        "book_id": "reinforcement",
        "level": "intermediate",
        "title": "Value Functions & Bellman Equations",
        "learn": "V^π(s): expected return from state s under policy π. Q^π(s,a): expected return from action a in state s. Bellman equations: V^π(s) = ∑P(s'|s,π(s))[R + γV^π(s')]. Optimal: V*(s) = max_a Q*(s,a).",
        "try_code": """import numpy as np
# Value iteration
def value_iteration(mdp, gamma=0.9, theta=1e-6):
    V = np.zeros(mdp.n_states)
    while True:
        delta = 0
        for s in range(mdp.n_states):
            v = V[s]
            # Bellman optimality backup
            V[s] = max([sum([p * (r + gamma * V[s1]) 
                        for p, s1, r in mdp.transitions(s, a)])
                       for a in range(mdp.n_actions)])
            delta = max(delta, abs(v - V[s]))
        if delta < theta: break
    return V

# Extract policy: π(s) = argmax_a Q(s,a)
def extract_policy(mdp, V, gamma=0.9):
    policy = np.zeros(mdp.n_states, dtype=int)
    for s in range(mdp.n_states):
        q_values = [sum([p * (r + gamma * V[s1]) 
                    for p, s1, r in mdp.transitions(s, a)])
                   for a in range(mdp.n_actions)]
        policy[s] = np.argmax(q_values)
    return policy""",
        "try_demo": "rl_value_iter",
        "prerequisites": ["rl_mdp"]
    })
    
    items.append({
        "id": "rl_qlearning",
        "book_id": "reinforcement",
        "level": "intermediate",
        "title": "Q-Learning",
        "learn": "Off-policy TD control. Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]. Converges to Q* under conditions: visit all (s,a) infinitely, decaying learning rate. Model-free: learns from experience without transition model.",
        "try_code": """import numpy as np
class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])  # Explore
        return np.argmax(self.Q[state])  # Exploit
    
    def update(self, state, action, reward, next_state):
        target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

# Training loop
agent = QLearning(n_states=25, n_actions=4)
for episode in range(1000):
    state = env.reset()
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state""",
        "try_demo": "rl_qlearning",
        "prerequisites": ["rl_value"]
    })
    
    items.append({
        "id": "rl_policy_gradient",
        "book_id": "reinforcement",
        "level": "advanced",
        "title": "Policy Gradient Methods",
        "learn": "Directly optimize policy π_θ(a|s). REINFORCE: ∇J(θ) = E[∇logπ_θ(a|s) Q^π(s,a)]. Monte Carlo estimate with baseline to reduce variance. Actor-Critic: learn V(s) as baseline. Advantage: A(s,a) = Q(s,a) - V(s).",
        "try_code": """import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim), nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.fc(state)

def reinforce(policy, env, episodes=1000, gamma=0.99):
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    
    for ep in range(episodes):
        states, actions, rewards = [], [], []
        state = env.reset()
        
        # Generate episode
        while not done:
            probs = policy(torch.tensor(state, dtype=torch.float32))
            action = torch.multinomial(probs, 1).item()
            next_state, reward, done = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        # Update policy
        loss = 0
        for s, a, G in zip(states, actions, returns):
            probs = policy(torch.tensor(s, dtype=torch.float32))
            log_prob = torch.log(probs[a])
            loss -= log_prob * G  # Gradient ascent
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()""",
        "try_demo": None,
        "prerequisites": ["rl_qlearning"]
    })
    
    items.append({
        "id": "rl_dqn",
        "book_id": "reinforcement",
        "level": "advanced",
        "title": "Deep Q-Networks (DQN)",
        "learn": "Q-learning with neural network function approximation. Experience replay: break correlations. Target network: stabilize learning. Loss: (r + γ max_a' Q_target(s',a') - Q(s,a))². Applications: Atari games, robotics.",
        "try_code": """import torch
import torch.nn as nn
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Training
policy_net = DQN(state_dim=4, action_dim=2)
target_net = DQN(state_dim=4, action_dim=2)
target_net.load_state_dict(policy_net.state_dict())
optimizer = torch.optim.Adam(policy_net.parameters())
buffer = ReplayBuffer()

# ... collect experience and train with TD loss""",
        "try_demo": None,
        "prerequisites": ["rl_qlearning"]
    })
    
    items.append({
        "id": "rl_mcts",
        "book_id": "reinforcement",
        "level": "expert",
        "title": "Monte Carlo Tree Search",
        "learn": "Combines tree search with Monte Carlo sampling. Four phases: Selection (UCT), Expansion, Simulation (rollout), Backpropagation. UCT: balance exploration/exploitation. AlphaGo combines MCTS with deep learning. Applications: games, planning.",
        "try_code": """import math
import random

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
    
    def uct_select(self, c=1.41):
        # Upper Confidence Bound for Trees
        return max(self.children, 
                  key=lambda n: n.value/n.visits + 
                       c * math.sqrt(math.log(self.visits) / n.visits))
    
    def expand(self, actions):
        for action in actions:
            next_state = self.state.apply_action(action)
            child = MCTSNode(next_state, parent=self)
            self.children.append(child)
    
    def rollout(self):
        state = self.state
        while not state.is_terminal():
            action = random.choice(state.legal_actions())
            state = state.apply_action(action)
        return state.reward()
    
    def backpropagate(self, value):
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.backpropagate(value)

def mcts(root, n_iterations=1000):
    for _ in range(n_iterations):
        # Selection
        node = root
        while node.children and not node.state.is_terminal():
            node = node.uct_select()
        
        # Expansion
        if not node.state.is_terminal():
            node.expand(node.state.legal_actions())
            node = random.choice(node.children)
        
        # Simulation
        value = node.rollout()
        
        # Backpropagation
        node.backpropagate(value)
    
    return max(root.children, key=lambda n: n.visits)""",
        "try_demo": None,
        "prerequisites": ["rl_policy_gradient", "search_adversarial"]
    })
    
    # ============================================================================
    # PROBABILISTIC REASONING (8 items)
    # ============================================================================
    
    items.append({
        "id": "prob_bayes",
        "book_id": "probabilistic",
        "level": "basics",
        "title": "Bayesian Reasoning",
        "learn": "Bayes' rule: P(H|E) = P(E|H)P(H)/P(E). Prior P(H), likelihood P(E|H), posterior P(H|E). Reasoning under uncertainty. Applications: diagnosis, spam filtering, robotics.",
        "try_code": """# Medical diagnosis example
def bayes_rule(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

# Disease test
P_disease = 0.01  # Prior: 1% have disease
P_pos_given_disease = 0.95  # Sensitivity
P_pos_given_healthy = 0.05  # False positive rate

P_pos = P_pos_given_disease * P_disease + P_pos_given_healthy * (1 - P_disease)

P_disease_given_pos = bayes_rule(P_disease, P_pos_given_disease, P_pos)
print(f"P(disease|positive test) = {P_disease_given_pos:.3f}")  # ~0.16""",
        "try_demo": None,
        "prerequisites": []
    })
    
    items.append({
        "id": "prob_independence",
        "book_id": "probabilistic",
        "level": "basics",
        "title": "Independence & Conditional Independence",
        "learn": "Independence: P(X,Y) = P(X)P(Y). Conditional independence: P(X,Y|Z) = P(X|Z)P(Y|Z). Key for compact representations. Naive Bayes assumes features independent given class. Enables efficient inference.",
        "try_code": """# Naive Bayes classifier
import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_prior = {c: (y == c).mean() for c in self.classes}
        self.feature_probs = {}
        
        for c in self.classes:
            X_c = X[y == c]
            # Assume Gaussian features
            self.feature_probs[c] = {
                'mean': X_c.mean(axis=0),
                'std': X_c.std(axis=0) + 1e-6
            }
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                prior = np.log(self.class_prior[c])
                # Product of likelihoods (sum in log space)
                likelihood = -0.5 * np.sum(
                    ((x - self.feature_probs[c]['mean']) / 
                     self.feature_probs[c]['std'])**2)
                posteriors[c] = prior + likelihood
            predictions.append(max(posteriors, key=posteriors.get))
        return predictions""",
        "try_demo": None,
        "prerequisites": ["prob_bayes"]
    })
    
    items.append({
        "id": "prob_reasoning",
        "book_id": "probabilistic",
        "level": "intermediate",
        "title": "Bayesian Networks",
        "learn": "Directed acyclic graph (DAG) encoding conditional independencies. Nodes: random variables. Edges: dependencies. Joint: P(X1,...,Xn) = ∏P(Xi|Parents(Xi)). Compact representation. Inference via belief propagation, variable elimination.",
        "try_code": """# Bayesian Network structure
class BayesianNetwork:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
    
    def add_node(self, name, parents, cpt):
        \"\"\"Add node with conditional probability table.\"\"\"
        self.nodes[name] = {'parents': parents, 'cpt': cpt}
        for parent in parents:
            if parent not in self.edges:
                self.edges[parent] = []
            self.edges[parent].append(name)
    
    def query(self, var, evidence={}):
        \"\"\"Simple inference via enumeration.\"\"\"
        # Simplified: in practice use variable elimination or belief prop
        pass

# Example: Alarm network
# Burglary → Alarm ← Earthquake
#             ↓
#           Call
bn = BayesianNetwork()
bn.add_node('Burglary', [], {'T': 0.001})
bn.add_node('Earthquake', [], {'T': 0.002})
bn.add_node('Alarm', ['Burglary', 'Earthquake'], {
    (True, True): 0.95, (True, False): 0.94,
    (False, True): 0.29, (False, False): 0.001
})
bn.add_node('Call', ['Alarm'], {True: 0.90, False: 0.05})""",
        "try_demo": None,
        "prerequisites": ["prob_independence"]
    })
    
    items.append({
        "id": "prob_hmm",
        "book_id": "probabilistic",
        "level": "intermediate",
        "title": "Hidden Markov Models",
        "learn": "Temporal probabilistic model. Hidden states X_t, observations Y_t. Transition P(X_t|X_{t-1}), emission P(Y_t|X_t). Forward algorithm: filtering P(X_t|y_1:t). Viterbi: most likely sequence. Baum-Welch: parameter learning.",
        "try_code": """import numpy as np

class HMM:
    def __init__(self, n_states, n_obs):
        self.n_states = n_states
        self.n_obs = n_obs
        self.transition = np.random.rand(n_states, n_states)
        self.transition /= self.transition.sum(axis=1, keepdims=True)
        self.emission = np.random.rand(n_states, n_obs)
        self.emission /= self.emission.sum(axis=1, keepdims=True)
        self.initial = np.ones(n_states) / n_states
    
    def forward(self, observations):
        \"\"\"Forward algorithm for filtering.\"\"\"
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        # Initialize
        alpha[0] = self.initial * self.emission[:, observations[0]]
        alpha[0] /= alpha[0].sum()
        
        # Recurse
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.transition) * self.emission[:, observations[t]]
            alpha[t] /= alpha[t].sum()
        
        return alpha
    
    def viterbi(self, observations):
        \"\"\"Most likely state sequence.\"\"\"
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Initialize
        delta[0] = self.initial * self.emission[:, observations[0]]
        
        # Recurse
        for t in range(1, T):
            for s in range(self.n_states):
                probs = delta[t-1] * self.transition[:, s]
                psi[t, s] = np.argmax(probs)
                delta[t, s] = np.max(probs) * self.emission[s, observations[t]]
        
        # Backtrack
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states

# Example: Weather model
hmm = HMM(n_states=2, n_obs=3)  # Sunny/Rainy, Walk/Shop/Clean
observations = [0, 1, 2, 1, 0]  # Observation sequence
states = hmm.viterbi(observations)
print(f"Most likely states: {states}")""",
        "try_demo": None,
        "prerequisites": ["prob_reasoning"]
    })
    
    items.append({
        "id": "prob_particle_filter",
        "book_id": "probabilistic",
        "level": "advanced",
        "title": "Particle Filters",
        "learn": "Sequential Monte Carlo for non-linear/non-Gaussian systems. Represent belief as particles (samples). Propagate via motion model, weight by observation likelihood, resample. Effective for robot localization, tracking.",
        "try_code": """import numpy as np

class ParticleFilter:
    def __init__(self, n_particles, state_dim):
        self.n_particles = n_particles
        self.particles = np.random.randn(n_particles, state_dim)
        self.weights = np.ones(n_particles) / n_particles
    
    def predict(self, motion_model, control):
        \"\"\"Propagate particles via motion model.\"\"\"
        for i in range(self.n_particles):
            self.particles[i] = motion_model(self.particles[i], control)
            # Add noise
            self.particles[i] += np.random.randn(*self.particles[i].shape) * 0.1
    
    def update(self, observation, sensor_model):
        \"\"\"Reweight particles based on observation likelihood.\"\"\"
        for i in range(self.n_particles):
            # Likelihood of observation given particle state
            self.weights[i] = sensor_model(self.particles[i], observation)
        
        # Normalize
        self.weights /= self.weights.sum()
    
    def resample(self):
        \"\"\"Resample particles proportional to weights.\"\"\"
        indices = np.random.choice(self.n_particles, self.n_particles, 
                                  p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def estimate(self):
        \"\"\"Weighted mean estimate.\"\"\"
        return np.average(self.particles, weights=self.weights, axis=0)

# Example: Robot localization
def motion_model(state, control):
    # state = [x, y, theta], control = [v, w] (velocity, angular)
    x, y, theta = state
    v, w = control
    dt = 0.1
    return np.array([
        x + v * np.cos(theta) * dt,
        y + v * np.sin(theta) * dt,
        theta + w * dt
    ])

def sensor_model(state, observation):
    # Gaussian likelihood
    distance = np.linalg.norm(state[:2] - observation)
    return np.exp(-0.5 * distance**2 / 1.0)

pf = ParticleFilter(n_particles=1000, state_dim=3)
# ... prediction/update loop""",
        "try_demo": None,
        "prerequisites": ["prob_hmm"]
    })
    
    items.append({
        "id": "prob_em",
        "book_id": "probabilistic",
        "level": "advanced",
        "title": "Expectation-Maximization",
        "learn": "Unsupervised learning for latent variable models. E-step: compute expected sufficient statistics given current parameters. M-step: optimize parameters. Guaranteed to increase likelihood. Applications: GMM, HMM training, missing data.",
        "try_code": """import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, n_components, n_features):
        self.K = n_components
        self.pi = np.ones(n_components) / n_components
        self.mu = np.random.randn(n_components, n_features)
        self.sigma = np.array([np.eye(n_features) for _ in range(n_components)])
    
    def e_step(self, X):
        \"\"\"Compute responsibilities (posterior over latent).\"\"\"
        N = len(X)
        gamma = np.zeros((N, self.K))
        
        for k in range(self.K):
            gamma[:, k] = self.pi[k] * multivariate_normal.pdf(
                X, mean=self.mu[k], cov=self.sigma[k])
        
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma
    
    def m_step(self, X, gamma):
        \"\"\"Maximize expected log-likelihood.\"\"\"
        N = len(X)
        N_k = gamma.sum(axis=0)
        
        # Update parameters
        self.pi = N_k / N
        self.mu = (gamma.T @ X) / N_k[:, np.newaxis]
        
        for k in range(self.K):
            diff = X - self.mu[k]
            self.sigma[k] = (gamma[:, k] * diff.T @ diff) / N_k[k]
    
    def fit(self, X, max_iter=100, tol=1e-4):
        for i in range(max_iter):
            gamma = self.e_step(X)
            self.m_step(X, gamma)
            # Check convergence via log-likelihood
        return self

# Usage
gmm = GMM(n_components=3, n_features=2)
X = np.random.randn(100, 2)
gmm.fit(X)
print(f"Cluster means: {gmm.mu}")""",
        "try_demo": None,
        "prerequisites": ["prob_reasoning"]
    })
    
    items.append({
        "id": "prob_dbns",
        "book_id": "probabilistic",
        "level": "advanced",
        "title": "Dynamic Bayesian Networks",
        "learn": "Temporal extension of Bayesian networks. Template for time-slice connections. HMMs are simple DBNs. Kalman filters: linear-Gaussian DBNs. Inference: forward filtering, smoothing (forward-backward), MPE (Viterbi).",
        "try_code": """# Dynamic Bayesian Network structure
class DBN:
    def __init__(self, intra_slice, inter_slice):
        \"\"\"
        intra_slice: edges within time slice
        inter_slice: edges between consecutive slices
        \"\"\"
        self.intra = intra_slice  # Structure at time t
        self.inter = inter_slice  # X_t → X_{t+1}
    
    def forward_filter(self, observations, T):
        \"\"\"Online inference: P(X_t | obs_1:t).\"\"\"
        beliefs = []
        belief = self.prior()  # P(X_0)
        
        for t in range(T):
            # Predict: P(X_t | obs_1:t-1) using inter-slice
            predicted = self.predict(belief)
            
            # Update: P(X_t | obs_1:t) using intra-slice
            belief = self.update(predicted, observations[t])
            beliefs.append(belief)
        
        return beliefs
    
    def smooth(self, observations):
        \"\"\"Offline inference: P(X_t | obs_1:T).\"\"\"
        # Forward pass
        forward = self.forward_filter(observations, len(observations))
        
        # Backward pass
        backward = []
        # ... similar to forward but in reverse
        
        # Combine: smoothed = forward * backward (normalized)
        return forward  # Simplified

# Example: Robot SLAM
# States: robot pose + landmark positions
# Observations: landmark measurements""",
        "try_demo": None,
        "prerequisites": ["prob_hmm", "prob_reasoning"]
    })
    
    items.append({
        "id": "clustering_ml",
        "book_id": "probabilistic",
        "level": "basics",
        "title": "Clustering Algorithms",
        "learn": "Unsupervised grouping. K-means: iterative centroid assignment. Hierarchical: agglomerative (bottom-up) or divisive (top-down). DBSCAN: density-based. Evaluation: silhouette score, within-cluster variance.",
        "try_code": """import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100):
        self.K = n_clusters
        self.max_iter = max_iter
    
    def fit(self, X):
        # Initialize centroids randomly
        idx = np.random.choice(len(X), self.K, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iter):
            # Assign points to nearest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([X[labels == k].mean(axis=0) 
                                     for k in range(self.K)])
            
            # Check convergence
            if np.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids
        
        self.labels_ = labels
        return self
    
    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

# Hierarchical clustering
def agglomerative_clustering(X, n_clusters):
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(X, method='ward')  # Minimize within-cluster variance
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    return labels

# DBSCAN
def dbscan(X, eps=0.5, min_samples=5):
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    return clustering.labels_  # -1 for noise points""",
        "try_demo": "ai_clustering",
        "prerequisites": ["prob_independence"]
    })
    
    return items


def main():
    """Generate and save enriched curriculum."""
    items = generate_curriculum_items()
    
    # Save to cache
    cache_dir = Path(__file__).parent.parent / '.cache'
    cache_dir.mkdir(exist_ok=True)
    
    output_file = cache_dir / 'ai_concepts_enriched.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=2)
    
    # Statistics
    by_level = {}
    by_book = {}
    for item in items:
        by_level[item['level']] = by_level.get(item['level'], 0) + 1
        by_book[item['book_id']] = by_book.get(item['book_id'], 0) + 1
    
    print(f"\n{'='*70}")
    print(f"✅ Generated {len(items)} curriculum items for AI Concepts Lab")
    print(f"{'='*70}")
    print(f"\nBy Level:")
    for level in ['basics', 'intermediate', 'advanced', 'expert']:
        print(f"  {level}: {by_level.get(level, 0)}")
    
    print(f"\nBy Topic:")
    for book_id, count in by_book.items():
        print(f"  {book_id}: {count}")
    
    print(f"\nSaved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
