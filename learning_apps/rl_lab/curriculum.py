"""
Curriculum: Reinforcement Learning — Sutton & Barto, "Reinforcement Learning: An Introduction".
MDPs, value functions, Bellman equations, TD, Q-learning, policy gradient. Uses ml_toolbox.ai_concepts.reinforcement_learning where available.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "sutton", "name": "Sutton & Barto", "short": "Sutton", "color": "#2563eb"},
    {"id": "mdp", "name": "MDPs & Value Functions", "short": "MDP & V", "color": "#059669"},
    {"id": "td", "name": "Temporal-Difference Learning", "short": "TD", "color": "#7c3aed"},
    {"id": "control", "name": "Control & Q-Learning", "short": "Control", "color": "#d97706"},
    {"id": "policy", "name": "Policy Gradient & Beyond", "short": "Policy", "color": "#dc2626"},
]

CURRICULUM: List[Dict[str, Any]] = [
    # Fundamentals
    {"id": "rl_mdp", "book_id": "sutton", "level": "basics", "title": "Markov Decision Processes (MDPs)",
     "learn": "MDP = (S, A, P, R, γ): states S, actions A, transition probabilities P(s'|s,a), rewards R(s,a), discount γ. Goal: find policy π that maximizes expected return. Foundation of RL.",
     "try_code": "# Simple MDP example\nstates = ['s0', 's1', 's2']\nactions = ['left', 'right']\ntransitions = {('s0', 'right'): [('s1', 0.8), ('s0', 0.2)]}\nrewards = {('s0', 'right'): 1.0}\ngamma = 0.9",
     "try_demo": "rl_mdp"},
    {"id": "rl_policy", "book_id": "sutton", "level": "basics", "title": "Policies and Returns",
     "learn": "Policy π(a|s): probability of action a in state s. Deterministic: π(s) = a. Stochastic: π(a|s) ∈ [0,1]. Return G_t = Σ γ^k * r_{t+k+1}. Goal: maximize expected return.",
     "try_code": "# Policy representations\nimport numpy as np\n\n# Deterministic policy\ndetpolicy = {0: 'right', 1: 'left', 2: 'right'}\n\n# Stochastic policy (softmax)\ndef softmax_policy(Q, state, temp=1.0):\n    exp_Q = np.exp(Q[state] / temp)\n    return exp_Q / exp_Q.sum()",
     "try_demo": "rl_policy"},
    {"id": "rl_value_function", "book_id": "sutton", "level": "basics", "title": "Value Functions",
     "learn": "State value: V(s) = E[Σ γ^t * r_t | s_0=s]. Action value: Q(s,a) = E[Σ γ^t * r_t | s_0=s, a_0=a]. Bellman equation: V(s) = Σ_a π(a|s) * Σ_s' P(s'|s,a) * [R + γ*V(s')].",
     "try_code": "import numpy as np\n\n# Value function for simple MDP\nV = np.zeros(n_states)\nfor s in range(n_states):\n    V[s] = sum(policy[s, a] * (reward[s, a] + gamma * sum(\n        transition[s, a, s_next] * V[s_next] for s_next in range(n_states)\n    )) for a in range(n_actions))",
     "try_demo": "rl_value"},
    {"id": "rl_bellman", "book_id": "sutton", "level": "intermediate", "title": "Bellman Equations",
     "learn": "Recursive decomposition of value: V(s) = max_a [R(s,a) + γ * Σ P(s'|s,a) * V(s')]. Optimal value V*(s), optimal policy π*(s) = argmax_a Q*(s,a). Basis for dynamic programming.",
     "try_code": "# Bellman optimality operator\ndef bellman_operator(V, transitions, rewards, gamma):\n    V_new = np.zeros_like(V)\n    for s in range(len(V)):\n        V_new[s] = max(\n            rewards[s, a] + gamma * sum(\n                transitions[s, a, s_next] * V[s_next]\n                for s_next in range(len(V))\n            )\n            for a in range(n_actions)\n        )\n    return V_new",
     "try_demo": None},
    
    # Dynamic Programming
    {"id": "rl_policy_iteration", "book_id": "sutton", "level": "intermediate", "title": "Policy Iteration",
     "learn": "Iterate: (1) Policy Evaluation: compute V^π. (2) Policy Improvement: π'(s) = argmax_a Q^π(s,a). Converges to optimal policy π*. Requires model (transitions, rewards).",
     "try_code": "def policy_iteration(transitions, rewards, gamma, threshold=1e-6):\n    policy = np.random.choice(n_actions, n_states)\n    V = np.zeros(n_states)\n    \n    while True:\n        # Policy evaluation\n        while True:\n            delta = 0\n            for s in range(n_states):\n                v = V[s]\n                V[s] = compute_value(s, policy[s], V, transitions, rewards, gamma)\n                delta = max(delta, abs(v - V[s]))\n            if delta < threshold:\n                break\n        \n        # Policy improvement\n        policy_stable = True\n        for s in range(n_states):\n            old_action = policy[s]\n            policy[s] = argmax_action(s, V, transitions, rewards, gamma)\n            if old_action != policy[s]:\n                policy_stable = False\n        \n        if policy_stable:\n            break\n    \n    return policy, V",
     "try_demo": "rl_policy_iteration"},
    {"id": "rl_value_iteration", "book_id": "sutton", "level": "intermediate", "title": "Value Iteration",
     "learn": "Combine evaluation and improvement: V_{k+1}(s) = max_a [R(s,a) + γ * Σ P(s'|s,a) * V_k(s')]. Converges to V*. Extract policy: π(s) = argmax_a Q(s,a). Faster than policy iteration.",
     "try_code": "def value_iteration(transitions, rewards, gamma, threshold=1e-6):\n    V = np.zeros(n_states)\n    \n    while True:\n        delta = 0\n        for s in range(n_states):\n            v = V[s]\n            # Bellman optimality update\n            V[s] = max(\n                rewards[s, a] + gamma * sum(\n                    transitions[s, a, s_next] * V[s_next]\n                    for s_next in range(n_states)\n                )\n                for a in range(n_actions)\n            )\n            delta = max(delta, abs(v - V[s]))\n        \n        if delta < threshold:\n            break\n    \n    # Extract policy\n    policy = np.array([argmax_action(s, V, transitions, rewards, gamma) \n                       for s in range(n_states)])\n    return policy, V",
     "try_demo": "rl_value_iteration"},
    
    # Model-Free Methods
    {"id": "rl_monte_carlo", "book_id": "sutton", "level": "intermediate", "title": "Monte Carlo Methods",
     "learn": "Learn from complete episodes. Update V(s) using actual returns: V(s) ← V(s) + α * [G_t - V(s)]. First-visit vs every-visit. Model-free: no transition probabilities needed.",
     "try_code": "def monte_carlo_prediction(episodes, alpha=0.1, gamma=0.9):\n    V = {}\n    \n    for episode in episodes:\n        # episode = [(s0, a0, r0), (s1, a1, r1), ...]\n        G = 0\n        for t in reversed(range(len(episode))):\n            state, action, reward = episode[t]\n            G = gamma * G + reward\n            \n            if state not in [s for s, _, _ in episode[:t]]:\n                # First-visit\n                if state not in V:\n                    V[state] = 0\n                V[state] += alpha * (G - V[state])\n    \n    return V",
     "try_demo": "rl_monte_carlo"},
    {"id": "rl_q_learning", "book_id": "sutton", "level": "intermediate", "title": "Q-Learning (Off-Policy TD)",
     "learn": "Learn Q(s,a) using: Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]. Off-policy: learn optimal Q* while following ε-greedy. Converges to Q* with appropriate α decay.",
     "try_code": "class QLearning:\n    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):\n        self.Q = np.zeros((n_states, n_actions))\n        self.alpha = alpha\n        self.gamma = gamma\n        self.epsilon = epsilon\n    \n    def get_action(self, state):\n        if np.random.random() < self.epsilon:\n            return np.random.randint(len(self.Q[state]))\n        return np.argmax(self.Q[state])\n    \n    def update(self, state, action, reward, next_state):\n        # Q-learning update\n        td_target = reward + self.gamma * np.max(self.Q[next_state])\n        td_error = td_target - self.Q[state, action]\n        self.Q[state, action] += self.alpha * td_error",
     "try_demo": "rl_q_learning"},
    {"id": "rl_sarsa", "book_id": "sutton", "level": "intermediate", "title": "SARSA (On-Policy TD)",
     "learn": "On-policy TD: Q(s,a) ← Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]. Uses actual next action a' (not max). Learns value of policy being followed. More conservative than Q-learning.",
     "try_code": "class SARSA:\n    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):\n        self.Q = np.zeros((n_states, n_actions))\n        self.alpha = alpha\n        self.gamma = gamma\n        self.epsilon = epsilon\n    \n    def get_action(self, state):\n        if np.random.random() < self.epsilon:\n            return np.random.randint(len(self.Q[state]))\n        return np.argmax(self.Q[state])\n    \n    def update(self, state, action, reward, next_state, next_action):\n        # SARSA update (uses next_action, not max)\n        td_target = reward + self.gamma * self.Q[next_state, next_action]\n        td_error = td_target - self.Q[state, action]\n        self.Q[state, action] += self.alpha * td_error",
     "try_demo": "rl_sarsa"},
    {"id": "rl_exploration", "book_id": "sutton", "level": "intermediate", "title": "Exploration Strategies",
     "learn": "Exploration-exploitation tradeoff. ε-greedy: random action with prob ε. Boltzmann: π(a|s) ∝ exp(Q(s,a)/τ). UCB: choose a = argmax[Q(s,a) + c*√(log t / N(s,a))]. Optimistic initialization.",
     "try_code": "# Exploration strategies\nimport numpy as np\n\ndef epsilon_greedy(Q, state, epsilon=0.1):\n    if np.random.random() < epsilon:\n        return np.random.randint(len(Q[state]))\n    return np.argmax(Q[state])\n\ndef boltzmann(Q, state, temperature=1.0):\n    exp_Q = np.exp(Q[state] / temperature)\n    probs = exp_Q / exp_Q.sum()\n    return np.random.choice(len(Q[state]), p=probs)\n\ndef ucb(Q, state, counts, total_steps, c=2.0):\n    ucb_values = Q[state] + c * np.sqrt(np.log(total_steps) / (counts[state] + 1e-5))\n    return np.argmax(ucb_values)",
     "try_demo": "rl_exploration"},
    
    # Advanced: TD(λ) and Deep RL
    {"id": "rl_td_lambda", "book_id": "sutton", "level": "advanced", "title": "TD(λ) and Eligibility Traces",
     "learn": "Bridge between TD and MC: λ=0 is TD(0), λ=1 is MC. Eligibility trace e(s,a) tracks recent visits. Update: Q(s,a) ← Q(s,a) + α * δ * e(s,a). Faster learning, credit assignment.",
     "try_code": "class TDLambda:\n    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, lambda_=0.8):\n        self.Q = np.zeros((n_states, n_actions))\n        self.e = np.zeros((n_states, n_actions))  # Eligibility traces\n        self.alpha = alpha\n        self.gamma = gamma\n        self.lambda_ = lambda_\n    \n    def update(self, state, action, reward, next_state, next_action):\n        # TD error\n        delta = reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action]\n        \n        # Update eligibility trace\n        self.e[state, action] += 1  # Accumulating traces\n        \n        # Update Q for all states\n        self.Q += self.alpha * delta * self.e\n        \n        # Decay eligibility traces\n        self.e *= self.gamma * self.lambda_",
     "try_demo": None},
    {"id": "rl_dqn", "book_id": "sutton", "level": "advanced", "title": "Deep Q-Networks (DQN)",
     "learn": "Approximate Q(s,a) with neural network. Experience replay: store (s,a,r,s') in buffer, sample mini-batches. Target network: Q_target updated slowly. Stabilizes training. Breakthrough for Atari games.",
     "try_code": "import torch\nimport torch.nn as nn\n\nclass DQN(nn.Module):\n    def __init__(self, state_dim, action_dim, hidden=128):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(state_dim, hidden),\n            nn.ReLU(),\n            nn.Linear(hidden, hidden),\n            nn.ReLU(),\n            nn.Linear(hidden, action_dim)\n        )\n    \n    def forward(self, state):\n        return self.net(state)\n\n# DQN update\ndef dqn_update(q_net, target_net, batch, optimizer, gamma=0.99):\n    states, actions, rewards, next_states, dones = batch\n    \n    # Current Q values\n    q_values = q_net(states).gather(1, actions)\n    \n    # Target Q values\n    with torch.no_grad():\n        next_q_values = target_net(next_states).max(1)[0]\n        targets = rewards + gamma * next_q_values * (1 - dones)\n    \n    loss = nn.MSELoss()(q_values, targets.unsqueeze(1))\n    optimizer.zero_grad()\n    loss.backward()\n    optimizer.step()",
     "try_demo": "rl_dqn"},
    {"id": "rl_policy_gradient", "book_id": "sutton", "level": "advanced", "title": "Policy Gradient Methods",
     "learn": "Directly optimize policy π_θ(a|s). REINFORCE: ∇_θ J(θ) = E[∇_θ log π_θ(a|s) * G_t]. Actor-only, high variance. Use baseline to reduce variance: G_t - b(s).",
     "try_code": "import torch\nimport torch.nn as nn\n\nclass PolicyNetwork(nn.Module):\n    def __init__(self, state_dim, action_dim, hidden=128):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(state_dim, hidden),\n            nn.ReLU(),\n            nn.Linear(hidden, action_dim),\n            nn.Softmax(dim=-1)\n        )\n    \n    def forward(self, state):\n        return self.net(state)\n\n# REINFORCE update\ndef reinforce_update(policy, episode, optimizer, gamma=0.99):\n    states, actions, rewards = zip(*episode)\n    \n    # Compute returns\n    returns = []\n    G = 0\n    for r in reversed(rewards):\n        G = r + gamma * G\n        returns.insert(0, G)\n    returns = torch.tensor(returns)\n    \n    # Policy gradient\n    loss = 0\n    for state, action, G in zip(states, actions, returns):\n        probs = policy(state)\n        loss -= torch.log(probs[action]) * G\n    \n    optimizer.zero_grad()\n    loss.backward()\n    optimizer.step()",
     "try_demo": "rl_policy_gradient"},
    {"id": "rl_actor_critic", "book_id": "sutton", "level": "advanced", "title": "Actor-Critic Methods",
     "learn": "Actor: policy π_θ(a|s). Critic: value V_w(s) or Q_w(s,a). TD error δ = r + γ*V(s') - V(s) guides both. Reduces variance vs REINFORCE. A3C, A2C, PPO are variants.",
     "try_code": "class ActorCritic(nn.Module):\n    def __init__(self, state_dim, action_dim, hidden=128):\n        super().__init__()\n        self.shared = nn.Sequential(\n            nn.Linear(state_dim, hidden),\n            nn.ReLU()\n        )\n        self.actor = nn.Sequential(\n            nn.Linear(hidden, action_dim),\n            nn.Softmax(dim=-1)\n        )\n        self.critic = nn.Linear(hidden, 1)\n    \n    def forward(self, state):\n        features = self.shared(state)\n        policy = self.actor(features)\n        value = self.critic(features)\n        return policy, value\n\ndef actor_critic_update(model, state, action, reward, next_state, done, optimizer, gamma=0.99):\n    policy, value = model(state)\n    _, next_value = model(next_state)\n    \n    # TD error\n    td_target = reward + gamma * next_value * (1 - done)\n    td_error = td_target - value\n    \n    # Actor loss (policy gradient)\n    actor_loss = -torch.log(policy[action]) * td_error.detach()\n    \n    # Critic loss (value function)\n    critic_loss = td_error.pow(2)\n    \n    loss = actor_loss + critic_loss\n    optimizer.zero_grad()\n    loss.backward()\n    optimizer.step()",
     "try_demo": "rl_actor_critic"},
    {"id": "rl_reward_shaping", "book_id": "sutton", "level": "advanced", "title": "Reward Shaping",
     "learn": "Add potential-based rewards to guide learning: r'(s,a,s') = r(s,a,s') + γ*Φ(s') - Φ(s). Preserves optimal policy if Φ is potential function. Speeds up learning without changing solution.",
     "try_code": "# Reward shaping example\ndef potential_based_shaping(state, action, next_state, reward, gamma=0.9):\n    # Define potential function (e.g., distance to goal)\n    def potential(s):\n        return -distance_to_goal(s)  # Negative distance\n    \n    # Shaped reward\n    shaped_reward = reward + gamma * potential(next_state) - potential(state)\n    return shaped_reward",
     "try_demo": None},
    
    # Expert
    {"id": "rl_ppo", "book_id": "sutton", "level": "expert", "title": "Proximal Policy Optimization (PPO)",
     "learn": "Constrain policy updates to trust region. Clipped objective: L^CLIP(θ) = min(r_t(θ)*A_t, clip(r_t(θ), 1-ε, 1+ε)*A_t). Stable, sample-efficient. State-of-the-art for many tasks.",
     "try_code": "def ppo_update(policy, old_policy, states, actions, advantages, optimizer, epsilon=0.2):\n    # Importance sampling ratio\n    probs = policy(states).gather(1, actions)\n    old_probs = old_policy(states).gather(1, actions).detach()\n    ratio = probs / old_probs\n    \n    # Clipped surrogate objective\n    surr1 = ratio * advantages\n    surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages\n    loss = -torch.min(surr1, surr2).mean()\n    \n    optimizer.zero_grad()\n    loss.backward()\n    optimizer.step()",
     "try_demo": None},
    {"id": "rl_model_based", "book_id": "sutton", "level": "expert", "title": "Model-Based RL",
     "learn": "Learn model of environment: P(s'|s,a), R(s,a). Plan using model (Dyna-Q, MCTS). Simulate experience for faster learning. Trade-off: model accuracy vs sample efficiency.",
     "try_code": "class DynaQ:\n    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, planning_steps=10):\n        self.Q = np.zeros((n_states, n_actions))\n        self.model = {}  # (s,a) -> (r, s')\n        self.planning_steps = planning_steps\n        self.alpha = alpha\n        self.gamma = gamma\n    \n    def update(self, state, action, reward, next_state):\n        # Q-learning update (direct RL)\n        td_target = reward + self.gamma * np.max(self.Q[next_state])\n        self.Q[state, action] += self.alpha * (td_target - self.Q[state, action])\n        \n        # Update model\n        self.model[(state, action)] = (reward, next_state)\n        \n        # Planning: sample from model\n        for _ in range(self.planning_steps):\n            s, a = random.choice(list(self.model.keys()))\n            r, s_next = self.model[(s, a)]\n            td_target = r + self.gamma * np.max(self.Q[s_next])\n            self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])",
     "try_demo": None},
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
