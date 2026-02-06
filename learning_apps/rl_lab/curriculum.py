"""
Curriculum: Reinforcement Learning — Sutton & Barto, "Reinforcement Learning: An Introduction".
MDPs, value functions, Bellman equations, TD, Q-learning, policy gradient. Uses ml_toolbox.ai_concepts.reinforcement_learning where available.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "mdp", "name": "MDPs & Value Functions", "short": "MDP & V", "color": "#2563eb"},
    {"id": "td", "name": "Temporal-Difference Learning", "short": "TD", "color": "#059669"},
    {"id": "control", "name": "Control & Q-Learning", "short": "Control", "color": "#7c3aed"},
    {"id": "policy", "name": "Policy Gradient & Beyond", "short": "Policy", "color": "#dc2626"},
]

CURRICULUM: List[Dict[str, Any]] = [
    {"id": "rl_mdp", "book_id": "mdp", "level": "basics", "title": "Markov Decision Processes",
     "learn": "MDP: S, A, P, R, γ. States, actions, transition probabilities, rewards, discount. Policy π(a|s).",
     "try_code": "from ml_toolbox.ai_concepts.reinforcement_learning import ...  # MDP helpers if available",
     "try_demo": None},
    {"id": "rl_value", "book_id": "mdp", "level": "intermediate", "title": "Value Functions V(s) and Q(s,a)",
     "learn": "V^π(s) = E[sum γ^t R_t]. Q^π(s,a) = E[R + γ V(s')]. Bellman equation for V and Q.",
     "try_code": "from ml_toolbox.ai_concepts.reinforcement_learning import ...",
     "try_demo": "rl_value_iteration"},
    {"id": "rl_bellman", "book_id": "mdp", "level": "intermediate", "title": "Bellman Optimality",
     "learn": "V*(s) = max_a Q*(s,a). Q*(s,a) = R + γ sum P(s'|s,a) V*(s'). Optimal policy from Q*.",
     "try_code": "from ml_toolbox.ai_concepts.reinforcement_learning import ...",
     "try_demo": "rl_value_iteration"},
    {"id": "rl_td", "book_id": "td", "level": "intermediate", "title": "Temporal-Difference Learning",
     "learn": "TD(0): V(s) ← V(s) + α[R + γV(s') - V(s)]. Bootstrapping: update from successor estimate.",
     "try_code": "from ml_toolbox.ai_concepts.reinforcement_learning import ...",
     "try_demo": "rl_td0"},
    {"id": "rl_qlearning", "book_id": "control", "level": "intermediate", "title": "Q-Learning (off-policy TD control)",
     "learn": "Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]. Learns optimal Q without following optimal policy.",
     "try_code": "from ml_toolbox.ai_concepts.reinforcement_learning import QLearning",
     "try_demo": "rl_qlearning"},
    {"id": "rl_sarsa", "book_id": "control", "level": "advanced", "title": "SARSA (on-policy TD control)",
     "learn": "Q(s,a) ← Q(s,a) + α[R + γ Q(s',a') - Q(s,a)]. a' from current policy. On-policy.",
     "try_code": "from ml_toolbox.ai_concepts.reinforcement_learning import ...",
     "try_demo": "rl_sarsa"},
    {"id": "rl_policy_gradient", "book_id": "policy", "level": "advanced", "title": "Policy Gradient",
     "learn": "∇J(θ) ∝ E[∇log π(a|s;θ) G_t]. REINFORCE and actor-critic. Sutton & Barto Ch. 13.",
     "try_code": "from ml_toolbox.ai_concepts.reinforcement_learning import ...",
     "try_demo": "rl_policy_gradient"},
    {"id": "rl_dqn", "book_id": "policy", "level": "expert", "title": "Deep Q-Networks (DQN)",
     "learn": "Q(s,a) approximated by neural network. Experience replay, target network. Extension of Sutton & Barto.",
     "try_code": "# DQN typically via external lib (e.g. stable-baselines3)",
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
