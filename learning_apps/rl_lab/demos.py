"""Demos for RL Lab. Uses ml_toolbox.ai_concepts.reinforcement_learning."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np


def rl_qlearning():
    """Q-Learning demo on a simple gridworld."""
    try:
        from ml_toolbox.ai_concepts.reinforcement_learning import QLearning
        # Simple MDP: 4 states (0=start, 3=goal), 2 actions (0=left, 1=right)
        ql = QLearning(n_states=4, n_actions=2, learning_rate=0.1, discount_factor=0.9, epsilon=0.1)
        
        # Simulate episodes
        for _ in range(100):
            state = 0
            for _ in range(10):
                action = ql.select_action(state)
                # Transition logic
                if action == 1:  # right
                    next_state = min(state + 1, 3)
                else:  # left
                    next_state = max(state - 1, 0)
                reward = 10 if next_state == 3 else -0.1
                done = next_state == 3
                ql.update(state, action, reward, next_state, done)
                if done:
                    break
                state = next_state
        
        out = "Q-Learning on 4-state gridworld (100 episodes)\n"
        out += "States: 0=start, 3=goal; Actions: 0=left, 1=right\n"
        out += f"Q-table shape: {ql.Q.shape}\n"
        out += "Q-values:\n"
        for s in range(4):
            out += f"  State {s}: left={ql.Q[s,0]:.2f}, right={ql.Q[s,1]:.2f}\n"
        out += "\nOptimal actions: " + str([np.argmax(ql.Q[s]) for s in range(4)])
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def rl_value_iteration():
    """Value iteration demo."""
    try:
        # Simple value iteration
        n_states = 4
        gamma = 0.9
        V = np.zeros(n_states)
        rewards = np.array([-1, -1, -1, 10])  # Goal at state 3
        
        for _ in range(50):
            V_new = np.zeros(n_states)
            for s in range(n_states):
                if s == 3:  # Terminal
                    V_new[s] = rewards[s]
                else:
                    left = max(s - 1, 0)
                    right = min(s + 1, 3)
                    V_new[s] = rewards[s] + gamma * max(V[left], V[right])
            V = V_new
        
        out = "Value Iteration (4 states, γ=0.9)\n"
        out += "Rewards: [-1, -1, -1, 10] (goal at state 3)\n"
        out += f"Converged V(s): {V.round(2).tolist()}\n"
        out += "Interpretation: Higher value = closer to goal"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def rl_td0():
    """TD(0) value estimation demo."""
    try:
        n_states = 4
        V = np.zeros(n_states)
        alpha = 0.1
        gamma = 0.9
        
        # Simulate random walks
        np.random.seed(42)
        for _ in range(200):
            state = 0
            for _ in range(20):
                action = np.random.choice([0, 1])  # Random policy
                next_state = max(0, min(3, state + (1 if action else -1)))
                reward = 10 if next_state == 3 else -0.1
                # TD(0) update
                V[state] = V[state] + alpha * (reward + gamma * V[next_state] - V[state])
                if next_state == 3:
                    break
                state = next_state
        
        out = "TD(0) Value Estimation (random policy)\n"
        out += "4 states, α=0.1, γ=0.9, 200 episodes\n"
        out += f"Learned V(s): {V.round(2).tolist()}\n"
        out += "Values increase toward goal (state 3)"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def rl_sarsa():
    """SARSA (on-policy TD control) demo."""
    try:
        n_states, n_actions = 4, 2
        Q = np.zeros((n_states, n_actions))
        alpha, gamma, epsilon = 0.1, 0.9, 0.1
        
        def eps_greedy(s):
            if np.random.random() < epsilon:
                return np.random.randint(n_actions)
            return np.argmax(Q[s])
        
        np.random.seed(42)
        for _ in range(200):
            s = 0
            a = eps_greedy(s)
            for _ in range(20):
                s_next = max(0, min(3, s + (1 if a else -1)))
                r = 10 if s_next == 3 else -0.1
                a_next = eps_greedy(s_next)
                # SARSA update
                Q[s, a] = Q[s, a] + alpha * (r + gamma * Q[s_next, a_next] - Q[s, a])
                if s_next == 3:
                    break
                s, a = s_next, a_next
        
        out = "SARSA (on-policy TD control)\n"
        out += "4 states, 2 actions, 200 episodes, ε=0.1\n"
        out += "Q-values:\n"
        for s in range(4):
            out += f"  State {s}: left={Q[s,0]:.2f}, right={Q[s,1]:.2f}\n"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def rl_policy_gradient():
    """Simple REINFORCE policy gradient demo."""
    try:
        # Simple bandit problem
        np.random.seed(42)
        n_actions = 3
        true_rewards = [0.2, 0.5, 0.8]  # Action 2 is best
        theta = np.zeros(n_actions)  # Softmax policy params
        alpha = 0.1
        
        rewards_history = []
        for ep in range(100):
            # Softmax policy
            probs = np.exp(theta) / np.sum(np.exp(theta))
            action = np.random.choice(n_actions, p=probs)
            reward = 1 if np.random.random() < true_rewards[action] else 0
            rewards_history.append(reward)
            
            # REINFORCE update
            for a in range(n_actions):
                if a == action:
                    theta[a] += alpha * reward * (1 - probs[a])
                else:
                    theta[a] -= alpha * reward * probs[a]
        
        final_probs = np.exp(theta) / np.sum(np.exp(theta))
        out = "REINFORCE Policy Gradient (3-arm bandit)\n"
        out += f"True reward probs: {true_rewards}\n"
        out += f"Learned policy π(a): {final_probs.round(3).tolist()}\n"
        out += f"Best action: {np.argmax(final_probs)} (expected: 2)\n"
        out += f"Avg reward last 20: {np.mean(rewards_history[-20:]):.2f}"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


DEMO_HANDLERS = {
    "rl_qlearning": rl_qlearning,
    "rl_value_iteration": rl_value_iteration,
    "rl_td0": rl_td0,
    "rl_sarsa": rl_sarsa,
    "rl_policy_gradient": rl_policy_gradient,
}


def run_demo(demo_id: str):
    if demo_id in DEMO_HANDLERS:
        try:
            return DEMO_HANDLERS[demo_id]()
        except Exception as e:
            return {"ok": False, "output": "", "error": str(e)}
    return {"ok": False, "output": "", "error": f"No demo: {demo_id}"}
