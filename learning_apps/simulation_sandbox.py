"""
SIMULATE Mode — Consequence Sandboxes.

Interactive what-if environments where learners tweak parameters and watch
algorithms behave in real-time. Modify an algorithm mid-execution, break it,
fix it, and build intuition through experimentation.
"""
import json
import math
import numpy as np
from typing import Dict, Any, List, Optional
from flask import request, jsonify


# ---------------------------------------------------------------------------
# Simulation Environments
# ---------------------------------------------------------------------------

def sim_gridworld(params: Dict) -> Dict:
    """Interactive gridworld where user sets Q-values and watches agent behave.

    params:
        grid_size: int (default 5)
        q_overrides: dict of "r,c,a" -> value  (optional manual Q-values)
        gamma: discount factor (0-1)
        reward_pos: [r, c] for goal
        obstacle_pos: list of [r,c] for walls
        episodes: int (how many episodes to simulate)
    """
    size = params.get("grid_size", 5)
    gamma = params.get("gamma", 0.9)
    episodes = min(params.get("episodes", 100), 500)
    reward_pos = tuple(params.get("reward_pos", [size - 1, size - 1]))
    obstacles = set(tuple(o) for o in params.get("obstacle_pos", []))
    q_overrides = params.get("q_overrides", {})

    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U
    action_names = ["right", "left", "down", "up"]

    # Initialize Q-table
    Q = np.zeros((size, size, 4))

    # Apply user overrides
    for key, val in q_overrides.items():
        parts = key.split(",")
        if len(parts) == 3:
            r, c, a = int(parts[0]), int(parts[1]), int(parts[2])
            if 0 <= r < size and 0 <= c < size and 0 <= a < 4:
                Q[r, c, a] = val

    # Q-learning simulation
    alpha = 0.1
    epsilon = params.get("epsilon", 0.1)
    episode_rewards = []

    for ep in range(episodes):
        state = (0, 0)
        total_reward = 0
        for step in range(size * size * 2):
            r, c = state
            if state == reward_pos:
                break

            # Epsilon-greedy
            if np.random.random() < epsilon:
                action = np.random.randint(4)
            else:
                action = int(np.argmax(Q[r, c]))

            dr, dc = actions[action]
            nr, nc = r + dr, c + dc

            # Boundary and obstacle check
            if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in obstacles:
                next_state = (nr, nc)
            else:
                next_state = state

            # Reward
            reward = 10.0 if next_state == reward_pos else -0.1

            # Q-update
            nr2, nc2 = next_state
            Q[r, c, action] += alpha * (reward + gamma * np.max(Q[nr2, nc2]) - Q[r, c, action])

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

    # Derive policy
    policy = {}
    for r in range(size):
        for c in range(size):
            if (r, c) in obstacles:
                policy[f"{r},{c}"] = "wall"
            elif (r, c) == reward_pos:
                policy[f"{r},{c}"] = "goal"
            else:
                best = int(np.argmax(Q[r, c]))
                policy[f"{r},{c}"] = action_names[best]

    return {
        "ok": True,
        "type": "gridworld",
        "grid_size": size,
        "q_table": {f"{r},{c}": Q[r, c].tolist() for r in range(size) for c in range(size)},
        "policy": policy,
        "episode_rewards": episode_rewards[-20:],  # last 20 episodes
        "mean_reward": float(np.mean(episode_rewards[-20:])),
        "params": {"gamma": gamma, "epsilon": epsilon, "episodes": episodes},
    }


def sim_gradient_descent(params: Dict) -> Dict:
    """Watch gradient descent navigate a loss landscape.

    params:
        landscape: "quadratic" | "rastrigin" | "saddle" | "rosenbrock"
        learning_rate: float
        momentum: float (0 = no momentum)
        start_x, start_y: starting point
        steps: int
    """
    landscape = params.get("landscape", "quadratic")
    lr = params.get("learning_rate", 0.01)
    momentum = params.get("momentum", 0.0)
    x = params.get("start_x", 3.0)
    y = params.get("start_y", 3.0)
    steps = min(params.get("steps", 200), 1000)

    def loss_and_grad(x, y, landscape):
        if landscape == "quadratic":
            # Simple bowl: f = x^2 + y^2
            return x**2 + y**2, 2*x, 2*y
        elif landscape == "rastrigin":
            # Lots of local minima
            A = 10
            f = A * 2 + (x**2 - A * math.cos(2 * math.pi * x)) + (y**2 - A * math.cos(2 * math.pi * y))
            dx = 2*x + A * 2 * math.pi * math.sin(2 * math.pi * x)
            dy = 2*y + A * 2 * math.pi * math.sin(2 * math.pi * y)
            return f, dx, dy
        elif landscape == "saddle":
            # Saddle point at origin: f = x^2 - y^2
            return x**2 - y**2, 2*x, -2*y
        elif landscape == "rosenbrock":
            # Narrow curved valley: f = (1-x)^2 + 100(y-x^2)^2
            f = (1 - x)**2 + 100 * (y - x**2)**2
            dx = -2 * (1 - x) + 100 * 2 * (y - x**2) * (-2 * x)
            dy = 100 * 2 * (y - x**2)
            return f, dx, dy
        else:
            return x**2 + y**2, 2*x, 2*y

    trajectory = []
    vx, vy = 0.0, 0.0

    for step in range(steps):
        f, dx, dy = loss_and_grad(x, y, landscape)
        trajectory.append({"step": step, "x": round(x, 6), "y": round(y, 6), "loss": round(f, 6)})

        vx = momentum * vx - lr * dx
        vy = momentum * vy - lr * dy
        x += vx
        y += vy

        # Clamp to prevent explosion
        x = max(-10, min(10, x))
        y = max(-10, min(10, y))

    return {
        "ok": True,
        "type": "gradient_descent",
        "landscape": landscape,
        "trajectory": trajectory,
        "final_loss": trajectory[-1]["loss"],
        "converged": trajectory[-1]["loss"] < 0.01,
        "params": {"learning_rate": lr, "momentum": momentum, "steps": steps},
    }


def sim_neural_network(params: Dict) -> Dict:
    """Live neural network: tweak weights/architecture, see decision boundary shift.

    params:
        dataset: "xor" | "circles" | "spiral"
        hidden_sizes: list of ints (e.g., [4, 4])
        activation: "relu" | "sigmoid" | "tanh"
        learning_rate: float
        epochs: int
    """
    dataset = params.get("dataset", "xor")
    hidden_sizes = params.get("hidden_sizes", [4])
    activation = params.get("activation", "relu")
    lr = params.get("learning_rate", 0.1)
    epochs = min(params.get("epochs", 200), 1000)

    # Generate dataset
    np.random.seed(42)
    if dataset == "xor":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 25, dtype=float)
        Y = np.array([0, 1, 1, 0] * 25, dtype=float)
        X += np.random.randn(*X.shape) * 0.1
    elif dataset == "circles":
        n = 100
        theta = np.linspace(0, 2 * np.pi, n)
        X_inner = np.column_stack([np.cos(theta) * 0.5, np.sin(theta) * 0.5]) + np.random.randn(n, 2) * 0.1
        X_outer = np.column_stack([np.cos(theta) * 1.5, np.sin(theta) * 1.5]) + np.random.randn(n, 2) * 0.1
        X = np.vstack([X_inner, X_outer])
        Y = np.array([0] * n + [1] * n, dtype=float)
    else:  # spiral
        n = 100
        t = np.linspace(0, 4 * np.pi, n)
        X_a = np.column_stack([t * np.cos(t), t * np.sin(t)]) / (4 * np.pi) + np.random.randn(n, 2) * 0.05
        X_b = np.column_stack([t * np.cos(t + np.pi), t * np.sin(t + np.pi)]) / (4 * np.pi) + np.random.randn(n, 2) * 0.05
        X = np.vstack([X_a, X_b])
        Y = np.array([0] * n + [1] * n, dtype=float)

    def act_fn(x, name):
        if name == "relu":
            return np.maximum(0, x)
        elif name == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif name == "tanh":
            return np.tanh(x)
        return x

    def act_deriv(x, name):
        if name == "relu":
            return (x > 0).astype(float)
        elif name == "sigmoid":
            s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return s * (1 - s)
        elif name == "tanh":
            return 1 - np.tanh(x) ** 2
        return np.ones_like(x)

    # Build network
    layers = [2] + hidden_sizes + [1]
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i])
        b = np.zeros(layers[i + 1])
        weights.append(w)
        biases.append(b)

    loss_history = []
    accuracy_history = []

    for epoch in range(epochs):
        # Forward pass
        activations = [X]
        pre_activations = []
        a = X
        for i in range(len(weights)):
            z = a @ weights[i] + biases[i]
            pre_activations.append(z)
            if i < len(weights) - 1:
                a = act_fn(z, activation)
            else:
                a = 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # sigmoid output
            activations.append(a)

        # Loss
        preds = activations[-1].flatten()
        preds_clipped = np.clip(preds, 1e-7, 1 - 1e-7)
        loss = -np.mean(Y * np.log(preds_clipped) + (1 - Y) * np.log(1 - preds_clipped))
        acc = np.mean((preds > 0.5).astype(float) == Y)
        loss_history.append(round(float(loss), 4))
        accuracy_history.append(round(float(acc), 4))

        # Backward pass
        delta = (preds - Y).reshape(-1, 1) / len(Y)
        for i in range(len(weights) - 1, -1, -1):
            dw = activations[i].T @ delta
            db = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ weights[i].T) * act_deriv(pre_activations[i - 1], activation)
            weights[i] -= lr * dw
            biases[i] -= lr * db

    # Generate decision boundary grid
    grid_size = 20
    x_range = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, grid_size)
    y_range = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, grid_size)
    boundary = []
    for xi in x_range:
        row = []
        for yi in y_range:
            a = np.array([[xi, yi]])
            for i in range(len(weights)):
                z = a @ weights[i] + biases[i]
                if i < len(weights) - 1:
                    a = act_fn(z, activation)
                else:
                    a = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            row.append(round(float(a[0, 0]), 3))
        boundary.append(row)

    return {
        "ok": True,
        "type": "neural_network",
        "dataset": dataset,
        "loss_history": loss_history[::max(1, epochs // 50)],  # subsample
        "accuracy_history": accuracy_history[::max(1, epochs // 50)],
        "final_loss": loss_history[-1],
        "final_accuracy": accuracy_history[-1],
        "decision_boundary": boundary,
        "boundary_x_range": [float(x_range[0]), float(x_range[-1])],
        "boundary_y_range": [float(y_range[0]), float(y_range[-1])],
        "data_points": [{"x": float(X[i, 0]), "y": float(X[i, 1]), "label": int(Y[i])} for i in range(len(Y))],
        "params": {"hidden_sizes": hidden_sizes, "activation": activation, "learning_rate": lr, "epochs": epochs},
    }


def sim_sorting_algorithm(params: Dict) -> Dict:
    """Watch a sorting algorithm step by step. Modify comparisons, break it.

    params:
        algorithm: "bubble" | "quick" | "merge" | "insertion"
        array: list of ints (or auto-generated)
        array_size: int (if array not provided)
        swap_override: optional dict to force a wrong swap at a specific step
    """
    algorithm = params.get("algorithm", "bubble")
    arr = params.get("array", None)
    if arr is None:
        size = min(params.get("array_size", 15), 50)
        arr = list(np.random.permutation(size) + 1)
    else:
        arr = list(arr[:50])

    swap_override = params.get("swap_override", {})
    steps = []
    a = list(arr)

    if algorithm == "bubble":
        for i in range(len(a)):
            for j in range(len(a) - 1 - i):
                step_key = str(len(steps))
                comparing = [j, j + 1]
                swapped = False
                if step_key in swap_override:
                    # Force a swap (or non-swap) at this step
                    if swap_override[step_key]:
                        a[j], a[j + 1] = a[j + 1], a[j]
                        swapped = True
                elif a[j] > a[j + 1]:
                    a[j], a[j + 1] = a[j + 1], a[j]
                    swapped = True
                steps.append({
                    "step": len(steps),
                    "comparing": comparing,
                    "swapped": swapped,
                    "array": list(a),
                })

    elif algorithm == "insertion":
        for i in range(1, len(a)):
            key = a[i]
            j = i - 1
            while j >= 0 and a[j] > key:
                a[j + 1] = a[j]
                steps.append({
                    "step": len(steps),
                    "comparing": [j, j + 1],
                    "swapped": True,
                    "array": list(a),
                })
                j -= 1
            a[j + 1] = key
            steps.append({
                "step": len(steps),
                "comparing": [j + 1, i],
                "swapped": False,
                "array": list(a),
                "inserted": j + 1,
            })

    elif algorithm == "quick":
        def quicksort(a, lo, hi):
            if lo < hi:
                pivot = a[hi]
                i = lo
                for j in range(lo, hi):
                    steps.append({
                        "step": len(steps),
                        "comparing": [j, hi],
                        "pivot": hi,
                        "array": list(a),
                        "swapped": a[j] <= pivot,
                    })
                    if a[j] <= pivot:
                        a[i], a[j] = a[j], a[i]
                        i += 1
                a[i], a[hi] = a[hi], a[i]
                steps.append({
                    "step": len(steps),
                    "comparing": [i, hi],
                    "pivot_placed": i,
                    "array": list(a),
                    "swapped": True,
                })
                quicksort(a, lo, i - 1)
                quicksort(a, i + 1, hi)
        quicksort(a, 0, len(a) - 1)

    else:  # merge sort
        def mergesort(a, lo, hi):
            if lo < hi:
                mid = (lo + hi) // 2
                mergesort(a, lo, mid)
                mergesort(a, mid + 1, hi)
                merged = []
                i, j = lo, mid + 1
                while i <= mid and j <= hi:
                    steps.append({
                        "step": len(steps),
                        "comparing": [i, j],
                        "array": list(a),
                        "swapped": a[i] > a[j],
                    })
                    if a[i] <= a[j]:
                        merged.append(a[i]); i += 1
                    else:
                        merged.append(a[j]); j += 1
                merged.extend(a[i:mid + 1])
                merged.extend(a[j:hi + 1])
                a[lo:hi + 1] = merged
                steps.append({
                    "step": len(steps),
                    "comparing": [],
                    "merge_range": [lo, hi],
                    "array": list(a),
                    "swapped": False,
                })
        mergesort(a, 0, len(a) - 1)

    is_sorted = all(a[i] <= a[i + 1] for i in range(len(a) - 1))

    return {
        "ok": True,
        "type": "sorting",
        "algorithm": algorithm,
        "original": list(arr),
        "final": list(a),
        "is_sorted": is_sorted,
        "total_steps": len(steps),
        "steps": steps[:200],  # cap at 200 for response size
    }


def sim_bayesian_update(params: Dict) -> Dict:
    """Tweak priors with sliders and watch the posterior update in real-time.

    params:
        prior_alpha: float (Beta prior alpha)
        prior_beta: float (Beta prior beta)
        observations: list of 0/1 (or summary: successes, trials)
        successes: int
        trials: int
    """
    alpha = params.get("prior_alpha", 1.0)
    beta_param = params.get("prior_beta", 1.0)
    observations = params.get("observations", [])
    successes = params.get("successes", sum(observations) if observations else 0)
    trials = params.get("trials", len(observations) if observations else 0)

    # Posterior parameters (Beta-Binomial conjugate)
    post_alpha = alpha + successes
    post_beta = beta_param + (trials - successes)

    # Generate PDF points
    from math import lgamma as _lgamma

    def beta_pdf(x, a, b):
        if x <= 0 or x >= 1:
            return 0
        try:
            log_pdf = (a - 1) * math.log(x) + (b - 1) * math.log(1 - x) - (
                _lgamma(a) + _lgamma(b) - _lgamma(a + b)
            )
            return math.exp(log_pdf)
        except (ValueError, OverflowError):
            return 0

    x_vals = [i / 100 for i in range(1, 100)]
    prior_pdf = [round(beta_pdf(x, alpha, beta_param), 6) for x in x_vals]
    posterior_pdf = [round(beta_pdf(x, post_alpha, post_beta), 6) for x in x_vals]

    # Summary statistics
    prior_mean = alpha / (alpha + beta_param)
    post_mean = post_alpha / (post_alpha + post_beta)
    post_mode = (post_alpha - 1) / (post_alpha + post_beta - 2) if (post_alpha > 1 and post_beta > 1) else post_mean

    # Credible interval (approximate: use Beta quantiles via normal approx)
    post_var = (post_alpha * post_beta) / ((post_alpha + post_beta) ** 2 * (post_alpha + post_beta + 1))
    post_std = math.sqrt(post_var)
    ci_low = max(0, round(post_mean - 1.96 * post_std, 4))
    ci_high = min(1, round(post_mean + 1.96 * post_std, 4))

    return {
        "ok": True,
        "type": "bayesian_update",
        "prior": {"alpha": alpha, "beta": beta_param, "mean": round(prior_mean, 4)},
        "posterior": {
            "alpha": post_alpha, "beta": post_beta,
            "mean": round(post_mean, 4), "mode": round(post_mode, 4),
            "ci_95": [ci_low, ci_high],
        },
        "data": {"successes": successes, "trials": trials},
        "x_vals": x_vals,
        "prior_pdf": prior_pdf,
        "posterior_pdf": posterior_pdf,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

SIMULATIONS = {
    "gridworld": sim_gridworld,
    "gradient_descent": sim_gradient_descent,
    "neural_network": sim_neural_network,
    "sorting": sim_sorting_algorithm,
    "bayesian_update": sim_bayesian_update,
}


def run_simulation(sim_type: str, params: Dict) -> Dict:
    fn = SIMULATIONS.get(sim_type)
    if not fn:
        return {"ok": False, "error": f"Unknown simulation: {sim_type}. Available: {list(SIMULATIONS.keys())}"}
    try:
        return fn(params)
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Flask Route Registration
# ---------------------------------------------------------------------------

def register_simulation_routes(app):
    """Register SIMULATE mode routes on a Flask app."""

    @app.route("/api/simulate/run", methods=["POST"])
    def api_simulate_run():
        body = request.get_json(force=True)
        sim_type = body.get("type", "")
        params = body.get("params", {})
        return jsonify(run_simulation(sim_type, params))

    @app.route("/api/simulate/types")
    def api_simulate_types():
        return jsonify({
            "ok": True,
            "types": [
                {"id": "gridworld", "name": "RL Gridworld", "desc": "Set Q-values, watch agent navigate. Tweak gamma, epsilon, obstacles."},
                {"id": "gradient_descent", "name": "Gradient Descent", "desc": "Watch GD on different landscapes: quadratic, Rastrigin, saddle, Rosenbrock."},
                {"id": "neural_network", "name": "Neural Network", "desc": "Tweak architecture, activation, LR. See decision boundary evolve on XOR/circles/spiral."},
                {"id": "sorting", "name": "Sorting Algorithms", "desc": "Step through bubble/quick/merge/insertion sort. Override swaps, break it."},
                {"id": "bayesian_update", "name": "Bayesian Update", "desc": "Slide priors, add observations, watch the posterior shift in real-time."},
            ],
        })
