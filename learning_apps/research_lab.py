"""
RESEARCH Mode — Paper Reproduction Lab.

Pick a paper → extract key claims → reproduce the core experiment in the
playground → compare results → build a personal annotated paper trail.
"""
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from flask import request, jsonify

DATA_DIR = Path(__file__).resolve().parent / ".data"
PAPERS_FILE = DATA_DIR / "paper_trail.json"

# ---------------------------------------------------------------------------
# Curated Paper Catalog — foundational ML/CS papers with reproduction guides
# ---------------------------------------------------------------------------
PAPER_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "attention_is_all_you_need",
        "title": "Attention Is All You Need",
        "authors": "Vaswani et al., 2017",
        "arxiv": "https://arxiv.org/abs/1706.03762",
        "key_claims": [
            "Self-attention can replace recurrence entirely for sequence transduction",
            "Multi-head attention allows the model to attend to different representation subspaces",
            "Positional encoding via sinusoids preserves sequence order information",
        ],
        "reproduction_guide": {
            "core_experiment": "Implement scaled dot-product attention and multi-head attention from scratch.",
            "steps": [
                "Implement scaled_dot_product_attention(Q, K, V) → softmax(QK^T/√d_k)V",
                "Implement multi_head_attention with h=8 heads, d_model=512",
                "Add sinusoidal positional encoding",
                "Test on a small sequence-to-sequence task (e.g., reverse a string)",
            ],
            "expected_results": "The attention weights should show interpretable patterns — e.g., attending to the reversed position.",
            "starter_code": """import numpy as np

def scaled_dot_product_attention(Q, K, V):
    \"\"\"Implement scaled dot-product attention.\"\"\"
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    # Apply softmax
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ V, weights

# Test it
np.random.seed(42)
seq_len, d_k = 4, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)
output, weights = scaled_dot_product_attention(Q, K, V)
print("Attention weights shape:", weights.shape)
print("Attention weights:\\n", weights.round(3))
""",
        },
        "prerequisite_topics": ["linear algebra", "softmax", "matrix multiplication"],
        "labs": ["llm_engineers_lab", "deep_learning_lab", "math_for_ml_lab"],
        "difficulty": "intermediate",
    },
    {
        "id": "playing_atari_with_deep_rl",
        "title": "Playing Atari with Deep Reinforcement Learning",
        "authors": "Mnih et al., 2013 (DeepMind)",
        "arxiv": "https://arxiv.org/abs/1312.5602",
        "key_claims": [
            "A CNN can learn to play Atari games directly from raw pixels",
            "Experience replay breaks temporal correlations and stabilizes training",
            "A fixed target network prevents oscillations in Q-value updates",
        ],
        "reproduction_guide": {
            "core_experiment": "Implement DQN with experience replay on a simple environment (CartPole).",
            "steps": [
                "Implement a replay buffer that stores (s, a, r, s', done) tuples",
                "Implement a Q-network (2-layer MLP for CartPole)",
                "Implement epsilon-greedy action selection with decaying epsilon",
                "Train with target network updates every C steps",
                "Plot the reward curve over episodes",
            ],
            "expected_results": "CartPole should be solved (avg reward > 195) within 500 episodes.",
            "starter_code": """import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions),
                np.array(rewards), np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

# Simple Q-network (replace with your neural net)
class SimpleQNetwork:
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.W1 = np.random.randn(state_dim, 64) * 0.1
        self.W2 = np.random.randn(64, action_dim) * 0.1
        self.lr = lr

    def predict(self, state):
        h = np.maximum(0, state @ self.W1)  # ReLU
        return h @ self.W2

# Test the buffer
buf = ReplayBuffer(1000)
for i in range(100):
    buf.push(np.random.randn(4), 0, 1.0, np.random.randn(4), False)
states, actions, rewards, next_states, dones = buf.sample(32)
print(f"Sampled batch: states {states.shape}, rewards {rewards.shape}")
""",
        },
        "prerequisite_topics": ["Q-learning", "neural networks", "epsilon-greedy"],
        "labs": ["rl_lab", "deep_learning_lab"],
        "difficulty": "advanced",
    },
    {
        "id": "batch_normalization",
        "title": "Batch Normalization: Accelerating Deep Network Training",
        "authors": "Ioffe & Szegedy, 2015",
        "arxiv": "https://arxiv.org/abs/1502.03167",
        "key_claims": [
            "Normalizing layer inputs reduces internal covariate shift",
            "Batch norm allows higher learning rates and faster convergence",
            "Batch norm has a regularizing effect, reducing the need for dropout",
        ],
        "reproduction_guide": {
            "core_experiment": "Implement batch normalization and compare training with/without it.",
            "steps": [
                "Implement batch_norm(x, gamma, beta) for a mini-batch",
                "Train a 3-layer MLP on synthetic data WITH batch norm",
                "Train the same network WITHOUT batch norm",
                "Compare convergence speed and final accuracy",
            ],
            "expected_results": "With batch norm, the network should converge ~2-3x faster and tolerate higher learning rates.",
            "starter_code": """import numpy as np

def batch_norm(x, gamma, beta, eps=1e-5):
    \"\"\"Batch normalization for a mini-batch x of shape (batch, features).\"\"\"
    mean = x.mean(axis=0)
    var = x.var(axis=0)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta, mean, var

# Test
np.random.seed(42)
batch = np.random.randn(32, 10) * 5 + 3  # shifted and scaled
gamma = np.ones(10)
beta = np.zeros(10)
normed, mu, var = batch_norm(batch, gamma, beta)
print(f"Before: mean={batch.mean(axis=0)[:3].round(2)}, std={batch.std(axis=0)[:3].round(2)}")
print(f"After:  mean={normed.mean(axis=0)[:3].round(2)}, std={normed.std(axis=0)[:3].round(2)}")
""",
        },
        "prerequisite_topics": ["neural networks", "gradient descent", "regularization"],
        "labs": ["deep_learning_lab", "math_for_ml_lab"],
        "difficulty": "intermediate",
    },
    {
        "id": "word2vec",
        "title": "Efficient Estimation of Word Representations in Vector Space",
        "authors": "Mikolov et al., 2013",
        "arxiv": "https://arxiv.org/abs/1301.3781",
        "key_claims": [
            "Word embeddings capture semantic relationships via vector arithmetic",
            "Skip-gram with negative sampling is efficient for large vocabularies",
            "king - man + woman ≈ queen demonstrates learned analogies",
        ],
        "reproduction_guide": {
            "core_experiment": "Implement skip-gram word2vec with negative sampling on a small corpus.",
            "steps": [
                "Build vocabulary from a small text corpus",
                "Implement skip-gram pair generation with context window",
                "Implement negative sampling loss",
                "Train embeddings and test with analogy tasks",
            ],
            "expected_results": "Similar words should have high cosine similarity. Simple analogies should partially work.",
            "starter_code": """import numpy as np
from collections import Counter

def build_vocab(text, min_count=1):
    words = text.lower().split()
    counts = Counter(words)
    vocab = {w: i for i, (w, c) in enumerate(counts.most_common()) if c >= min_count}
    return vocab, words

def skip_gram_pairs(words, vocab, window=2):
    pairs = []
    indices = [vocab[w] for w in words if w in vocab]
    for i, center in enumerate(indices):
        for j in range(max(0, i-window), min(len(indices), i+window+1)):
            if i != j:
                pairs.append((center, indices[j]))
    return pairs

# Test
text = "the king sat on the throne the queen sat beside the king"
vocab, words = build_vocab(text)
pairs = skip_gram_pairs(words, vocab)
print(f"Vocab size: {len(vocab)}")
print(f"Skip-gram pairs: {len(pairs)}")
print(f"Sample: {pairs[:5]}")
""",
        },
        "prerequisite_topics": ["embeddings", "softmax", "dot product"],
        "labs": ["llm_engineers_lab", "deep_learning_lab"],
        "difficulty": "intermediate",
    },
    {
        "id": "gan",
        "title": "Generative Adversarial Networks",
        "authors": "Goodfellow et al., 2014",
        "arxiv": "https://arxiv.org/abs/1406.2661",
        "key_claims": [
            "A generator and discriminator trained adversarially can produce realistic samples",
            "The minimax game converges when the generator distribution matches the data distribution",
            "The discriminator's optimal strategy is D*(x) = p_data(x) / (p_data(x) + p_g(x))",
        ],
        "reproduction_guide": {
            "core_experiment": "Implement a simple GAN that learns to generate samples from a 1D Gaussian.",
            "steps": [
                "Implement a 2-layer generator: z ~ N(0,1) → G(z) → fake samples",
                "Implement a 2-layer discriminator: x → D(x) → probability",
                "Train with alternating gradient descent on the minimax objective",
                "Plot the generated vs real distributions over training",
            ],
            "expected_results": "Generated samples should converge to match the target Gaussian after ~1000 steps.",
            "starter_code": """import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class SimpleGenerator:
    def __init__(self, z_dim=1, hidden=16, out_dim=1):
        self.W1 = np.random.randn(z_dim, hidden) * 0.1
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, out_dim) * 0.1
        self.b2 = np.zeros(out_dim)

    def forward(self, z):
        h = np.maximum(0, z @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

# Test
G = SimpleGenerator()
z = np.random.randn(100, 1)
fake = G.forward(z)
print(f"Generated samples: mean={fake.mean():.3f}, std={fake.std():.3f}")
print(f"Target: mean=5.0, std=1.0")
""",
        },
        "prerequisite_topics": ["neural networks", "gradient descent", "probability distributions"],
        "labs": ["deep_learning_lab", "probabilistic_ml_lab"],
        "difficulty": "advanced",
    },
]


def _load_trail() -> Dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PAPERS_FILE.exists():
        return json.loads(PAPERS_FILE.read_text())
    return {}


def _save_trail(data: Dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PAPERS_FILE.write_text(json.dumps(data, indent=2, default=str))


def list_papers(difficulty: str = None, lab: str = None) -> List[Dict]:
    """List available papers, optionally filtered."""
    papers = PAPER_CATALOG
    if difficulty:
        papers = [p for p in papers if p["difficulty"] == difficulty]
    if lab:
        papers = [p for p in papers if lab in p["labs"]]
    return [{k: v for k, v in p.items() if k != "reproduction_guide"} for p in papers]


def get_paper(paper_id: str) -> Optional[Dict]:
    return next((p for p in PAPER_CATALOG if p["id"] == paper_id), None)


def start_reproduction(user_id: str, paper_id: str) -> Dict:
    """Start reproducing a paper."""
    paper = get_paper(paper_id)
    if not paper:
        return {"ok": False, "error": "Paper not found"}

    trail = _load_trail()
    key = f"{user_id}::{paper_id}"

    if key in trail:
        return {"ok": True, "entry": trail[key], "paper": paper, "message": "Already started"}

    entry = {
        "paper_id": paper_id,
        "user_id": user_id,
        "started_at": datetime.now().isoformat(),
        "status": "in_progress",
        "notes": "",
        "code": paper["reproduction_guide"]["starter_code"],
        "results": None,
        "annotations": [],
        "claims_verified": {claim: None for claim in paper["key_claims"]},
    }
    trail[key] = entry
    _save_trail(trail)
    return {"ok": True, "entry": entry, "paper": paper}


def save_reproduction(user_id: str, paper_id: str, code: str = None,
                      notes: str = None, results: str = None,
                      claim_idx: int = None, verified: bool = None) -> Dict:
    """Save progress on a reproduction."""
    trail = _load_trail()
    key = f"{user_id}::{paper_id}"
    if key not in trail:
        return {"ok": False, "error": "Not started"}

    entry = trail[key]
    if code is not None:
        entry["code"] = code
    if notes is not None:
        entry["notes"] = notes
    if results is not None:
        entry["results"] = results
        entry["last_run"] = datetime.now().isoformat()
    if claim_idx is not None and verified is not None:
        paper = get_paper(paper_id)
        if paper and 0 <= claim_idx < len(paper["key_claims"]):
            claim = paper["key_claims"][claim_idx]
            entry["claims_verified"][claim] = verified

    # Check completion
    if all(v is not None for v in entry["claims_verified"].values()):
        entry["status"] = "completed"
        entry["completed_at"] = datetime.now().isoformat()

    trail[key] = entry
    _save_trail(trail)
    return {"ok": True, "entry": entry}


def add_annotation(user_id: str, paper_id: str, text: str, section: str = "") -> Dict:
    """Add a personal annotation to a paper."""
    trail = _load_trail()
    key = f"{user_id}::{paper_id}"
    if key not in trail:
        return {"ok": False, "error": "Not started"}

    trail[key]["annotations"].append({
        "text": text,
        "section": section,
        "timestamp": datetime.now().isoformat(),
    })
    _save_trail(trail)
    return {"ok": True, "annotation_count": len(trail[key]["annotations"])}


def get_paper_trail(user_id: str) -> List[Dict]:
    """Get a user's complete paper trail."""
    trail = _load_trail()
    entries = []
    for key, entry in trail.items():
        if entry["user_id"] == user_id:
            paper = get_paper(entry["paper_id"])
            entries.append({
                **entry,
                "paper_title": paper["title"] if paper else "Unknown",
                "paper_authors": paper["authors"] if paper else "",
            })
    return sorted(entries, key=lambda x: x["started_at"], reverse=True)


# ---------------------------------------------------------------------------
# Flask Route Registration
# ---------------------------------------------------------------------------

def register_research_routes(app):
    """Register RESEARCH mode routes on a Flask app."""

    @app.route("/api/research/papers")
    def api_research_papers():
        difficulty = request.args.get("difficulty")
        lab = request.args.get("lab")
        return jsonify({"ok": True, "papers": list_papers(difficulty, lab)})

    @app.route("/api/research/paper/<paper_id>")
    def api_research_paper(paper_id):
        paper = get_paper(paper_id)
        if not paper:
            return jsonify({"ok": False, "error": "Not found"})
        return jsonify({"ok": True, "paper": paper})

    @app.route("/api/research/start", methods=["POST"])
    def api_research_start():
        body = request.get_json(force=True)
        user_id = body.get("user_id", "default")
        paper_id = body.get("paper_id")
        return jsonify(start_reproduction(user_id, paper_id))

    @app.route("/api/research/save", methods=["POST"])
    def api_research_save():
        body = request.get_json(force=True)
        return jsonify(save_reproduction(
            body.get("user_id", "default"),
            body.get("paper_id"),
            code=body.get("code"),
            notes=body.get("notes"),
            results=body.get("results"),
            claim_idx=body.get("claim_idx"),
            verified=body.get("verified"),
        ))

    @app.route("/api/research/annotate", methods=["POST"])
    def api_research_annotate():
        body = request.get_json(force=True)
        return jsonify(add_annotation(
            body.get("user_id", "default"),
            body.get("paper_id"),
            body.get("text", ""),
            body.get("section", ""),
        ))

    @app.route("/api/research/trail")
    def api_research_trail():
        user_id = request.args.get("user_id", "default")
        return jsonify({"ok": True, "trail": get_paper_trail(user_id)})
