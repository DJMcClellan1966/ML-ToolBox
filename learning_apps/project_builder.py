"""
BUILD Mode — Project-Based Learning Engine.

Multi-step capstone projects that span labs. Learners build real artifacts
(trading bots, search engines, classifiers) with scaffolded milestones,
AI code review, and cross-lab integration.
"""
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from flask import request, jsonify

DATA_DIR = Path(__file__).resolve().parent / ".data"
PROJECTS_FILE = DATA_DIR / "user_projects.json"

# ---------------------------------------------------------------------------
# Project Catalog — curated capstone projects across labs
# ---------------------------------------------------------------------------
PROJECT_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "image_classifier",
        "title": "End-to-End Image Classifier",
        "desc": "Build a CNN image classifier from scratch: data loading, model architecture, training loop, evaluation, and deployment.",
        "difficulty": "intermediate",
        "labs": ["deep_learning_lab", "practical_ml_lab", "math_for_ml_lab"],
        "tags": ["deep learning", "CNN", "production"],
        "estimated_hours": 12,
        "milestones": [
            {"id": "m1", "title": "Data Pipeline", "desc": "Load and preprocess an image dataset. Apply augmentation. Split into train/val/test.", "deliverable": "data_pipeline.py"},
            {"id": "m2", "title": "Model Architecture", "desc": "Design a convolutional neural network. Explain each layer's purpose.", "deliverable": "model.py"},
            {"id": "m3", "title": "Training Loop", "desc": "Implement forward pass, loss computation, backprop, and optimizer step. Track metrics.", "deliverable": "train.py"},
            {"id": "m4", "title": "Evaluation & Visualization", "desc": "Confusion matrix, per-class accuracy, grad-CAM visualization of what the model sees.", "deliverable": "evaluate.py"},
            {"id": "m5", "title": "Production API", "desc": "Wrap the model in a Flask API. Accept an image, return predictions with confidence.", "deliverable": "api.py"},
        ],
    },
    {
        "id": "rl_trading_bot",
        "title": "RL Trading Bot",
        "desc": "Build a reinforcement learning agent that learns a trading strategy on historical stock data.",
        "difficulty": "advanced",
        "labs": ["rl_lab", "practical_ml_lab", "math_for_ml_lab"],
        "tags": ["reinforcement learning", "finance", "Q-learning"],
        "estimated_hours": 16,
        "milestones": [
            {"id": "m1", "title": "Market Environment", "desc": "Create an OpenAI Gym-style environment with state=price features, actions=buy/sell/hold, reward=PnL.", "deliverable": "env.py"},
            {"id": "m2", "title": "Q-Learning Agent", "desc": "Implement tabular Q-learning with epsilon-greedy exploration on discretized states.", "deliverable": "q_agent.py"},
            {"id": "m3", "title": "Deep Q-Network", "desc": "Replace the table with a neural network. Add experience replay and target network.", "deliverable": "dqn_agent.py"},
            {"id": "m4", "title": "Backtesting", "desc": "Run the trained agent on held-out data. Compare returns vs buy-and-hold baseline.", "deliverable": "backtest.py"},
            {"id": "m5", "title": "Risk Analysis", "desc": "Compute Sharpe ratio, max drawdown, and visualize the equity curve.", "deliverable": "risk.py"},
        ],
    },
    {
        "id": "search_engine",
        "title": "Build a Search Engine",
        "desc": "Implement a search engine from scratch: crawling, indexing, ranking, and a web UI.",
        "difficulty": "intermediate",
        "labs": ["clrs_algorithms_lab", "python_practice_lab", "llm_engineers_lab"],
        "tags": ["algorithms", "information retrieval", "TF-IDF"],
        "estimated_hours": 10,
        "milestones": [
            {"id": "m1", "title": "Document Loader", "desc": "Parse a corpus of text documents. Tokenize, stem, remove stop words.", "deliverable": "loader.py"},
            {"id": "m2", "title": "Inverted Index", "desc": "Build an inverted index mapping terms to document IDs with term frequencies.", "deliverable": "index.py"},
            {"id": "m3", "title": "TF-IDF Ranking", "desc": "Score documents by TF-IDF relevance to a query. Return top-k results.", "deliverable": "ranker.py"},
            {"id": "m4", "title": "PageRank (bonus)", "desc": "Build a link graph between documents and compute PageRank scores.", "deliverable": "pagerank.py"},
            {"id": "m5", "title": "Web UI", "desc": "Create a simple Flask search interface with query box and ranked results.", "deliverable": "app.py"},
        ],
    },
    {
        "id": "bayesian_ab_tester",
        "title": "Bayesian A/B Testing Platform",
        "desc": "Build a Bayesian A/B testing framework with conjugate priors, credible intervals, and decision rules.",
        "difficulty": "intermediate",
        "labs": ["probabilistic_ml_lab", "math_for_ml_lab", "practical_ml_lab"],
        "tags": ["Bayesian", "statistics", "experimentation"],
        "estimated_hours": 8,
        "milestones": [
            {"id": "m1", "title": "Beta-Binomial Model", "desc": "Implement the Beta-Binomial conjugate model for conversion rate estimation.", "deliverable": "model.py"},
            {"id": "m2", "title": "Posterior Inference", "desc": "Compute posterior distributions, credible intervals, and P(A > B) analytically.", "deliverable": "inference.py"},
            {"id": "m3", "title": "Visualization", "desc": "Plot prior → posterior evolution, overlap regions, and expected loss.", "deliverable": "visualize.py"},
            {"id": "m4", "title": "Decision Engine", "desc": "Implement stopping rules: expected loss threshold, ROPE, and sample size estimation.", "deliverable": "decision.py"},
        ],
    },
    {
        "id": "llm_rag_chatbot",
        "title": "RAG Chatbot from Scratch",
        "desc": "Build a Retrieval-Augmented Generation chatbot over your own documents.",
        "difficulty": "advanced",
        "labs": ["llm_engineers_lab", "clrs_algorithms_lab", "practical_ml_lab"],
        "tags": ["LLM", "RAG", "embeddings", "chatbot"],
        "estimated_hours": 14,
        "milestones": [
            {"id": "m1", "title": "Document Chunker", "desc": "Split documents into overlapping chunks. Handle PDFs, markdown, and plain text.", "deliverable": "chunker.py"},
            {"id": "m2", "title": "Embedding Store", "desc": "Embed chunks using sentence transformers. Store in a vector database (FAISS or simple cosine).", "deliverable": "embeddings.py"},
            {"id": "m3", "title": "Retrieval Pipeline", "desc": "Given a query, retrieve top-k relevant chunks. Re-rank by cross-encoder score.", "deliverable": "retrieval.py"},
            {"id": "m4", "title": "Generation", "desc": "Feed retrieved context + query to an LLM. Implement prompt engineering and guardrails.", "deliverable": "generate.py"},
            {"id": "m5", "title": "Evaluation", "desc": "Measure retrieval recall, answer faithfulness, and hallucination rate on a test set.", "deliverable": "evaluate.py"},
        ],
    },
    {
        "id": "minigrad",
        "title": "Build Your Own Autograd Engine",
        "desc": "Implement a tiny automatic differentiation library (like micrograd) and train a neural net with it.",
        "difficulty": "expert",
        "labs": ["deep_learning_lab", "math_for_ml_lab", "sicp_lab"],
        "tags": ["autograd", "backpropagation", "from scratch"],
        "estimated_hours": 10,
        "milestones": [
            {"id": "m1", "title": "Value Class", "desc": "Implement a Value node with data, grad, and backward function. Support +, *, -, /, **.", "deliverable": "engine.py"},
            {"id": "m2", "title": "Backpropagation", "desc": "Implement topological sort and reverse-mode autodiff. Verify gradients numerically.", "deliverable": "engine.py"},
            {"id": "m3", "title": "Neural Network", "desc": "Build Neuron, Layer, and MLP classes on top of the engine. Forward pass + loss.", "deliverable": "nn.py"},
            {"id": "m4", "title": "Training", "desc": "Train the MLP on a toy dataset (moons, circles). Visualize decision boundary evolution.", "deliverable": "train.py"},
        ],
    },
]


def _load_projects() -> Dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PROJECTS_FILE.exists():
        return json.loads(PROJECTS_FILE.read_text())
    return {}


def _save_projects(data: Dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_FILE.write_text(json.dumps(data, indent=2, default=str))


def _user_key(user_id: str, project_id: str) -> str:
    return f"{user_id}::{project_id}"


# ---------------------------------------------------------------------------
# API Functions
# ---------------------------------------------------------------------------

def list_projects() -> List[Dict]:
    """Return the full project catalog."""
    return PROJECT_CATALOG


def start_project(user_id: str, project_id: str) -> Dict:
    """Start a project for a user."""
    catalog_entry = next((p for p in PROJECT_CATALOG if p["id"] == project_id), None)
    if not catalog_entry:
        return {"ok": False, "error": "Project not found"}

    data = _load_projects()
    key = _user_key(user_id, project_id)
    if key in data:
        return {"ok": True, "project": data[key], "message": "Already started"}

    project = {
        "project_id": project_id,
        "user_id": user_id,
        "started_at": datetime.now().isoformat(),
        "status": "in_progress",
        "milestones": {
            m["id"]: {"status": "not_started", "code": "", "review": None, "submitted_at": None}
            for m in catalog_entry["milestones"]
        },
    }
    data[key] = project
    _save_projects(data)
    return {"ok": True, "project": project}


def submit_milestone(user_id: str, project_id: str, milestone_id: str, code: str) -> Dict:
    """Submit code for a milestone, get AI review."""
    data = _load_projects()
    key = _user_key(user_id, project_id)
    if key not in data:
        return {"ok": False, "error": "Project not started"}

    project = data[key]
    if milestone_id not in project["milestones"]:
        return {"ok": False, "error": "Milestone not found"}

    # Find catalog entry for context
    catalog_entry = next((p for p in PROJECT_CATALOG if p["id"] == project_id), None)
    milestone_def = next((m for m in catalog_entry["milestones"] if m["id"] == milestone_id), None)

    # Generate review (LLM or rule-based)
    review = _review_code(code, milestone_def, catalog_entry)

    project["milestones"][milestone_id] = {
        "status": "completed" if review["passes"] else "needs_revision",
        "code": code,
        "review": review,
        "submitted_at": datetime.now().isoformat(),
    }

    # Check if all milestones complete
    if all(m["status"] == "completed" for m in project["milestones"].values()):
        project["status"] = "completed"
        project["completed_at"] = datetime.now().isoformat()

    data[key] = project
    _save_projects(data)
    return {"ok": True, "review": review, "project": project}


def get_user_project(user_id: str, project_id: str) -> Dict:
    """Get a user's project state."""
    data = _load_projects()
    key = _user_key(user_id, project_id)
    if key not in data:
        return {"ok": False, "error": "Not started"}
    return {"ok": True, "project": data[key]}


def get_all_user_projects(user_id: str) -> List[Dict]:
    """Get all projects a user has started."""
    data = _load_projects()
    return [v for k, v in data.items() if k.startswith(f"{user_id}::")]


def _review_code(code: str, milestone: Dict, project: Dict) -> Dict:
    """AI code review — tries LLM, falls back to static analysis."""
    issues = []
    suggestions = []
    passes = True

    # Static checks
    if len(code.strip()) < 20:
        issues.append("Submission is too short to be a meaningful implementation.")
        passes = False

    if not any(kw in code for kw in ["def ", "class ", "import ", "="]):
        issues.append("No functions, classes, or assignments found.")
        passes = False

    # Check for key concepts based on milestone
    title_lower = milestone["title"].lower()
    if "class" in milestone["desc"].lower() and "class " not in code:
        suggestions.append(f"Consider using a class structure for {milestone['title']}.")

    if "test" in milestone["desc"].lower() and "assert" not in code and "test" not in code.lower():
        suggestions.append("Add tests or assertions to verify correctness.")

    if "visualiz" in milestone["desc"].lower() and "plt" not in code and "plot" not in code.lower():
        suggestions.append("Consider adding visualization (matplotlib) for the results.")

    # Try LLM review
    llm_review = _llm_review(code, milestone, project)
    if llm_review:
        return llm_review

    return {
        "passes": passes,
        "issues": issues,
        "suggestions": suggestions,
        "summary": "Code reviewed with static analysis." + (" All basic checks passed!" if passes else " Some issues found."),
        "reviewer": "static",
    }


def _llm_review(code: str, milestone: Dict, project: Dict) -> Optional[Dict]:
    """Try to get an LLM code review."""
    try:
        import importlib
        if importlib.util.find_spec("ollama"):
            ollama = importlib.import_module("ollama")
            prompt = f"""You are a senior engineer reviewing a student's code submission.

Project: {project['title']}
Milestone: {milestone['title']} — {milestone['desc']}

Code:
```python
{code[:3000]}
```

Review the code. Respond in JSON:
{{"passes": true/false, "issues": ["..."], "suggestions": ["..."], "summary": "One paragraph review"}}"""
            resp = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
            text = resp["message"]["content"]
            # Try to parse JSON from response
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                review = json.loads(match.group())
                review["reviewer"] = "llm"
                return review
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Flask Route Registration
# ---------------------------------------------------------------------------

def register_project_routes(app):
    """Register BUILD mode routes on a Flask app."""

    @app.route("/api/projects/catalog")
    def api_project_catalog():
        return jsonify({"ok": True, "projects": list_projects()})

    @app.route("/api/projects/start", methods=["POST"])
    def api_project_start():
        body = request.get_json(force=True)
        user_id = body.get("user_id", "default")
        project_id = body.get("project_id")
        return jsonify(start_project(user_id, project_id))

    @app.route("/api/projects/submit", methods=["POST"])
    def api_project_submit():
        body = request.get_json(force=True)
        user_id = body.get("user_id", "default")
        project_id = body.get("project_id")
        milestone_id = body.get("milestone_id")
        code = body.get("code", "")
        return jsonify(submit_milestone(user_id, project_id, milestone_id, code))

    @app.route("/api/projects/status")
    def api_project_status():
        user_id = request.args.get("user_id", "default")
        project_id = request.args.get("project_id")
        if project_id:
            return jsonify(get_user_project(user_id, project_id))
        return jsonify({"ok": True, "projects": get_all_user_projects(user_id)})
