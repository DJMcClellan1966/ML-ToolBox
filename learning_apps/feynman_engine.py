"""
TEACH Mode — Feynman Engine.

Flips the dynamic: the learner teaches the AI, which plays dumb strategically.
The system critiques explanations, asks probing follow-ups, and scores
understanding depth. "If you can't explain it simply, you don't understand it."
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from flask import request, jsonify

DATA_DIR = Path(__file__).resolve().parent / ".data"
TEACHBACKS_FILE = DATA_DIR / "teachback_sessions.json"

# ---------------------------------------------------------------------------
# Probing follow-up templates — used when LLM is unavailable
# ---------------------------------------------------------------------------
PROBE_TEMPLATES = {
    "analogy": "Can you explain '{concept}' using an analogy a 10-year-old would understand?",
    "edge_case": "What happens to '{concept}' in the edge case where {edge}?",
    "why": "You said {claim}. But *why* is that true? What's the deeper reason?",
    "connect": "How does '{concept}' relate to {related_concept}?",
    "counter": "A student says: '{misconception}'. How would you correct them?",
    "simplify": "You used the term '{jargon}'. Can you explain that without any jargon?",
    "predict": "If we changed {variable}, what would happen and why?",
}

# Topic-specific probes
TOPIC_PROBES: Dict[str, List[Dict[str, str]]] = {
    "backpropagation": [
        {"type": "edge_case", "edge": "the learning rate is extremely large", "concept": "backpropagation"},
        {"type": "why", "claim": "we multiply gradients along the chain", "concept": "chain rule"},
        {"type": "counter", "misconception": "Backprop changes the weights going forward through the network"},
        {"type": "simplify", "jargon": "computational graph"},
        {"type": "predict", "variable": "the activation function from ReLU to sigmoid", "concept": "gradient flow"},
    ],
    "gradient_descent": [
        {"type": "analogy", "concept": "gradient descent"},
        {"type": "edge_case", "edge": "the loss surface has many local minima", "concept": "gradient descent"},
        {"type": "counter", "misconception": "Gradient descent always finds the global minimum"},
        {"type": "predict", "variable": "the batch size from 1 to the full dataset", "concept": "convergence"},
    ],
    "q_learning": [
        {"type": "analogy", "concept": "Q-learning"},
        {"type": "edge_case", "edge": "epsilon is set to 0 from the start", "concept": "exploration"},
        {"type": "why", "claim": "we use a discount factor gamma", "concept": "temporal difference"},
        {"type": "counter", "misconception": "Q-learning requires a model of the environment"},
    ],
    "bayesian_inference": [
        {"type": "analogy", "concept": "Bayesian inference"},
        {"type": "simplify", "jargon": "conjugate prior"},
        {"type": "edge_case", "edge": "we have no prior knowledge at all", "concept": "prior selection"},
        {"type": "counter", "misconception": "Bayesian methods always need MCMC to compute the posterior"},
    ],
    "transformers": [
        {"type": "why", "claim": "attention is computed as softmax(QK^T/√d)V", "concept": "scaled dot-product attention"},
        {"type": "edge_case", "edge": "the sequence length is 100,000 tokens", "concept": "self-attention"},
        {"type": "simplify", "jargon": "multi-head attention"},
        {"type": "counter", "misconception": "Transformers process tokens sequentially like RNNs"},
    ],
    "default": [
        {"type": "analogy", "concept": "this concept"},
        {"type": "why", "claim": "this is important", "concept": "the topic"},
        {"type": "simplify", "jargon": "the technical terms you used"},
    ],
}


def _load_sessions() -> Dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TEACHBACKS_FILE.exists():
        return json.loads(TEACHBACKS_FILE.read_text())
    return {}


def _save_sessions(data: Dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEACHBACKS_FILE.write_text(json.dumps(data, indent=2, default=str))


def _get_probes(topic: str) -> List[Dict]:
    """Get probing questions for a topic."""
    key = topic.lower().replace(" ", "_").replace("-", "_")
    probes = TOPIC_PROBES.get(key, TOPIC_PROBES["default"])
    return probes


def _format_probe(probe: Dict, topic: str) -> str:
    """Format a probe template into a question."""
    template = PROBE_TEMPLATES.get(probe["type"], "Tell me more about '{concept}'.")
    params = {**probe, "concept": topic}
    try:
        return template.format(**params)
    except KeyError:
        return f"Can you explain more about {topic}?"


def _score_explanation(explanation: str, topic: str) -> Dict:
    """Score an explanation on multiple dimensions."""
    words = explanation.split()
    word_count = len(words)
    sentences = [s.strip() for s in explanation.replace("!", ".").replace("?", ".").split(".") if s.strip()]

    scores = {
        "clarity": 0.0,
        "depth": 0.0,
        "accuracy_proxy": 0.0,
        "completeness": 0.0,
        "accessibility": 0.0,
    }

    # Clarity: penalize very long or very short explanations
    if 30 <= word_count <= 300:
        scores["clarity"] = min(1.0, word_count / 100)
    elif word_count > 300:
        scores["clarity"] = max(0.3, 1.0 - (word_count - 300) / 500)
    else:
        scores["clarity"] = word_count / 30

    # Depth: look for causal language, examples, connections
    depth_markers = ["because", "therefore", "since", "this means", "for example",
                     "in other words", "the reason", "as a result", "leads to",
                     "depends on", "analogous", "similar to", "contrast"]
    depth_count = sum(1 for m in depth_markers if m in explanation.lower())
    scores["depth"] = min(1.0, depth_count / 4)

    # Accuracy proxy: presence of key technical terms (not a real accuracy check)
    scores["accuracy_proxy"] = min(1.0, 0.3 + len(sentences) * 0.1)

    # Completeness: multiple sentences covering different aspects
    scores["completeness"] = min(1.0, len(sentences) / 5)

    # Accessibility: use of analogies, simple language
    access_markers = ["like", "imagine", "think of", "picture", "analogy",
                      "simple", "basically", "in plain", "everyday"]
    access_count = sum(1 for m in access_markers if m in explanation.lower())
    scores["accessibility"] = min(1.0, access_count / 3)

    # Overall Feynman Score
    weights = {"clarity": 0.2, "depth": 0.3, "accuracy_proxy": 0.15,
               "completeness": 0.2, "accessibility": 0.15}
    overall = sum(scores[k] * weights[k] for k in weights)
    scores["overall"] = round(overall, 3)

    return scores


def start_teachback(user_id: str, topic: str, lab_id: str = "") -> Dict:
    """Start a teach-back session."""
    session_id = f"{user_id}_{int(time.time())}"
    probes = _get_probes(topic)

    sessions = _load_sessions()
    sessions[session_id] = {
        "user_id": user_id,
        "topic": topic,
        "lab_id": lab_id,
        "started_at": datetime.now().isoformat(),
        "status": "active",
        "exchanges": [],
        "probes_used": 0,
        "total_probes": len(probes),
        "scores": None,
    }
    _save_sessions(sessions)

    # First probe: "Explain this to me"
    opening = f"I'm a curious student who knows nothing about {topic}. Can you explain it to me from scratch?"

    return {
        "ok": True,
        "session_id": session_id,
        "ai_message": opening,
        "probes_remaining": len(probes),
    }


def continue_teachback(session_id: str, user_explanation: str) -> Dict:
    """Process a user's explanation and return a probing follow-up."""
    sessions = _load_sessions()
    if session_id not in sessions:
        return {"ok": False, "error": "Session not found"}

    session = sessions[session_id]
    if session["status"] != "active":
        return {"ok": False, "error": "Session already completed"}

    topic = session["topic"]
    probes = _get_probes(topic)
    probe_idx = session["probes_used"]

    # Score this explanation
    scores = _score_explanation(user_explanation, topic)

    # Record exchange
    session["exchanges"].append({
        "user": user_explanation,
        "scores": scores,
        "timestamp": datetime.now().isoformat(),
    })

    # Try LLM follow-up first
    llm_response = _llm_followup(user_explanation, topic, session["exchanges"])

    if llm_response:
        ai_message = llm_response
    elif probe_idx < len(probes):
        ai_message = _format_probe(probes[probe_idx], topic)
    else:
        ai_message = None  # Session complete

    session["probes_used"] = probe_idx + 1

    # Check if session is done
    if session["probes_used"] >= len(probes) or ai_message is None:
        session["status"] = "completed"
        session["completed_at"] = datetime.now().isoformat()
        # Compute final scores as average across all exchanges
        all_scores = [ex["scores"] for ex in session["exchanges"]]
        final = {}
        for key in all_scores[0]:
            final[key] = round(sum(s[key] for s in all_scores) / len(all_scores), 3)
        session["scores"] = final
        ai_message = ai_message or _generate_feedback(final, topic)
        session["exchanges"].append({"ai_feedback": ai_message})

    sessions[session_id] = session
    _save_sessions(sessions)

    return {
        "ok": True,
        "session_id": session_id,
        "ai_message": ai_message,
        "scores": scores,
        "probes_remaining": max(0, len(probes) - session["probes_used"]),
        "status": session["status"],
        "final_scores": session["scores"] if session["status"] == "completed" else None,
    }


def _generate_feedback(scores: Dict, topic: str) -> str:
    """Generate final feedback based on scores."""
    overall = scores.get("overall", 0)
    parts = []

    if overall >= 0.7:
        parts.append(f"Excellent teach-back on {topic}! You demonstrated strong understanding.")
    elif overall >= 0.4:
        parts.append(f"Good effort on {topic}. You've got the basics but there's room to deepen.")
    else:
        parts.append(f"Keep working on {topic}. Teaching it back revealed some gaps to fill.")

    weakest = min(scores, key=lambda k: scores[k] if k != "overall" else 999)
    if weakest == "clarity":
        parts.append("💡 Tip: Try to structure your explanation more clearly — start simple, build up.")
    elif weakest == "depth":
        parts.append("💡 Tip: Go deeper — explain *why* things work, not just *what* they are.")
    elif weakest == "accessibility":
        parts.append("💡 Tip: Use more analogies and simple language. Imagine your audience is a beginner.")
    elif weakest == "completeness":
        parts.append("💡 Tip: Cover more aspects of the topic — you may have missed key pieces.")

    return " ".join(parts)


def _llm_followup(explanation: str, topic: str, history: List[Dict]) -> Optional[str]:
    """Use LLM to generate a strategic follow-up question."""
    try:
        import importlib
        if importlib.util.find_spec("ollama"):
            ollama = importlib.import_module("ollama")
            history_text = "\n".join(
                f"Student: {ex['user']}" for ex in history[-3:] if "user" in ex
            )
            prompt = f"""You are playing the role of a curious but skeptical student learning about {topic}.
The student (who is actually the teacher) just explained:

"{explanation}"

Previous exchanges:
{history_text}

Ask ONE strategic follow-up question that:
1. Tests whether they truly understand (not just memorized)
2. Pushes them toward edge cases or deeper reasoning
3. Sounds natural and curious, not adversarial

Keep it to 1-2 sentences. Don't say "great explanation" — play dumb."""
            resp = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
            return resp["message"]["content"]
    except Exception:
        pass
    return None


def get_teachback_history(user_id: str) -> List[Dict]:
    """Get all teach-back sessions for a user."""
    sessions = _load_sessions()
    user_sessions = []
    for sid, s in sessions.items():
        if s["user_id"] == user_id:
            user_sessions.append({
                "session_id": sid,
                "topic": s["topic"],
                "lab_id": s["lab_id"],
                "status": s["status"],
                "started_at": s["started_at"],
                "scores": s.get("scores"),
                "exchange_count": len(s["exchanges"]),
            })
    return sorted(user_sessions, key=lambda x: x["started_at"], reverse=True)


# ---------------------------------------------------------------------------
# Flask Route Registration
# ---------------------------------------------------------------------------

def register_feynman_routes(app):
    """Register TEACH mode routes on a Flask app."""

    @app.route("/api/feynman/start", methods=["POST"])
    def api_feynman_start():
        body = request.get_json(force=True)
        user_id = body.get("user_id", "default")
        topic = body.get("topic", "")
        lab_id = body.get("lab_id", "")
        if not topic:
            return jsonify({"ok": False, "error": "Topic required"})
        return jsonify(start_teachback(user_id, topic, lab_id))

    @app.route("/api/feynman/respond", methods=["POST"])
    def api_feynman_respond():
        body = request.get_json(force=True)
        session_id = body.get("session_id", "")
        explanation = body.get("explanation", "")
        if not session_id or not explanation:
            return jsonify({"ok": False, "error": "session_id and explanation required"})
        return jsonify(continue_teachback(session_id, explanation))

    @app.route("/api/feynman/history")
    def api_feynman_history():
        user_id = request.args.get("user_id", "default")
        return jsonify({"ok": True, "sessions": get_teachback_history(user_id)})
