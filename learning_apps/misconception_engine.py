"""
Misconception Detection Engine.

Provides diagnostic micro-quizzes that detect common misconceptions and
route learners to targeted prerequisites and adaptive paths.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

DATA_DIR = Path(__file__).parent / ".cache"
DATA_DIR.mkdir(exist_ok=True)
PROFILE_FILE = DATA_DIR / "misconception_profiles.json"


@dataclass
class DiagnosticQuestion:
    id: str
    prompt: str
    choices: List[str]
    correct: int
    misconceptions: Dict[int, str]
    tags: List[str]


# Minimal seed bank (expandable)
QUESTION_BANK: List[DiagnosticQuestion] = [
    DiagnosticQuestion(
        id="dp_bellman",
        prompt="Which statement best describes the Bellman optimality equation?",
        choices=[
            "It averages over actions under a fixed policy",
            "It selects the best action to maximize expected return",
            "It only applies to episodic tasks",
            "It requires supervised labels"
        ],
        correct=1,
        misconceptions={
            0: "confusing_policy_vs_optimal",
            2: "thinking_episodic_only",
            3: "confusing_supervised_rl"
        },
        tags=["reinforcement_learning", "bellman_equation", "optimality"]
    ),
    DiagnosticQuestion(
        id="svm_margin",
        prompt="Why does maximizing margin often improve generalization in SVMs?",
        choices=[
            "It reduces training time",
            "It increases the number of features",
            "It creates a more robust decision boundary",
            "It guarantees zero training error"
        ],
        correct=2,
        misconceptions={
            0: "thinking_compute_overfit",
            1: "confusing_features_vs_margin",
            3: "assuming_hard_margin_always"
        },
        tags=["svm", "generalization", "margin"]
    ),
    DiagnosticQuestion(
        id="backprop_chain",
        prompt="What is the main role of the chain rule in backpropagation?",
        choices=[
            "It normalizes gradients",
            "It propagates error signals through layers",
            "It reduces model size",
            "It selects activation functions"
        ],
        correct=1,
        misconceptions={
            0: "thinking_normalization",
            2: "confusing_compression",
            3: "confusing_architecture_choice"
        },
        tags=["backprop", "chain_rule", "deep_learning"]
    )
]


def _load_profiles() -> Dict[str, Any]:
    if PROFILE_FILE.exists():
        try:
            return json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"users": {}}
    return {"users": {}}


def _save_profiles(data: Dict[str, Any]) -> None:
    try:
        PROFILE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def get_quiz(goal: Optional[str] = None, limit: int = 3) -> Dict[str, Any]:
    # For now: basic filter by goal keywords if provided
    items = QUESTION_BANK
    if goal:
        goal_l = goal.lower()
        items = [q for q in QUESTION_BANK if any(t in goal_l for t in q.tags)] or QUESTION_BANK
    selected = items[:limit]
    return {
        "ok": True,
        "quiz": [
            {
                "id": q.id,
                "prompt": q.prompt,
                "choices": q.choices,
                "tags": q.tags
            } for q in selected
        ]
    }


def submit_quiz(user_id: str, answers: Dict[str, int]) -> Dict[str, Any]:
    misconceptions: Dict[str, int] = {}
    tags: Dict[str, int] = {}

    for q in QUESTION_BANK:
        if q.id not in answers:
            continue
        chosen = answers[q.id]
        for t in q.tags:
            tags[t] = tags.get(t, 0) + 1
        if chosen != q.correct:
            tag = q.misconceptions.get(chosen, "unknown_misconception")
            misconceptions[tag] = misconceptions.get(tag, 0) + 1

    data = _load_profiles()
    user = data["users"].setdefault(user_id, {"events": []})
    user["events"].append({
        "ts": datetime.utcnow().isoformat(),
        "answers": answers,
        "misconceptions": misconceptions,
        "tags": tags
    })
    _save_profiles(data)

    return {
        "ok": True,
        "user_id": user_id,
        "misconceptions": misconceptions,
        "tags": tags
    }


def register_misconception_routes(app):
    from flask import request, jsonify

    @app.route("/api/diagnostic/quiz")
    def api_diagnostic_quiz():
        goal = request.args.get("goal", None)
        limit = int(request.args.get("limit", 3))
        return jsonify(get_quiz(goal, limit))

    @app.route("/api/diagnostic/submit", methods=["POST"])
    def api_diagnostic_submit():
        data = request.get_json(silent=True) or {}
        user_id = data.get("user_id", "default")
        answers = data.get("answers", {})
        result = submit_quiz(user_id, answers)

        # Recommend a path based on misconception tags
        goal = " ".join(result.get("tags", {}).keys())
        try:
            from learning_apps.unified_curriculum import recommend_path
            result["recommended_path"] = recommend_path(user_id, goal, 6)
        except Exception:
            result["recommended_path"] = {"ok": False, "error": "Unified path unavailable"}

        return jsonify(result)

    return app
