"""
JOURNAL Mode — Metacognitive Layer.

Learning journal with reflections, synthesis prompts, spaced reflection,
and AI-powered insight analysis. The layer that turns consumption into wisdom.
"""
import json
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from flask import request, jsonify

DATA_DIR = Path(__file__).resolve().parent / ".data"
JOURNAL_FILE = DATA_DIR / "learning_journal.json"

# ---------------------------------------------------------------------------
# Synthesis Prompts — cross-lab reflection triggers
# ---------------------------------------------------------------------------
SYNTHESIS_PROMPTS = [
    "Connect something from {lab_a} to something from {lab_b}. What's the common thread?",
    "What's the most counterintuitive thing you've learned this week? Why did it surprise you?",
    "If you had to explain your learning journey so far to a stranger, what's the 30-second version?",
    "What's one thing you thought you understood but realized you didn't?",
    "Pick a concept you learned recently. What question about it are you *not* able to answer yet?",
    "How would you use {concept} in a real-world project? Be specific.",
    "What's the connection between gradient descent and reinforcement learning? (Hint: both optimize.)",
    "If you could go back to when you started, what would you tell yourself?",
    "What's the most useful mental model you've built so far?",
    "Draw a concept map in words: pick 5 concepts and describe how they connect.",
    "What would break if you removed {concept} from machine learning entirely?",
    "Describe a moment this week when something 'clicked'. What triggered it?",
    "What are you most confused about right now? Be specific and honest.",
    "Compare two things you've learned that seem unrelated but share a deep connection.",
    "What's one thing you've learned that changed how you think about problems outside of ML?",
]

ENTRY_TYPES = [
    "reflection",      # Open-ended reflection on a topic
    "confusion",       # What's confusing right now
    "aha_moment",      # Something that clicked
    "synthesis",       # Cross-topic/cross-lab connection
    "goal_check",      # Am I on track?
    "weekly_review",   # Weekly summary
    "question",        # Unresolved question
]


def _load_journal() -> Dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if JOURNAL_FILE.exists():
        return json.loads(JOURNAL_FILE.read_text())
    return {}


def _save_journal(data: Dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JOURNAL_FILE.write_text(json.dumps(data, indent=2, default=str))


def _ensure_user(data: Dict, user_id: str) -> Dict:
    if user_id not in data:
        data[user_id] = {
            "entries": [],
            "goals": [],
            "streaks": {"current": 0, "best": 0, "last_entry_date": None},
        }
    return data[user_id]


def add_entry(user_id: str, entry_type: str, topic: str, content: str,
              lab_id: str = "", mood: str = "") -> Dict:
    """Add a journal entry."""
    if entry_type not in ENTRY_TYPES:
        return {"ok": False, "error": f"Unknown type. Use one of: {ENTRY_TYPES}"}
    if not content.strip():
        return {"ok": False, "error": "Content cannot be empty"}

    data = _load_journal()
    user = _ensure_user(data, user_id)

    entry = {
        "id": f"j_{int(time.time())}_{random.randint(100, 999)}",
        "type": entry_type,
        "topic": topic,
        "content": content,
        "lab_id": lab_id,
        "mood": mood,
        "timestamp": datetime.now().isoformat(),
        "ai_insight": None,
        "revisited": False,
        "revisit_note": None,
    }

    # Generate AI insight
    entry["ai_insight"] = _generate_insight(entry, user["entries"][-10:])

    user["entries"].append(entry)

    # Update streak
    today = datetime.now().strftime("%Y-%m-%d")
    if user["streaks"]["last_entry_date"] != today:
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        if user["streaks"]["last_entry_date"] == yesterday:
            user["streaks"]["current"] += 1
        else:
            user["streaks"]["current"] = 1
        user["streaks"]["best"] = max(user["streaks"]["best"], user["streaks"]["current"])
        user["streaks"]["last_entry_date"] = today

    _save_journal(data)
    return {"ok": True, "entry": entry, "streak": user["streaks"]}


def get_entries(user_id: str, entry_type: str = None, limit: int = 50) -> List[Dict]:
    """Get journal entries, optionally filtered."""
    data = _load_journal()
    user = _ensure_user(data, user_id)
    entries = user["entries"]
    if entry_type:
        entries = [e for e in entries if e["type"] == entry_type]
    return entries[-limit:]


def set_goal(user_id: str, goal: str, target_date: str = "") -> Dict:
    """Set a learning goal with optional target date."""
    data = _load_journal()
    user = _ensure_user(data, user_id)

    goal_entry = {
        "id": f"g_{int(time.time())}",
        "goal": goal,
        "target_date": target_date,
        "created_at": datetime.now().isoformat(),
        "status": "active",
        "check_ins": [],
    }
    user["goals"].append(goal_entry)
    _save_journal(data)
    return {"ok": True, "goal": goal_entry}


def check_in_goal(user_id: str, goal_id: str, note: str, progress_pct: int = 0) -> Dict:
    """Check in on a goal."""
    data = _load_journal()
    user = _ensure_user(data, user_id)

    for goal in user["goals"]:
        if goal["id"] == goal_id:
            goal["check_ins"].append({
                "note": note,
                "progress_pct": min(100, max(0, progress_pct)),
                "timestamp": datetime.now().isoformat(),
            })
            if progress_pct >= 100:
                goal["status"] = "completed"
                goal["completed_at"] = datetime.now().isoformat()
            _save_journal(data)
            return {"ok": True, "goal": goal}

    return {"ok": False, "error": "Goal not found"}


def get_goals(user_id: str, status: str = None) -> List[Dict]:
    """Get learning goals."""
    data = _load_journal()
    user = _ensure_user(data, user_id)
    goals = user["goals"]
    if status:
        goals = [g for g in goals if g["status"] == status]
    return goals


def get_synthesis_prompt(user_id: str) -> Dict:
    """Get a personalized synthesis prompt based on user's activity."""
    data = _load_journal()
    user = _ensure_user(data, user_id)

    # Find labs the user has journaled about
    labs = list(set(e.get("lab_id", "") for e in user["entries"] if e.get("lab_id")))
    concepts = list(set(e.get("topic", "") for e in user["entries"] if e.get("topic")))

    prompt = random.choice(SYNTHESIS_PROMPTS)

    # Substitute placeholders
    if "{lab_a}" in prompt and len(labs) >= 2:
        a, b = random.sample(labs, 2)
        prompt = prompt.replace("{lab_a}", a.replace("_", " ").title())
        prompt = prompt.replace("{lab_b}", b.replace("_", " ").title())
    elif "{lab_a}" in prompt:
        prompt = "What's the most important connection you've seen between two different topics you've studied?"

    if "{concept}" in prompt and concepts:
        prompt = prompt.replace("{concept}", random.choice(concepts))
    elif "{concept}" in prompt:
        prompt = prompt.replace("{concept}", "a core concept")

    return {"ok": True, "prompt": prompt}


def get_spaced_reflections(user_id: str, limit: int = 3) -> List[Dict]:
    """Get old journal entries for spaced reflection — revisit with fresh eyes."""
    data = _load_journal()
    user = _ensure_user(data, user_id)

    candidates = []
    now = datetime.now()
    for entry in user["entries"]:
        if entry.get("revisited"):
            continue
        entry_date = datetime.fromisoformat(entry["timestamp"])
        age_days = (now - entry_date).days
        # Revisit entries 7-30 days old
        if 7 <= age_days <= 90:
            candidates.append(entry)

    # Sort by age, prefer older entries
    candidates.sort(key=lambda e: e["timestamp"])
    return candidates[:limit]


def revisit_entry(user_id: str, entry_id: str, note: str) -> Dict:
    """Add a revisit note to an old journal entry."""
    data = _load_journal()
    user = _ensure_user(data, user_id)

    for entry in user["entries"]:
        if entry["id"] == entry_id:
            entry["revisited"] = True
            entry["revisit_note"] = note
            entry["revisited_at"] = datetime.now().isoformat()
            _save_journal(data)
            return {"ok": True, "entry": entry}

    return {"ok": False, "error": "Entry not found"}


def get_journal_stats(user_id: str) -> Dict:
    """Get journal statistics."""
    data = _load_journal()
    user = _ensure_user(data, user_id)

    entries = user["entries"]
    type_counts = {}
    for e in entries:
        t = e["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    # Topics frequency
    topic_counts = {}
    for e in entries:
        t = e.get("topic", "")
        if t:
            topic_counts[t] = topic_counts.get(t, 0) + 1
    top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_entries": len(entries),
        "type_breakdown": type_counts,
        "top_topics": top_topics,
        "streak": user["streaks"],
        "active_goals": len([g for g in user["goals"] if g["status"] == "active"]),
        "completed_goals": len([g for g in user["goals"] if g["status"] == "completed"]),
    }


def _generate_insight(entry: Dict, recent_entries: List[Dict]) -> Optional[str]:
    """Generate an AI insight about the journal entry."""
    # Try LLM first
    try:
        import importlib
        if importlib.util.find_spec("ollama"):
            ollama = importlib.import_module("ollama")
            context = "\n".join(f"- [{e['type']}] {e['topic']}: {e['content'][:100]}" for e in recent_entries[-5:])
            prompt = f"""A student wrote this learning journal entry:

Type: {entry['type']}
Topic: {entry['topic']}
Entry: {entry['content']}

Recent entries for context:
{context}

Give a brief (1-2 sentence) insightful observation. Look for:
- Patterns in their learning
- Connections they might be missing
- Encouragement tied to specific growth
Don't be generic. Be specific to what they wrote."""
            resp = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
            return resp["message"]["content"]
    except Exception:
        pass

    # Rule-based fallback
    content_lower = entry["content"].lower()
    if entry["type"] == "confusion":
        return "Naming your confusion is the first step to resolving it. Try teaching this topic to solidify understanding."
    elif entry["type"] == "aha_moment":
        return "Great insight! Write down what specifically triggered this — it reveals your best learning conditions."
    elif "connect" in content_lower or "relat" in content_lower:
        return "Cross-topic connections are a sign of deep learning. Keep building these bridges."
    elif len(entry["content"].split()) > 100:
        return "Detailed reflection shows strong engagement. Try distilling this to a single key takeaway."
    else:
        return "Consistent journaling builds metacognitive skill. Your future self will thank you."


# ---------------------------------------------------------------------------
# Flask Route Registration
# ---------------------------------------------------------------------------

def register_journal_routes(app):
    """Register JOURNAL mode routes on a Flask app."""

    @app.route("/api/journal/entry", methods=["POST"])
    def api_journal_add():
        body = request.get_json(force=True)
        return jsonify(add_entry(
            user_id=body.get("user_id", "default"),
            entry_type=body.get("type", "reflection"),
            topic=body.get("topic", ""),
            content=body.get("content", ""),
            lab_id=body.get("lab_id", ""),
            mood=body.get("mood", ""),
        ))

    @app.route("/api/journal/entries")
    def api_journal_entries():
        user_id = request.args.get("user_id", "default")
        entry_type = request.args.get("type")
        limit = int(request.args.get("limit", 50))
        return jsonify({"ok": True, "entries": get_entries(user_id, entry_type, limit)})

    @app.route("/api/journal/goal", methods=["POST"])
    def api_journal_goal():
        body = request.get_json(force=True)
        return jsonify(set_goal(
            body.get("user_id", "default"),
            body.get("goal", ""),
            body.get("target_date", ""),
        ))

    @app.route("/api/journal/goal/checkin", methods=["POST"])
    def api_journal_goal_checkin():
        body = request.get_json(force=True)
        return jsonify(check_in_goal(
            body.get("user_id", "default"),
            body.get("goal_id", ""),
            body.get("note", ""),
            body.get("progress_pct", 0),
        ))

    @app.route("/api/journal/goals")
    def api_journal_goals():
        user_id = request.args.get("user_id", "default")
        status = request.args.get("status")
        return jsonify({"ok": True, "goals": get_goals(user_id, status)})

    @app.route("/api/journal/prompt")
    def api_journal_prompt():
        user_id = request.args.get("user_id", "default")
        return jsonify(get_synthesis_prompt(user_id))

    @app.route("/api/journal/reflections")
    def api_journal_reflections():
        user_id = request.args.get("user_id", "default")
        return jsonify({"ok": True, "reflections": get_spaced_reflections(user_id)})

    @app.route("/api/journal/revisit", methods=["POST"])
    def api_journal_revisit():
        body = request.get_json(force=True)
        return jsonify(revisit_entry(
            body.get("user_id", "default"),
            body.get("entry_id", ""),
            body.get("note", ""),
        ))

    @app.route("/api/journal/stats")
    def api_journal_stats():
        user_id = request.args.get("user_id", "default")
        return jsonify({"ok": True, "stats": get_journal_stats(user_id)})
