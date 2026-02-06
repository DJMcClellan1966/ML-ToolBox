"""
Progress Tracking for Learning Apps.
Stores user progress in a local JSON file.
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

PROGRESS_FILE = Path(__file__).parent / "user_progress.json"


def _load_progress() -> Dict[str, Any]:
    """Load progress from file."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"users": {}}
    return {"users": {}}


def _save_progress(data: Dict[str, Any]) -> None:
    """Save progress to file."""
    try:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:
        pass


def mark_topic_started(user_id: str, lab_id: str, topic_id: str) -> Dict[str, Any]:
    """Mark a topic as started (in-progress)."""
    data = _load_progress()
    if user_id not in data["users"]:
        data["users"][user_id] = {"labs": {}, "created_at": datetime.now().isoformat()}
    if lab_id not in data["users"][user_id]["labs"]:
        data["users"][user_id]["labs"][lab_id] = {"topics": {}}
    
    topics = data["users"][user_id]["labs"][lab_id]["topics"]
    if topic_id not in topics:
        topics[topic_id] = {
            "status": "in-progress",
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "demo_runs": 0
        }
    elif topics[topic_id]["status"] == "not-started":
        topics[topic_id]["status"] = "in-progress"
        topics[topic_id]["started_at"] = datetime.now().isoformat()
    
    _save_progress(data)
    return {"ok": True, "status": topics[topic_id]["status"]}


def mark_topic_completed(user_id: str, lab_id: str, topic_id: str) -> Dict[str, Any]:
    """Mark a topic as completed."""
    data = _load_progress()
    if user_id not in data["users"]:
        data["users"][user_id] = {"labs": {}, "created_at": datetime.now().isoformat()}
    if lab_id not in data["users"][user_id]["labs"]:
        data["users"][user_id]["labs"][lab_id] = {"topics": {}}
    
    topics = data["users"][user_id]["labs"][lab_id]["topics"]
    if topic_id not in topics:
        topics[topic_id] = {
            "status": "completed",
            "started_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "demo_runs": 0
        }
    else:
        topics[topic_id]["status"] = "completed"
        topics[topic_id]["completed_at"] = datetime.now().isoformat()
    
    _save_progress(data)
    return {"ok": True, "status": "completed"}


def record_demo_run(user_id: str, lab_id: str, topic_id: str) -> Dict[str, Any]:
    """Record a demo run for a topic."""
    data = _load_progress()
    if user_id not in data["users"]:
        data["users"][user_id] = {"labs": {}, "created_at": datetime.now().isoformat()}
    if lab_id not in data["users"][user_id]["labs"]:
        data["users"][user_id]["labs"][lab_id] = {"topics": {}}
    
    topics = data["users"][user_id]["labs"][lab_id]["topics"]
    if topic_id not in topics:
        topics[topic_id] = {
            "status": "in-progress",
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "demo_runs": 1
        }
    else:
        topics[topic_id]["demo_runs"] = topics[topic_id].get("demo_runs", 0) + 1
    
    _save_progress(data)
    return {"ok": True, "demo_runs": topics[topic_id]["demo_runs"]}


def get_user_progress(user_id: str) -> Dict[str, Any]:
    """Get all progress for a user."""
    data = _load_progress()
    if user_id not in data["users"]:
        return {"ok": True, "labs": {}, "stats": {"total_topics": 0, "completed": 0, "in_progress": 0}}
    
    user_data = data["users"][user_id]
    stats = {"total_topics": 0, "completed": 0, "in_progress": 0, "demo_runs": 0}
    
    for lab_id, lab_data in user_data.get("labs", {}).items():
        for topic_id, topic_data in lab_data.get("topics", {}).items():
            stats["total_topics"] += 1
            if topic_data.get("status") == "completed":
                stats["completed"] += 1
            elif topic_data.get("status") == "in-progress":
                stats["in_progress"] += 1
            stats["demo_runs"] += topic_data.get("demo_runs", 0)
    
    return {"ok": True, "labs": user_data.get("labs", {}), "stats": stats}


def get_lab_progress(user_id: str, lab_id: str) -> Dict[str, Any]:
    """Get progress for a specific lab."""
    data = _load_progress()
    if user_id not in data["users"]:
        return {"ok": True, "topics": {}, "stats": {"total": 0, "completed": 0, "in_progress": 0}}
    
    labs = data["users"][user_id].get("labs", {})
    if lab_id not in labs:
        return {"ok": True, "topics": {}, "stats": {"total": 0, "completed": 0, "in_progress": 0}}
    
    topics = labs[lab_id].get("topics", {})
    stats = {"total": len(topics), "completed": 0, "in_progress": 0}
    for t in topics.values():
        if t.get("status") == "completed":
            stats["completed"] += 1
        elif t.get("status") == "in-progress":
            stats["in_progress"] += 1
    
    return {"ok": True, "topics": topics, "stats": stats}


def get_topic_status(user_id: str, lab_id: str, topic_id: str) -> str:
    """Get status of a single topic: 'not-started', 'in-progress', or 'completed'."""
    data = _load_progress()
    try:
        return data["users"][user_id]["labs"][lab_id]["topics"][topic_id]["status"]
    except KeyError:
        return "not-started"


def reset_progress(user_id: str, lab_id: Optional[str] = None) -> Dict[str, Any]:
    """Reset progress for a user (all labs or specific lab)."""
    data = _load_progress()
    if user_id not in data["users"]:
        return {"ok": True, "message": "No progress to reset"}
    
    if lab_id:
        if lab_id in data["users"][user_id].get("labs", {}):
            del data["users"][user_id]["labs"][lab_id]
    else:
        data["users"][user_id]["labs"] = {}
    
    _save_progress(data)
    return {"ok": True, "message": "Progress reset"}
