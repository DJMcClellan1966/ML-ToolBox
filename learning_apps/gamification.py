"""
Gamification System for Learning Apps
Tracks achievements, badges, and leaderboards for users.
"""
import json
from pathlib import Path
from typing import Dict, Any, List

GAMIFY_FILE = Path(__file__).parent / "user_gamification.json"

BADGES = [
    {"id": "starter", "name": "Getting Started", "desc": "Complete your first topic."},
    {"id": "explorer", "name": "Explorer", "desc": "Complete topics in 3 different labs."},
    {"id": "streak", "name": "Streak", "desc": "Complete topics 5 days in a row."},
    {"id": "demo_runner", "name": "Demo Runner", "desc": "Run 5 demos."},
    {"id": "completionist", "name": "Completionist", "desc": "Complete all topics in a lab."},
]


def _load_gamify() -> Dict[str, Any]:
    if GAMIFY_FILE.exists():
        try:
            with open(GAMIFY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"users": {}}
    return {"users": {}}


def _save_gamify(data: Dict[str, Any]) -> None:
    try:
        with open(GAMIFY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def get_user_badges(user_id: str) -> List[str]:
    data = _load_gamify()
    return data["users"].get(user_id, {}).get("badges", [])


def award_badge(user_id: str, badge_id: str) -> None:
    data = _load_gamify()
    if user_id not in data["users"]:
        data["users"][user_id] = {"badges": [], "achievements": []}
    if badge_id not in data["users"][user_id]["badges"]:
        data["users"][user_id]["badges"].append(badge_id)
    _save_gamify(data)


def get_leaderboard() -> List[Dict[str, Any]]:
    data = _load_gamify()
    leaderboard = []
    for user, info in data["users"].items():
        leaderboard.append({"user": user, "badges": len(info.get("badges", []))})
    leaderboard.sort(key=lambda x: -x["badges"])
    return leaderboard[:10]


def check_and_award_badges(user_id: str, progress: Dict[str, Any]) -> None:
    """
    Check progress and award badges automatically.
    """
    # Starter badge
    if progress.get("completed", 0) >= 1:
        award_badge(user_id, "starter")
    # Explorer badge
    if len(progress.get("labs", {})) >= 3:
        award_badge(user_id, "explorer")
    # Completionist badge
    for lab, lab_data in progress.get("labs", {}).items():
        topics = lab_data.get("topics", {})
        if topics and all(t.get("status") == "completed" for t in topics.values()):
            award_badge(user_id, "completionist")
    # Demo Runner badge
    if progress.get("demo_runs", 0) >= 5:
        award_badge(user_id, "demo_runner")
    # Streak badge (simple: completed >= 5)
    if progress.get("completed", 0) >= 5:
        award_badge(user_id, "streak")


def get_badge_info(badge_id: str) -> Dict[str, Any]:
    for b in BADGES:
        if b["id"] == badge_id:
            return b
    return {"id": badge_id, "name": badge_id, "desc": ""}
