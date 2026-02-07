"""
User Profile and Recommendation Engine for Personalized Learning Paths.
Stores user preferences, progress, and recommends topics/labs.
"""
import json
from pathlib import Path
from typing import Dict, Any, List

PROFILE_FILE = Path(__file__).parent / "user_profiles.json"


def _load_profiles() -> Dict[str, Any]:
    if PROFILE_FILE.exists():
        try:
            with open(PROFILE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"users": {}}
    return {"users": {}}


def _save_profiles(data: Dict[str, Any]) -> None:
    try:
        with open(PROFILE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def get_user_profile(user_id: str) -> Dict[str, Any]:
    data = _load_profiles()
    return data["users"].get(user_id, {})


def set_user_profile(user_id: str, profile: Dict[str, Any]) -> None:
    data = _load_profiles()
    data["users"][user_id] = profile
    _save_profiles(data)


def recommend_topics(user_id: str, all_topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Recommend topics based on user progress and interests.
    Simple rule-based: prioritize not-started, match interests, adjust for difficulty.
    """
    profile = get_user_profile(user_id)
    interests = profile.get("interests", [])
    progress = profile.get("progress", {})
    preferred_level = profile.get("preferred_level", "intermediate")
    
    # Score topics
    scored = []
    for topic in all_topics:
        score = 0
        if topic["id"] not in progress or progress[topic["id"]] == "not-started":
            score += 2
        if topic.get("level") == preferred_level:
            score += 1
        if any(interest in topic.get("title", "").lower() for interest in interests):
            score += 2
        scored.append((score, topic))
    scored.sort(reverse=True)
    return [t for s, t in scored[:10]]


def update_user_interests(user_id: str, interests: List[str]) -> None:
    profile = get_user_profile(user_id)
    profile["interests"] = interests
    set_user_profile(user_id, profile)


def update_user_preferred_level(user_id: str, level: str) -> None:
    profile = get_user_profile(user_id)
    profile["preferred_level"] = level
    set_user_profile(user_id, profile)


def update_user_progress(user_id: str, topic_id: str, status: str) -> None:
    profile = get_user_profile(user_id)
    if "progress" not in profile:
        profile["progress"] = {}
    profile["progress"][topic_id] = status
    set_user_profile(user_id, profile)
