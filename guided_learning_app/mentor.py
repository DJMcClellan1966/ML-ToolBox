from __future__ import annotations

from typing import Dict, List


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _keyword_hits(answer: str, keywords: List[str]) -> List[str]:
    answer_norm = _normalize(answer)
    hits = []
    for keyword in keywords:
        if keyword.lower() in answer_norm:
            hits.append(keyword)
    return hits


def evaluate_answer(lesson: Dict, answer: str) -> Dict:
    keywords = lesson.get("check", {}).get("keywords", [])
    question = lesson.get("check", {}).get("question", "")
    hits = _keyword_hits(answer, keywords)

    min_hits = max(2, int(len(keywords) * 0.6)) if keywords else 1
    is_correct = len(hits) >= min_hits

    missing = [kw for kw in keywords if kw not in hits]

    if not answer.strip():
        feedback = "Try explaining it in your own words. A short paragraph is enough."
    elif is_correct:
        feedback = (
            "Great job! You captured the key ideas. "
            "If you want, add one short example to strengthen your understanding."
        )
    else:
        feedback = (
            "You're close. Include more of the key ideas in your answer. "
            "Look at the hints and try again."
        )

    hints = []
    if missing:
        hints = [f"Mention: {kw}" for kw in missing[:3]]

    return {
        "question": question,
        "is_correct": is_correct,
        "hits": hits,
        "missing": missing,
        "feedback": feedback,
        "hints": hints,
    }
