from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Dict

from flask import Flask, redirect, render_template, request, session, url_for

from curriculum import get_lessons, get_lesson, get_next_lesson_id
from mentor import evaluate_answer


APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / ".data"
PROGRESS_PATH = DATA_DIR / "progress.json"

app = Flask(__name__)
app.secret_key = os.environ.get("GUIDED_LEARNING_SECRET", "guided-learning-secret")


def _ensure_progress_file() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not PROGRESS_PATH.exists():
        PROGRESS_PATH.write_text("{}", encoding="utf-8")


def _load_progress() -> Dict:
    _ensure_progress_file()
    try:
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_progress(data: Dict) -> None:
    _ensure_progress_file()
    PROGRESS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _get_session_id() -> str:
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


def _get_user_progress() -> Dict:
    sid = _get_session_id()
    data = _load_progress()
    return data.get(sid, {"completed": [], "attempts": {}, "last_lesson": None})


def _set_user_progress(progress: Dict) -> None:
    sid = _get_session_id()
    data = _load_progress()
    data[sid] = progress
    _save_progress(data)


@app.route("/")
def index():
    lessons = get_lessons()
    progress = _get_user_progress()
    completed = set(progress.get("completed", []))
    total = len(lessons)
    done = len(completed)
    percent = int((done / total) * 100) if total else 0
    last_lesson = progress.get("last_lesson")
    return render_template(
        "index.html",
        lessons=lessons,
        completed=completed,
        total=total,
        done=done,
        percent=percent,
        last_lesson=last_lesson,
    )


@app.route("/lesson/<lesson_id>")
def lesson(lesson_id: str):
    lesson_data = get_lesson(lesson_id)
    if not lesson_data:
        return redirect(url_for("index"))

    progress = _get_user_progress()
    completed = set(progress.get("completed", []))
    attempts = progress.get("attempts", {}).get(lesson_id, 0)

    return render_template(
        "lesson.html",
        lesson=lesson_data,
        completed=lesson_id in completed,
        attempts=attempts,
        next_lesson_id=get_next_lesson_id(lesson_id),
    )


@app.route("/check", methods=["POST"])
def check_answer():
    lesson_id = request.form.get("lesson_id", "")
    answer = request.form.get("answer", "")
    lesson_data = get_lesson(lesson_id)
    if not lesson_data:
        return redirect(url_for("index"))

    progress = _get_user_progress()
    attempts = progress.get("attempts", {})
    attempts[lesson_id] = attempts.get(lesson_id, 0) + 1
    progress["attempts"] = attempts

    result = evaluate_answer(lesson_data, answer)

    if result["is_correct"]:
        completed = set(progress.get("completed", []))
        completed.add(lesson_id)
        progress["completed"] = sorted(completed)
        progress["last_lesson"] = lesson_id

    _set_user_progress(progress)

    return render_template(
        "lesson.html",
        lesson=lesson_data,
        completed=result["is_correct"],
        attempts=attempts[lesson_id],
        result=result,
        user_answer=answer,
        next_lesson_id=get_next_lesson_id(lesson_id),
    )


@app.route("/reset", methods=["POST"])
def reset_progress():
    progress = {"completed": [], "attempts": {}, "last_lesson": None}
    _set_user_progress(progress)
    return redirect(url_for("index"))


if __name__ == "__main__":
    print("Guided Learning App running at http://127.0.0.1:5050/")
    app.run(host="127.0.0.1", port=5050, debug=False)
