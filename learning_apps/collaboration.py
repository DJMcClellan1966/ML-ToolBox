"""
Basic Real-Time Collaboration System for Learning Apps
Implements simple group chat per lab using in-memory storage (for demo).
"""
from flask import request, jsonify
from datetime import datetime

# In-memory chat storage (reset on server restart)
CHAT_HISTORY = {}


def get_chat_history(lab_id: str) -> list:
    return CHAT_HISTORY.get(lab_id, [])[-50:]


def post_chat_message(lab_id: str, user: str, message: str) -> dict:
    if lab_id not in CHAT_HISTORY:
        CHAT_HISTORY[lab_id] = []
    msg = {
        "user": user,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    CHAT_HISTORY[lab_id].append(msg)
    return msg


def register_collab_routes(app, lab_id: str):
    @app.route(f"/api/collab/chat", methods=["GET"])
    def api_collab_chat():
        return jsonify({"ok": True, "history": get_chat_history(lab_id)})

    @app.route(f"/api/collab/chat", methods=["POST"])
    def api_collab_post():
        data = request.get_json(silent=True) or {}
        user = data.get("user", "anon")
        message = data.get("message", "")
        if not message:
            return jsonify({"ok": False, "error": "Message required"}), 400
        msg = post_chat_message(lab_id, user, message)
        return jsonify({"ok": True, "message": msg})
