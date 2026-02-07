"""
Voice/Video Tutor Placeholder API for Future Integration
Provides endpoints for starting/joining sessions and status checks.
"""
from flask import request, jsonify

# Placeholder in-memory session store
SESSIONS = {}


def register_voice_video_routes(app):
    @app.route("/api/voice_video/start", methods=["POST"])
    def api_voice_video_start():
        user = request.get_json(silent=True).get("user", "anon")
        session_id = f"session_{user}_{len(SESSIONS)+1}"
        SESSIONS[session_id] = {"user": user, "status": "active"}
        return jsonify({"ok": True, "session_id": session_id})

    @app.route("/api/voice_video/join", methods=["POST"])
    def api_voice_video_join():
        session_id = request.get_json(silent=True).get("session_id")
        if session_id in SESSIONS:
            SESSIONS[session_id]["status"] = "joined"
            return jsonify({"ok": True, "session_id": session_id, "status": "joined"})
        return jsonify({"ok": False, "error": "Session not found"})

    @app.route("/api/voice_video/status")
    def api_voice_video_status():
        session_id = request.args.get("session_id")
        if session_id in SESSIONS:
            return jsonify({"ok": True, "status": SESSIONS[session_id]["status"]})
        return jsonify({"ok": False, "error": "Session not found"})
