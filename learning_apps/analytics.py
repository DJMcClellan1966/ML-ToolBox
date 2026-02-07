"""
Default Analytics Dashboard for Learning Apps
Provides user progress stats, topic completion, demo runs, and badge summary.
"""
from flask import request, jsonify

# Uses progress and gamification modules
try:
    from learning_apps import progress as progress_module
except Exception:
    progress_module = None
try:
    from learning_apps import gamification
except Exception:
    gamification = None


def register_analytics_routes(app):
    @app.route("/api/analytics/dashboard")
    def api_analytics_dashboard():
        user_id = request.args.get("user", "default")
        stats = {}
        if progress_module:
            stats = progress_module.get_user_progress(user_id).get("stats", {})
        badges = []
        if gamification:
            badges = gamification.get_user_badges(user_id)
        return jsonify({
            "ok": True,
            "stats": stats,
            "badges": badges
        })
