"""
PORTFOLIO Mode — Exportable Proof of Work.

Generates a static HTML portfolio site from a learner's completed work:
projects built, papers reproduced, topics mastered, code written,
Feynman scores, and skill certificates.
"""
import json
import time
import html
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from flask import request, jsonify, Response

DATA_DIR = Path(__file__).resolve().parent / ".data"


def _load_json(path: Path) -> Any:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def gather_portfolio_data(user_id: str) -> Dict:
    """Gather all portfolio data for a user from every system."""
    portfolio = {
        "user_id": user_id,
        "generated_at": datetime.now().isoformat(),
        "projects": [],
        "papers": [],
        "teachbacks": [],
        "journal_entries": [],
        "challenges_won": [],
        "progress": {},
        "badges": [],
        "skills": {},
    }

    # Projects (BUILD mode)
    projects_data = _load_json(DATA_DIR / "user_projects.json")
    for key, proj in projects_data.items():
        if proj.get("user_id") == user_id:
            completed_milestones = sum(
                1 for m in proj.get("milestones", {}).values()
                if m.get("status") == "completed"
            )
            total_milestones = len(proj.get("milestones", {}))
            portfolio["projects"].append({
                "project_id": proj["project_id"],
                "status": proj.get("status", "in_progress"),
                "started_at": proj.get("started_at"),
                "completed_at": proj.get("completed_at"),
                "milestones_completed": completed_milestones,
                "milestones_total": total_milestones,
                "code_samples": {
                    mid: m.get("code", "")[:500]
                    for mid, m in proj.get("milestones", {}).items()
                    if m.get("code")
                },
            })

    # Papers (RESEARCH mode)
    papers_data = _load_json(DATA_DIR / "paper_trail.json")
    for key, entry in papers_data.items():
        if entry.get("user_id") == user_id:
            claims_verified = sum(
                1 for v in entry.get("claims_verified", {}).values() if v is True
            )
            portfolio["papers"].append({
                "paper_id": entry["paper_id"],
                "status": entry.get("status"),
                "started_at": entry.get("started_at"),
                "claims_verified": claims_verified,
                "claims_total": len(entry.get("claims_verified", {})),
                "has_code": bool(entry.get("code")),
                "annotation_count": len(entry.get("annotations", [])),
            })

    # Teach-backs (TEACH mode)
    teachback_data = _load_json(DATA_DIR / "teachback_sessions.json")
    for sid, session in teachback_data.items():
        if session.get("user_id") == user_id and session.get("status") == "completed":
            portfolio["teachbacks"].append({
                "topic": session["topic"],
                "lab_id": session.get("lab_id"),
                "scores": session.get("scores"),
                "exchange_count": len(session.get("exchanges", [])),
                "completed_at": session.get("completed_at"),
            })

    # Journal entries (JOURNAL mode)
    journal_data = _load_json(DATA_DIR / "learning_journal.json")
    user_journal = journal_data.get(user_id, {})
    entries = user_journal.get("entries", [])
    portfolio["journal_entries"] = [
        {"topic": e.get("topic"), "type": e.get("type"), "timestamp": e.get("timestamp")}
        for e in entries[-20:]  # last 20
    ]

    # Challenge results (COMPETE mode)
    arena_data = _load_json(DATA_DIR / "challenge_arena.json")
    for key, sub in arena_data.items():
        if sub.get("user_id") == user_id and sub.get("all_passed"):
            portfolio["challenges_won"].append({
                "challenge_id": sub.get("challenge_id"),
                "solved_at": sub.get("submitted_at"),
                "execution_time_ms": sub.get("execution_time_ms"),
            })

    # Progress data
    progress_file = Path(__file__).resolve().parent / "user_progress.json"
    if progress_file.exists():
        progress_data = json.loads(progress_file.read_text())
        user_progress = progress_data.get(user_id, {})
        for lab_id, topics in user_progress.items():
            if isinstance(topics, dict):
                completed = sum(1 for t in topics.values() if isinstance(t, dict) and t.get("status") == "completed")
                total = len(topics)
                if total > 0:
                    portfolio["progress"][lab_id] = {
                        "completed": completed,
                        "total": total,
                        "pct": round(100 * completed / total),
                    }

    # Badges
    try:
        from learning_apps import gamification
        badges = gamification.get_user_badges(user_id)
        portfolio["badges"] = [gamification.get_badge_info(b) for b in badges]
    except Exception:
        pass

    # Compute skill summary
    skill_evidence = {}
    for proj in portfolio["projects"]:
        if proj["status"] == "completed":
            skill_evidence.setdefault("building", []).append(proj["project_id"])
    for paper in portfolio["papers"]:
        if paper["status"] == "completed":
            skill_evidence.setdefault("research", []).append(paper["paper_id"])
    for tb in portfolio["teachbacks"]:
        overall = (tb.get("scores") or {}).get("overall", 0)
        if overall >= 0.6:
            skill_evidence.setdefault("teaching", []).append(tb["topic"])
    for lab_id, prog in portfolio["progress"].items():
        if prog["pct"] >= 80:
            skill_evidence.setdefault("mastery", []).append(lab_id)
    portfolio["skills"] = skill_evidence

    return portfolio


def generate_portfolio_html(portfolio: Dict) -> str:
    """Generate a standalone HTML portfolio page."""
    user = html.escape(portfolio["user_id"])
    date = portfolio["generated_at"][:10]

    projects_html = ""
    for p in portfolio["projects"]:
        status_color = "#22c55e" if p["status"] == "completed" else "#f59e0b"
        projects_html += f"""
        <div class="card">
            <h3>{html.escape(p['project_id'].replace('_', ' ').title())}
                <span class="badge" style="background:{status_color}">{p['status']}</span>
            </h3>
            <p>Milestones: {p['milestones_completed']}/{p['milestones_total']}</p>
        </div>"""

    papers_html = ""
    for p in portfolio["papers"]:
        papers_html += f"""
        <div class="card">
            <h3>{html.escape(p['paper_id'].replace('_', ' ').title())}</h3>
            <p>Claims verified: {p['claims_verified']}/{p['claims_total']}
               {'✅' if p['status'] == 'completed' else '🔄'}</p>
        </div>"""

    teachbacks_html = ""
    for t in portfolio["teachbacks"]:
        score = (t.get("scores") or {}).get("overall", 0)
        teachbacks_html += f"""
        <div class="card">
            <h3>{html.escape(t['topic'])}</h3>
            <p>Feynman Score: <strong>{score:.0%}</strong> | Exchanges: {t['exchange_count']}</p>
        </div>"""

    progress_html = ""
    for lab_id, prog in portfolio["progress"].items():
        pct = prog["pct"]
        progress_html += f"""
        <div class="card">
            <h3>{html.escape(lab_id.replace('_', ' ').title())}</h3>
            <div class="progress-bar"><div class="fill" style="width:{pct}%"></div></div>
            <p>{prog['completed']}/{prog['total']} topics ({pct}%)</p>
        </div>"""

    skills_html = ""
    for skill, evidence in portfolio.get("skills", {}).items():
        skills_html += f'<span class="skill-tag">{html.escape(skill)} ({len(evidence)})</span> '

    challenges_html = f"<p><strong>{len(portfolio['challenges_won'])}</strong> challenges solved</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{user}'s ML Learning Portfolio</title>
<style>
  :root {{ --bg: #0f172a; --card: #1e293b; --text: #f1f5f9; --muted: #94a3b8;
           --accent: #3b82f6; --success: #22c55e; --border: #334155; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); padding: 40px 24px; }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  h1 {{ font-size: 2.2rem; margin-bottom: 8px; }}
  h2 {{ font-size: 1.4rem; margin: 32px 0 16px; border-bottom: 2px solid var(--accent); padding-bottom: 8px; }}
  .subtitle {{ color: var(--muted); margin-bottom: 32px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 16px; }}
  .card h3 {{ font-size: 1.1rem; margin-bottom: 8px; display: flex; align-items: center; gap: 10px; }}
  .badge {{ font-size: 0.75rem; padding: 2px 10px; border-radius: 10px; color: white; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
  .progress-bar {{ background: var(--border); border-radius: 6px; height: 8px; margin: 8px 0; }}
  .fill {{ background: var(--accent); height: 100%; border-radius: 6px; }}
  .skill-tag {{ display: inline-block; background: rgba(59,130,246,0.15); color: var(--accent); padding: 6px 14px; border-radius: 20px; margin: 4px; font-weight: 600; }}
  .stats {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 32px; }}
  .stat {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px 24px; flex: 1; min-width: 140px; text-align: center; }}
  .stat h3 {{ font-size: 2rem; color: var(--accent); }}
  .stat p {{ color: var(--muted); font-size: 0.9rem; }}
  .footer {{ margin-top: 48px; text-align: center; color: var(--muted); font-size: 0.85rem; }}
</style>
</head>
<body>
<div class="container">
  <h1>🎓 {user}'s ML Learning Portfolio</h1>
  <p class="subtitle">Generated {date} from ML Learning Platform</p>

  <div class="stats">
    <div class="stat"><h3>{len(portfolio['projects'])}</h3><p>Projects</p></div>
    <div class="stat"><h3>{len(portfolio['papers'])}</h3><p>Papers</p></div>
    <div class="stat"><h3>{len(portfolio['teachbacks'])}</h3><p>Teach-backs</p></div>
    <div class="stat"><h3>{len(portfolio['challenges_won'])}</h3><p>Challenges</p></div>
    <div class="stat"><h3>{sum(p['pct'] for p in portfolio['progress'].values()) // max(1, len(portfolio['progress']))}%</h3><p>Avg Mastery</p></div>
  </div>

  <h2>🛠️ Skills</h2>
  <div class="card">{skills_html or '<p>Complete projects, teach topics, and reproduce papers to build your skill profile.</p>'}</div>

  <h2>🏗️ Projects Built</h2>
  <div class="grid">{projects_html or '<div class="card"><p>No projects yet. Start one in BUILD mode!</p></div>'}</div>

  <h2>📄 Papers Reproduced</h2>
  <div class="grid">{papers_html or '<div class="card"><p>No papers yet. Start in RESEARCH mode!</p></div>'}</div>

  <h2>🎤 Teach-backs</h2>
  <div class="grid">{teachbacks_html or '<div class="card"><p>No teach-backs yet. Try TEACH mode!</p></div>'}</div>

  <h2>🏆 Challenge Arena</h2>
  <div class="card">{challenges_html}</div>

  <h2>📊 Lab Progress</h2>
  <div class="grid">{progress_html or '<div class="card"><p>Start learning in any lab to track progress.</p></div>'}</div>

  <div class="footer">
    <p>Generated by ML Learning Platform • {date}</p>
  </div>
</div>
</body>
</html>"""


def generate_certificate(user_id: str, skill: str, evidence: List[str]) -> str:
    """Generate a skill certificate HTML."""
    date = datetime.now().strftime("%B %d, %Y")
    items = "".join(f"<li>{html.escape(e.replace('_', ' ').title())}</li>" for e in evidence)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Certificate — {html.escape(skill)}</title>
<style>
  body {{ font-family: Georgia, serif; background: #f8fafc; display: flex; justify-content: center; padding: 40px; }}
  .cert {{ background: white; border: 3px double #1e293b; border-radius: 8px; padding: 60px; max-width: 700px; text-align: center; }}
  h1 {{ font-size: 2rem; margin-bottom: 8px; color: #1e293b; }}
  h2 {{ font-size: 1.5rem; color: #3b82f6; margin: 20px 0; }}
  .name {{ font-size: 1.8rem; color: #1e293b; font-weight: bold; margin: 20px 0; }}
  ul {{ text-align: left; display: inline-block; margin: 20px 0; }}
  .date {{ color: #64748b; margin-top: 24px; }}
</style></head>
<body><div class="cert">
  <h1>🎓 Certificate of Achievement</h1>
  <p>This certifies that</p>
  <div class="name">{html.escape(user_id)}</div>
  <p>has demonstrated proficiency in</p>
  <h2>{html.escape(skill.title())}</h2>
  <p>Supported by the following evidence:</p>
  <ul>{items}</ul>
  <p class="date">Awarded {date} • ML Learning Platform</p>
</div></body></html>"""


# ---------------------------------------------------------------------------
# Flask Route Registration
# ---------------------------------------------------------------------------

def register_portfolio_routes(app):
    """Register PORTFOLIO mode routes on a Flask app."""

    @app.route("/api/portfolio/data")
    def api_portfolio_data():
        user_id = request.args.get("user_id", "default")
        return jsonify({"ok": True, "portfolio": gather_portfolio_data(user_id)})

    @app.route("/api/portfolio/export")
    def api_portfolio_export():
        user_id = request.args.get("user_id", "default")
        portfolio = gather_portfolio_data(user_id)
        html_content = generate_portfolio_html(portfolio)
        return Response(html_content, mimetype="text/html",
                        headers={"Content-Disposition": f"inline; filename={user_id}_portfolio.html"})

    @app.route("/api/portfolio/certificate")
    def api_portfolio_certificate():
        user_id = request.args.get("user_id", "default")
        skill = request.args.get("skill", "")
        if not skill:
            return jsonify({"ok": False, "error": "Skill parameter required"})
        portfolio = gather_portfolio_data(user_id)
        evidence = portfolio.get("skills", {}).get(skill, [])
        if not evidence:
            return jsonify({"ok": False, "error": f"No evidence for skill: {skill}"})
        html_content = generate_certificate(user_id, skill, evidence)
        return Response(html_content, mimetype="text/html")
