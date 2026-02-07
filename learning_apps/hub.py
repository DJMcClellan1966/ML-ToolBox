"""
Learning Apps Hub — Single entry point for all labs using Flask Blueprints.
Run from repo root: python learning_apps/hub.py
Open http://127.0.0.1:5000
All labs accessible at /lab/<lab_id>/
"""
import sys
import importlib
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

app = Flask(__name__)

# --- Register All Lab Blueprints ---
from learning_apps.blueprint_factory import create_lab_blueprint

def register_lab_blueprints():
    """Register all labs as blueprints under /lab/<lab_id>/"""
    lab_configs = [
        {"id": "ml_learning_lab", "title": "ML Learning Lab", 
         "desc": "Knuth, Skiena, Sedgewick, Bishop/Goodfellow/R&N, Info theory, Compass, Build Knuth Machine.",
         "module_path": "ml_learning_lab"},
        {"id": "clrs_algorithms_lab", "title": "CLRS Algorithms Lab",
         "desc": "Introduction to Algorithms: DP, Greedy, Graph.",
         "module_path": "learning_apps.clrs_algorithms_lab"},
        {"id": "deep_learning_lab", "title": "Deep Learning Lab",
         "desc": "Goodfellow, Bishop, ESL, Burkov.",
         "module_path": "learning_apps.deep_learning_lab"},
        {"id": "ai_concepts_lab", "title": "AI Concepts Lab",
         "desc": "Russell & Norvig: game theory, search, RL, probabilistic reasoning.",
         "module_path": "learning_apps.ai_concepts_lab"},
        {"id": "cross_domain_lab", "title": "Cross-Domain Lab",
         "desc": "Quantum, stat mech, linguistics, precognition, self-organization.",
         "module_path": "learning_apps.cross_domain_lab"},
        {"id": "python_practice_lab", "title": "Python Practice Lab",
         "desc": "Reed & Zelle: problem decomposition, algorithms, code organization.",
         "module_path": "learning_apps.python_practice_lab"},
        {"id": "sicp_lab", "title": "SICP Lab",
         "desc": "Structure and Interpretation of Computer Programs.",
         "module_path": "learning_apps.sicp_lab"},
        {"id": "practical_ml_lab", "title": "Practical ML Lab",
         "desc": "Hands-On ML (Géron): features, tuning, ensembles, production.",
         "module_path": "learning_apps.practical_ml_lab"},
        {"id": "rl_lab", "title": "RL Lab",
         "desc": "Sutton & Barto: MDPs, TD, Q-learning, policy gradient.",
         "module_path": "learning_apps.rl_lab"},
        {"id": "probabilistic_ml_lab", "title": "Probabilistic ML Lab",
         "desc": "Murphy: graphical models, EM, variational inference, Bayesian.",
         "module_path": "learning_apps.probabilistic_ml_lab"},
        {"id": "ml_theory_lab", "title": "ML Theory Lab",
         "desc": "Shalev-Shwartz & Ben-David: PAC, VC dimension, generalization.",
         "module_path": "learning_apps.ml_theory_lab"},
        {"id": "llm_engineers_lab", "title": "LLM Engineers Lab",
         "desc": "Handbook + Build Your Own LLM: RAG, prompts, eval, safety.",
         "module_path": "learning_apps.llm_engineers_lab"},
        {"id": "math_for_ml_lab", "title": "Math for ML Lab",
         "desc": "Linear algebra, calculus, probability, optimization.",
         "module_path": "learning_apps.math_for_ml_lab"},
    ]
    
    registered = []
    for cfg in lab_configs:
        try:
            # Try to import curriculum and demos
            curriculum_module = None
            demos_module = None
            try:
                curriculum_module = importlib.import_module(f"{cfg['module_path']}.curriculum")
            except:
                pass
            try:
                demos_module = importlib.import_module(f"{cfg['module_path']}.demos")
            except:
                pass
            
            # Create and register blueprint
            bp = create_lab_blueprint(
                lab_id=cfg["id"],
                title=cfg["title"],
                description=cfg["desc"],
                curriculum_module=curriculum_module,
                demos_module=demos_module
            )
            app.register_blueprint(bp)
            registered.append(cfg["id"])
        except Exception as e:
            print(f"Warning: Could not register {cfg['id']}: {e}")
    
    return registered

REGISTERED_LABS = register_lab_blueprints()
print(f"Registered {len(REGISTERED_LABS)} lab blueprints: {', '.join(REGISTERED_LABS)}")

# Practice problem generator
try:
    from learning_apps.practice_llm import register_practice_routes
    register_practice_routes(app)
except Exception:
    pass

# Voice/video tutor placeholder
try:
    from learning_apps.voice_video import register_voice_video_routes
    register_voice_video_routes(app)
except Exception:
    pass

# Analytics dashboard
try:
    from learning_apps.analytics import register_analytics_routes
    register_analytics_routes(app)
except Exception:
    pass

# Gamification system
try:
    from learning_apps import gamification
except Exception:
    gamification = None

@app.route("/api/gamification/badges")
def api_gamification_badges():
    user_id = request.args.get("user", "default")
    if gamification:
        badges = gamification.get_user_badges(user_id)
        badge_info = [gamification.get_badge_info(b) for b in badges]
        return jsonify({"ok": True, "badges": badge_info})
    return jsonify({"ok": False, "error": "Gamification not available"})

@app.route("/api/gamification/leaderboard")
def api_gamification_leaderboard():
    if gamification:
        return jsonify({"ok": True, "leaderboard": gamification.get_leaderboard()})
    return jsonify({"ok": False, "error": "Gamification not available"})

LABS = [
    {"name": "ML Learning Lab", "path": "ml_learning_lab", "url": "/lab/ml_learning_lab/",
     "desc": "Knuth, Skiena, Sedgewick, Bishop/Goodfellow/R&N, Info theory, Compass, Build Knuth Machine.", "icon": "🧠"},
    {"name": "CLRS Algorithms Lab", "path": "clrs_algorithms_lab", "url": "/lab/clrs_algorithms_lab/",
     "desc": "Introduction to Algorithms: DP, Greedy, Graph.", "icon": "📊"},
    {"name": "Deep Learning Lab", "path": "deep_learning_lab", "url": "/lab/deep_learning_lab/",
     "desc": "Goodfellow, Bishop, ESL, Burkov.", "icon": "🔮"},
    {"name": "AI Concepts Lab", "path": "ai_concepts_lab", "url": "/lab/ai_concepts_lab/",
     "desc": "Russell & Norvig: game theory, search, RL, probabilistic reasoning.", "icon": "🤖"},
    {"name": "Cross-Domain Lab", "path": "cross_domain_lab", "url": "/lab/cross_domain_lab/",
     "desc": "Quantum, stat mech, linguistics, precognition, self-organization.", "icon": "🌐"},
    {"name": "Python Practice Lab", "path": "python_practice_lab", "url": "/lab/python_practice_lab/",
     "desc": "Reed & Zelle: problem decomposition, algorithms, code organization.", "icon": "🐍"},
    {"name": "SICP Lab", "path": "sicp_lab", "url": "/lab/sicp_lab/",
     "desc": "Structure and Interpretation of Computer Programs.", "icon": "📖"},
    {"name": "Practical ML Lab", "path": "practical_ml_lab", "url": "/lab/practical_ml_lab/",
     "desc": "Hands-On ML (Géron): features, tuning, ensembles, production.", "icon": "🛠️"},
    {"name": "RL Lab", "path": "rl_lab", "url": "/lab/rl_lab/",
     "desc": "Sutton & Barto: MDPs, TD, Q-learning, policy gradient.", "icon": "🎮"},
    {"name": "Probabilistic ML Lab", "path": "probabilistic_ml_lab", "url": "/lab/probabilistic_ml_lab/",
     "desc": "Murphy: graphical models, EM, variational inference, Bayesian.", "icon": "📈"},
    {"name": "ML Theory Lab", "path": "ml_theory_lab", "url": "/lab/ml_theory_lab/",
     "desc": "Shalev-Shwartz & Ben-David: PAC, VC dimension, generalization.", "icon": "📐"},
    {"name": "LLM Engineers Lab", "path": "llm_engineers_lab", "url": "/lab/llm_engineers_lab/",
     "desc": "Handbook + Build Your Own LLM: RAG, prompts, eval, safety.", "icon": "💬"},
    {"name": "Math for ML Lab", "path": "math_for_ml_lab", "url": "/lab/math_for_ml_lab/",
     "desc": "Linear algebra, calculus, probability, optimization.", "icon": "➗"},
]


def _load_all_curricula():
    """Load curricula from all labs for global search."""
    all_items = []
    for lab in LABS:
        path = lab["path"]
        try:
            if path == "ml_learning_lab":
                continue  # Different structure
            mod = importlib.import_module(f"learning_apps.{path}.curriculum")
            items = mod.get_curriculum()
            for item in items:
                item["lab_name"] = lab["name"]
                item["lab_url"] = lab["url"]
                item["lab_path"] = path
            all_items.extend(items)
        except Exception:
            pass
    return all_items


@app.route("/api/search")
def api_search():
    """Global search across all labs."""
    q = request.args.get("q", "").lower().strip()
    if not q:
        return jsonify({"ok": True, "results": [], "query": q})
    
    all_items = _load_all_curricula()
    results = []
    for item in all_items:
        title = item.get("title", "").lower()
        learn = item.get("learn", "").lower()
        if q in title or q in learn:
            results.append({
                "id": item.get("id"),
                "title": item.get("title"),
                "learn": item.get("learn", "")[:150] + "..." if len(item.get("learn", "")) > 150 else item.get("learn", ""),
                "level": item.get("level"),
                "lab_name": item.get("lab_name"),
                "lab_url": item.get("lab_url"),
                "lab_path": item.get("lab_path"),
                "has_demo": bool(item.get("try_demo")),
            })
    return jsonify({"ok": True, "results": results[:20], "query": q, "total": len(results)})


@app.route("/api/labs")
def api_labs():
    """List all labs with their info."""
    return jsonify({"ok": True, "labs": LABS})


@app.route("/api/progress/all")
def api_progress_all():
    """Get progress across all labs."""
    try:
        from learning_apps import progress as progress_module
        user_id = request.args.get("user", "default")
        return jsonify(progress_module.get_user_progress(user_id))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/tutors")
def api_tutors():
    """Get all AI tutor characters."""
    try:
        from learning_apps.ai_tutor import TUTOR_CHARACTERS
        tutors = []
        for lab_id, char in TUTOR_CHARACTERS.items():
            if lab_id != "default":
                # Find the matching lab
                lab = next((l for l in LABS if l["path"] == lab_id), None)
                tutors.append({
                    "lab_id": lab_id,
                    "lab_name": lab["name"] if lab else lab_id,
                    "lab_url": lab["url"] if lab else f"/lab/{lab_id}/",
                    "id": char["id"],
                    "name": char["name"],
                    "title": char["title"],
                    "avatar": char["avatar"],
                    "quote": char["quote"],
                    "strengths": char["personality"]["strengths"]
                })
        return jsonify({"ok": True, "tutors": tutors})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "tutors": []})


LEARNING_PATHS = [
    {
        "id": "beginner_to_expert",
        "name": "Beginner to Expert",
        "desc": "A complete journey from programming basics to advanced AI and ML.",
        "labs": [
            "python_practice_lab",
            "math_for_ml_lab",
            "clrs_algorithms_lab",
            "practical_ml_lab",
            "deep_learning_lab",
            "ai_concepts_lab",
            "rl_lab",
            "llm_engineers_lab"
        ]
    },
    {
        "id": "ml_specialist",
        "name": "ML Specialist",
        "desc": "Focus on practical and theoretical machine learning skills.",
        "labs": [
            "practical_ml_lab",
            "deep_learning_lab",
            "probabilistic_ml_lab",
            "ml_theory_lab"
        ]
    },
    {
        "id": "ai_foundations",
        "name": "AI Foundations",
        "desc": "Core AI concepts, algorithms, and reasoning.",
        "labs": [
            "ai_concepts_lab",
            "clrs_algorithms_lab",
            "math_for_ml_lab"
        ]
    },
    {
        "id": "career_prep",
        "name": "Career & Interview Prep",
        "desc": "Sharpen your skills for interviews and real-world projects.",
        "labs": [
            "python_practice_lab",
            "clrs_algorithms_lab",
            "ml_theory_lab"
        ]
    },
    {
        "id": "project_based",
        "name": "Project-Based Learning",
        "desc": "Build real projects across labs for hands-on experience.",
        "labs": [
            "practical_ml_lab",
            "llm_engineers_lab",
            "cross_domain_lab"
        ]
    },
    {
        "id": "knuth_taocp",
        "name": "Donald Knuth: The Art of Computer Programming",
        "desc": "Master algorithms through TAOCP: Random numbers, sorting, searching, combinatorics, and the Knuth Machine.",
        "labs": [
            "ml_learning_lab",
            "clrs_algorithms_lab",
            "sicp_lab",
            "math_for_ml_lab"
        ]
    },
    {
        "id": "algorithms_master",
        "name": "Algorithms Mastery",
        "desc": "Deep dive into classical algorithms: CLRS, Knuth, Skiena, and Sedgewick.",
        "labs": [
            "clrs_algorithms_lab",
            "ml_learning_lab",
            "python_practice_lab",
            "sicp_lab"
        ]
    },
    {
        "id": "rl_research",
        "name": "Reinforcement Learning & Decision Making",
        "desc": "Master RL from basics to advanced: MDPs, Q-learning, policy gradients, and real-world applications.",
        "labs": [
            "rl_lab",
            "ai_concepts_lab",
            "probabilistic_ml_lab",
            "deep_learning_lab"
        ]
    },
    {
        "id": "probabilistic_ai",
        "name": "Probabilistic AI & Uncertainty",
        "desc": "Bayesian methods, graphical models, variational inference, and probabilistic reasoning.",
        "labs": [
            "probabilistic_ml_lab",
            "ai_concepts_lab",
            "math_for_ml_lab",
            "ml_theory_lab"
        ]
    },
    {
        "id": "research_frontiers",
        "name": "Research Frontiers",
        "desc": "Explore cutting-edge topics: quantum ML, LLMs, cross-domain thinking, and emerging paradigms.",
        "labs": [
            "cross_domain_lab",
            "llm_engineers_lab",
            "deep_learning_lab",
            "probabilistic_ml_lab"
        ]
    }
]

@app.route("/api/paths")
def api_paths():
    """List all guided learning paths."""
    return jsonify({"ok": True, "paths": LEARNING_PATHS})


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Learning Apps Hub</title>
  <style>
    :root {
      --bg-primary: #0f172a;
      --bg-secondary: #1e293b;
      --bg-tertiary: #334155;
      --text-primary: #f1f5f9;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --accent: #3b82f6;
      --accent-hover: #2563eb;
      --success: #22c55e;
      --warning: #f59e0b;
      --border: #475569;
      --radius: 12px;
    }
    .light-theme {
      --bg-primary: #f8fafc;
      --bg-secondary: #ffffff;
      --bg-tertiary: #e2e8f0;
      --text-primary: #1e293b;
      --text-secondary: #475569;
      --text-muted: #94a3b8;
      --border: #cbd5e1;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: var(--bg-primary);
      color: var(--text-primary);
      min-height: 100vh;
      line-height: 1.6;
    }
    
    .header {
      background: var(--bg-secondary);
      border-bottom: 1px solid var(--border);
      padding: 20px 24px;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .header-inner {
      max-width: 1200px;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }
    .header h1 {
      font-size: 1.8rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .header .icon { font-size: 1.5rem; }
    .header-desc { color: var(--text-secondary); font-size: 0.95rem; margin-top: 4px; }
    .header-actions { display: flex; gap: 10px; align-items: center; }
    
    .search-box {
      display: flex;
      align-items: center;
      background: var(--bg-primary);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 8px 16px;
      min-width: 300px;
    }
    .search-box input {
      background: transparent;
      border: none;
      color: var(--text-primary);
      font-size: 1rem;
      outline: none;
      flex: 1;
      padding: 4px 0;
    }
    .search-box input::placeholder { color: var(--text-muted); }
    .search-box .icon { color: var(--text-muted); margin-right: 8px; }
    
    .btn {
      padding: 10px 16px;
      border-radius: 8px;
      font-size: 0.95rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
      border: 1px solid var(--border);
      background: var(--bg-tertiary);
      color: var(--text-primary);
    }
    .btn:hover { background: var(--accent); border-color: var(--accent); color: white; }
    
    .main {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }
    
    .stats-bar {
      display: flex;
      gap: 24px;
      margin-bottom: 24px;
      flex-wrap: wrap;
    }
    .stat-card {
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px 24px;
      flex: 1;
      min-width: 200px;
    }
    .stat-card h3 {
      font-size: 2rem;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .stat-card p { color: var(--text-secondary); font-size: 0.9rem; }
    .stat-card.accent h3 { color: var(--accent); }
    .stat-card.success h3 { color: var(--success); }
    .stat-card.warning h3 { color: var(--warning); }
    
    .labs-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
      gap: 20px;
    }
    
    .lab-card {
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px;
      transition: all 0.2s;
      cursor: pointer;
    }
    .lab-card:hover {
      border-color: var(--accent);
      transform: translateY(-3px);
      box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .lab-card .lab-header {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }
    .lab-card .lab-icon {
      font-size: 2rem;
      background: var(--bg-tertiary);
      width: 50px;
      height: 50px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 12px;
    }
    .lab-card h2 { font-size: 1.2rem; font-weight: 600; }
    .lab-card .port { color: var(--text-muted); font-size: 0.85rem; margin-left: 8px; }
    .lab-card .desc {
      color: var(--text-secondary);
      font-size: 0.9rem;
      line-height: 1.5;
      margin-bottom: 16px;
    }
    .lab-card .status {
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
      padding: 10px 14px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 0.9rem;
      color: #fff;
      text-align: center;
      cursor: pointer;
      transition: all 0.3s;
    }
    .lab-card .status:hover { 
      transform: scale(1.02);
      box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    .lab-card .cmd {
      background: var(--bg-primary);
      padding: 10px 14px;
      border-radius: 8px;
      font-family: 'Fira Code', Consolas, monospace;
      font-size: 0.85rem;
      color: var(--text-secondary);
      overflow-x: auto;
    }
    .lab-card .cmd:hover { color: var(--text-primary); }
    
    .search-results {
      display: none;
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      margin-bottom: 24px;
      max-height: 400px;
      overflow-y: auto;
    }
    .search-results.active { display: block; }
    .search-results-header {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .search-results-header h3 { font-size: 1rem; }
    .search-results-header .close {
      background: transparent;
      border: none;
      color: var(--text-secondary);
      font-size: 1.2rem;
      cursor: pointer;
    }
    .search-result {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      cursor: pointer;
      transition: background 0.2s;
    }
    .search-result:last-child { border-bottom: none; }
    .search-result:hover { background: var(--bg-tertiary); }
    .search-result h4 {
      font-size: 1rem;
      margin-bottom: 4px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .search-result .level-badge {
      font-size: 0.7rem;
      padding: 2px 8px;
      border-radius: 10px;
      text-transform: uppercase;
    }
    .level-badge.basics { background: #dcfce7; color: #166534; }
    .level-badge.intermediate { background: #dbeafe; color: #1e40af; }
    .level-badge.advanced { background: #fef3c7; color: #92400e; }
    .level-badge.expert { background: #fce7f3; color: #9d174d; }
    .search-result p { color: var(--text-secondary); font-size: 0.85rem; }
    .search-result .lab-tag {
      color: var(--accent);
      font-size: 0.8rem;
      margin-top: 6px;
    }
    
    /* Tutors Carousel */
    .tutors-carousel {
      position: relative;
      margin-bottom: 24px;
    }
    .tutors-scroll {
      display: flex;
      gap: 16px;
      overflow-x: auto;
      padding: 8px 4px;
      scroll-behavior: smooth;
      -webkit-overflow-scrolling: touch;
    }
    .tutors-scroll::-webkit-scrollbar {
      height: 6px;
    }
    .tutors-scroll::-webkit-scrollbar-track {
      background: var(--bg-tertiary);
      border-radius: 3px;
    }
    .tutors-scroll::-webkit-scrollbar-thumb {
      background: var(--accent);
      border-radius: 3px;
    }
    .tutor-card {
      flex: 0 0 280px;
      background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px;
      cursor: pointer;
      transition: all 0.3s;
      position: relative;
      overflow: hidden;
    }
    .tutor-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 4px;
      background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    .tutor-card:hover {
      transform: translateY(-4px);
      box-shadow: 0 12px 40px rgba(59, 130, 246, 0.2);
      border-color: var(--accent);
    }
    .tutor-avatar-large {
      width: 60px;
      height: 60px;
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 28px;
      margin-bottom: 12px;
    }
    .tutor-card h3 {
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 4px;
    }
    .tutor-card .tutor-title {
      color: var(--accent);
      font-size: 0.85rem;
      margin-bottom: 10px;
    }
    .tutor-card .tutor-quote {
      font-style: italic;
      color: var(--text-secondary);
      font-size: 0.85rem;
      line-height: 1.4;
      border-left: 2px solid var(--accent);
      padding-left: 10px;
      margin-bottom: 12px;
    }
    .tutor-card .tutor-strengths {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .tutor-card .strength-tag {
      background: rgba(59, 130, 246, 0.15);
      color: var(--accent);
      padding: 4px 10px;
      border-radius: 12px;
      font-size: 0.75rem;
    }
    
    @media (max-width: 768px) {
      .header-inner { flex-direction: column; align-items: flex-start; }
      .search-box { min-width: 100%; }
      .labs-grid { grid-template-columns: 1fr; }
      .stat-card { min-width: 100%; }
      .tutor-card { flex: 0 0 260px; }
    }
  </style>
</head>
<body>
  <header class="header">
    <div class="header-inner">
      <div>
        <h1><span class="icon">📚</span> Learning Apps Hub</h1>
        <p class="header-desc">13 specialized labs with AI tutors inspired by the greatest minds in CS & ML</p>
      </div>
      <div class="header-actions">
        <div class="search-box">
          <span class="icon">🔍</span>
          <input type="text" id="global-search" placeholder="Search all topics (Ctrl+K)...">
        </div>
        <button class="btn" id="theme-toggle">🌙</button>
      </div>
    </div>
  </header>
  
  <main class="main">
    <!-- Search Results -->
    <div class="search-results" id="search-results">
      <div class="search-results-header">
        <h3 id="search-count">Results</h3>
        <button class="close" id="close-search">&times;</button>
      </div>
      <div id="search-results-list"></div>
    </div>
    
    <!-- Stats -->
    <div class="stats-bar">
      <div class="stat-card accent">
        <h3>{{ labs|length }}</h3>
        <p>Learning Labs</p>
      </div>
      <div class="stat-card success">
        <h3 id="total-topics">0</h3>
        <p>Total Topics</p>
      </div>
      <div class="stat-card warning">
        <h3 id="user-completed">0</h3>
        <p>Completed</p>
      </div>
    </div>
    
    <!-- Learning Paths Section -->
    <div class="section-header" style="margin: 32px 0 16px;">
      <h2 style="font-size: 1.4rem; display: flex; align-items: center; gap: 10px;">
        <span>🗺️</span> Guided Learning Paths
      </h2>
      <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 4px;">
        Follow a recommended sequence of labs for your goals
      </p>
    </div>
    <div class="labs-grid" id="paths-list" style="margin-bottom: 32px;">
      <!-- Paths loaded dynamically -->
    </div>

    <!-- AI Tutors Section -->
    <div class="section-header" style="margin: 32px 0 16px;">
      <h2 style="font-size: 1.4rem; display: flex; align-items: center; gap: 10px;">
        <span>🎓</span> Meet Your AI Tutors
      </h2>
      <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 4px;">
        Learn from the greatest minds in CS & ML — each lab features a personalized AI mentor
      </p>
    </div>
    <div class="tutors-carousel" id="tutors-carousel">
      <div class="tutors-scroll" id="tutors-scroll">
        <!-- Tutors loaded dynamically -->
      </div>
    </div>
    
    <!-- Section Header for Labs -->
    <div class="section-header" style="margin: 32px 0 16px;">
      <h2 style="font-size: 1.4rem; display: flex; align-items: center; gap: 10px;">
        <span>📚</span> Learning Labs
      </h2>
      <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 4px;">
        Choose a lab to start your learning journey
      </p>
    </div>
    
    <!-- Labs Grid -->
    <div class="labs-grid">
      {% for lab in labs %}
      <a href="{{ lab.url }}" class="lab-card">
        <div class="lab-header">
          <div class="lab-icon">{{ lab.icon }}</div>
          <div>
            <h2>{{ lab.name }}</h2>
          </div>
        </div>
        <p class="desc">{{ lab.desc }}</p>
        <div class="status">Enter Lab →</div>
      </a>
      {% endfor %}
    </div>
  </main>
  
  <script>
    const api = (path) => fetch(path).then(r => r.json());
    const labs = {{ labs|tojson }};
    
    // Theme
    function initTheme() {
      if (localStorage.getItem('theme') === 'light') document.body.classList.add('light-theme');
      updateThemeIcon();
    }
    function updateThemeIcon() {
      document.getElementById('theme-toggle').textContent = 
        document.body.classList.contains('light-theme') ? '☀️' : '🌙';
    }
    document.getElementById('theme-toggle').onclick = () => {
      document.body.classList.toggle('light-theme');
      localStorage.setItem('theme', document.body.classList.contains('light-theme') ? 'light' : 'dark');
      updateThemeIcon();
    };
    initTheme();
    
    // Search
    let searchTimeout = null;
    document.getElementById('global-search').oninput = (e) => {
      clearTimeout(searchTimeout);
      const q = e.target.value.trim();
      if (!q) {
        document.getElementById('search-results').classList.remove('active');
        return;
      }
      searchTimeout = setTimeout(async () => {
        const data = await api('/api/search?q=' + encodeURIComponent(q));
        if (data.ok) {
          document.getElementById('search-count').textContent = 
            data.total + ' result' + (data.total !== 1 ? 's' : '') + ' for "' + q + '"';
          const list = document.getElementById('search-results-list');
          list.innerHTML = '';
          data.results.forEach(r => {
            const div = document.createElement('div');
            div.className = 'search-result';
            div.innerHTML = `
              <h4>${r.title} <span class="level-badge ${r.level}">${r.level}</span></h4>
              <p>${r.learn}</p>
              <div class="lab-tag">📚 ${r.lab_name} ${r.has_demo ? '• ▶ Has Demo' : ''}</div>
            `;
            div.onclick = () => window.location.href = '/lab/' + r.lab_path + '/';
            list.appendChild(div);
          });
          document.getElementById('search-results').classList.add('active');
        }
      }, 300);
    };
    
    document.getElementById('close-search').onclick = () => {
      document.getElementById('search-results').classList.remove('active');
      document.getElementById('global-search').value = '';
    };
    
    // Keyboard shortcut
    document.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        document.getElementById('global-search').focus();
      }
      if (e.key === 'Escape') {
        document.getElementById('lab-modal').classList.remove('active');
        document.getElementById('search-results').classList.remove('active');
      }
    });
    
    // Load stats
    async function loadStats() {
      try {
        const data = await api('/api/search?q=');  // Get total by empty search
        // Count total topics by loading all curricula
        let total = 0;
        const labs = {{ labs|tojson }};
        for (const lab of labs) {
          try {
            const r = await fetch(lab.url + 'api/curriculum');
            if (r.ok) {
              const d = await r.json();
              if (d.ok) total += (d.items || []).length;
            }
          } catch {}
        }
        document.getElementById('total-topics').textContent = total || '80+';
        
        const progress = await api('/api/progress/all');
        if (progress.ok && progress.stats) {
          document.getElementById('user-completed').textContent = progress.stats.completed || 0;
        }
      } catch {}
    }
    
    // Load AI Tutors
    async function loadTutors() {
      try {
        const data = await api('/api/tutors');
        if (data.ok && data.tutors) {
          const scroll = document.getElementById('tutors-scroll');
          scroll.innerHTML = '';
          data.tutors.forEach(tutor => {
            const card = document.createElement('div');
            card.className = 'tutor-card';
            card.innerHTML = `
              <div class="tutor-avatar-large">${tutor.avatar}</div>
              <h3>${tutor.name}</h3>
              <div class="tutor-title">${tutor.title}</div>
              <div class="tutor-quote">"${tutor.quote.length > 100 ? tutor.quote.substring(0, 100) + '...' : tutor.quote}"</div>
              <div class="tutor-strengths">
                ${(tutor.strengths || []).slice(0, 3).map(s => `<span class="strength-tag">${s}</span>`).join('')}
              </div>
            `;
            card.onclick = () => window.location.href = '/lab/' + tutor.lab_id + '/';
            card.title = 'Click to learn with ' + tutor.name + ' in ' + tutor.lab_name;
            scroll.appendChild(card);
          });
        }
      } catch (e) {
        console.log('Tutors not available:', e);
      }
    }
    
    // Load Guided Learning Paths
    async function loadPaths() {
      try {
        const data = await api('/api/paths');
        if (data.ok && data.paths) {
          const div = document.getElementById('paths-list');
          div.innerHTML = '';
          data.paths.forEach(path => {
            const el = document.createElement('a');
            el.href = '/lab/' + path.labs[0] + '/';
            el.className = 'lab-card';
            el.innerHTML = `<h3>${path.name}</h3><p>${path.desc}</p><div style="margin-top:8px;">` +
              path.labs.map(lab => `<span class="level-badge" style="margin-right:6px;">${lab.replace(/_/g,' ').replace('lab','Lab')}</span>`).join('') +
              `</div>`;
            div.appendChild(el);
          });
        }
      } catch {}
    }
    
    loadStats();
    loadTutors();
    loadPaths();
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML, labs=LABS)

if __name__ == "__main__":
    port = int(__import__("os").environ.get("PORT", 5000))
    print("Learning Apps Hub — http://127.0.0.1:{}/".format(port))
    app.run(host="127.0.0.1", port=port, debug=False)
