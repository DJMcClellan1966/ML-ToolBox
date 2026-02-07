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

# Spaced Repetition System
try:
    from learning_apps.spaced_repetition import register_srs_routes
    register_srs_routes(app)
except Exception:
    pass

# Code Playground
try:
    from learning_apps.code_playground import register_playground_routes
    register_playground_routes(app)
except Exception:
    pass

# Intelligent Demo System
try:
    from learning_apps.intelligent_demos import register_intelligent_demo_routes
    register_intelligent_demo_routes(app)
    print("✨ Intelligent Demo System loaded")
except Exception:
    pass

# D3.js Visualizations
try:
    from learning_apps.visualizations import register_visualization_routes
    register_visualization_routes(app)
    print("📊 D3.js Visualizations loaded")
except Exception:
    pass

# Deep Learning Experience
try:
    from learning_apps.deep_learning_experience import register_deep_learning_routes
    register_deep_learning_routes(app)
    print("🧠 Deep Learning Experience loaded")
except Exception:
    pass

# Unified Curriculum Brain
try:
  from learning_apps.unified_curriculum import register_unified_routes
  register_unified_routes(app)
  print("🧭 Unified Curriculum Brain loaded")
except Exception:
  pass

# Misconception Diagnostics
try:
  from learning_apps.misconception_engine import register_misconception_routes
  register_misconception_routes(app)
  print("🧩 Misconception Diagnostics loaded")
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

    /* Unified Curriculum Brain */
    .unified-panel {
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px;
      margin-bottom: 32px;
    }
    .unified-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin-top: 12px;
    }
    .unified-card {
      background: var(--bg-tertiary);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
    }
    .unified-card h4 { font-size: 1rem; margin-bottom: 10px; }
    .unified-input, .unified-textarea {
      width: 100%;
      background: var(--bg-primary);
      color: var(--text-primary);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 0.95rem;
      outline: none;
    }
    .unified-textarea { min-height: 90px; resize: vertical; }
    .unified-actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
    .unified-results {
      margin-top: 12px;
      background: var(--bg-primary);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      max-height: 260px;
      overflow-y: auto;
      font-size: 0.9rem;
      color: var(--text-secondary);
    }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; background: rgba(59,130,246,0.15); color: var(--accent); margin-right: 6px; }
    .diagnostic-question { margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid var(--border); }
    .diagnostic-choice { display: block; margin-top: 6px; }
    
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

    <!-- Unified Curriculum Brain -->
    <div class="section-header" style="margin: 20px 0 12px;">
      <h2 style="font-size: 1.4rem; display: flex; align-items: center; gap: 10px;">
        <span>🧭</span> Unified Curriculum Brain
      </h2>
      <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 4px;">
        Search across all books, get a combined answer, and generate an adaptive path.
      </p>
    </div>
    <div class="unified-panel">
      <div class="unified-grid">
        <div class="unified-card">
          <h4>🔎 Unified Search</h4>
          <input class="unified-input" id="unified-query" placeholder="Ask anything across all books...">
          <div class="unified-actions">
            <button class="btn" onclick="runUnifiedSearch()">Search</button>
            <button class="btn" onclick="buildUnifiedIndex()">Build Index</button>
          </div>
          <div class="unified-results" id="unified-search-results">No results yet.</div>
        </div>
        <div class="unified-card">
          <h4>🧠 Combined Answer</h4>
          <textarea class="unified-textarea" id="unified-question" placeholder="Explain backprop and link it to optimization and information theory..."></textarea>
          <div class="unified-actions">
            <button class="btn" onclick="runUnifiedSynthesis()">Synthesize</button>
          </div>
          <div class="unified-results" id="unified-synthesis-results">No synthesis yet.</div>
        </div>
        <div class="unified-card">
          <h4>🗺️ Adaptive Path</h4>
          <input class="unified-input" id="unified-goal" placeholder="Goal: become strong in RL and optimization">
          <div class="unified-actions">
            <button class="btn" onclick="runUnifiedPath()">Recommend</button>
          </div>
          <div class="unified-results" id="unified-path-results">No path yet.</div>
        </div>
      </div>
    </div>

    <!-- Misconception Diagnostics -->
    <div class="section-header" style="margin: 20px 0 12px;">
      <h2 style="font-size: 1.4rem; display: flex; align-items: center; gap: 10px;">
        <span>🧩</span> Misconception Diagnostics
      </h2>
      <p style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 4px;">
        A short diagnostic that finds misunderstandings and routes you to the right prerequisites.
      </p>
    </div>
    <div class="unified-panel">
      <div class="unified-grid">
        <div class="unified-card">
          <h4>⚡ Quick Diagnostic</h4>
          <input class="unified-input" id="diag-goal" placeholder="Goal: deep learning, RL, SVM, etc.">
          <div class="unified-actions">
            <button class="btn" onclick="loadDiagnosticQuiz()">Load Quiz</button>
            <button class="btn" onclick="submitDiagnosticQuiz()">Submit</button>
          </div>
          <div class="unified-results" id="diag-quiz">No quiz loaded.</div>
        </div>
        <div class="unified-card">
          <h4>🧭 Recommended Fixes</h4>
          <div class="unified-results" id="diag-results">No results yet.</div>
        </div>
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

    // Unified Curriculum Brain
    async function buildUnifiedIndex() {
      const el = document.getElementById('unified-search-results');
      el.textContent = 'Building index...';
      try {
        const r = await fetch('/api/unified/index', { method: 'POST' });
        const d = await r.json();
        el.textContent = d.ok ? ('Index ready. Chunks: ' + d.chunks) : 'Index build failed.';
      } catch (e) { el.textContent = 'Error: ' + e.message; }
    }

    async function runUnifiedSearch() {
      const q = document.getElementById('unified-query').value.trim();
      const el = document.getElementById('unified-search-results');
      if (!q) { el.textContent = 'Enter a query.'; return; }
      el.textContent = 'Searching...';
      try {
        const r = await fetch('/api/unified/search?q=' + encodeURIComponent(q) + '&top_k=6');
        const d = await r.json();
        if (!d.ok) { el.textContent = 'Search failed.'; return; }
        el.innerHTML = d.results.map(r =>
          `<div style="margin-bottom:10px;">
            <div><span class="pill">${r.meta?.kind || 'text'}</span><b>${r.title}</b></div>
            <div>${(r.text || '').slice(0, 200)}...</div>
          </div>`
        ).join('') || 'No results.';
      } catch (e) { el.textContent = 'Error: ' + e.message; }
    }

    async function runUnifiedSynthesis() {
      const q = document.getElementById('unified-question').value.trim();
      const el = document.getElementById('unified-synthesis-results');
      if (!q) { el.textContent = 'Enter a question.'; return; }
      el.textContent = 'Synthesizing...';
      try {
        const r = await fetch('/api/unified/synthesize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q, top_k: 6, use_llm: true })
        });
        const d = await r.json();
        el.textContent = d.ok ? (d.answer || 'No answer.') : 'Synthesis failed.';
      } catch (e) { el.textContent = 'Error: ' + e.message; }
    }

    async function runUnifiedPath() {
      const goal = document.getElementById('unified-goal').value.trim();
      const el = document.getElementById('unified-path-results');
      el.textContent = 'Calculating path...';
      try {
        const r = await fetch('/api/unified/path?user_id=default&goal=' + encodeURIComponent(goal) + '&limit=8');
        const d = await r.json();
        if (!d.ok) { el.textContent = 'Path failed.'; return; }
        el.innerHTML = (d.recommendations || []).map(x =>
          `<div style="margin-bottom:8px;">
             <span class="pill">${x.level}</span><b>${x.title}</b>
             <div style="color:var(--text-muted);">${x.lab_id} • ${x.book_name || ''}</div>
           </div>`
        ).join('') || 'No recommendations.';
      } catch (e) { el.textContent = 'Error: ' + e.message; }
    }

    // Misconception Diagnostics
    let diagQuiz = [];
    async function loadDiagnosticQuiz() {
      const goal = document.getElementById('diag-goal').value.trim();
      const el = document.getElementById('diag-quiz');
      el.textContent = 'Loading quiz...';
      try {
        const r = await fetch('/api/diagnostic/quiz?goal=' + encodeURIComponent(goal) + '&limit=3');
        const d = await r.json();
        if (!d.ok) { el.textContent = 'Quiz unavailable.'; return; }
        diagQuiz = d.quiz || [];
        if (!diagQuiz.length) { el.textContent = 'No questions found.'; return; }
        el.innerHTML = diagQuiz.map((q, idx) => {
          const choices = q.choices.map((c, i) =>
            `<label class="diagnostic-choice"><input type="radio" name="q_${idx}" value="${i}"> ${c}</label>`
          ).join('');
          return `<div class="diagnostic-question"><b>${q.prompt}</b>${choices}</div>`;
        }).join('');
      } catch (e) { el.textContent = 'Error: ' + e.message; }
    }

    async function submitDiagnosticQuiz() {
      const el = document.getElementById('diag-results');
      if (!diagQuiz.length) { el.textContent = 'Load the quiz first.'; return; }
      const answers = {};
      diagQuiz.forEach((q, idx) => {
        const sel = document.querySelector(`input[name="q_${idx}"]:checked`);
        if (sel) answers[q.id] = parseInt(sel.value, 10);
      });
      el.textContent = 'Analyzing...';
      try {
        const r = await fetch('/api/diagnostic/submit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: 'default', answers })
        });
        const d = await r.json();
        const mis = d.misconceptions || {};
        const misHtml = Object.keys(mis).length
          ? '<div><b>Detected misconceptions:</b> ' + Object.entries(mis).map(([k,v]) => `<span class="pill">${k} (${v})</span>`).join(' ') + '</div>'
          : '<div><b>No misconceptions detected.</b></div>';
        const recs = d.recommended_path?.recommendations || [];
        const recHtml = recs.map(x =>
          `<div style="margin-top:8px;">
             <span class="pill">${x.level}</span><b>${x.title}</b>
             <div style="color:var(--text-muted);">${x.lab_id} • ${x.book_name || ''}</div>
           </div>`
        ).join('') || '<div>No recommendations.</div>';
        el.innerHTML = misHtml + '<div style="margin-top:10px;"><b>Next best topics:</b></div>' + recHtml;
      } catch (e) { el.textContent = 'Error: ' + e.message; }
    }
    
    loadStats();
    loadTutors();
    loadPaths();
  </script>
  
  <!-- SRS Widget Styles -->
  <style>
    .floating-widgets { position: fixed; bottom: 24px; right: 24px; z-index: 100; display: flex; flex-direction: column; gap: 12px; }
    .widget-btn { 
      display: flex; align-items: center; gap: 8px;
      padding: 12px 20px; border-radius: 24px; border: none;
      cursor: pointer; font-weight: 600; color: white;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      transition: transform 0.2s;
    }
    .widget-btn:hover { transform: scale(1.05); }
    .srs-widget { background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%); }
    .playground-widget { background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%); }
    .widget-badge { background: #ef4444; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
    .widget-modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85); z-index: 200; padding: 24px; overflow-y: auto; }
    .widget-modal.active { display: block; }
    .modal-container { max-width: 600px; margin: 40px auto; background: var(--bg-secondary); border-radius: 16px; overflow: hidden; }
    .modal-header { display: flex; justify-content: space-between; align-items: center; padding: 16px 24px; border-bottom: 1px solid var(--border); }
    .modal-close { background: transparent; border: none; color: var(--text-primary); font-size: 1.5rem; cursor: pointer; }
    .modal-content { padding: 24px; }
    .srs-card-display { text-align: center; padding: 32px; }
    .srs-question { font-size: 1.3rem; margin-bottom: 24px; }
    .srs-answer { background: var(--bg-tertiary); padding: 20px; border-radius: 12px; margin: 16px 0; display: none; }
    .srs-answer.revealed { display: block; }
    .srs-ratings { display: flex; gap: 8px; justify-content: center; margin-top: 20px; }
    .srs-rating { padding: 10px 16px; border-radius: 8px; border: none; cursor: pointer; font-weight: 600; color: white; }
    .srs-rating.again { background: #ef4444; }
    .srs-rating.hard { background: #f59e0b; }
    .srs-rating.good { background: #22c55e; }
    .srs-rating.easy { background: #3b82f6; }
    .reveal-btn { background: var(--accent); color: white; border: none; padding: 12px 32px; border-radius: 8px; cursor: pointer; }
    .playground-container { max-width: 1000px; margin: 40px auto; }
    .playground-editor textarea { width: 100%; height: 250px; background: var(--bg-primary); border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-family: 'Fira Code', Consolas, monospace; font-size: 14px; color: var(--text-primary); resize: vertical; }
    .playground-output pre { background: var(--bg-primary); border-radius: 8px; padding: 12px; height: 200px; overflow-y: auto; font-family: monospace; font-size: 13px; margin: 0; white-space: pre-wrap; }
    .run-btn { background: #22c55e; color: white; border: none; padding: 10px 24px; border-radius: 8px; cursor: pointer; font-weight: 600; margin-top: 12px; }
    .challenge-select { background: var(--bg-tertiary); border: 1px solid var(--border); color: var(--text-primary); padding: 8px 12px; border-radius: 8px; margin-bottom: 12px; }
  </style>
  
  <!-- Floating Widget Buttons -->
  <div class="floating-widgets">
    <button class="widget-btn srs-widget" onclick="openSRS()">📚 Review <span class="widget-badge" id="srs-due">0</span></button>
    <button class="widget-btn playground-widget" onclick="openPlayground()">💻 Code</button>
  </div>
  
  <!-- SRS Modal -->
  <div class="widget-modal" id="srs-modal">
    <div class="modal-container">
      <div class="modal-header"><h2>📚 Spaced Repetition Review</h2><button class="modal-close" onclick="closeSRS()">×</button></div>
      <div class="modal-content srs-card-display">
        <div id="srs-content">
          <p id="srs-q" class="srs-question">Loading...</p>
          <div id="srs-a" class="srs-answer"></div>
          <button class="reveal-btn" id="reveal-btn" onclick="revealSRS()">Show Answer</button>
          <div class="srs-ratings" id="srs-ratings" style="display:none;">
            <button class="srs-rating again" onclick="rateSRS(1)">Again</button>
            <button class="srs-rating hard" onclick="rateSRS(3)">Hard</button>
            <button class="srs-rating good" onclick="rateSRS(4)">Good</button>
            <button class="srs-rating easy" onclick="rateSRS(5)">Easy</button>
          </div>
        </div>
        <p id="srs-empty" style="display:none;">🎉 All caught up! No cards due for review.</p>
      </div>
    </div>
  </div>
  
  <!-- Playground Modal -->
  <div class="widget-modal" id="playground-modal">
    <div class="playground-container modal-container">
      <div class="modal-header"><h2>💻 Code Playground</h2><button class="modal-close" onclick="closePlayground()">×</button></div>
      <div class="modal-content">
        <select class="challenge-select" id="challenge-sel" onchange="loadChallenge()"><option value="">Free Coding</option></select>
        <p id="challenge-info" style="color:var(--text-secondary);margin-bottom:12px;"></p>
        <textarea id="pg-code" placeholder="# Write Python code here..."></textarea>
        <div style="display:flex;gap:12px;align-items:center;margin-top:12px;">
          <button class="run-btn" onclick="runCode()">▶ Run</button>
          <button class="run-btn" id="submit-btn" style="display:none;background:#6366f1;" onclick="submitCode()">✓ Submit</button>
        </div>
        <h4 style="margin-top:16px;">Output</h4>
        <pre id="pg-output">Run code to see output...</pre>
        <div id="test-results" style="margin-top:12px;"></div>
      </div>
    </div>
  </div>
  
  <script>
    // SRS
    let srsCards = [], srsIdx = 0;
    async function loadSRSCards() {
      try {
        const r = await fetch('/api/srs/cards?user=default&limit=50');
        const d = await r.json();
        srsCards = d.cards || [];
        document.getElementById('srs-due').textContent = srsCards.length;
      } catch {}
    }
    function openSRS() { document.getElementById('srs-modal').classList.add('active'); srsIdx = 0; showSRSCard(); }
    function closeSRS() { document.getElementById('srs-modal').classList.remove('active'); }
    function showSRSCard() {
      if (srsIdx >= srsCards.length) {
        document.getElementById('srs-content').style.display = 'none';
        document.getElementById('srs-empty').style.display = 'block';
        return;
      }
      document.getElementById('srs-content').style.display = 'block';
      document.getElementById('srs-empty').style.display = 'none';
      const c = srsCards[srsIdx];
      document.getElementById('srs-q').textContent = c.question;
      document.getElementById('srs-a').textContent = c.answer;
      document.getElementById('srs-a').classList.remove('revealed');
      document.getElementById('reveal-btn').style.display = 'inline-block';
      document.getElementById('srs-ratings').style.display = 'none';
    }
    function revealSRS() {
      document.getElementById('srs-a').classList.add('revealed');
      document.getElementById('reveal-btn').style.display = 'none';
      document.getElementById('srs-ratings').style.display = 'flex';
    }
    async function rateSRS(q) {
      try { await fetch('/api/srs/review', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({card_id: srsCards[srsIdx].id, quality: q}) }); } catch {}
      srsIdx++;
      document.getElementById('srs-due').textContent = Math.max(0, srsCards.length - srsIdx);
      showSRSCard();
    }
    loadSRSCards();
    
    // Playground
    let currentChal = null;
    async function loadChallenges() {
      try {
        const r = await fetch('/api/playground/challenges');
        const d = await r.json();
        const sel = document.getElementById('challenge-sel');
        (d.challenges || []).forEach(c => {
          const o = document.createElement('option');
          o.value = c.id; o.textContent = '[' + c.difficulty + '] ' + c.title;
          sel.appendChild(o);
        });
      } catch {}
    }
    async function loadChallenge() {
      const id = document.getElementById('challenge-sel').value;
      if (!id) { currentChal = null; document.getElementById('pg-code').value = ''; document.getElementById('challenge-info').textContent = ''; document.getElementById('submit-btn').style.display = 'none'; return; }
      try {
        const r = await fetch('/api/playground/challenge/' + id);
        const d = await r.json();
        currentChal = d.challenge;
        document.getElementById('pg-code').value = currentChal.starter_code;
        document.getElementById('challenge-info').textContent = currentChal.description;
        document.getElementById('submit-btn').style.display = 'inline-block';
      } catch {}
    }
    function openPlayground() { document.getElementById('playground-modal').classList.add('active'); loadChallenges(); }
    function closePlayground() { document.getElementById('playground-modal').classList.remove('active'); }
    async function runCode() {
      const code = document.getElementById('pg-code').value;
      const out = document.getElementById('pg-output');
      out.textContent = 'Running...';
      try {
        const r = await fetch('/api/playground/run', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({code}) });
        const d = await r.json();
        out.textContent = d.ok ? (d.output || '(no output)') : d.error;
        out.style.color = d.ok ? 'var(--text-primary)' : '#ef4444';
        if (d.execution_time_ms) out.textContent += '\\n⏱ ' + d.execution_time_ms + 'ms';
      } catch (e) { out.textContent = 'Error: ' + e.message; out.style.color = '#ef4444'; }
    }
    async function submitCode() {
      if (!currentChal) return;
      const res = document.getElementById('test-results');
      res.innerHTML = 'Testing...';
      try {
        const r = await fetch('/api/playground/submit', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({challenge_id: currentChal.id, code: document.getElementById('pg-code').value}) });
        const d = await r.json();
        let h = '<p><b>' + d.passed_count + '/' + d.total_count + ' tests passed</b></p>';
        (d.tests || []).forEach(t => { h += '<p style="color:' + (t.passed ? '#22c55e' : '#ef4444') + '">' + (t.passed ? '✅' : '❌') + ' Test ' + t.test + '</p>'; });
        if (d.all_passed) h = '🎉 <b style="color:#22c55e">All tests passed!</b>' + h;
        res.innerHTML = h;
      } catch (e) { res.innerHTML = 'Error: ' + e.message; }
    }
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
