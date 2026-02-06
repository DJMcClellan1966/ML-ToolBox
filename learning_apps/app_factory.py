"""
Shared App Factory for Learning Apps.
Creates Flask apps with consistent routes, modern UI, progress tracking, and Compass integration.
Usage:
    from learning_apps.app_factory import create_lab_app
    app = create_lab_app("Deep Learning Lab", "...", 5003, "deep_learning_lab", curriculum, demos)
"""
import sys
from pathlib import Path
from typing import Optional, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from flask import Flask, request, jsonify, render_template_string
except ImportError:
    print("Install Flask: pip install flask")
    sys.exit(1)


def create_lab_app(
    title: str,
    description: str,
    port: int,
    lab_id: str,
    curriculum_module: Any,
    demos_module: Any
) -> Flask:
    """
    Factory to create a learning lab Flask app.
    
    Args:
        title: App title (e.g. "Deep Learning Lab")
        description: Short description
        port: Port number
        lab_id: Unique lab identifier (e.g. "deep_learning_lab")
        curriculum_module: Module with get_curriculum, get_books, get_levels, get_by_book, get_by_level
        demos_module: Module with run_demo function
    
    Returns:
        Configured Flask app
    """
    app = Flask(__name__)
    app.config["APP_TITLE"] = title
    app.config["APP_DESCRIPTION"] = description
    app.config["LAB_ID"] = lab_id
    app.config["PORT"] = port
    
    # Check curriculum availability
    curriculum_available = False
    get_curriculum = get_books = get_levels = get_by_book = get_by_level = get_item = None
    if curriculum_module:
        try:
            get_curriculum = getattr(curriculum_module, "get_curriculum", None)
            get_books = getattr(curriculum_module, "get_books", None)
            get_levels = getattr(curriculum_module, "get_levels", None)
            get_by_book = getattr(curriculum_module, "get_by_book", None)
            get_by_level = getattr(curriculum_module, "get_by_level", None)
            get_item = getattr(curriculum_module, "get_item", None)
            curriculum_available = all([get_curriculum, get_books, get_levels])
        except Exception:
            pass
    
    # Check demos availability
    demos_available = False
    run_demo = None
    if demos_module:
        try:
            run_demo = getattr(demos_module, "run_demo", None)
            demos_available = run_demo is not None
        except Exception:
            pass
    
    if not demos_available:
        def run_demo(demo_id):
            return {"ok": False, "output": "", "error": "Demos not available"}
    
    # Try to register Compass routes
    compass_html = ""
    try:
        from learning_apps.compass_api import register_compass_routes, get_compass_html_snippet
        register_compass_routes(app)
        compass_html = get_compass_html_snippet()
    except Exception:
        pass
    
    # Try to register AI Tutor routes
    tutor_html = ""
    tutor_available = False
    try:
        from learning_apps.ai_tutor import register_tutor_routes, get_tutor_html_snippet
        register_tutor_routes(app, lab_id)
        tutor_html = get_tutor_html_snippet()
        tutor_available = True
    except Exception:
        pass
    
    # Try to import progress tracking
    progress_available = False
    try:
        from learning_apps import progress as progress_module
        progress_available = True
    except Exception:
        progress_module = None
    
    # --- API Routes ---
    
    @app.route("/api/health")
    def api_health():
        return jsonify({
            "ok": True,
            "curriculum": curriculum_available,
            "demos": demos_available,
            "progress": progress_available,
            "tutor": tutor_available,
            "lab_id": lab_id
        })
    
    @app.route("/api/curriculum")
    def api_curriculum():
        if not curriculum_available:
            return jsonify({"ok": False, "items": [], "books": [], "levels": []}), 503
        items = get_curriculum()
        # Attach progress status if available
        user_id = request.args.get("user", "default")
        if progress_available and progress_module:
            lab_progress = progress_module.get_lab_progress(user_id, lab_id)
            topic_statuses = lab_progress.get("topics", {})
            for item in items:
                item["progress_status"] = topic_statuses.get(item["id"], {}).get("status", "not-started")
        return jsonify({
            "ok": True,
            "items": items,
            "books": get_books(),
            "levels": get_levels()
        })
    
    @app.route("/api/curriculum/book/<book_id>")
    def api_book(book_id):
        if not curriculum_available:
            return jsonify({"ok": False, "items": []}), 503
        return jsonify({"ok": True, "items": get_by_book(book_id)})
    
    @app.route("/api/curriculum/level/<level>")
    def api_level(level):
        if not curriculum_available:
            return jsonify({"ok": False, "items": []}), 503
        return jsonify({"ok": True, "items": get_by_level(level)})
    
    @app.route("/api/try/<demo_id>", methods=["GET", "POST"])
    def api_try(demo_id):
        if not demos_available:
            return jsonify({"ok": False, "error": "Demos not available"}), 503
        # Record demo run for progress
        user_id = request.args.get("user", "default")
        topic_id = request.args.get("topic", demo_id)
        if progress_available and progress_module:
            progress_module.record_demo_run(user_id, lab_id, topic_id)
        return jsonify(run_demo(demo_id))
    
    # --- Progress Routes ---
    
    @app.route("/api/progress")
    def api_progress():
        user_id = request.args.get("user", "default")
        if not progress_available:
            return jsonify({"ok": False, "error": "Progress tracking not available"}), 503
        return jsonify(progress_module.get_lab_progress(user_id, lab_id))
    
    @app.route("/api/progress/start/<topic_id>", methods=["POST"])
    def api_start_topic(topic_id):
        user_id = request.args.get("user", "default")
        if not progress_available:
            return jsonify({"ok": False}), 503
        return jsonify(progress_module.mark_topic_started(user_id, lab_id, topic_id))
    
    @app.route("/api/progress/complete/<topic_id>", methods=["POST"])
    def api_complete_topic(topic_id):
        user_id = request.args.get("user", "default")
        if not progress_available:
            return jsonify({"ok": False}), 503
        return jsonify(progress_module.mark_topic_completed(user_id, lab_id, topic_id))
    
    @app.route("/api/progress/reset", methods=["POST"])
    def api_reset_progress():
        user_id = request.args.get("user", "default")
        if not progress_available:
            return jsonify({"ok": False}), 503
        return jsonify(progress_module.reset_progress(user_id, lab_id))
    
    # --- Main Route ---
    
    @app.route("/")
    def index():
        return render_template_string(_get_modern_html(title, description, lab_id, compass_html, tutor_html))
    
    return app


def _get_modern_html(title: str, description: str, lab_id: str, compass_html: str, tutor_html: str = "") -> str:
    """Generate modern, responsive HTML for the lab."""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
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
      --error: #ef4444;
      --border: #475569;
      --radius: 12px;
      --radius-sm: 8px;
    }}
    .light-theme {{
      --bg-primary: #f8fafc;
      --bg-secondary: #ffffff;
      --bg-tertiary: #e2e8f0;
      --text-primary: #1e293b;
      --text-secondary: #475569;
      --text-muted: #94a3b8;
      --border: #cbd5e1;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: var(--bg-primary);
      color: var(--text-primary);
      min-height: 100vh;
      line-height: 1.6;
    }}
    
    /* Header */
    .header {{
      background: var(--bg-secondary);
      border-bottom: 1px solid var(--border);
      padding: 16px 24px;
      position: sticky;
      top: 0;
      z-index: 100;
    }}
    .header-inner {{
      max-width: 1200px;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .header h1 {{
      font-size: 1.5rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .header h1 .icon {{ font-size: 1.3rem; }}
    .header-desc {{ color: var(--text-secondary); font-size: 0.9rem; }}
    .header-actions {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    .theme-toggle, .search-btn {{
      background: var(--bg-tertiary);
      border: 1px solid var(--border);
      color: var(--text-primary);
      padding: 8px 12px;
      border-radius: var(--radius-sm);
      cursor: pointer;
      font-size: 1rem;
      transition: all 0.2s;
    }}
    .theme-toggle:hover, .search-btn:hover {{
      background: var(--accent);
      border-color: var(--accent);
    }}
    
    /* Progress Bar */
    .progress-bar {{
      background: var(--bg-secondary);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
    }}
    .progress-inner {{
      max-width: 1200px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }}
    .progress-track {{
      flex: 1;
      min-width: 200px;
      height: 8px;
      background: var(--bg-tertiary);
      border-radius: 4px;
      overflow: hidden;
    }}
    .progress-fill {{
      height: 100%;
      background: linear-gradient(90deg, var(--success), var(--accent));
      border-radius: 4px;
      transition: width 0.3s ease;
    }}
    .progress-stats {{
      display: flex;
      gap: 16px;
      font-size: 0.85rem;
    }}
    .progress-stat {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .stat-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }}
    .stat-dot.completed {{ background: var(--success); }}
    .stat-dot.in-progress {{ background: var(--warning); }}
    
    /* Main Layout */
    .main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      grid-template-columns: 280px 1fr;
      gap: 24px;
    }}
    @media (max-width: 900px) {{
      .main {{ grid-template-columns: 1fr; }}
    }}
    
    /* Sidebar */
    .sidebar {{
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .sidebar-card {{
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px;
    }}
    .sidebar-title {{
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 12px;
    }}
    .filter-btn {{
      display: block;
      width: 100%;
      text-align: left;
      padding: 10px 14px;
      margin-bottom: 6px;
      background: transparent;
      border: 1px solid transparent;
      border-radius: var(--radius-sm);
      color: var(--text-primary);
      cursor: pointer;
      transition: all 0.2s;
      font-size: 0.95rem;
    }}
    .filter-btn:hover {{
      background: var(--bg-tertiary);
    }}
    .filter-btn.active {{
      background: var(--accent);
      color: white;
    }}
    .filter-btn .count {{
      float: right;
      color: var(--text-muted);
      font-size: 0.85rem;
    }}
    .filter-btn.active .count {{ color: rgba(255,255,255,0.7); }}
    
    /* Content */
    .content {{
      min-width: 0;
    }}
    
    /* Breadcrumbs */
    .breadcrumbs {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.9rem;
      color: var(--text-secondary);
      margin-bottom: 16px;
    }}
    .breadcrumbs a {{
      color: var(--accent);
      text-decoration: none;
    }}
    .breadcrumbs a:hover {{ text-decoration: underline; }}
    
    /* Topic Grid */
    .topic-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 16px;
    }}
    
    /* Topic Card */
    .topic-card {{
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px;
      cursor: pointer;
      transition: all 0.2s;
      position: relative;
    }}
    .topic-card:hover {{
      border-color: var(--accent);
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }}
    .topic-card.completed {{
      border-left: 4px solid var(--success);
    }}
    .topic-card.in-progress {{
      border-left: 4px solid var(--warning);
    }}
    .topic-card .level-badge {{
      display: inline-block;
      padding: 4px 10px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border-radius: 20px;
      margin-bottom: 10px;
    }}
    .level-badge.basics {{ background: #dcfce7; color: #166534; }}
    .level-badge.intermediate {{ background: #dbeafe; color: #1e40af; }}
    .level-badge.advanced {{ background: #fef3c7; color: #92400e; }}
    .level-badge.expert {{ background: #fce7f3; color: #9d174d; }}
    .light-theme .level-badge.basics {{ background: #bbf7d0; }}
    .light-theme .level-badge.intermediate {{ background: #bfdbfe; }}
    .light-theme .level-badge.advanced {{ background: #fde68a; }}
    .light-theme .level-badge.expert {{ background: #fbcfe8; }}
    .topic-card h3 {{
      font-size: 1.1rem;
      margin-bottom: 8px;
      color: var(--text-primary);
    }}
    .topic-card .description {{
      font-size: 0.9rem;
      color: var(--text-secondary);
      line-height: 1.5;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }}
    .topic-card .meta {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 12px;
      font-size: 0.85rem;
    }}
    .topic-card .book-tag {{
      color: var(--text-muted);
    }}
    .topic-card .has-demo {{
      color: var(--success);
      font-weight: 500;
    }}
    .status-icon {{
      position: absolute;
      top: 12px;
      right: 12px;
      font-size: 1.1rem;
    }}
    
    /* Topic Detail Modal */
    .modal-overlay {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.7);
      z-index: 200;
      justify-content: center;
      align-items: center;
      padding: 24px;
    }}
    .modal-overlay.active {{ display: flex; }}
    .modal {{
      background: var(--bg-secondary);
      border-radius: var(--radius);
      max-width: 700px;
      width: 100%;
      max-height: 90vh;
      overflow-y: auto;
      position: relative;
    }}
    .modal-header {{
      padding: 20px 24px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
    }}
    .modal-header h2 {{ font-size: 1.3rem; }}
    .modal-close {{
      background: transparent;
      border: none;
      color: var(--text-secondary);
      font-size: 1.5rem;
      cursor: pointer;
      padding: 4px;
      line-height: 1;
    }}
    .modal-close:hover {{ color: var(--text-primary); }}
    .modal-body {{ padding: 24px; }}
    .modal-section {{
      margin-bottom: 20px;
    }}
    .modal-section h4 {{
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }}
    .learn-content {{
      background: var(--bg-primary);
      border-radius: var(--radius-sm);
      padding: 16px;
      line-height: 1.7;
    }}
    .code-block {{
      background: var(--bg-primary);
      border-radius: var(--radius-sm);
      padding: 16px;
      font-family: 'Fira Code', 'Consolas', monospace;
      font-size: 0.9rem;
      overflow-x: auto;
      position: relative;
    }}
    .copy-btn {{
      position: absolute;
      top: 8px;
      right: 8px;
      background: var(--bg-tertiary);
      border: none;
      color: var(--text-secondary);
      padding: 6px 10px;
      border-radius: 4px;
      font-size: 0.8rem;
      cursor: pointer;
    }}
    .copy-btn:hover {{ background: var(--accent); color: white; }}
    .demo-actions {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .btn {{
      padding: 12px 24px;
      border-radius: var(--radius-sm);
      font-size: 0.95rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
      border: none;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .btn-primary {{
      background: var(--accent);
      color: white;
    }}
    .btn-primary:hover {{ background: var(--accent-hover); }}
    .btn-primary:disabled {{
      background: var(--bg-tertiary);
      color: var(--text-muted);
      cursor: not-allowed;
    }}
    .btn-success {{
      background: var(--success);
      color: white;
    }}
    .btn-success:hover {{ background: #16a34a; }}
    .btn-outline {{
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-primary);
    }}
    .btn-outline:hover {{
      background: var(--bg-tertiary);
    }}
    .demo-output {{
      margin-top: 16px;
      border-radius: var(--radius-sm);
      overflow: hidden;
    }}
    .output-header {{
      background: var(--bg-tertiary);
      padding: 10px 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.85rem;
    }}
    .output-header .status {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .output-header .status.ok {{ color: var(--success); }}
    .output-header .status.error {{ color: var(--error); }}
    .output-content {{
      background: var(--bg-primary);
      padding: 16px;
      font-family: 'Fira Code', 'Consolas', monospace;
      font-size: 0.9rem;
      max-height: 300px;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    
    /* Search Modal */
    .search-modal {{
      max-width: 600px;
    }}
    .search-input {{
      width: 100%;
      padding: 14px 18px;
      font-size: 1.1rem;
      border: 2px solid var(--border);
      border-radius: var(--radius-sm);
      background: var(--bg-primary);
      color: var(--text-primary);
      outline: none;
    }}
    .search-input:focus {{
      border-color: var(--accent);
    }}
    .search-results {{
      margin-top: 16px;
      max-height: 400px;
      overflow-y: auto;
    }}
    .search-result {{
      padding: 12px 16px;
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: background 0.2s;
    }}
    .search-result:hover {{
      background: var(--bg-tertiary);
    }}
    .search-result h4 {{ margin-bottom: 4px; }}
    .search-result p {{
      font-size: 0.85rem;
      color: var(--text-secondary);
    }}
    
    /* Loading Spinner */
    .spinner {{
      display: inline-block;
      width: 20px;
      height: 20px;
      border: 2px solid var(--bg-tertiary);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
    
    /* Compass Panel */
    .compass-panel {{
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px;
      margin-top: 24px;
    }}
    .compass-panel h3 {{
      font-size: 1rem;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .compass-row {{
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}
    .compass-input {{
      flex: 1;
      min-width: 200px;
      padding: 10px 14px;
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      background: var(--bg-primary);
      color: var(--text-primary);
      font-size: 0.95rem;
    }}
    .compass-input:focus {{
      outline: none;
      border-color: var(--accent);
    }}
    #compass-out {{
      background: var(--bg-primary);
      border-radius: var(--radius-sm);
      padding: 16px;
      min-height: 80px;
      font-size: 0.9rem;
      white-space: pre-wrap;
    }}
    
    /* Responsive */
    @media (max-width: 600px) {{
      .header-inner {{ flex-direction: column; align-items: flex-start; }}
      .topic-grid {{ grid-template-columns: 1fr; }}
      .main {{ padding: 16px; }}
      .modal {{ margin: 12px; }}
    }}
  </style>
</head>
<body>
  <!-- Header -->
  <header class="header">
    <div class="header-inner">
      <div>
        <h1><span class="icon">📚</span> {title}</h1>
        <p class="header-desc">{description}</p>
      </div>
      <div class="header-actions">
        <button class="search-btn" id="search-btn" title="Search topics (Ctrl+K)">🔍 Search</button>
        <button class="theme-toggle" id="theme-toggle" title="Toggle theme">🌙</button>
      </div>
    </div>
  </header>
  
  <!-- Progress Bar -->
  <div class="progress-bar">
    <div class="progress-inner">
      <div class="progress-track">
        <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
      </div>
      <div class="progress-stats">
        <span class="progress-stat"><span class="stat-dot completed"></span> <span id="stat-completed">0</span> completed</span>
        <span class="progress-stat"><span class="stat-dot in-progress"></span> <span id="stat-in-progress">0</span> in progress</span>
      </div>
    </div>
  </div>
  
  <!-- Main Content -->
  <main class="main">
    <!-- Sidebar -->
    <aside class="sidebar">
      <div class="sidebar-card">
        <div class="sidebar-title">Filter by Level</div>
        <button class="filter-btn active" data-filter="all">All Topics <span class="count" id="count-all">0</span></button>
        <button class="filter-btn" data-filter="basics">Basics <span class="count" id="count-basics">0</span></button>
        <button class="filter-btn" data-filter="intermediate">Intermediate <span class="count" id="count-intermediate">0</span></button>
        <button class="filter-btn" data-filter="advanced">Advanced <span class="count" id="count-advanced">0</span></button>
        <button class="filter-btn" data-filter="expert">Expert <span class="count" id="count-expert">0</span></button>
      </div>
      <div class="sidebar-card">
        <div class="sidebar-title">Filter by Book</div>
        <div id="book-filters"></div>
      </div>
      <div class="sidebar-card">
        <div class="sidebar-title">Progress</div>
        <button class="btn btn-outline" style="width:100%" id="reset-progress-btn">🔄 Reset Progress</button>
      </div>
    </aside>
    
    <!-- Content Area -->
    <div class="content">
      <div class="breadcrumbs" id="breadcrumbs">
        <a href="http://127.0.0.1:5000" target="_blank">Hub</a>
        <span>›</span>
        <span>{title}</span>
      </div>
      <div class="topic-grid" id="topic-grid"></div>
      
      <!-- Compass Panel -->
      <div class="compass-panel">
        <h3>🧭 ML Compass</h3>
        <div class="compass-row">
          <input type="text" class="compass-input" id="compass-concept" placeholder="Concept (e.g. entropy, attention)">
          <button class="btn btn-primary" id="compass-explain-btn">Explain</button>
          <button class="btn btn-outline" id="compass-guidance-btn">Guidance</button>
        </div>
        <div class="compass-row">
          <input type="text" class="compass-input" id="compass-oracle-input" placeholder="Problem (e.g. my model overfits)">
          <button class="btn btn-primary" id="compass-oracle-btn">Ask Oracle</button>
        </div>
        <div class="compass-row">
          <input type="text" class="compass-input" id="compass-generate-topic" placeholder="Topic for code (e.g. logistic regression)">
          <button class="btn btn-success" id="compass-generate-btn">Generate Code</button>
        </div>
        <div id="compass-out"></div>
      </div>
    </div>
  </main>
  
  <!-- Topic Detail Modal -->
  <div class="modal-overlay" id="topic-modal">
    <div class="modal">
      <div class="modal-header">
        <div>
          <span class="level-badge" id="modal-level"></span>
          <h2 id="modal-title"></h2>
        </div>
        <button class="modal-close" id="modal-close">&times;</button>
      </div>
      <div class="modal-body">
        <div class="modal-section">
          <h4>📖 Learn</h4>
          <div class="learn-content" id="modal-learn"></div>
        </div>
        <div class="modal-section">
          <h4>💻 Code Example</h4>
          <div class="code-block">
            <button class="copy-btn" id="copy-code-btn">Copy</button>
            <pre id="modal-code"></pre>
          </div>
        </div>
        <div class="modal-section">
          <h4>🚀 Try It</h4>
          <div class="demo-actions">
            <button class="btn btn-primary" id="run-demo-btn" disabled>
              <span class="btn-text">▶ Run Demo</span>
            </button>
            <button class="btn btn-success" id="mark-complete-btn">✓ Mark Complete</button>
          </div>
          <div class="demo-output" id="demo-output" style="display:none">
            <div class="output-header">
              <span class="status" id="output-status"><span class="spinner"></span> Running...</span>
              <button class="copy-btn" id="copy-output-btn">Copy</button>
            </div>
            <div class="output-content" id="output-content"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
  
  <!-- Search Modal -->
  <div class="modal-overlay" id="search-modal">
    <div class="modal search-modal">
      <div class="modal-header">
        <h2>🔍 Search Topics</h2>
        <button class="modal-close" id="search-close">&times;</button>
      </div>
      <div class="modal-body">
        <input type="text" class="search-input" id="search-input" placeholder="Type to search...">
        <div class="search-results" id="search-results"></div>
      </div>
    </div>
  </div>
  
  <script>
    const LAB_ID = "{lab_id}";
    const api = (path, opts = {{}}) => fetch(path, {{headers: {{'Content-Type': 'application/json'}}, ...opts}}).then(r => r.json());
    
    // State
    let curriculum = {{items: [], books: [], levels: []}};
    let currentFilter = {{type: 'all', value: null}};
    let currentTopic = null;
    
    // Theme
    function initTheme() {{
      const saved = localStorage.getItem('theme');
      if (saved === 'light') document.body.classList.add('light-theme');
      updateThemeIcon();
    }}
    function updateThemeIcon() {{
      const btn = document.getElementById('theme-toggle');
      btn.textContent = document.body.classList.contains('light-theme') ? '☀️' : '🌙';
    }}
    document.getElementById('theme-toggle').onclick = () => {{
      document.body.classList.toggle('light-theme');
      localStorage.setItem('theme', document.body.classList.contains('light-theme') ? 'light' : 'dark');
      updateThemeIcon();
    }};
    initTheme();
    
    // Load curriculum
    async function loadCurriculum() {{
      const data = await api('/api/curriculum');
      if (data.ok) {{
        curriculum = {{items: data.items || [], books: data.books || [], levels: data.levels || []}};
        updateCounts();
        renderBookFilters();
        renderTopics();
        updateProgress();
      }}
    }}
    
    function updateCounts() {{
      document.getElementById('count-all').textContent = curriculum.items.length;
      ['basics', 'intermediate', 'advanced', 'expert'].forEach(level => {{
        const count = curriculum.items.filter(i => i.level === level).length;
        const el = document.getElementById('count-' + level);
        if (el) el.textContent = count;
      }});
    }}
    
    function renderBookFilters() {{
      const container = document.getElementById('book-filters');
      container.innerHTML = '';
      curriculum.books.forEach(book => {{
        const btn = document.createElement('button');
        btn.className = 'filter-btn';
        btn.dataset.filter = 'book';
        btn.dataset.bookId = book.id;
        const count = curriculum.items.filter(i => i.book_id === book.id).length;
        btn.innerHTML = (book.short || book.name) + ' <span class="count">' + count + '</span>';
        btn.onclick = () => setFilter('book', book.id);
        container.appendChild(btn);
      }});
    }}
    
    function setFilter(type, value) {{
      currentFilter = {{type, value}};
      document.querySelectorAll('.filter-btn').forEach(btn => {{
        btn.classList.remove('active');
        if (type === 'all' && btn.dataset.filter === 'all') btn.classList.add('active');
        else if (type === 'level' && btn.dataset.filter === value) btn.classList.add('active');
        else if (type === 'book' && btn.dataset.bookId === value) btn.classList.add('active');
      }});
      renderTopics();
    }}
    
    // Level filter buttons
    document.querySelectorAll('.filter-btn[data-filter]').forEach(btn => {{
      if (btn.dataset.filter === 'all') btn.onclick = () => setFilter('all', null);
      else if (['basics', 'intermediate', 'advanced', 'expert'].includes(btn.dataset.filter)) {{
        btn.onclick = () => setFilter('level', btn.dataset.filter);
      }}
    }});
    
    function renderTopics() {{
      const grid = document.getElementById('topic-grid');
      grid.innerHTML = '';
      let items = curriculum.items;
      if (currentFilter.type === 'level') {{
        items = items.filter(i => i.level === currentFilter.value);
      }} else if (currentFilter.type === 'book') {{
        items = items.filter(i => i.book_id === currentFilter.value);
      }}
      
      items.forEach(item => {{
        const card = document.createElement('div');
        card.className = 'topic-card ' + (item.progress_status || '');
        card.innerHTML = `
          <span class="status-icon">${{item.progress_status === 'completed' ? '✅' : item.progress_status === 'in-progress' ? '🔄' : ''}}</span>
          <span class="level-badge ${{item.level}}">${{item.level}}</span>
          <h3>${{item.title}}</h3>
          <p class="description">${{item.learn || ''}}</p>
          <div class="meta">
            <span class="book-tag">${{getBookName(item.book_id)}}</span>
            ${{item.try_demo ? '<span class="has-demo">▶ Demo</span>' : ''}}
          </div>
        `;
        card.onclick = () => openTopic(item);
        grid.appendChild(card);
      }});
    }}
    
    function getBookName(bookId) {{
      const book = curriculum.books.find(b => b.id === bookId);
      return book ? (book.short || book.name) : bookId;
    }}
    
    function updateProgress() {{
      const completed = curriculum.items.filter(i => i.progress_status === 'completed').length;
      const inProgress = curriculum.items.filter(i => i.progress_status === 'in-progress').length;
      const total = curriculum.items.length;
      const pct = total > 0 ? Math.round((completed / total) * 100) : 0;
      document.getElementById('progress-fill').style.width = pct + '%';
      document.getElementById('stat-completed').textContent = completed;
      document.getElementById('stat-in-progress').textContent = inProgress;
    }}
    
    // Topic Modal
    function openTopic(item) {{
      currentTopic = item;
      document.getElementById('modal-level').textContent = item.level;
      document.getElementById('modal-level').className = 'level-badge ' + item.level;
      document.getElementById('modal-title').textContent = item.title;
      document.getElementById('modal-learn').textContent = item.learn || 'No description available.';
      document.getElementById('modal-code').textContent = item.try_code || '# No code example';
      document.getElementById('run-demo-btn').disabled = !item.try_demo;
      document.getElementById('demo-output').style.display = 'none';
      document.getElementById('topic-modal').classList.add('active');
      // Mark as started
      api('/api/progress/start/' + item.id, {{method: 'POST'}});
    }}
    
    document.getElementById('modal-close').onclick = () => {{
      document.getElementById('topic-modal').classList.remove('active');
      loadCurriculum(); // Refresh to show updated progress
    }};
    document.getElementById('topic-modal').onclick = (e) => {{
      if (e.target.id === 'topic-modal') {{
        document.getElementById('topic-modal').classList.remove('active');
        loadCurriculum();
      }}
    }};
    
    // Run demo
    document.getElementById('run-demo-btn').onclick = async () => {{
      if (!currentTopic || !currentTopic.try_demo) return;
      const output = document.getElementById('demo-output');
      const status = document.getElementById('output-status');
      const content = document.getElementById('output-content');
      output.style.display = 'block';
      status.innerHTML = '<span class="spinner"></span> Running...';
      status.className = 'status';
      content.textContent = '';
      
      try {{
        const start = performance.now();
        const result = await api('/api/try/' + currentTopic.try_demo + '?topic=' + currentTopic.id);
        const elapsed = ((performance.now() - start) / 1000).toFixed(2);
        if (result.ok) {{
          status.innerHTML = '✓ Completed in ' + elapsed + 's';
          status.className = 'status ok';
          content.textContent = result.output || 'Demo ran successfully.';
        }} else {{
          status.innerHTML = '✗ Error';
          status.className = 'status error';
          content.textContent = result.error || 'Unknown error';
        }}
      }} catch (err) {{
        status.innerHTML = '✗ Error';
        status.className = 'status error';
        content.textContent = err.message;
      }}
    }};
    
    // Mark complete
    document.getElementById('mark-complete-btn').onclick = async () => {{
      if (!currentTopic) return;
      await api('/api/progress/complete/' + currentTopic.id, {{method: 'POST'}});
      document.getElementById('topic-modal').classList.remove('active');
      loadCurriculum();
    }};
    
    // Copy buttons
    document.getElementById('copy-code-btn').onclick = () => {{
      navigator.clipboard.writeText(document.getElementById('modal-code').textContent);
      document.getElementById('copy-code-btn').textContent = 'Copied!';
      setTimeout(() => document.getElementById('copy-code-btn').textContent = 'Copy', 1500);
    }};
    document.getElementById('copy-output-btn').onclick = () => {{
      navigator.clipboard.writeText(document.getElementById('output-content').textContent);
      document.getElementById('copy-output-btn').textContent = 'Copied!';
      setTimeout(() => document.getElementById('copy-output-btn').textContent = 'Copy', 1500);
    }};
    
    // Reset progress
    document.getElementById('reset-progress-btn').onclick = async () => {{
      if (confirm('Reset all progress for this lab?')) {{
        await api('/api/progress/reset', {{method: 'POST'}});
        loadCurriculum();
      }}
    }};
    
    // Search
    document.getElementById('search-btn').onclick = () => {{
      document.getElementById('search-modal').classList.add('active');
      document.getElementById('search-input').focus();
    }};
    document.getElementById('search-close').onclick = () => {{
      document.getElementById('search-modal').classList.remove('active');
    }};
    document.getElementById('search-modal').onclick = (e) => {{
      if (e.target.id === 'search-modal') document.getElementById('search-modal').classList.remove('active');
    }};
    document.getElementById('search-input').oninput = (e) => {{
      const q = e.target.value.toLowerCase();
      const results = document.getElementById('search-results');
      results.innerHTML = '';
      if (!q) return;
      const matches = curriculum.items.filter(i => 
        i.title.toLowerCase().includes(q) || 
        (i.learn && i.learn.toLowerCase().includes(q))
      );
      matches.slice(0, 10).forEach(item => {{
        const div = document.createElement('div');
        div.className = 'search-result';
        div.innerHTML = '<h4>' + item.title + '</h4><p>' + (item.learn || '').slice(0, 100) + '...</p>';
        div.onclick = () => {{
          document.getElementById('search-modal').classList.remove('active');
          openTopic(item);
        }};
        results.appendChild(div);
      }});
    }};
    
    // Keyboard shortcut
    document.addEventListener('keydown', (e) => {{
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {{
        e.preventDefault();
        document.getElementById('search-btn').click();
      }}
      if (e.key === 'Escape') {{
        document.getElementById('topic-modal').classList.remove('active');
        document.getElementById('search-modal').classList.remove('active');
      }}
    }});
    
    // Compass functionality
    function showCompass(txt, err) {{
      const el = document.getElementById('compass-out');
      el.textContent = txt || '';
      el.style.borderLeft = err ? '3px solid var(--error)' : '3px solid var(--success)';
    }}
    
    document.getElementById('compass-explain-btn').onclick = async () => {{
      const concept = document.getElementById('compass-concept').value || 'entropy';
      showCompass('Loading...');
      try {{
        const d = await api('/api/explain/' + encodeURIComponent(concept));
        if (d.ok) {{
          let s = '';
          for (const [k, v] of Object.entries(d.views || {{}})) s += k + ': ' + v + '\\n';
          showCompass(s || 'Explained: ' + concept);
        }} else showCompass('Unknown concept. Try: ' + (d.available || []).join(', '), true);
      }} catch (e) {{ showCompass(e.message, true); }}
    }};
    
    document.getElementById('compass-guidance-btn').onclick = async () => {{
      showCompass('Loading...');
      try {{
        const d = await api('/api/guidance');
        if (d.ok) {{
          let s = 'Avoid: ' + (d.avoid || []).map(x => x.pattern).join(', ');
          s += '\\nEncourage: ' + (d.encourage || []).map(x => x.pattern).join(', ');
          showCompass(s);
        }} else showCompass(d.error || 'Error', true);
      }} catch (e) {{ showCompass(e.message, true); }}
    }};
    
    document.getElementById('compass-oracle-btn').onclick = async () => {{
      const desc = document.getElementById('compass-oracle-input').value || 'My model overfits';
      showCompass('Loading...');
      try {{
        const d = await api('/api/oracle', {{method: 'POST', body: JSON.stringify({{description: desc}})}});
        if (d.ok) showCompass('Pattern: ' + d.pattern + '\\nSuggestion: ' + d.suggestion + '\\nWhy: ' + d.why);
        else showCompass(d.error || 'Error', true);
      }} catch (e) {{ showCompass(e.message, true); }}
    }};
    
    document.getElementById('compass-generate-btn').onclick = async () => {{
      const topic = document.getElementById('compass-generate-topic').value || 'linear regression';
      showCompass('Generating...');
      try {{
        const d = await api('/api/generate_code', {{method: 'POST', body: JSON.stringify({{topic: topic}})}});
        if (d.ok) showCompass(d.code || 'Code generated');
        else showCompass(d.error || 'No code', true);
      }} catch (e) {{ showCompass(e.message, true); }}
    }};
    
    // Init
    loadCurriculum();
  </script>
  
  {tutor_html}
  
</body>
</html>
'''
