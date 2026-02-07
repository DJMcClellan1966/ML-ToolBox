"""
Blueprint Factory for Learning Labs.
Creates Flask Blueprints for each lab to run under a single server.
Usage:
    from learning_apps.blueprint_factory import create_lab_blueprint
    bp = create_lab_blueprint("clrs", "CLRS Algorithms Lab", "...", curriculum, demos)
    app.register_blueprint(bp, url_prefix="/lab/clrs")
"""
import sys
from pathlib import Path
from typing import Optional, Any

from flask import Blueprint, request, jsonify, render_template_string


def create_lab_blueprint(
    lab_id: str,
    title: str,
    description: str,
    curriculum_module: Any = None,
    demos_module: Any = None
) -> Blueprint:
    """
    Factory to create a learning lab Flask Blueprint.
    
    Args:
        lab_id: Unique lab identifier (e.g. "clrs_algorithms_lab")
        title: App title (e.g. "CLRS Algorithms Lab")
        description: Short description
        curriculum_module: Module with get_curriculum, get_books, get_levels, etc.
        demos_module: Module with run_demo function
    
    Returns:
        Configured Flask Blueprint
    """
    bp = Blueprint(lab_id, __name__, url_prefix=f"/lab/{lab_id}")
    
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
    
    # --- API Routes ---
    
    @bp.route("/api/curriculum")
    def api_curriculum():
        if not curriculum_available:
            return jsonify({"ok": False, "items": [], "books": [], "levels": []}), 503
        items = get_curriculum()
        return jsonify({
            "ok": True,
            "items": items,
            "books": get_books(),
            "levels": get_levels()
        })
    
    @bp.route("/api/try/<demo_id>", methods=["GET", "POST"])
    def api_try(demo_id):
        if not demos_available:
            return jsonify({"ok": False, "error": "Demos not available"}), 503
        return jsonify(run_demo(demo_id))
    
    @bp.route("/api/curriculum/book/<book_id>")
    def api_book(book_id):
        if not curriculum_available:
            return jsonify({"ok": False, "items": []}), 503
        return jsonify({"ok": True, "items": get_by_book(book_id)})
    
    @bp.route("/api/curriculum/level/<level>")
    def api_level(level):
        if not curriculum_available:
            return jsonify({"ok": False, "items": []}), 503
        return jsonify({"ok": True, "items": get_by_level(level)})
    
    # --- Main Route ---
    
    @bp.route("/")
    def index():
        return render_template_string(_get_lab_html(title, description, lab_id))
    
    return bp


def _get_lab_html(title: str, description: str, lab_id: str) -> str:
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
      --accent: #6366f1;
      --accent-hover: #818cf8;
      --success: #22c55e;
      --warning: #f59e0b;
      --error: #ef4444;
      --border: #475569;
      --radius: 12px;
    }}
    .light-theme {{
      --bg-primary: #f8fafc;
      --bg-secondary: #ffffff;
      --bg-tertiary: #e2e8f0;
      --text-primary: #1e293b;
      --text-secondary: #64748b;
      --border: #cbd5e1;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: var(--bg-primary);
      color: var(--text-primary);
      min-height: 100vh;
      line-height: 1.6;
    }}
    header {{
      background: var(--bg-secondary);
      border-bottom: 1px solid var(--border);
      padding: 16px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    header h1 {{
      font-size: 1.5rem;
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .back-btn {{
      background: var(--bg-tertiary);
      border: 1px solid var(--border);
      color: var(--text-primary);
      padding: 8px 16px;
      border-radius: 8px;
      cursor: pointer;
      text-decoration: none;
      font-size: 0.9rem;
    }}
    .back-btn:hover {{ background: var(--accent); }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    .description {{
      color: var(--text-secondary);
      margin-bottom: 24px;
      font-size: 1.1rem;
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      margin-bottom: 24px;
      flex-wrap: wrap;
    }}
    .tab {{
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      color: var(--text-primary);
      padding: 10px 20px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.95rem;
    }}
    .tab:hover {{ background: var(--bg-tertiary); }}
    .tab.active {{ background: var(--accent); border-color: var(--accent); }}
    .panel {{ display: none; }}
    .panel.active {{ display: block; }}
    .topic-list {{
      display: grid;
      gap: 16px;
    }}
    .topic-card {{
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .topic-card:hover {{
      border-color: var(--accent);
      transform: translateY(-2px);
    }}
    .topic-card h3 {{
      margin-bottom: 8px;
      color: var(--text-primary);
    }}
    .topic-card p {{
      color: var(--text-secondary);
      font-size: 0.95rem;
    }}
    .level-badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 12px;
      font-size: 0.75rem;
      font-weight: 600;
      margin-right: 8px;
    }}
    .level-basics {{ background: #22c55e33; color: #22c55e; }}
    .level-intermediate {{ background: #f59e0b33; color: #f59e0b; }}
    .level-advanced {{ background: #ef444433; color: #ef4444; }}
    .demo-btn {{
      background: var(--accent);
      border: none;
      color: white;
      padding: 6px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.85rem;
      margin-top: 12px;
    }}
    .demo-btn:hover {{ background: var(--accent-hover); }}
    .output {{
      background: var(--bg-tertiary);
      border-radius: 8px;
      padding: 16px;
      margin-top: 12px;
      font-family: 'Fira Code', Consolas, monospace;
      font-size: 0.9rem;
      white-space: pre-wrap;
      max-height: 300px;
      overflow-y: auto;
    }}
    .loading {{ opacity: 0.6; }}
    .error {{ color: var(--error); }}
    #theme-toggle {{
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-primary);
      padding: 8px 12px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 1.2rem;
    }}
  </style>
</head>
<body>
  <header>
    <h1>📚 {title}</h1>
    <div style="display:flex;gap:12px;align-items:center;">
      <button id="theme-toggle">🌙</button>
      <a href="/" class="back-btn">← Back to Hub</a>
    </div>
  </header>
  
  <main>
    <p class="description">{description}</p>
    
    <div class="tabs">
      <button class="tab active" data-tab="all">All Topics</button>
      <button class="tab" data-tab="by-level">By Level</button>
      <button class="tab" data-tab="by-book">By Book</button>
    </div>
    
    <div id="panel-all" class="panel active">
      <div id="topic-list" class="topic-list">
        <p style="color:var(--text-secondary)">Loading curriculum...</p>
      </div>
    </div>
    
    <div id="panel-by-level" class="panel">
      <div class="tabs" id="level-tabs"></div>
      <div id="level-topics" class="topic-list"></div>
    </div>
    
    <div id="panel-by-book" class="panel">
      <div class="tabs" id="book-tabs"></div>
      <div id="book-topics" class="topic-list"></div>
    </div>
  </main>
  
  <script>
    const api = (path, opts) => fetch(path, opts).then(r => r.json());
    let curriculum = [];
    let books = [];
    let levels = [];
    
    // Theme
    function initTheme() {{
      if (localStorage.getItem('theme') === 'light') document.body.classList.add('light-theme');
      updateThemeIcon();
    }}
    function updateThemeIcon() {{
      document.getElementById('theme-toggle').textContent = 
        document.body.classList.contains('light-theme') ? '☀️' : '🌙';
    }}
    document.getElementById('theme-toggle').onclick = () => {{
      document.body.classList.toggle('light-theme');
      localStorage.setItem('theme', document.body.classList.contains('light-theme') ? 'light' : 'dark');
      updateThemeIcon();
    }};
    initTheme();
    
    // Tabs
    document.querySelectorAll('.tabs').forEach(tabContainer => {{
      tabContainer.addEventListener('click', (e) => {{
        if (e.target.classList.contains('tab') && e.target.dataset.tab) {{
          document.querySelectorAll('.tab[data-tab]').forEach(t => t.classList.remove('active'));
          e.target.classList.add('active');
          document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
          document.getElementById('panel-' + e.target.dataset.tab).classList.add('active');
        }}
      }});
    }});
    
    // Render topic card
    function renderTopic(item) {{
      const levelClass = 'level-' + (item.level || 'basics');
      const hasDemo = item.try_demo ? `<button class="demo-btn" data-demo="${{item.try_demo}}">▶ Run Demo</button>` : '';
      return `
        <div class="topic-card" data-id="${{item.id}}">
          <span class="level-badge ${{levelClass}}">${{item.level || 'basics'}}</span>
          <h3>${{item.title}}</h3>
          <p>${{item.learn || ''}}</p>
          ${{hasDemo}}
          <div class="output" style="display:none"></div>
        </div>
      `;
    }}
    
    // Load curriculum
    async function loadCurriculum() {{
      try {{
        const data = await api('api/curriculum');
        if (data.ok) {{
          curriculum = data.items || [];
          books = data.books || [];
          levels = data.levels || [];
          
          // Render all topics
          document.getElementById('topic-list').innerHTML = 
            curriculum.map(renderTopic).join('') || '<p>No topics found</p>';
          
          // Render level tabs
          const levelTabs = document.getElementById('level-tabs');
          levelTabs.innerHTML = levels.map((lvl, i) => 
            `<button class="tab ${{i === 0 ? 'active' : ''}}" data-level="${{lvl}}">${{lvl}}</button>`
          ).join('');
          if (levels.length) filterByLevel(levels[0]);
          
          // Render book tabs
          const bookTabs = document.getElementById('book-tabs');
          bookTabs.innerHTML = books.map((book, i) => 
            `<button class="tab ${{i === 0 ? 'active' : ''}}" data-book="${{book}}">${{book}}</button>`
          ).join('');
          if (books.length) filterByBook(books[0]);
          
          // Add tab click handlers
          levelTabs.onclick = (e) => {{
            if (e.target.dataset.level) {{
              levelTabs.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
              e.target.classList.add('active');
              filterByLevel(e.target.dataset.level);
            }}
          }};
          bookTabs.onclick = (e) => {{
            if (e.target.dataset.book) {{
              bookTabs.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
              e.target.classList.add('active');
              filterByBook(e.target.dataset.book);
            }}
          }};
        }}
      }} catch (e) {{
        document.getElementById('topic-list').innerHTML = '<p class="error">Failed to load curriculum</p>';
      }}
    }}
    
    function filterByLevel(level) {{
      const items = curriculum.filter(c => c.level === level);
      document.getElementById('level-topics').innerHTML = items.map(renderTopic).join('') || '<p>No topics</p>';
    }}
    
    function filterByBook(book) {{
      const items = curriculum.filter(c => c.book === book);
      document.getElementById('book-topics').innerHTML = items.map(renderTopic).join('') || '<p>No topics</p>';
    }}
    
    // Demo runner
    document.addEventListener('click', async (e) => {{
      if (e.target.classList.contains('demo-btn')) {{
        const demoId = e.target.dataset.demo;
        const output = e.target.parentElement.querySelector('.output');
        output.style.display = 'block';
        output.textContent = 'Running...';
        output.classList.add('loading');
        
        try {{
          const data = await api('api/try/' + demoId, {{ method: 'POST' }});
          output.classList.remove('loading');
          if (data.error) {{
            output.classList.add('error');
            output.textContent = data.error;
          }} else {{
            output.classList.remove('error');
            output.textContent = data.output || 'Done!';
          }}
        }} catch (err) {{
          output.classList.remove('loading');
          output.classList.add('error');
          output.textContent = 'Error: ' + err.message;
        }}
      }}
    }});
    
    loadCurriculum();
  </script>
</body>
</html>
'''
