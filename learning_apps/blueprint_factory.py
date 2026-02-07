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
    
    @bp.route("/api/intelligent-try/<demo_id>", methods=["GET", "POST"])
    def api_intelligent_try(demo_id):
        """Run demo with full intelligent context (explanations, visualizations, etc.)"""
        try:
            from learning_apps.intelligent_demos import run_intelligent_demo
            params = request.get_json(silent=True) if request.method == "POST" else {}
            result = run_intelligent_demo(demo_id, params)
            
            # If intelligent demo not found, fall back to basic demo
            if not result.get("ok") and demos_available:
                basic_result = run_demo(demo_id)
                return jsonify({
                    **basic_result,
                    "intelligent_available": False,
                    "message": "Basic demo only - no enriched content available"
                })
            
            return jsonify(result)
        except Exception as e:
            # Fall back to basic demo
            if demos_available:
                return jsonify({**run_demo(demo_id), "intelligent_available": False})
            return jsonify({"ok": False, "error": str(e)}), 500
    
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
    .demo-actions {{
      display: flex;
      gap: 8px;
      margin-top: 12px;
      flex-wrap: wrap;
    }}
    .demo-btn {{
      background: var(--accent);
      border: none;
      color: white;
      padding: 8px 14px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.85rem;
      transition: all 0.2s;
    }}
    .demo-btn:hover {{ background: var(--accent-hover); transform: translateY(-1px); }}
    .demo-btn.intelligent {{
      background: linear-gradient(135deg, #8b5cf6, #6366f1);
      box-shadow: 0 2px 8px rgba(139, 92, 246, 0.3);
    }}
    .demo-btn.intelligent:hover {{
      box-shadow: 0 4px 16px rgba(139, 92, 246, 0.5);
    }}
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
    
    /* Intelligent Demo Panel */
    .intelligent-panel {{
      margin-top: 16px;
    }}
    .intelligent-demo {{
      background: linear-gradient(135deg, #1a1a2e, #16213e);
      border: 1px solid #6366f1;
      border-radius: 16px;
      padding: 20px;
    }}
    .intelligent-demo .demo-header {{
      border-bottom: 1px solid #334155;
      padding-bottom: 16px;
      margin-bottom: 16px;
    }}
    .intelligent-demo .demo-header h3 {{
      color: #e2e8f0;
      margin: 0 0 8px 0;
      font-size: 1.3rem;
    }}
    .intelligent-demo .demo-meta {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
    }}
    .intelligent-demo .book-tag, .intelligent-demo .complexity-tag {{
      background: rgba(99, 102, 241, 0.2);
      padding: 4px 12px;
      border-radius: 16px;
      font-size: 0.8rem;
      color: #a5b4fc;
    }}
    .demo-section {{
      margin-bottom: 20px;
      padding: 16px;
      background: rgba(0, 0, 0, 0.2);
      border-radius: 12px;
    }}
    .demo-section h4 {{
      color: #6366f1;
      margin: 0 0 12px 0;
      font-size: 1rem;
    }}
    .prereq-list {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .prereq-tag {{
      background: #22c55e33;
      color: #22c55e;
      padding: 4px 12px;
      border-radius: 16px;
      font-size: 0.8rem;
    }}
    .markdown-content {{
      color: #e2e8f0;
      line-height: 1.7;
    }}
    .markdown-content h2, .markdown-content h3 {{
      color: #f8fafc;
      margin-top: 16px;
    }}
    .markdown-content code {{
      background: #0f172a;
      padding: 2px 6px;
      border-radius: 4px;
      color: #e94560;
    }}
    .markdown-content table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0;
    }}
    .markdown-content th, .markdown-content td {{
      border: 1px solid #334155;
      padding: 8px 12px;
      text-align: left;
    }}
    .markdown-content th {{
      background: #1e293b;
      color: #f8fafc;
    }}
    .steps-container {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .algo-step {{
      display: flex;
      gap: 16px;
      align-items: flex-start;
    }}
    .step-num {{
      background: #6366f1;
      color: white;
      width: 28px;
      height: 28px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: bold;
      font-size: 0.85rem;
      flex-shrink: 0;
    }}
    .step-content {{
      flex: 1;
    }}
    .step-content strong {{
      color: #f8fafc;
    }}
    .step-content p {{
      color: #94a3b8;
      margin: 4px 0;
    }}
    .step-content code {{
      display: block;
      background: #0f172a;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 0.85rem;
      color: #22c55e;
      margin-top: 8px;
    }}
    .output-box {{
      background: #0f172a;
      padding: 16px;
      border-radius: 8px;
      font-family: 'Fira Code', monospace;
      color: #22c55e;
    }}
    .related-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 12px;
    }}
    .related-concept {{
      background: #1e293b;
      padding: 12px;
      border-radius: 8px;
      border-left: 3px solid #6366f1;
    }}
    .related-concept strong {{
      color: #f8fafc;
      display: block;
      margin-bottom: 4px;
    }}
    .views-available {{
      color: #94a3b8;
      font-size: 0.8rem;
    }}
    .challenges-list {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .challenge-item {{
      background: #1e293b;
      padding: 12px;
      border-radius: 8px;
      border-left: 3px solid #f59e0b;
    }}
    .challenge-type {{
      background: #f59e0b33;
      color: #f59e0b;
      padding: 2px 8px;
      border-radius: 8px;
      font-size: 0.75rem;
      text-transform: uppercase;
    }}
    .challenge-item p {{
      color: #e2e8f0;
      margin: 8px 0 0 0;
    }}
    .socratic-question {{
      background: linear-gradient(135deg, #8b5cf6, #6366f1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-size: 1.1rem;
      font-style: italic;
    }}
    .loading-spinner {{
      text-align: center;
      padding: 40px;
      color: #94a3b8;
      font-size: 1.1rem;
    }}
    
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
      const hasDemo = item.try_demo ? `
        <div class="demo-actions">
          <button class="demo-btn" data-demo="${{item.try_demo}}">▶ Quick Run</button>
          <button class="demo-btn intelligent" data-demo="${{item.try_demo}}" data-intelligent="true">🧠 Deep Learn</button>
        </div>
      ` : '';
      return `
        <div class="topic-card" data-id="${{item.id}}">
          <span class="level-badge ${{levelClass}}">${{item.level || 'basics'}}</span>
          <h3>${{item.title}}</h3>
          <p>${{item.learn || ''}}</p>
          ${{hasDemo}}
          <div class="output" style="display:none"></div>
          <div class="intelligent-panel" style="display:none"></div>
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
        const isIntelligent = e.target.dataset.intelligent === 'true';
        const card = e.target.closest('.topic-card');
        const output = card.querySelector('.output');
        const intelligentPanel = card.querySelector('.intelligent-panel');
        
        if (isIntelligent) {{
          // Intelligent demo with full educational context
          intelligentPanel.style.display = 'block';
          intelligentPanel.innerHTML = '<div class="loading-spinner">🧠 Loading deep learning experience...</div>';
          
          try {{
            const data = await api('api/intelligent-try/' + demoId, {{ method: 'POST' }});
            if (data.ok) {{
              intelligentPanel.innerHTML = renderIntelligentDemo(data);
            }} else {{
              // Fall back to basic output
              output.style.display = 'block';
              output.textContent = data.output || data.error || 'Demo completed';
              intelligentPanel.style.display = 'none';
            }}
          }} catch (err) {{
            intelligentPanel.innerHTML = '<div class="error">Error: ' + err.message + '</div>';
          }}
        }} else {{
          // Quick run - basic demo
          output.style.display = 'block';
          output.textContent = 'Running...';
          output.classList.add('loading');
          intelligentPanel.style.display = 'none';
          
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
      }}
    }});
    
    // Render intelligent demo with all educational content
    function renderIntelligentDemo(data) {{
      const prereqs = (data.prerequisites || []).map(p => `<span class="prereq-tag">${{p}}</span>`).join('');
      const steps = (data.algorithm_steps || []).map((s, i) => `
        <div class="algo-step">
          <div class="step-num">${{i + 1}}</div>
          <div class="step-content">
            <strong>${{s.title}}</strong>
            <p>${{s.description}}</p>
            ${{s.state ? `<code>${{s.state}}</code>` : ''}}
          </div>
        </div>
      `).join('');
      
      const related = (data.related_concepts || []).map(r => `
        <div class="related-concept">
          <strong>${{r.concept}}</strong>
          ${{Object.keys(r.views || {{}}).length > 0 ? 
            '<span class="views-available">📚 ' + Object.keys(r.views).join(', ') + '</span>' : ''}}
        </div>
      `).join('');
      
      const challenges = (data.practice_challenges || []).map(c => `
        <div class="challenge-item">
          <span class="challenge-type">${{c.type}}</span>
          <p>${{c.prompt}}</p>
        </div>
      `).join('');
      
      return `
        <div class="intelligent-demo">
          <div class="demo-header">
            <h3>🧠 ${{data.title || 'Demo'}}</h3>
            <div class="demo-meta">
              <span class="book-tag">📖 ${{data.book || ''}}</span>
              <span class="complexity-tag">⏱️ ${{data.complexity?.time || ''}} | 💾 ${{data.complexity?.space || ''}}</span>
            </div>
          </div>
          
          <div class="demo-section prereqs">
            <h4>📋 Prerequisites</h4>
            <div class="prereq-list">${{prereqs || '<span class="none">None</span>'}}</div>
          </div>
          
          <div class="demo-section pre-explanation">
            <h4>🎯 What You're About to Learn</h4>
            <div class="markdown-content">${{data.pre_explanation || ''}}</div>
          </div>
          
          <div class="demo-section algorithm-steps">
            <h4>📊 Step-by-Step Breakdown</h4>
            <div class="steps-container">${{steps || '<p>No steps available</p>'}}</div>
          </div>
          
          <div class="demo-section demo-output">
            <h4>▶ Demo Result</h4>
            <div class="output-box">${{data.demo_output || 'No output'}}</div>
          </div>
          
          <div class="demo-section post-explanation">
            <h4>📈 What Just Happened</h4>
            <div class="markdown-content">${{data.post_explanation || ''}}</div>
          </div>
          
          <div class="demo-section related-concepts">
            <h4>🔗 Related Concepts</h4>
            <div class="related-grid">${{related || '<p>None</p>'}}</div>
          </div>
          
          <div class="demo-section practice">
            <h4>💪 Practice Challenges</h4>
            <div class="challenges-list">${{challenges || '<p>None</p>'}}</div>
          </div>
          
          ${{data.socratic_question ? `
            <div class="demo-section socratic">
              <h4>🤔 Think About This</h4>
              <div class="socratic-question">${{data.socratic_question}}</div>
            </div>
          ` : ''}}
        </div>
      `;
    }}
    
    loadCurriculum();
  </script>
  
  <!-- SRS Widget -->
  <style>
    .srs-container {{ position: fixed; bottom: 80px; right: 24px; z-index: 100; }}
    .srs-btn {{ 
      background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
      color: white; border: none; padding: 12px 20px; 
      border-radius: 24px; cursor: pointer; font-weight: 600;
      box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
      display: flex; align-items: center; gap: 8px;
    }}
    .srs-btn:hover {{ transform: scale(1.05); }}
    .srs-badge {{ background: #ef4444; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }}
    .playground-btn {{
      position: fixed; bottom: 140px; right: 24px; z-index: 100;
      background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
      color: white; border: none; padding: 12px 20px;
      border-radius: 24px; cursor: pointer; font-weight: 600;
      box-shadow: 0 4px 12px rgba(34, 197, 94, 0.4);
    }}
    .playground-btn:hover {{ transform: scale(1.05); }}
  </style>
  
  <div class="srs-container">
    <button class="srs-btn" onclick="window.location.href='/'">
      📚 Review <span class="srs-badge" id="srs-count">0</span>
    </button>
  </div>
  
  <button class="playground-btn" onclick="window.location.href='/'">
    💻 Playground
  </button>
  
  <script>
    // Load SRS count
    fetch('/api/srs/cards?user=default&limit=100')
      .then(r => r.json())
      .then(d => {{ document.getElementById('srs-count').textContent = d.count || 0; }})
      .catch(() => {{}});
  </script>
</body>
</html>
'''
