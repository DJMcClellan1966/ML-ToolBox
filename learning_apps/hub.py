"""
Learning Apps Hub — Single entry point listing all labs with links and global search.
Run from repo root: python learning_apps/hub.py
Open http://127.0.0.1:5000
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flask import Flask, render_template_string, request, jsonify
import importlib

app = Flask(__name__)

LABS = [
    {"name": "ML Learning Lab", "port": 5001, "path": "ml_learning_lab", "cmd": "python ml_learning_lab/app.py",
     "desc": "Knuth, Skiena, Sedgewick, Bishop/Goodfellow/R&N, Info theory, Compass, Build Knuth Machine.", "icon": "🧠"},
    {"name": "CLRS Algorithms Lab", "port": 5002, "path": "clrs_algorithms_lab", "cmd": "python learning_apps/clrs_algorithms_lab/app.py",
     "desc": "Introduction to Algorithms: DP, Greedy, Graph.", "icon": "📊"},
    {"name": "Deep Learning Lab", "port": 5003, "path": "deep_learning_lab", "cmd": "python learning_apps/deep_learning_lab/app.py",
     "desc": "Goodfellow, Bishop, ESL, Burkov.", "icon": "🔮"},
    {"name": "AI Concepts Lab", "port": 5004, "path": "ai_concepts_lab", "cmd": "python learning_apps/ai_concepts_lab/app.py",
     "desc": "Russell & Norvig: game theory, search, RL, probabilistic reasoning.", "icon": "🤖"},
    {"name": "Cross-Domain Lab", "port": 5005, "path": "cross_domain_lab", "cmd": "python learning_apps/cross_domain_lab/app.py",
     "desc": "Quantum, stat mech, linguistics, precognition, self-organization.", "icon": "🌐"},
    {"name": "Python Practice Lab", "port": 5006, "path": "python_practice_lab", "cmd": "python learning_apps/python_practice_lab/app.py",
     "desc": "Reed & Zelle: problem decomposition, algorithms, code organization.", "icon": "🐍"},
    {"name": "SICP Lab", "port": 5007, "path": "sicp_lab", "cmd": "python learning_apps/sicp_lab/app.py",
     "desc": "Structure and Interpretation of Computer Programs.", "icon": "📖"},
    {"name": "Practical ML Lab", "port": 5008, "path": "practical_ml_lab", "cmd": "python learning_apps/practical_ml_lab/app.py",
     "desc": "Hands-On ML (Géron): features, tuning, ensembles, production.", "icon": "🛠️"},
    {"name": "RL Lab", "port": 5009, "path": "rl_lab", "cmd": "python learning_apps/rl_lab/app.py",
     "desc": "Sutton & Barto: MDPs, TD, Q-learning, policy gradient.", "icon": "🎮"},
    {"name": "Probabilistic ML Lab", "port": 5010, "path": "probabilistic_ml_lab", "cmd": "python learning_apps/probabilistic_ml_lab/app.py",
     "desc": "Murphy: graphical models, EM, variational inference, Bayesian.", "icon": "📈"},
    {"name": "ML Theory Lab", "port": 5011, "path": "ml_theory_lab", "cmd": "python learning_apps/ml_theory_lab/app.py",
     "desc": "Shalev-Shwartz & Ben-David: PAC, VC dimension, generalization.", "icon": "📐"},
    {"name": "LLM Engineers Lab", "port": 5012, "path": "llm_engineers_lab", "cmd": "python learning_apps/llm_engineers_lab/app.py",
     "desc": "Handbook + Build Your Own LLM: RAG, prompts, eval, safety.", "icon": "💬"},
    {"name": "Math for ML Lab", "port": 5013, "path": "math_for_ml_lab", "cmd": "python learning_apps/math_for_ml_lab/app.py",
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
                item["lab_port"] = lab["port"]
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
                "lab_port": item.get("lab_port"),
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
                    "lab_port": lab["port"] if lab else 5000,
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
    
    .modal-overlay {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.7);
      z-index: 200;
      justify-content: center;
      align-items: center;
      padding: 24px;
    }
    .modal-overlay.active { display: flex; }
    .modal {
      background: var(--bg-secondary);
      border-radius: var(--radius);
      max-width: 500px;
      width: 100%;
      padding: 24px;
    }
    .modal h2 { margin-bottom: 16px; }
    .modal p { color: var(--text-secondary); margin-bottom: 12px; }
    .modal .cmd-big {
      background: var(--bg-primary);
      padding: 16px;
      border-radius: 8px;
      font-family: 'Fira Code', Consolas, monospace;
      font-size: 1rem;
      margin-bottom: 16px;
    }
    .modal-actions { display: flex; gap: 12px; justify-content: flex-end; }
    .modal-actions .btn-primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
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
      <div class="lab-card" data-port="{{ lab.port }}" data-cmd="{{ lab.cmd }}" data-name="{{ lab.name }}">
        <div class="lab-header">
          <div class="lab-icon">{{ lab.icon }}</div>
          <div>
            <h2>{{ lab.name }} <span class="port">:{{ lab.port }}</span></h2>
          </div>
        </div>
        <p class="desc">{{ lab.desc }}</p>
        <div class="cmd">{{ lab.cmd }}</div>
      </div>
      {% endfor %}
    </div>
  </main>
  
  <!-- Lab Modal -->
  <div class="modal-overlay" id="lab-modal">
    <div class="modal">
      <h2 id="modal-lab-name"></h2>
      <p>Run this command from the repo root, then click Open:</p>
      <div class="cmd-big" id="modal-cmd"></div>
      <div class="modal-actions">
        <button class="btn" id="modal-close">Close</button>
        <button class="btn btn-primary" id="modal-open">Open Lab →</button>
      </div>
    </div>
  </div>
  
  <script>
    const api = (path) => fetch(path).then(r => r.json());
    let selectedLab = null;
    
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
    
    // Lab cards
    document.querySelectorAll('.lab-card').forEach(card => {
      card.onclick = () => {
        selectedLab = {
          port: card.dataset.port,
          cmd: card.dataset.cmd,
          name: card.dataset.name
        };
        document.getElementById('modal-lab-name').textContent = selectedLab.name;
        document.getElementById('modal-cmd').textContent = selectedLab.cmd;
        document.getElementById('lab-modal').classList.add('active');
      };
    });
    
    document.getElementById('modal-close').onclick = () => {
      document.getElementById('lab-modal').classList.remove('active');
    };
    document.getElementById('modal-open').onclick = () => {
      if (selectedLab) window.open('http://127.0.0.1:' + selectedLab.port, '_blank');
    };
    document.getElementById('lab-modal').onclick = (e) => {
      if (e.target.id === 'lab-modal') document.getElementById('lab-modal').classList.remove('active');
    };
    
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
            div.onclick = () => window.open('http://127.0.0.1:' + r.lab_port, '_blank');
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
            const r = await fetch('http://127.0.0.1:' + lab.port + '/api/curriculum');
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
            card.onclick = () => window.open('http://127.0.0.1:' + tutor.lab_port, '_blank');
            card.title = 'Click to learn with ' + tutor.name + ' in ' + tutor.lab_name;
            scroll.appendChild(card);
          });
        }
      } catch (e) {
        console.log('Tutors not available:', e);
      }
    }
    
    loadStats();
    loadTutors();
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
