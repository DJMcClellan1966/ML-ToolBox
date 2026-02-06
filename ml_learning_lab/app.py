"""
ML Learning Lab — Learn, Decide, Practice, Think · By Book · By Level · Build Knuth Machine.
Integrates ml_toolbox and ML_Compass. Run from repo root: python ml_learning_lab/app.py
"""
import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEARNING_APP_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LEARNING_APP_DIR) not in sys.path:
    sys.path.insert(0, str(LEARNING_APP_DIR))

# ML_Compass (core of learning flow)
try:
    from ML_Compass.oracle import suggest as oracle_suggest, suggest_from_description
    from ML_Compass.explainers import explain_concept
    from ML_Compass.pattern_guidance import get_avoid_encourage
    from ML_Compass.socratic import debate_and_question
    from ML_Compass.theory_channel import channel_capacity_bits, recommend_redundancy
    COMPASS_AVAILABLE = True
except Exception as e:
    COMPASS_AVAILABLE = False
    _compass_err = str(e)

# Optional: ml_toolbox for Practice demos
try:
    from ml_toolbox import MLToolbox
    import numpy as np
    TOOLBOX_AVAILABLE = True
except Exception:
    TOOLBOX_AVAILABLE = False
    MLToolbox = None
    np = None

try:
    from flask import Flask, request, jsonify, render_template_string
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

if not FLASK_AVAILABLE:
    print("Install Flask: pip install flask")
    sys.exit(1)

app = Flask(__name__)


# ---------- API routes (used by the front-end) ----------

@app.route("/api/health")
def api_health():
    return jsonify({
        "ok": True,
        "compass": COMPASS_AVAILABLE,
        "toolbox": TOOLBOX_AVAILABLE,
    })


@app.route("/api/explain/<concept>")
def api_explain(concept):
    if not COMPASS_AVAILABLE:
        return jsonify({"ok": False, "error": "ML_Compass not available"}), 503
    out = explain_concept(concept)
    return jsonify(out)


@app.route("/api/oracle", methods=["POST"])
def api_oracle():
    if not COMPASS_AVAILABLE:
        return jsonify({"ok": False, "error": "ML_Compass not available"}), 503
    body = request.get_json() or {}
    if body.get("description"):
        out = suggest_from_description(body["description"])
    else:
        profile = body.get("profile", {})
        if not profile and body.get("tabular"):
            profile = {
                "tabular": True,
                "classification": body.get("classification", True),
                "n_samples": body.get("n_samples", "medium"),
            }
        out = oracle_suggest(profile)
    return jsonify(out)


@app.route("/api/guidance")
def api_guidance():
    if not COMPASS_AVAILABLE:
        return jsonify({"ok": False, "error": "ML_Compass not available"}), 503
    minimal = request.args.get("minimal", "false").lower() == "true"
    return jsonify(get_avoid_encourage(minimal=minimal))


@app.route("/api/debate", methods=["POST"])
def api_debate():
    if not COMPASS_AVAILABLE:
        return jsonify({"ok": False, "error": "ML_Compass not available"}), 503
    body = request.get_json() or {}
    statement = body.get("statement", body.get("text", ""))
    if not statement and request.content_type and "application/json" not in request.content_type:
        statement = request.data.decode("utf-8") if request.data else ""
    out = debate_and_question(statement or "I used a single model.")
    return jsonify(out)


@app.route("/api/capacity")
def api_capacity():
    if not COMPASS_AVAILABLE:
        return jsonify({"ok": False, "error": "ML_Compass not available"}), 503
    try:
        signal = float(request.args.get("signal_power", 10))
        noise = float(request.args.get("noise_power", 1))
        C = channel_capacity_bits(signal, noise)
        rec = recommend_redundancy(signal, noise)
        return jsonify({"ok": True, "channel_capacity_bits": C, "recommend_redundancy": rec})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------- Curriculum (books, levels, learn/try) ----------
try:
    from curriculum import (
        get_curriculum, get_books, get_levels,
        get_by_book, get_by_level, get_item,
    )
    from demos import run_demo, knuth_machine_run
    CURRICULUM_AVAILABLE = True
except Exception:
    CURRICULUM_AVAILABLE = False


@app.route("/api/curriculum")
def api_curriculum():
    if not CURRICULUM_AVAILABLE:
        return jsonify({"ok": False, "items": [], "books": [], "levels": []}), 503
    return jsonify({"ok": True, "items": get_curriculum(), "books": get_books(), "levels": get_levels()})


@app.route("/api/curriculum/book/<book_id>")
def api_curriculum_book(book_id):
    if not CURRICULUM_AVAILABLE:
        return jsonify({"ok": False, "items": []}), 503
    return jsonify({"ok": True, "items": get_by_book(book_id)})


@app.route("/api/curriculum/level/<level>")
def api_curriculum_level(level):
    if not CURRICULUM_AVAILABLE:
        return jsonify({"ok": False, "items": []}), 503
    return jsonify({"ok": True, "items": get_by_level(level)})


@app.route("/api/curriculum/item/<item_id>")
def api_curriculum_item(item_id):
    if not CURRICULUM_AVAILABLE:
        return jsonify({"ok": False, "item": None}), 503
    item = get_item(item_id)
    return jsonify({"ok": item is not None, "item": item})


@app.route("/api/try/<demo_id>", methods=["GET", "POST"])
def api_try_demo(demo_id):
    if not CURRICULUM_AVAILABLE:
        return jsonify({"ok": False, "error": "Demos not available"}), 503
    result = run_demo(demo_id)
    return jsonify(result)


@app.route("/api/knuth_machine", methods=["POST"])
def api_knuth_machine():
    if not CURRICULUM_AVAILABLE:
        return jsonify({"ok": False, "error": "Knuth Machine not available"}), 503
    body = request.get_json() or {}
    steps = body.get("steps", ["lcg", "shuffle", "sample"])
    params = body.get("params", {"seed": 42, "lcg_n": 10, "sample_k": 5})
    result = knuth_machine_run(steps, params)
    return jsonify(result)


@app.route("/api/practice/demo")
def api_practice_demo():
    """Run a minimal fit/predict demo. Uses sklearn if toolbox not available."""
    if TOOLBOX_AVAILABLE and MLToolbox is not None:
        try:
            toolbox = MLToolbox(check_dependencies=False)
            X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]])
            y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
            result = toolbox.fit(X, y, task_type="classification", use_cache=False)
            model = result.get("model")
            if model is not None:
                pred = toolbox.predict(model, np.array([[2.5, 3.5]]), use_cache=False)
                return jsonify({
                    "ok": True,
                    "source": "ml_toolbox",
                    "message": "Trained with MLToolbox.fit(); prediction for [2.5, 3.5]: " + str(pred),
                    "metrics": {k: v for k, v in result.items() if k != "model" and isinstance(v, (int, float, str))},
                })
        except Exception as e:
            pass
    # Fallback: sklearn
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        pred = clf.predict(X[:1])
        return jsonify({
            "ok": True,
            "source": "sklearn",
            "message": "Fallback: trained RandomForest on synthetic data. First-sample prediction: " + str(pred.tolist()),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------- Serve the single-page app ----------

INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ML Toolbox Learning App</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: linear-gradient(160deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
      min-height: 100vh;
      color: #e2e8f0;
    }
    .container { max-width: 900px; margin: 0 auto; padding: 24px; }
    header {
      text-align: center;
      padding: 32px 0 24px;
      border-bottom: 1px solid #334155;
    }
    header h1 { font-size: 1.85rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px; }
    header p { color: #94a3b8; font-size: 1rem; }
    .tabs {
      display: flex;
      gap: 8px;
      margin: 24px 0 20px;
      flex-wrap: wrap;
    }
    .tabs button {
      padding: 10px 18px;
      border: 1px solid #475569;
      border-radius: 8px;
      background: #1e293b;
      color: #e2e8f0;
      cursor: pointer;
      font-size: 0.95rem;
    }
    .tabs button:hover { background: #334155; }
    .tabs button.active { background: #3b82f6; border-color: #3b82f6; }
    .panel { display: none; }
    .panel.active { display: block; }
    .panel h2 { font-size: 1.25rem; margin-bottom: 16px; color: #f1f5f9; }
    .card {
      background: #1e293b;
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 16px;
      border: 1px solid #334155;
    }
    label { display: block; margin-bottom: 6px; color: #94a3b8; font-size: 0.9rem; }
    input[type="text"], textarea, select {
      width: 100%;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid #475569;
      background: #0f172a;
      color: #e2e8f0;
      font-size: 0.95rem;
    }
    textarea { min-height: 100px; resize: vertical; }
    .btn {
      padding: 10px 20px;
      border-radius: 8px;
      border: none;
      font-size: 0.95rem;
      cursor: pointer;
      margin-top: 10px;
    }
    .btn-primary { background: #3b82f6; color: white; }
    .btn-primary:hover { background: #2563eb; }
    .btn-secondary { background: #475569; color: white; margin-left: 8px; }
    .btn-secondary:hover { background: #64748b; }
    .out {
      margin-top: 16px;
      padding: 14px;
      border-radius: 8px;
      background: #0f172a;
      border: 1px solid #334155;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 0.9rem;
      max-height: 320px;
      overflow-y: auto;
    }
    .out.success { border-color: #22c55e; }
    .out.error { border-color: #ef4444; color: #fca5a5; }
    .out.empty { color: #64748b; }
    .concept-list { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
    .concept-list button {
      padding: 8px 14px;
      border-radius: 8px;
      border: 1px solid #475569;
      background: #1e293b;
      color: #cbd5e1;
      cursor: pointer;
      font-size: 0.9rem;
    }
    .concept-list button:hover { background: #334155; }
    .views { margin-top: 12px; }
    .views h4 { color: #94a3b8; margin: 12px 0 4px; font-size: 0.85rem; }
    .views p { margin-bottom: 8px; line-height: 1.5; }
    .status { font-size: 0.8rem; color: #64748b; margin-top: 12px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>ML Toolbox Learning App</h1>
      <p>Learn concepts · Decide what to do · Practice with the toolbox · Think with Socratic debate</p>
    </header>

    <div class="tabs" role="tablist">
      <button type="button" class="tab active" data-tab="learn">Compass</button>
      <button type="button" class="tab" data-tab="books">By Book</button>
      <button type="button" class="tab" data-tab="levels">By Level</button>
      <button type="button" class="tab" data-tab="knuth-machine">Build Knuth Machine</button>
    </div>

    <!-- Compass: Learn, Decide, Practice, Think -->
    <div id="panel-learn" class="panel active">
      <h2>Learn — concepts</h2>
      <div class="card">
        <label>Pick a concept (cross-domain: theory, ML, practice)</label>
        <div class="concept-list">
          <button data-concept="entropy">entropy</button>
          <button data-concept="bias_variance">bias_variance</button>
          <button data-concept="capacity">capacity</button>
        </div>
        <div id="learn-out" class="out empty">Click a concept to see explanations.</div>
      </div>
      <h2 style="margin-top:20px;">Decide — oracle & guidance</h2>
      <div class="card">
        <label>Describe your problem</label>
        <textarea id="decide-input" placeholder="e.g. I have tabular data, 5000 rows, classification. My model overfits."></textarea>
        <button type="button" class="btn btn-primary" id="decide-oracle">Ask Oracle</button>
        <button type="button" class="btn btn-secondary" id="decide-guidance">Show avoid / encourage</button>
        <div id="decide-out" class="out empty">Result here.</div>
      </div>
      <h2 style="margin-top:20px;">Practice — run a demo</h2>
      <div class="card">
        <button type="button" class="btn btn-primary" id="practice-run">Run toolbox demo</button>
        <div id="practice-out" class="out empty">Output here.</div>
      </div>
      <h2 style="margin-top:20px;">Think — debate</h2>
      <div class="card">
        <label>Your statement</label>
        <textarea id="think-input" placeholder="e.g. I used an ensemble of three models."></textarea>
        <button type="button" class="btn btn-primary" id="think-btn">Get debate + question</button>
        <div id="think-out" class="out empty">Debate and question here.</div>
      </div>
    </div>

    <!-- By Book -->
    <div id="panel-books" class="panel">
      <h2>Learn by book</h2>
      <div class="card">
        <label>Choose a book</label>
        <div id="book-list" class="concept-list"></div>
        <div id="book-topics"></div>
        <div id="topic-detail" style="margin-top:16px;display:none;">
          <h3 id="topic-title" style="margin-bottom:8px;"></h3>
          <p><strong>Learn</strong></p>
          <p id="topic-learn" class="out" style="max-height:120px;"></p>
          <p><strong>Try it</strong></p>
          <button type="button" class="btn btn-primary" id="topic-try-btn">Run demo</button>
          <pre id="topic-code" style="background:#0f172a;padding:10px;border-radius:8px;margin-top:8px;font-size:0.85rem;overflow:auto;"></pre>
          <div id="topic-try-out" class="out empty" style="margin-top:8px;"></div>
        </div>
      </div>
    </div>

    <!-- By Level -->
    <div id="panel-levels" class="panel">
      <h2>Learn by level (basics → expert)</h2>
      <div class="card">
        <div id="level-tabs" class="tabs" style="margin-bottom:12px;"></div>
        <div id="level-topics"></div>
        <div id="level-topic-detail" style="margin-top:16px;display:none;">
          <h3 id="level-topic-title" style="margin-bottom:8px;"></h3>
          <p><strong>Learn</strong></p>
          <p id="level-topic-learn" class="out" style="max-height:120px;"></p>
          <button type="button" class="btn btn-primary" id="level-topic-try-btn">Run demo</button>
          <pre id="level-topic-code" style="background:#0f172a;padding:10px;border-radius:8px;margin-top:8px;font-size:0.85rem;"></pre>
          <div id="level-topic-try-out" class="out empty" style="margin-top:8px;"></div>
        </div>
      </div>
    </div>

    <!-- Build Knuth Machine -->
    <div id="panel-knuth-machine" class="panel">
      <h2>Build a Knuth Machine</h2>
      <p style="color:#94a3b8;margin-bottom:16px;">Chain TAOCP-style steps: LCG → Shuffle → Sample. Reproducible randomness for ML.</p>
      <div class="card">
        <label>Pipeline steps</label>
        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px;">
          <label><input type="checkbox" id="km-lcg" checked> LCG (random numbers)</label>
          <label><input type="checkbox" id="km-shuffle" checked> Shuffle (Fisher-Yates)</label>
          <label><input type="checkbox" id="km-sample" checked> Sample (k from n)</label>
        </div>
        <label>Params: seed <input type="number" id="km-seed" value="42" style="width:80px;"> LCG n <input type="number" id="km-lcg-n" value="10" style="width:80px;"> Sample k <input type="number" id="km-sample-k" value="5" style="width:80px;"></label>
        <button type="button" class="btn btn-primary" id="knuth-machine-run" style="margin-top:12px;">Run pipeline</button>
        <div id="knuth-machine-out" class="out empty" style="margin-top:12px;"></div>
      </div>
    </div>

    <p class="status" id="status">Checking backend…</p>
  </div>

  <script>
    const api = (path, opts = {}) => fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts }).then(r => r.json());

    function showOut(id, text, isError = false) {
      const el = document.getElementById(id);
      el.textContent = text || '';
      el.className = 'out ' + (isError ? 'error' : text ? 'success' : 'empty');
    }

    function setStatus(msg) {
      const el = document.getElementById('status');
      if (el) el.textContent = msg;
    }

    // Tabs (Compass, By Book, By Level, Build Knuth Machine)
    document.querySelectorAll('.tab').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.panel').forEach(p => { p.classList.remove('active'); p.style.display = 'none'; });
        btn.classList.add('active');
        const panel = document.getElementById('panel-' + btn.dataset.tab);
        if (panel) { panel.classList.add('active'); panel.style.display = 'block'; }
      });
    });

    // Learn: concept buttons
    document.querySelectorAll('[data-concept]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const concept = btn.dataset.concept;
        showOut('learn-out', 'Loading…');
        try {
          const data = await api('/api/explain/' + concept);
          if (data.ok && data.views) {
            let s = '';
            for (const [k, v] of Object.entries(data.views)) s += k + ':\n' + v + '\n\n';
            showOut('learn-out', s.trim());
          } else {
            showOut('learn-out', 'Available: ' + (data.available || []).join(', ') || JSON.stringify(data), true);
          }
        } catch (e) {
          showOut('learn-out', 'Error: ' + e.message, true);
        }
      });
    });

    // Decide: oracle
    document.getElementById('decide-oracle').addEventListener('click', async () => {
      const desc = document.getElementById('decide-input').value.trim() || 'I have tabular data and need to classify.';
      showOut('decide-out', 'Loading…');
      try {
        const data = await api('/api/oracle', { method: 'POST', body: JSON.stringify({ description: desc }) });
        if (data.ok !== false) {
          let s = 'Pattern: ' + (data.pattern || '—') + '\n\nSuggestion: ' + (data.suggestion || '—') + '\n\nWhy: ' + (data.why || '—');
          showOut('decide-out', s);
        } else {
          showOut('decide-out', data.error || JSON.stringify(data), true);
        }
      } catch (e) {
        showOut('decide-out', 'Error: ' + e.message, true);
      }
    });

    document.getElementById('decide-guidance').addEventListener('click', async () => {
      showOut('decide-out', 'Loading…');
      try {
        const data = await api('/api/guidance');
        if (data.avoid || data.encourage) {
          let s = 'Avoid:\n' + (data.avoid || []).map(p => '  • ' + p.pattern + ': ' + (p.reason || '')).join('\n');
          s += '\n\nEncourage:\n' + (data.encourage || []).map(p => '  • ' + p.pattern + ': ' + (p.reason || '')).join('\n');
          showOut('decide-out', s);
        } else {
          showOut('decide-out', data.error || JSON.stringify(data), true);
        }
      } catch (e) {
        showOut('decide-out', 'Error: ' + e.message, true);
      }
    });

    // Practice
    document.getElementById('practice-run').addEventListener('click', async () => {
      showOut('practice-out', 'Running demo…');
      try {
        const data = await api('/api/practice/demo');
        if (data.ok) {
          showOut('practice-out', (data.message || '') + (data.metrics ? '\n\n' + JSON.stringify(data.metrics, null, 2) : ''));
        } else {
          showOut('practice-out', data.error || JSON.stringify(data), true);
        }
      } catch (e) {
        showOut('practice-out', 'Error: ' + e.message, true);
      }
    });

    // Think
    document.getElementById('think-btn').addEventListener('click', async () => {
      const statement = document.getElementById('think-input').value.trim() || 'I used an ensemble of three models.';
      showOut('think-out', 'Loading…');
      try {
        const data = await api('/api/debate', { method: 'POST', body: JSON.stringify({ statement: statement }) });
        if (data.ok && (data.debate || data.question)) {
          showOut('think-out', (data.debate || '') + (data.question ? '\n\nQuestion: ' + data.question : ''));
        } else {
          showOut('think-out', data.debate || data.error || JSON.stringify(data), true);
        }
      } catch (e) {
        showOut('think-out', 'Error: ' + e.message, true);
      }
    });

    // Health
    api('/api/health').then(d => {
      const parts = [];
      if (d.compass) parts.push('ML_Compass');
      else parts.push('no ML_Compass');
      if (d.toolbox) parts.push('ml_toolbox');
      else parts.push('no ml_toolbox (Practice uses sklearn)');
      setStatus('Backend: ' + parts.join(' · '));
    }).catch(() => setStatus('Backend not reachable. Start with: python ml_learning_lab/app.py'));

    // Curriculum: load once, then By Book + By Level
    let curriculum = { items: [], books: [], levels: [] };
    api('/api/curriculum').then(d => {
      if (d.ok) curriculum = { items: d.items || [], books: d.books || [], levels: d.levels || [] };
      const list = document.getElementById('book-list');
      if (list) (curriculum.books || []).forEach(b => {
        const btn = document.createElement('button');
        btn.textContent = b.short || b.name;
        btn.dataset.bookId = b.id;
        btn.style.borderLeftColor = b.color || '#475569';
        btn.style.borderLeftWidth = '3px';
        btn.style.borderLeftStyle = 'solid';
        btn.addEventListener('click', () => {
          const items = (curriculum.items || []).filter(i => i.book_id === b.id);
          const wrap = document.getElementById('book-topics');
          wrap.innerHTML = '';
          items.forEach(it => {
            const bt = document.createElement('button');
            bt.textContent = it.title + ' (' + it.level + ')';
            bt.style.marginRight = '8px'; bt.style.marginTop = '8px';
            bt.addEventListener('click', () => {
              document.getElementById('topic-detail').style.display = 'block';
              document.getElementById('topic-title').textContent = it.title;
              document.getElementById('topic-learn').textContent = it.learn || '';
              document.getElementById('topic-code').textContent = it.try_code || '';
              document.getElementById('topic-try-out').textContent = '';
              document.getElementById('topic-try-out').className = 'out empty';
              document.getElementById('topic-try-btn').dataset.demoId = it.try_demo || '';
            });
            wrap.appendChild(bt);
          });
        });
        list.appendChild(btn);
      });
      const levelTabs = document.getElementById('level-tabs');
      if (levelTabs) ['basics', 'intermediate', 'advanced', 'expert'].forEach(lev => {
        const btn = document.createElement('button');
        btn.textContent = lev;
        btn.dataset.level = lev;
        btn.addEventListener('click', () => {
          const items = (curriculum.items || []).filter(i => i.level === lev);
          const wrap = document.getElementById('level-topics');
          wrap.innerHTML = '';
          items.forEach(it => {
            const bt = document.createElement('button');
            bt.textContent = it.title;
            bt.style.marginRight = '8px'; bt.style.marginTop = '8px';
            bt.addEventListener('click', () => {
              document.getElementById('level-topic-detail').style.display = 'block';
              document.getElementById('level-topic-title').textContent = it.title;
              document.getElementById('level-topic-learn').textContent = it.learn || '';
              document.getElementById('level-topic-code').textContent = it.try_code || '';
              document.getElementById('level-topic-try-out').textContent = '';
              document.getElementById('level-topic-try-btn').dataset.demoId = it.try_demo || '';
            });
            wrap.appendChild(bt);
          });
        });
        levelTabs.appendChild(btn);
      });
    }).catch(() => {});

    document.getElementById('topic-try-btn') && document.getElementById('topic-try-btn').addEventListener('click', async () => {
      const id = document.getElementById('topic-try-btn').dataset.demoId;
      if (!id) { showOut('topic-try-out', 'No demo for this topic.', true); return; }
      showOut('topic-try-out', 'Running…');
      try {
        const d = await api('/api/try/' + id);
        showOut('topic-try-out', d.error ? d.error : d.output, !!d.error);
      } catch (e) { showOut('topic-try-out', e.message, true); }
    });

    document.getElementById('level-topic-try-btn') && document.getElementById('level-topic-try-btn').addEventListener('click', async () => {
      const id = document.getElementById('level-topic-try-btn').dataset.demoId;
      if (!id) { showOut('level-topic-try-out', 'No demo for this topic.', true); return; }
      showOut('level-topic-try-out', 'Running…');
      try {
        const d = await api('/api/try/' + id);
        showOut('level-topic-try-out', d.error ? d.error : d.output, !!d.error);
      } catch (e) { showOut('level-topic-try-out', e.message, true); }
    });

    // Build Knuth Machine
    document.getElementById('knuth-machine-run') && document.getElementById('knuth-machine-run').addEventListener('click', async () => {
      const steps = [];
      if (document.getElementById('km-lcg') && document.getElementById('km-lcg').checked) steps.push('lcg');
      if (document.getElementById('km-shuffle') && document.getElementById('km-shuffle').checked) steps.push('shuffle');
      if (document.getElementById('km-sample') && document.getElementById('km-sample').checked) steps.push('sample');
      const params = {
        seed: parseInt(document.getElementById('km-seed') && document.getElementById('km-seed').value, 10) || 42,
        lcg_n: parseInt(document.getElementById('km-lcg-n') && document.getElementById('km-lcg-n').value, 10) || 10,
        sample_k: parseInt(document.getElementById('km-sample-k') && document.getElementById('km-sample-k').value, 10) || 5
      };
      showOut('knuth-machine-out', 'Running pipeline…');
      try {
        const d = await api('/api/knuth_machine', { method: 'POST', body: JSON.stringify({ steps, params }) });
        showOut('knuth-machine-out', d.error ? d.error : d.output, !!d.error);
      } catch (e) { showOut('knuth-machine-out', e.message, true); }
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


def main():
    import os
    port = int(os.environ.get("PORT", 5001))
    print("ML Toolbox Learning App — http://127.0.0.1:{}/".format(port))
    print("Learn · Decide · Practice · Think")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
