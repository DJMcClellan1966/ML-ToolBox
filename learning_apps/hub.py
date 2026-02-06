"""
Learning Apps Hub — Single entry point listing all labs with links.
Run from repo root: python learning_apps/hub.py
Open http://127.0.0.1:5000
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flask import Flask, render_template_string

app = Flask(__name__)

LABS = [
    {"name": "ML Learning Lab", "port": 5001, "path": "ml_learning_lab", "cmd": "python ml_learning_lab/app.py",
     "desc": "Knuth, Skiena, Sedgewick, Bishop/Goodfellow/R&N, Info theory, Compass, Build Knuth Machine."},
    {"name": "CLRS Algorithms Lab", "port": 5002, "path": "clrs_algorithms_lab", "cmd": "python learning_apps/clrs_algorithms_lab/app.py",
     "desc": "Introduction to Algorithms: DP, Greedy, Graph."},
    {"name": "Deep Learning Lab", "port": 5003, "path": "deep_learning_lab", "cmd": "python learning_apps/deep_learning_lab/app.py",
     "desc": "Goodfellow, Bishop, ESL, Burkov."},
    {"name": "AI Concepts Lab", "port": 5004, "path": "ai_concepts_lab", "cmd": "python learning_apps/ai_concepts_lab/app.py",
     "desc": "Russell & Norvig: game theory, search, RL, probabilistic reasoning."},
    {"name": "Cross-Domain Lab", "port": 5005, "path": "cross_domain_lab", "cmd": "python learning_apps/cross_domain_lab/app.py",
     "desc": "Quantum, stat mech, linguistics, precognition, self-organization."},
    {"name": "Python Practice Lab", "port": 5006, "path": "python_practice_lab", "cmd": "python learning_apps/python_practice_lab/app.py",
     "desc": "Reed & Zelle: problem decomposition, algorithms, code organization."},
    {"name": "SICP Lab", "port": 5007, "path": "sicp_lab", "cmd": "python learning_apps/sicp_lab/app.py",
     "desc": "Structure and Interpretation of Computer Programs."},
    {"name": "Practical ML Lab", "port": 5008, "path": "practical_ml_lab", "cmd": "python learning_apps/practical_ml_lab/app.py",
     "desc": "Hands-On ML (Géron): features, tuning, ensembles, production."},
    {"name": "RL Lab", "port": 5009, "path": "rl_lab", "cmd": "python learning_apps/rl_lab/app.py",
     "desc": "Sutton & Barto: MDPs, TD, Q-learning, policy gradient."},
    {"name": "Probabilistic ML Lab", "port": 5010, "path": "probabilistic_ml_lab", "cmd": "python learning_apps/probabilistic_ml_lab/app.py",
     "desc": "Murphy: graphical models, EM, variational inference, Bayesian."},
    {"name": "ML Theory Lab", "port": 5011, "path": "ml_theory_lab", "cmd": "python learning_apps/ml_theory_lab/app.py",
     "desc": "Shalev-Shwartz & Ben-David: PAC, VC dimension, generalization."},
    {"name": "LLM Engineers Lab", "port": 5012, "path": "llm_engineers_lab", "cmd": "python learning_apps/llm_engineers_lab/app.py",
     "desc": "Handbook + Build Your Own LLM: RAG, prompts, eval, safety."},
    {"name": "Math for ML Lab", "port": 5013, "path": "math_for_ml_lab", "cmd": "python learning_apps/math_for_ml_lab/app.py",
     "desc": "Linear algebra, calculus, probability, optimization."},
]

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Learning Apps Hub</title>
  <style>
    *{margin:0;} body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;}
    .c{max-width:800px;margin:0 auto;}
    h1{font-size:1.6rem;margin-bottom:8px;} .sub{color:#94a3b8;margin-bottom:24px;}
    .card{background:#1e293b;border-radius:12px;padding:16px;margin:12px 0;border:1px solid #334155;}
    .card h2{font-size:1.1rem;margin-bottom:6px;} .card p{color:#94a3b8;font-size:0.9rem;}
    a{color:#60a5fa;text-decoration:none;} a:hover{text-decoration:underline;}
    .port{color:#64748b;font-size:0.85rem;} code{background:#0f172a;padding:2px 6px;border-radius:4px;font-size:0.85rem;}
  </style>
</head>
<body>
  <div class="c">
    <h1>Learning Apps Hub</h1>
    <p class="sub">Start a lab below, then open its link. Run each from repo root.</p>
    {% for lab in labs %}
    <div class="card">
      <h2><a href="http://127.0.0.1:{{ lab.port }}" target="_blank" rel="noopener">{{ lab.name }}</a> <span class="port">:{{ lab.port }}</span></h2>
      <p>{{ lab.desc }}</p>
      <p style="margin-top:8px;"><code>{{ lab.cmd }}</code></p>
    </div>
    {% endfor %}
  </div>
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
