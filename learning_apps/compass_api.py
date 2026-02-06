"""
Compass API for learning apps: explain, guidance, oracle, explain_rag, generate_code.
Labs call register_compass_routes(app) and include get_compass_html_snippet() in their HTML.
"""
import sys
from pathlib import Path

# Ensure repo root on path when this module is loaded (e.g. from learning_apps/xxx_lab/app.py)
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_COMPASS_AVAILABLE = False
_EXPLAIN_CONCEPT = None
_ORACLE_SUGGEST = None
_SUGGEST_FROM_DESC = None
_GET_AVOID_ENCOURAGE = None
_GET_CORPUS = None
_QUANTUM_ENHANCEMENTS = None
_AGENT_AVAILABLE = False
_ML_CODE_AGENT = None

try:
    from ML_Compass.explainers import explain_concept as _ec
    from ML_Compass.oracle import suggest as _os, suggest_from_description as _sfd
    from ML_Compass.pattern_guidance import get_avoid_encourage as _gae
    from ML_Compass.theory_corpus import get_corpus as _gc
    _EXPLAIN_CONCEPT = _ec
    _ORACLE_SUGGEST = _os
    _SUGGEST_FROM_DESC = _sfd
    _GET_AVOID_ENCOURAGE = _gae
    _GET_CORPUS = _gc
    _COMPASS_AVAILABLE = True
except Exception:
    pass

try:
    from ML_Compass import quantum_enhancements as _qe
    _QUANTUM_ENHANCEMENTS = _qe
except Exception:
    pass

try:
    from ml_toolbox.ai_agent import MLCodeAgent
    _ML_CODE_AGENT = MLCodeAgent
    _AGENT_AVAILABLE = True
except Exception:
    pass


def register_compass_routes(app):
    """Register /api/explain/<concept>, /api/oracle, /api/guidance, /api/explain_rag/<concept>, /api/generate_code on the given Flask app."""
    from flask import request, jsonify

    @app.route("/api/explain/<concept>")
    def api_explain(concept):
        if not _COMPASS_AVAILABLE:
            return jsonify({"ok": False, "error": "ML_Compass not available", "available": []}), 503
        return jsonify(_EXPLAIN_CONCEPT(concept))

    @app.route("/api/oracle", methods=["POST"])
    def api_oracle():
        if not _COMPASS_AVAILABLE:
            return jsonify({"ok": False, "error": "ML_Compass not available"}), 503
        body = request.get_json(silent=True) or {}
        if body.get("description"):
            return jsonify(_SUGGEST_FROM_DESC(body["description"]))
        return jsonify(_ORACLE_SUGGEST(body))

    @app.route("/api/guidance")
    def api_guidance():
        if not _COMPASS_AVAILABLE:
            return jsonify({"ok": False, "error": "ML_Compass not available"}), 503
        minimal = request.args.get("minimal", "false").lower() == "true"
        return jsonify(_GET_AVOID_ENCOURAGE(minimal=minimal))

    @app.route("/api/explain_rag/<concept>")
    def api_explain_rag(concept):
        if not _COMPASS_AVAILABLE:
            return jsonify({"ok": False, "error": "ML_Compass not available"}), 503
        if _QUANTUM_ENHANCEMENTS and hasattr(_QUANTUM_ENHANCEMENTS, "explain_concept_rag"):
            corpus = _GET_CORPUS()
            top_k = int(request.args.get("top_k", 3))
            out = _QUANTUM_ENHANCEMENTS.explain_concept_rag(concept, corpus, top_k=top_k)
            if out is not None:
                return jsonify({"ok": True, **out})
        return jsonify({"ok": False, "error": "RAG not available (quantum_kernel)", "concept": concept}), 503

    @app.route("/api/generate_code", methods=["POST"])
    def api_generate_code():
        if not _AGENT_AVAILABLE or _ML_CODE_AGENT is None:
            return jsonify({"ok": False, "error": "MLCodeAgent not available", "code": ""}), 503
        body = request.get_json(silent=True) or {}
        topic = body.get("topic", "").strip() or "simple linear regression"
        try:
            agent = _ML_CODE_AGENT()
            result = agent.build(topic)
            if result.get("success") and result.get("code"):
                return jsonify({"ok": True, "code": result["code"], "topic": topic})
            return jsonify({"ok": False, "error": result.get("error", "No code generated"), "code": ""}), 200
        except Exception as e:
            return jsonify({"ok": False, "error": str(e), "code": ""}), 200

    @app.route("/api/compass/status")
    def api_compass_status():
        return jsonify({
            "compass": _COMPASS_AVAILABLE,
            "rag": _QUANTUM_ENHANCEMENTS is not None and hasattr(_QUANTUM_ENHANCEMENTS, "explain_concept_rag"),
            "generate_code": _AGENT_AVAILABLE,
        })


def get_compass_html_snippet():
    """Return HTML + JS for the Compass panel (Explain concept, Guidance, Oracle, Generate code)."""
    return r"""
<div class="card" id="compass-panel" style="margin-top:20px;">
  <h2 style="font-size:1rem;margin-bottom:10px;">ML Compass</h2>
  <p style="color:#94a3b8;font-size:0.85rem;margin-bottom:10px;">Explain concept, get guidance, ask oracle, generate code.</p>
  <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;">
    <input type="text" id="compass-concept" placeholder="Concept (e.g. entropy)" style="padding:8px 12px;border-radius:8px;border:1px solid #475569;background:#1e293b;color:#e2e8f0;min-width:160px;">
    <button type="button" class="run" id="compass-explain-btn">Explain</button>
    <button type="button" class="run" id="compass-guidance-btn">Guidance</button>
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;">
    <input type="text" id="compass-oracle-input" placeholder="Problem (e.g. my model overfits)" style="padding:8px 12px;border-radius:8px;border:1px solid #475569;background:#1e293b;color:#e2e8f0;min-width:200px;">
    <button type="button" class="run" id="compass-oracle-btn">Ask Oracle</button>
  </div>
  <div style="margin-bottom:10px;">
    <input type="text" id="compass-generate-topic" placeholder="Topic for code (e.g. logistic regression)" style="padding:8px 12px;border-radius:8px;border:1px solid #475569;background:#1e293b;color:#e2e8f0;min-width:220px;">
    <button type="button" class="run" id="compass-generate-btn">Generate code</button>
  </div>
  <div id="compass-out" class="out" style="min-height:60px;"></div>
</div>
<script>
(function(){
  const api=(p,o={})=>fetch(p,{headers:{'Content-Type':'application/json'},...o}).then(r=>r.json());
  function showCompass(txt,err){const e=document.getElementById('compass-out');if(!e)return;e.textContent=txt||'';e.className='out '+(err?'err':'ok');}
  const explainBtn=document.getElementById('compass-explain-btn');
  if(explainBtn)explainBtn.onclick=async()=>{
    const concept=document.getElementById('compass-concept');const c=(concept&&concept.value)||'entropy';
    showCompass('Loading…');
    try{const d=await api('/api/explain/'+encodeURIComponent(c));if(d.ok){let s='';for(const [k,v] of Object.entries(d.views||{}))s+=k+': '+v+'\n';showCompass(s);}else showCompass('Unknown concept. Try: '+((d.available||[]).join(', ')),true);}catch(e){showCompass(e.message,true);}
  };
  const guidanceBtn=document.getElementById('compass-guidance-btn');
  if(guidanceBtn)guidanceBtn.onclick=async()=>{
    showCompass('Loading…');
    try{const d=await api('/api/guidance');if(d.ok){let s='Avoid: '+(d.avoid||[]).map(x=>x.pattern).join(', ')+'\nEncourage: '+(d.encourage||[]).map(x=>x.pattern).join(', ');showCompass(s);}else showCompass(d.error||'Error',true);}catch(e){showCompass(e.message,true);}
  };
  const oracleBtn=document.getElementById('compass-oracle-btn');
  if(oracleBtn)oracleBtn.onclick=async()=>{
    const inp=document.getElementById('compass-oracle-input');const desc=(inp&&inp.value)||'My model overfits';
    showCompass('Loading…');
    try{const d=await api('/api/oracle',{method:'POST',body:JSON.stringify({description:desc})});if(d.ok)showCompass('Pattern: '+d.pattern+'\nSuggestion: '+d.suggestion+'\nWhy: '+d.why);else showCompass(d.error||'Error',true);}catch(e){showCompass(e.message,true);}
  };
  const genBtn=document.getElementById('compass-generate-btn');
  if(genBtn)genBtn.onclick=async()=>{
    const inp=document.getElementById('compass-generate-topic');const topic=(inp&&inp.value)||'linear regression';
    showCompass('Generating…');
    try{const d=await api('/api/generate_code',{method:'POST',body:JSON.stringify({topic:topic})});if(d.ok)showCompass(d.code||'');else showCompass(d.error||'No code',true);}catch(e){showCompass(e.message,true);}
  };
})();
</script>
"""
