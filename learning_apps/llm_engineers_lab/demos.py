"""Demos for LLM Engineers Lab. Uses ml_toolbox.llm_engineering (run from repo root)."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def llm_rag():
    from ml_toolbox.llm_engineering import KnowledgeRetriever
    kr = KnowledgeRetriever()
    kr.add_document("doc1", "Machine learning uses data to train models.")
    kr.add_document("doc2", "Large language models use transformers and attention.")
    results = kr.retrieve("transformer model", top_k=2)
    out = "RAG retrieve('transformer model', top_k=2):\n"
    for r in results:
        out += f"  - {r.get('content', r.get('doc_id', ''))[:60]}...\n"
    return {"ok": True, "output": out.strip()}


def llm_prompt():
    from ml_toolbox.llm_engineering import PromptTemplate, PromptEngineer
    t = PromptTemplate("Summarize in one sentence: {text}")
    formatted = t.format(text="Machine learning is a subset of AI that learns from data.")
    pe = PromptEngineer()
    role_prompt = pe.add_role("Explain RAG.", "teacher")
    out = f"PromptTemplate: {formatted}\n\nPromptEngineer.add_role('Explain RAG.', 'teacher'): {role_prompt[:80]}..."
    return {"ok": True, "output": out}


DEMO_HANDLERS = {"llm_rag": llm_rag, "llm_prompt": llm_prompt}


def run_demo(demo_id: str):
    if demo_id not in DEMO_HANDLERS:
        return {"ok": False, "output": "", "error": f"Unknown demo: {demo_id}"}
    try:
        return DEMO_HANDLERS[demo_id]()
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}
