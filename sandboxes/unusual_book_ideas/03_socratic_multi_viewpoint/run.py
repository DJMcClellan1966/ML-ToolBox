"""Sandbox 03: Socratic Multi-Viewpoint - viability test."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def run():
    from ml_toolbox.agent_enhancements.socratic_method import SocraticQuestioner
    # Simulated "viewpoints" (no RAG needed for viability)
    viewpoints = {
        "theory": "Overfitting is a consequence of high model capacity relative to data. Bias-variance tradeoff explains it.",
        "practice": "In practice, use early stopping and validation set. Regularization and more data help.",
    }
    statement = "My model overfits."
    q = SocraticQuestioner()
    question = q.generate_question(statement, question_type="evidence")
    # Assemble "debate": two views + one Socratic question
    debate = f"View 1 (theory): {viewpoints['theory']}\nView 2 (practice): {viewpoints['practice']}\nQuestion: {question}"
    has_question = "?" in question
    return {
        "ok": True,
        "debate_length": len(debate),
        "has_socratic_question": has_question,
        "question_preview": question[:60] + "..." if len(question) > 60 else question,
    }

if __name__ == "__main__":
    try:
        result = run()
        print("PASS", result)
    except Exception as e:
        print("FAIL", str(e))
        sys.exit(1)
