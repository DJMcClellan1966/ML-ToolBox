"""
Socratic Multi-Viewpoint: debate snippet + follow-up question.
"""
from typing import Dict, Any, Optional

try:
    from ml_toolbox.agent_enhancements.socratic_method import SocraticQuestioner
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


def debate_and_question(
    statement: str,
    question_type: Optional[str] = None,
    viewpoints: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Return a short debate (theory vs practice) and a Socratic follow-up question.

    Args:
        statement: User statement to question (e.g. "I used an ensemble.")
        question_type: Optional - clarification, assumption, evidence, implication, alternative
        viewpoints: Optional dict of view_name -> text; default is theory + practice.

    Returns:
        {"ok": bool, "debate": str, "question": str, "has_question": bool}
    """
    if not _AVAILABLE:
        return {
            "ok": False,
            "debate": "SocraticQuestioner not available.",
            "question": "",
            "has_question": False,
        }
    q = SocraticQuestioner()
    question = q.generate_question(statement, question_type=question_type)
    if viewpoints is None:
        viewpoints = {
            "theory": "Ensembles reduce variance (Bishop).",
            "practice": "Start with 3 models, add more if needed.",
        }
    debate_lines = [f"{k}: {v}" for k, v in viewpoints.items()]
    debate_lines.append(f"Question: {question}")
    debate = " ".join(debate_lines)
    return {
        "ok": True,
        "debate": debate,
        "question": question,
        "has_question": "?" in question,
    }
