"""
Cross-domain explainers: one concept, multiple views (e.g. entropy: Shannon, Boltzmann, ML).
"""
from typing import Dict, Any

EXPLAINERS: Dict[str, Dict[str, str]] = {
    "entropy": {
        "information_theory": "Shannon entropy H = -sum p log p measures uncertainty in bits. Higher H means more randomness.",
        "thermodynamics": "Boltzmann linked entropy to disorder. In statistical mechanics, entropy counts microstates; same logarithmic form.",
        "ml": "In ML: decision tree splitting (information gain), softmax loss, regularization to encourage smooth distributions.",
    },
    "bias_variance": {
        "theory": "Bias is error from model being too simple; variance from being too sensitive to the training set. Tradeoff between underfitting and overfitting.",
        "practice": "Use validation set and cross-validation; regularize (L1/L2) or use ensemble to reduce variance.",
        "ml": "Bishop Ch. 3: balance capacity and regularization; more data often reduces variance.",
    },
    "capacity": {
        "information_theory": "Shannon channel capacity C = B log2(1 + S/N) bounds reliable transmission over a noisy channel.",
        "ml": "Model capacity: how complex a function the model can represent. Too high -> overfitting; too low -> underfitting.",
        "practice": "For teacher-student: capacity bounds how much knowledge you can compress into a smaller model.",
    },
}


def explain_concept(concept: str) -> Dict[str, Any]:
    """
    Return multiple views (e.g. information theory, physics, ML) for a concept.

    Args:
        concept: e.g. "entropy", "bias_variance", "capacity"

    Returns:
        {"ok": bool, "concept": str, "views": dict or None}
    """
    key = concept.lower().replace(" ", "_").replace("-", "_")
    if key in EXPLAINERS:
        return {"ok": True, "concept": concept, "views": EXPLAINERS[key]}
    return {"ok": False, "concept": concept, "views": None, "available": list(EXPLAINERS.keys())}
