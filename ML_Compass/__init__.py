"""
ML Compass - Decide. Understand. Build. Think.

Combined Top 4: Oracle + Explainers + Theory-as-Channel + Socratic.
"""
from .oracle import suggest as oracle_suggest
from .explainers import explain_concept
from .theory_channel import correct_predictions, channel_capacity_bits, recommend_redundancy
from .socratic import debate_and_question

__all__ = [
    "oracle_suggest",
    "explain_concept",
    "correct_predictions",
    "channel_capacity_bits",
    "recommend_redundancy",
    "debate_and_question",
]
