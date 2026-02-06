"""
ML Compass - Decide. Understand. Build. Think.

Combined Top 4: Oracle + Explainers + Theory-as-Channel + Socratic.
+ RAG enhancements, pattern guidance, theory corpus.
"""
from .oracle import suggest as oracle_suggest, suggest_from_description
from .explainers import explain_concept
from .theory_channel import correct_predictions, channel_capacity_bits, recommend_redundancy
from .socratic import debate_and_question
from .pattern_guidance import get_avoid_encourage, get_rules_for_nl_oracle
from .theory_corpus import get_corpus, get_default_corpus

__all__ = [
    "oracle_suggest",
    "suggest_from_description",
    "explain_concept",
    "correct_predictions",
    "channel_capacity_bits",
    "recommend_redundancy",
    "debate_and_question",
    "get_avoid_encourage",
    "get_rules_for_nl_oracle",
    "get_corpus",
    "get_default_corpus",
]
