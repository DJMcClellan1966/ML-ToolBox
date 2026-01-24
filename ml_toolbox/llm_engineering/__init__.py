"""
LLM Engineering Module - Best Practices from LLM Engineer's Handbook

Implements:
- Prompt Engineering
- RAG (Retrieval Augmented Generation)
- Fine-tuning Support
- Optimization
- Evaluation
- Safety & Guardrails
- Chain-of-Thought Reasoning
- Few-Shot Learning
"""
from .prompt_engineering import PromptEngineer, PromptTemplate
from .rag_system import RAGSystem, KnowledgeRetriever
from .llm_optimizer import LLMOptimizer
from .llm_evaluator import LLMEvaluator
from .safety_guardrails import SafetyGuardrails
from .chain_of_thought import ChainOfThoughtReasoner
from .few_shot_learning import FewShotLearner

__all__ = [
    'PromptEngineer',
    'PromptTemplate',
    'RAGSystem',
    'KnowledgeRetriever',
    'LLMOptimizer',
    'LLMEvaluator',
    'SafetyGuardrails',
    'ChainOfThoughtReasoner',
    'FewShotLearner'
]
