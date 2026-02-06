"""
Curriculum: LLM Engineers Handbook + Build Your Own LLM.
- Handbook: RAG, prompt engineering, chain-of-thought, few-shot, evaluation, safety, optimization (ml_toolbox.llm_engineering).
- Build your own LLM: architecture, tokenization, training/finetuning, scaling, inference, RAG apps.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "handbook_rag", "name": "RAG & Retrieval", "short": "RAG", "color": "#2563eb"},
    {"id": "handbook_prompts", "name": "Prompt Engineering", "short": "Prompts", "color": "#059669"},
    {"id": "handbook_eval_safety", "name": "Evaluation & Safety", "short": "Eval & Safety", "color": "#7c3aed"},
    {"id": "build_llm", "name": "Build Your Own LLM", "short": "Build LLM", "color": "#dc2626"},
]

CURRICULUM: List[Dict[str, Any]] = [
    {"id": "rag_retriever", "book_id": "handbook_rag", "level": "basics", "title": "RAG & Knowledge Retrieval",
     "learn": "Retrieval Augmented Generation: retrieve relevant docs, augment prompt, then generate. KnowledgeRetriever, RAGSystem. Query expansion + reranking improve quality (LLM-Engineers-Handbook).",
     "try_code": "from ml_toolbox.llm_engineering import RAGSystem, KnowledgeRetriever\nkr = KnowledgeRetriever()\nkr.add_document('d1', 'ML is useful.')\nprint(kr.retrieve('machine learning', top_k=2))",
     "try_demo": "llm_rag"},
    {"id": "rag_augment", "book_id": "handbook_rag", "level": "intermediate", "title": "Context Augmentation",
     "learn": "RAGSystem.augment_prompt(prompt, query, top_k): inject retrieved chunks into the prompt so the LLM answers with context.",
     "try_code": "from ml_toolbox.llm_engineering import RAGSystem\nrag = RAGSystem()\nrag.add_knowledge('k1', 'Transformers use attention.')\nprint(rag.augment_prompt('Explain:', 'attention', top_k=1)[:200])",
     "try_demo": None},
    {"id": "prompt_templates", "book_id": "handbook_prompts", "level": "basics", "title": "Prompt Templates",
     "learn": "PromptTemplate with {variable} placeholders. PromptEngineer, create_prompt(task_type, ...). Best practices from LLM Engineer's Handbook.",
     "try_code": "from ml_toolbox.llm_engineering import PromptTemplate\nt = PromptTemplate('Summarize: {text}')\nprint(t.format(text='Hello world'))",
     "try_demo": "llm_prompt"},
    {"id": "few_shot", "book_id": "handbook_prompts", "level": "intermediate", "title": "Few-Shot Learning",
     "learn": "Add 1–5 examples in the prompt so the model mimics the format. FewShotLearner, create_few_shot_prompt.",
     "try_code": "from ml_toolbox.llm_engineering import FewShotLearner\nlearner = FewShotLearner()\n# learner.add_example(...); learner.create_prompt(...)",
     "try_demo": None},
    {"id": "chain_of_thought", "book_id": "handbook_prompts", "level": "intermediate", "title": "Chain-of-Thought Prompting",
     "learn": "Ask the model to reason step by step. ChainOfThoughtReasoner, create_chain_of_thought_prompt. Improves reasoning tasks.",
     "try_code": "from ml_toolbox.llm_engineering import ChainOfThoughtReasoner, PromptEngineer\npe = PromptEngineer()\nprint(pe.create_chain_of_thought_prompt('Solve 2+3', ['Step 1: add', 'Step 2: answer'])[:150])",
     "try_demo": None},
    {"id": "llm_eval", "book_id": "handbook_eval_safety", "level": "intermediate", "title": "LLM Evaluation",
     "learn": "LLMEvaluator: relevance, completeness, clarity, structure. evaluate_response(prompt, response, expected).",
     "try_code": "from ml_toolbox.llm_engineering import LLMEvaluator\nev = LLMEvaluator()\nprint(ev.evaluate_response('What is 2+2?', '4', expected='4'))",
     "try_demo": None},
    {"id": "safety_guardrails", "book_id": "handbook_eval_safety", "level": "intermediate", "title": "Safety Guardrails",
     "learn": "SafetyGuardrails: check_prompt, check_response, sanitize_prompt, filter_response. Block harmful or PII.",
     "try_code": "from ml_toolbox.llm_engineering import SafetyGuardrails\nsg = SafetyGuardrails()\nprint(sg.check_prompt('Explain sorting.'))",
     "try_demo": None},
    {"id": "llm_optimizer", "book_id": "handbook_eval_safety", "level": "advanced", "title": "LLM Optimization",
     "learn": "LLMOptimizer: optimize_prompt_length, estimate_tokens, optimize_for_cost, cache_key, track_usage.",
     "try_code": "from ml_toolbox.llm_engineering import LLMOptimizer\nopt = LLMOptimizer()\nprint(opt.estimate_tokens('Hello world'))",
     "try_demo": None},
    {"id": "build_arch", "book_id": "build_llm", "level": "intermediate", "title": "Transformer Architecture",
     "learn": "Build your own LLM: self-attention, multi-head attention, FFN, layer norm, decoder stack. Token in → logits out.",
     "try_code": "# Conceptual; implementation via PyTorch/JAX or Hugging Face. See ml_toolbox.llm_engineering for app-level patterns.",
     "try_demo": None},
    {"id": "build_tokenize", "book_id": "build_llm", "level": "basics", "title": "Tokenization",
     "learn": "BPE, WordPiece, SentencePiece. Vocabulary size, subword units. Tokenizers determine context length and cost.",
     "try_code": "# Use tokenizers lib or Hugging Face; LLMOptimizer.estimate_tokens() for rough counts.",
     "try_demo": None},
    {"id": "build_train", "book_id": "build_llm", "level": "advanced", "title": "Pretraining & Finetuning",
     "learn": "Pretraining: next-token prediction on large corpus. Finetuning: supervised on tasks (SFT, RLHF). LoRA/QLoRA for efficiency.",
     "try_code": "# Training pipelines are external (e.g. Hugging Face, Axolotl). Handbook covers prompt/rag/eval.",
     "try_demo": None},
    {"id": "build_rag_app", "book_id": "build_llm", "level": "intermediate", "title": "Building an LLM App with RAG",
     "learn": "End-to-end: Prompt → RAG → Generation → Evaluation → Deployment. PromptRAGDeployPipeline in ml_toolbox.agent_pipelines.",
     "try_code": "from ml_toolbox.agent_pipelines.prompt_rag_deploy import PromptRAGDeployPipeline, PipelineStage",
     "try_demo": None},
]


def get_curriculum(): return list(CURRICULUM)
def get_books(): return list(BOOKS)
def get_levels(): return list(LEVELS)
def get_by_book(book_id: str): return [c for c in CURRICULUM if c["book_id"] == book_id]
def get_by_level(level: str): return [c for c in CURRICULUM if c["level"] == level]
def get_item(item_id: str):
    for c in CURRICULUM:
        if c["id"] == item_id: return c
    return None
