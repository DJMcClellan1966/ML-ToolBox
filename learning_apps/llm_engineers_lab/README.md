# LLM Engineers Lab

**Machine Learning / LLM Engineers Handbook** (RAG, prompt engineering, evaluation, safety) + **Build Your Own LLM** (architecture, tokenization, training, scaling, LLM apps).

- **Handbook**: Uses `ml_toolbox.llm_engineering` (RAGSystem, KnowledgeRetriever, PromptEngineer, PromptTemplate, FewShotLearner, ChainOfThoughtReasoner, LLMEvaluator, SafetyGuardrails, LLMOptimizer). Aligned with LLM-Engineers-Handbook ideas (query expansion, reranking, RAG pipeline).
- **Build your own LLM**: Curriculum on transformers, tokenization, pretraining/finetuning, scaling, and building LLM apps (e.g. PromptRAGDeployPipeline in `ml_toolbox.agent_pipelines`).

## Run (from repo root)

```bash
python learning_apps/llm_engineers_lab/app.py
```

Open **http://127.0.0.1:5012**

## Contents

- **RAG & Retrieval**: KnowledgeRetriever, RAGSystem, context augmentation
- **Prompt Engineering**: templates, few-shot, chain-of-thought
- **Evaluation & Safety**: LLMEvaluator, SafetyGuardrails, LLMOptimizer
- **Build Your Own LLM**: transformer architecture, tokenization, training/finetuning, RAG apps

Learn by book or level. Demos: RAG retrieve, prompt template/engineer.
