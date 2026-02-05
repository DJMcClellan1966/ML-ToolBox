# RAG Corpus for ML Toolbox

This directory contains **enlarged corpus content** for retrieval-augmented generation (RAG): concept summaries, guides, and reference material that can be indexed and retrieved when users ask about ML, preprocessing, or the toolbox.

## Purpose

- **Enlarge the corpus** beyond the codebase alone: add structured, educational text so RAG and the learning companion have more to retrieve.
- **No copyrighted book text**: all content here is original summaries and explanations suitable for indexing.
- **Chunk-friendly**: each file is organized with clear headings so chunking (by section or by paragraph) works well.

## What's Included

| File | Contents |
|------|----------|
| `ml_foundations.md` | Core ML concepts: bias-variance, overfitting, train/test, cross-validation, metrics |
| `information_theory_for_ml.md` | Entropy, mutual information, KL divergence, and how they are used in ML |
| `preprocessing_and_data_quality.md` | When to use which preprocessing, data quality, semantic dedup, safety |
| `algorithm_selection_guide.md` | Problem type → algorithm and preprocessing suggestions |
| `glossary.md` | Short definitions of common ML and toolbox terms |

## How to Use This Corpus

### 1. Index into your RAG

With `BetterKnowledgeRetriever` (or any RAG that accepts documents):

```python
from pathlib import Path
from better_rag_system import BetterKnowledgeRetriever

retriever = BetterKnowledgeRetriever()
corpus_dir = Path("corpus")

for md_file in corpus_dir.glob("*.md"):
    if md_file.name == "README.md":
        continue
    text = md_file.read_text(encoding="utf-8")
    # Optional: split into chunks by ## or ### headings for finer retrieval
    retriever.add_document(doc_id=md_file.stem, content=text, metadata={"source": str(md_file)})
```

For finer retrieval, split each file into sections (e.g. by `## ` or `### `) and add each section as a separate document with a stable `doc_id`.

### 2. Include Existing Repo Content

To **enlarge** further, also index:

- **Markdown docs** in the repo root and in `ml_toolbox/`, `ai/`, `quantum_kernel/`: e.g. `COMPARTMENT1_DATA_GUIDE.md`, `ml_toolbox/README.md`, `USE_CASES.md`, `WHAT_CAN_I_DO_WITH_THIS.md`.
- **Docstrings** from `ml_toolbox/textbook_concepts/*.py`: extract module and function docstrings as short documents (e.g. "Information theory: entropy, mutual information, ...").

A script can walk the repo, read `.md` files and selected `.py` docstrings, and call `retriever.add_document()` or `retriever.batch_add_documents()`.

### 3. Adding More Content

- Add new `.md` files under `corpus/` with clear headings.
- Keep paragraphs focused so that a single chunk (e.g. one section) answers one kind of question.
- Avoid copying copyrighted book text; write original summaries and definitions.

## Chunking Tips

- **By heading**: Split on `## ` or `### ` and give each chunk a `doc_id` like `ml_foundations#bias-variance`.
- **By paragraph**: For long sections, split into paragraphs and keep a `source` metadata (e.g. file + section title) so citations work.
- **Overlap**: Optional 1–2 sentence overlap between chunks can improve retrieval at boundaries.

Enlarging the corpus improves RAG answer quality by giving the retriever more relevant, on-topic text to return.
