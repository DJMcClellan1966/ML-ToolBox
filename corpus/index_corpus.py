"""
Index the corpus directory (and optionally other docs) into BetterKnowledgeRetriever.

Run from the repo root:
  python corpus/index_corpus.py

Or from anywhere with the repo on the path:
  python -c "import sys; sys.path.insert(0, 'path/to/ML-ToolBox'); from corpus.index_corpus import build_corpus_retriever; r = build_corpus_retriever(); print(r.get_stats())"
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional

# Allow running from repo root or from corpus/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CORPUS_DIR = Path(__file__).resolve().parent


def _chunk_by_heading(content: str, source_id: str) -> List[Dict]:
    """Split markdown into chunks by ## or ### headings for finer retrieval."""
    chunks = []
    current_title = ""
    current_body = []
    for line in content.splitlines():
        if line.startswith("## ") or line.startswith("### "):
            if current_body:
                chunks.append({
                    "id": f"{source_id}#{current_title[:50].replace(' ', '-').lower()}" if current_title else source_id,
                    "content": "\n".join(current_body).strip(),
                    "metadata": {"source": source_id, "section": current_title}
                })
            current_title = line.lstrip("# ").strip()
            current_body = [line]
        else:
            current_body.append(line)
    if current_body:
        chunks.append({
            "id": f"{source_id}#{current_title[:50].replace(' ', '-').lower()}" if current_title else source_id,
            "content": "\n".join(current_body).strip(),
            "metadata": {"source": source_id, "section": current_title}
        })
    return chunks if chunks else [{"id": source_id, "content": content.strip(), "metadata": {"source": source_id}}]


def build_corpus_retriever(
    corpus_dir: Optional[Path] = None,
    chunk_by_heading: bool = True,
    include_readme: bool = False,
    model_name: str = "all-MiniLM-L6-v2",
):
    """
    Build a BetterKnowledgeRetriever and fill it with corpus documents.

    Args:
        corpus_dir: Directory containing .md files (default: corpus/ next to this script).
        chunk_by_heading: If True, split each file by ## / ### for finer retrieval.
        include_readme: If True, include README.md in the corpus.
        model_name: Sentence transformer model name.

    Returns:
        BetterKnowledgeRetriever instance with corpus loaded.
    """
    try:
        from better_rag_system import BetterKnowledgeRetriever
    except ImportError:
        from ml_toolbox.llm_engineering.rag_system import KnowledgeRetriever
        # Fallback: return a simple retriever; caller can still add_document manually
        return KnowledgeRetriever()

    corpus_dir = corpus_dir or CORPUS_DIR
    retriever = BetterKnowledgeRetriever(model_name=model_name)
    documents_to_add = []

    for md_file in sorted(corpus_dir.glob("*.md")):
        if md_file.name == "README.md" and not include_readme:
            continue
        text = md_file.read_text(encoding="utf-8", errors="replace")
        source_id = md_file.stem
        if chunk_by_heading:
            for chunk in _chunk_by_heading(text, source_id):
                if chunk["content"]:
                    documents_to_add.append({
                        "id": chunk["id"],
                        "content": chunk["content"],
                        "metadata": chunk.get("metadata", {"source": source_id}),
                    })
        else:
            documents_to_add.append({
                "id": source_id,
                "content": text,
                "metadata": {"source": source_id},
            })

    if documents_to_add:
        retriever.batch_add_documents(documents_to_add)
    return retriever


if __name__ == "__main__":
    r = build_corpus_retriever()
    stats = r.get_stats() if hasattr(r, "get_stats") else {"documents": len(getattr(r, "documents", []))}
    print("Corpus index built.", stats)
