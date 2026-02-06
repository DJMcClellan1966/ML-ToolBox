"""
RAG enhancements: query expansion and cross-encoder reranking.
Inspired by LLM-Twin/LLM-Engineers-Handbook. Optional: requires sentence_transformers for reranking.
"""
from typing import Dict, Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Query expansion (template-based; no LLM required)
# ---------------------------------------------------------------------------


def expand_queries_concept(concept: str, n: int = 3) -> List[str]:
    """Generate multiple query phrasings for concept retrieval. Overcomes single-query similarity limits."""
    concept_clean = concept.replace("_", " ").strip()
    templates = [
        f"Explain {concept_clean} in machine learning and related theory.",
        f"What is {concept_clean}? Definition and role in ML.",
        f"{concept_clean} in information theory, statistics, and machine learning.",
    ]
    return list(dict.fromkeys(templates[:n]))  # dedup, preserve order


def expand_queries_statement(statement: str, n: int = 3) -> List[str]:
    """Generate multiple query phrasings for debate viewpoint retrieval."""
    templates = [
        statement,
        f"Viewpoints and arguments about: {statement}",
        f"Discussion or critique of: {statement}",
    ]
    return templates[:n]


# ---------------------------------------------------------------------------
# Reranker (optional: sentence_transformers CrossEncoder)
# ---------------------------------------------------------------------------

_RERANKER = None


def _get_reranker():
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    try:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        return _RERANKER
    except Exception:
        return None


def rerank(
    query: str,
    chunks: List[Dict[str, Any]],
    content_key: str = "content",
    keep_top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Rerank chunks by relevance to query using a cross-encoder.
    chunks: list of dicts with at least content_key (e.g. "content").
    Returns same dicts in reranked order.
    """
    model = _get_reranker()
    if model is None or not chunks:
        return chunks[:keep_top_k]
    pairs = [(query, c.get(content_key, "")) for c in chunks]
    scores = model.predict(pairs)
    scored = list(zip(scores, chunks, strict=False))
    scored.sort(key=lambda x: float(x[0]), reverse=True)
    return [c for _, c in scored[:keep_top_k]]


def is_reranker_available() -> bool:
    return _get_reranker() is not None


# ---------------------------------------------------------------------------
# Search + rerank using a kernel that has find_similar(query, candidates, top_k)
# ---------------------------------------------------------------------------

def search_and_rerank_concept(
    concept: str,
    corpus_chunks: List[Dict[str, str]],
    top_k: int,
    kernel: Any,
    expand_n: int = 3,
    use_rerank: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Multi-query retrieval then rerank for concept explanation.
    kernel must have: find_similar(query: str, candidates: List[str], top_k: int) -> List[Tuple[str, float]] or list of dicts with text/score.
    """
    if not corpus_chunks:
        return None
    candidates = [c["content"] for c in corpus_chunks]
    content_to_chunk = {c["content"]: c for c in corpus_chunks}
    queries = expand_queries_concept(concept, n=expand_n)
    collected: List[Tuple[Dict[str, str], float]] = []  # (chunk, score)
    seen_contents = set()
    for q in queries:
        try:
            similar = kernel.find_similar(q, candidates, top_k=top_k * 2)
        except Exception:
            continue
        for s in similar:
            text = s[0] if isinstance(s, (list, tuple)) else s.get("text", "")
            score = s[1] if isinstance(s, (list, tuple)) and len(s) > 1 else s.get("score", 0.0)
            if text not in seen_contents and text in content_to_chunk:
                seen_contents.add(text)
                collected.append((content_to_chunk[text], float(score)))
    if not collected:
        return None
    # Dedupe by chunk id or content; keep order by max score per chunk
    by_content: Dict[str, Tuple[Dict[str, str], float]] = {}
    for chunk, score in collected:
        key = chunk.get("id", chunk.get("content", ""))
        if key not in by_content or by_content[key][1] < score:
            by_content[key] = (chunk, score)
    chunks_only = [t[0] for t in by_content.values()]
    if use_rerank and len(chunks_only) > 1:
        main_query = f"Explain {concept.replace('_', ' ')} in machine learning and related theory."
        chunks_only = rerank(main_query, chunks_only, content_key="content", keep_top_k=top_k)
    else:
        chunks_only = chunks_only[:top_k]
    retrieved = [c["content"] for c in chunks_only]
    return {"ok": True, "concept": concept, "retrieved": retrieved, "source": "rag_expanded_reranked"}


def search_and_rerank_debate(
    statement: str,
    viewpoint_chunks: List[Dict[str, str]],
    top_k: int,
    kernel: Any,
    expand_n: int = 3,
    use_rerank: bool = True,
) -> Optional[Dict[str, Any]]:
    """Multi-query retrieval then rerank for debate viewpoints."""
    if not viewpoint_chunks:
        return None
    candidates = [c["content"] for c in viewpoint_chunks]
    content_to_chunk = {c["content"]: c for c in viewpoint_chunks}
    queries = expand_queries_statement(statement, n=expand_n)
    collected: List[Tuple[Dict[str, str], float]] = []
    seen = set()
    for q in queries:
        try:
            similar = kernel.find_similar(q, candidates, top_k=top_k * 2)
        except Exception:
            continue
        for s in similar:
            text = s[0] if isinstance(s, (list, tuple)) else s.get("text", "")
            score = s[1] if isinstance(s, (list, tuple)) and len(s) > 1 else s.get("score", 0.0)
            if text not in seen and text in content_to_chunk:
                seen.add(text)
                collected.append((content_to_chunk[text], float(score)))
    if not collected:
        return None
    by_content: Dict[str, Tuple[Dict[str, str], float]] = {}
    for chunk, score in collected:
        key = chunk.get("content", "")
        if key not in by_content or by_content[key][1] < score:
            by_content[key] = (chunk, score)
    chunks_only = [t[0] for t in by_content.values()]
    if use_rerank and len(chunks_only) > 1:
        chunks_only = rerank(statement, chunks_only, content_key="content", keep_top_k=top_k)
    else:
        chunks_only = chunks_only[:top_k]
    views = [{"viewpoint": c.get("viewpoint", "unknown"), "content": c["content"]} for c in chunks_only]
    return {"ok": True, "views": views, "source": "retrieved_expanded_reranked"}
