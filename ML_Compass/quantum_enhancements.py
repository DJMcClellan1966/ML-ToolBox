"""
Optional: use quantum_kernel to improve Oracle, Explainers, and Socratic.
ML Compass works without this; import and use when quantum_kernel is available.
Uses rag_enhancements for query expansion + reranking when available.
"""
from typing import Dict, Any, List, Optional

_KERNEL = None


def _get_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    try:
        from quantum_kernel import get_kernel
        _KERNEL = get_kernel()
        return _KERNEL
    except ImportError:
        return None


def _rerank_rules(description: str, rules_with_descriptions: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    """Optionally rerank top rule matches with cross-encoder. Returns up to top_n rules in order."""
    try:
        from .rag_enhancements import rerank
        chunk_like = [{"content": r["description"], "rule": r} for r in rules_with_descriptions]
        reranked = rerank(description, chunk_like, content_key="content", keep_top_k=min(top_n, len(chunk_like)))
        return [c["rule"] for c in reranked]
    except Exception:
        return rules_with_descriptions[:top_n]


def oracle_suggest_nl(description: str, rules_with_descriptions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Oracle path using natural language: embed description, find best-matching rule by similarity.
    Uses query expansion + optional rerank when rag_enhancements available.
    rules_with_descriptions: list of {"description": str, "pattern": str, "suggestion": str, "why": str}
    """
    kernel = _get_kernel()
    if kernel is None or not rules_with_descriptions:
        return None
    candidates = [r["description"] for r in rules_with_descriptions]
    try:
        # Get top 3 by similarity then rerank to pick best
        similar = kernel.find_similar(description, candidates, top_k=min(3, len(candidates)))
        if not similar:
            return None
        best_text = similar[0][0] if isinstance(similar[0], (list, tuple)) else similar[0].get("text", candidates[0])
        idx = next((i for i, c in enumerate(candidates) if c == best_text), 0)
        r = rules_with_descriptions[idx]
        # Optional: rerank top matches
        top_rules = [rules_with_descriptions[next((i for i, c in enumerate(candidates) if c == (s[0] if isinstance(s, (list, tuple)) else s.get("text", ""))), 0)] for s in similar]
        top_rules = _rerank_rules(description, top_rules, top_n=1)
        if top_rules:
            r = top_rules[0]
        return {"ok": True, "pattern": r["pattern"], "suggestion": r["suggestion"], "why": r["why"], "source": "nl_match"}
    except Exception:
        return None


def explain_concept_rag(concept: str, corpus_chunks: List[Dict[str, str]], top_k: int = 3) -> Optional[Dict[str, Any]]:
    """
    Explain concept by retrieving relevant chunks from corpus. Uses query expansion + rerank when kernel available.
    corpus_chunks: [{"id": str, "content": str}, ...]
    """
    kernel = _get_kernel()
    if kernel is None or not corpus_chunks:
        return None
    try:
        from .rag_enhancements import search_and_rerank_concept
        out = search_and_rerank_concept(concept, corpus_chunks, top_k, kernel, use_rerank=True)
        if out is not None:
            return out
    except Exception:
        pass
    # Fallback: single-query retrieval
    query = f"Explain {concept} in machine learning and related theory."
    candidates = [c["content"] for c in corpus_chunks]
    try:
        similar = kernel.find_similar(query, candidates, top_k=top_k)
        if not similar:
            return None
        retrieved = [s[0] if isinstance(s, (list, tuple)) else getattr(s, "text", s) for s in similar]
        return {"ok": True, "concept": concept, "retrieved": retrieved, "source": "rag"}
    except Exception:
        return None


def debate_retrieve(statement: str, viewpoint_chunks: List[Dict[str, str]], top_k: int = 3) -> Optional[Dict[str, Any]]:
    """
    Retrieve debate viewpoints relevant to the user statement. Uses query expansion + rerank when kernel available.
    viewpoint_chunks: [{"content": str, "viewpoint": str}, ...]
    """
    kernel = _get_kernel()
    if kernel is None or not viewpoint_chunks:
        return None
    try:
        from .rag_enhancements import search_and_rerank_debate
        out = search_and_rerank_debate(statement, viewpoint_chunks, top_k, kernel, use_rerank=True)
        if out is not None:
            return out
    except Exception:
        pass
    candidates = [c["content"] for c in viewpoint_chunks]
    try:
        similar = kernel.find_similar(statement, candidates, top_k=top_k)
        if not similar:
            return None
        views = []
        for s in similar:
            text = s[0] if isinstance(s, (list, tuple)) else getattr(s, "text", s)
            idx = next((i for i, c in enumerate(candidates) if c == text), 0)
            views.append({"viewpoint": viewpoint_chunks[idx].get("viewpoint", "unknown"), "content": text})
        return {"ok": True, "views": views, "source": "retrieved"}
    except Exception:
        return None


def is_available() -> bool:
    """Return True if quantum_kernel can be used."""
    return _get_kernel() is not None
