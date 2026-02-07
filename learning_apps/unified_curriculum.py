"""
Unified Curriculum Graph + Adaptive Tutor.

Builds a combined corpus index and concept graph from all lab curricula + corpus docs.
Provides:
- Vector search over all book content (local TF-IDF)
- Unified concept graph (nodes/edges)
- Adaptive learning path recommendations
- Cross-book synthesis summaries (LLM optional)
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import importlib
import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
LEARNING_APPS_DIR = Path(__file__).resolve().parent
CORPUS_DIR = REPO_ROOT / "corpus"
COMPASS_DIR = REPO_ROOT / "ML_Compass"
CACHE_DIR = LEARNING_APPS_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
INDEX_CACHE = CACHE_DIR / "unified_index.pkl"
GRAPH_CACHE = CACHE_DIR / "unified_graph.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------
# Utilities
# --------------------------------------------

STOPWORDS = {
    "the", "and", "or", "to", "of", "in", "a", "an", "for", "on", "with", "by",
    "from", "is", "are", "be", "as", "that", "this", "it", "at", "we", "you",
    "your", "our", "their", "not", "can", "will", "into", "via", "using"
}

LEVEL_ORDER_DEFAULT = ["basics", "intermediate", "advanced", "expert"]


def _llm_generate(prompt: str) -> str:
    if importlib.util.find_spec("ollama") is not None:
        try:
            ollama = importlib.import_module("ollama")
            response = ollama.chat(
                model=os.getenv("OLLAMA_MODEL", "llama3.2"),
                messages=[
                    {"role": "system", "content": "You are a concise, expert ML tutor."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.get("message", {}).get("content", "")
        except Exception:
            return ""
    if importlib.util.find_spec("openai") is not None and os.getenv("OPENAI_API_KEY"):
        try:
            openai_module = importlib.import_module("openai")
            OpenAI = getattr(openai_module, "OpenAI", None)
            if OpenAI is None:
                return ""
            client = OpenAI()
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a concise, expert ML tutor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return resp.choices[0].message.content or ""
        except Exception:
            return ""
    return ""


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return ""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def _chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    text = _normalize(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks


def _extract_title_from_md(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line.replace("# ", "").strip()
    return "Untitled"


# --------------------------------------------
# Curriculum Loading
# --------------------------------------------

@dataclass
class CurriculumItem:
    id: str
    title: str
    learn: str
    lab_id: str
    level: str
    book_id: str
    book_name: str


def _discover_curriculum_modules() -> List[Tuple[str, Path]]:
    modules = []
    for item in LEARNING_APPS_DIR.iterdir():
        if item.is_dir():
            curriculum = item / "curriculum.py"
            if curriculum.exists():
                modules.append((item.name, curriculum))
    return modules


def _load_curriculum_items() -> List[CurriculumItem]:
    items: List[CurriculumItem] = []
    for lab_id, module_path in _discover_curriculum_modules():
        try:
            module_name = f"learning_apps.{lab_id}.curriculum"
            module = __import__(module_name, fromlist=["*"])
        except Exception:
            continue

        curriculum = getattr(module, "CURRICULUM", []) or []
        books = {b.get("id"): b.get("name") for b in getattr(module, "BOOKS", [])}

        for c in curriculum:
            items.append(
                CurriculumItem(
                    id=str(c.get("id")),
                    title=str(c.get("title")),
                    learn=str(c.get("learn")),
                    lab_id=lab_id,
                    level=str(c.get("level")),
                    book_id=str(c.get("book_id")),
                    book_name=str(books.get(c.get("book_id"), c.get("book_id", "")))
                )
            )

    return items


# --------------------------------------------
# Corpus Index
# --------------------------------------------

@dataclass
class CorpusChunk:
    id: str
    text: str
    source: str
    title: str
    meta: Dict[str, Any]


def _collect_corpus_documents() -> List[Tuple[str, str, str]]:
    docs: List[Tuple[str, str, str]] = []
    for base in [CORPUS_DIR, COMPASS_DIR]:
        if not base.exists():
            continue
        for path in base.rglob("*.md"):
            text = _safe_read(path)
            if not text:
                continue
            title = _extract_title_from_md(text)
            docs.append((str(path), title, text))
    return docs


def _build_chunks(curriculum_items: List[CurriculumItem]) -> List[CorpusChunk]:
    chunks: List[CorpusChunk] = []

    # Curriculum items
    for item in curriculum_items:
        content = f"{item.title}\n\n{item.learn}"
        for i, chunk in enumerate(_chunk_text(content)):
            chunks.append(CorpusChunk(
                id=f"curriculum::{item.lab_id}::{item.id}::{i}",
                text=chunk,
                source=f"{item.lab_id}.curriculum",
                title=item.title,
                meta={
                    "lab_id": item.lab_id,
                    "level": item.level,
                    "book_id": item.book_id,
                    "book_name": item.book_name,
                    "topic_id": item.id,
                    "kind": "curriculum"
                }
            ))

    # Corpus docs
    for path, title, text in _collect_corpus_documents():
        for i, chunk in enumerate(_chunk_text(text)):
            chunks.append(CorpusChunk(
                id=f"corpus::{Path(path).name}::{i}",
                text=chunk,
                source=path,
                title=title,
                meta={"kind": "corpus"}
            ))

    return chunks


def build_or_load_index(force_rebuild: bool = False) -> Dict[str, Any]:
    if INDEX_CACHE.exists() and not force_rebuild:
        try:
            return joblib.load(INDEX_CACHE)
        except Exception:
            pass

    curriculum_items = _load_curriculum_items()
    chunks = _build_chunks(curriculum_items)
    texts = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(max_features=40000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts) if texts else None

    payload = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "chunks": chunks,
    }
    joblib.dump(payload, INDEX_CACHE)
    return payload


def search_corpus(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not query:
        return []

    payload = build_or_load_index()
    vectorizer = payload["vectorizer"]
    matrix = payload["matrix"]
    chunks = payload["chunks"]

    if matrix is None or vectorizer is None:
        return []

    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()
    best_idx = sims.argsort()[::-1][:top_k]

    results = []
    for idx in best_idx:
        chunk = chunks[idx]
        results.append({
            "id": chunk.id,
            "title": chunk.title,
            "source": chunk.source,
            "text": chunk.text,
            "score": round(float(sims[idx]), 5),
            "meta": chunk.meta
        })
    return results


# --------------------------------------------
# Graph Builder
# --------------------------------------------

def build_or_load_graph(force_rebuild: bool = False) -> Dict[str, Any]:
    if GRAPH_CACHE.exists() and not force_rebuild:
        try:
            return json.loads(GRAPH_CACHE.read_text(encoding="utf-8"))
        except Exception:
            pass

    items = _load_curriculum_items()
    nodes = []
    node_texts = []

    for item in items:
        node_id = f"{item.lab_id}::{item.id}"
        text = f"{item.title}. {item.learn}"
        node_texts.append(text)
        nodes.append({
            "id": node_id,
            "title": item.title,
            "lab_id": item.lab_id,
            "level": item.level,
            "book_id": item.book_id,
            "book_name": item.book_name,
            "keywords": _tokenize(text)[:12],
        })

    # Similarity edges
    edges: List[Dict[str, Any]] = []
    if node_texts:
        tfidf = TfidfVectorizer(max_features=20000)
        mat = tfidf.fit_transform(node_texts)
        sim = cosine_similarity(mat)
        threshold = 0.20
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if sim[i, j] >= threshold:
                    edges.append({
                        "source": nodes[i]["id"],
                        "target": nodes[j]["id"],
                        "type": "related",
                        "weight": round(float(sim[i, j]), 3)
                    })

    # Prerequisite edges based on levels within book
    level_rank = {lvl: i for i, lvl in enumerate(LEVEL_ORDER_DEFAULT)}
    items_by_book: Dict[str, List[CurriculumItem]] = {}
    for item in items:
        items_by_book.setdefault(f"{item.lab_id}::{item.book_id}", []).append(item)

    for _, book_items in items_by_book.items():
        book_items.sort(key=lambda x: level_rank.get(x.level, 99))
        for i in range(1, len(book_items)):
            prev = book_items[i - 1]
            curr = book_items[i]
            edges.append({
                "source": f"{prev.lab_id}::{prev.id}",
                "target": f"{curr.lab_id}::{curr.id}",
                "type": "prerequisite",
                "weight": 1.0
            })

    graph = {"nodes": nodes, "edges": edges}
    GRAPH_CACHE.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return graph


# --------------------------------------------
# Adaptive Tutor + Synthesis
# --------------------------------------------

def _get_progress_status_map(user_id: str) -> Dict[str, str]:
    try:
        from learning_apps.progress import get_user_progress
        data = get_user_progress(user_id)
        status_map: Dict[str, str] = {}
        for lab_id, lab_data in data.get("labs", {}).items():
            for topic_id, topic_data in lab_data.get("topics", {}).items():
                status_map[f"{lab_id}::{topic_id}"] = topic_data.get("status", "not-started")
        return status_map
    except Exception:
        return {}


def _prereq_satisfied(node_id: str, graph: Dict[str, Any], status_map: Dict[str, str]) -> bool:
    prereqs = [e for e in graph["edges"] if e["type"] == "prerequisite" and e["target"] == node_id]
    if not prereqs:
        return True
    for e in prereqs:
        if status_map.get(e["source"]) != "completed":
            return False
    return True


def recommend_path(user_id: str, goal_query: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
    graph = build_or_load_graph()
    status_map = _get_progress_status_map(user_id)

    # Relevance scores via corpus search
    relevance: Dict[str, float] = {}
    if goal_query:
        hits = search_corpus(goal_query, top_k=25)
        for h in hits:
            topic_id = h.get("meta", {}).get("topic_id")
            lab_id = h.get("meta", {}).get("lab_id")
            if topic_id and lab_id:
                key = f"{lab_id}::{topic_id}"
                relevance[key] = max(relevance.get(key, 0.0), h.get("score", 0.0))

    scored = []
    for node in graph["nodes"]:
        node_id = node["id"]
        status = status_map.get(node_id, "not-started")
        if status == "completed":
            continue
        prereq_ok = _prereq_satisfied(node_id, graph, status_map)
        base = 1.0 if prereq_ok else 0.0
        in_progress_bonus = 0.4 if status == "in-progress" else 0.0
        relevance_boost = relevance.get(node_id, 0.0) * 2.0
        score = base + in_progress_bonus + relevance_boost
        scored.append((score, node, status, prereq_ok))

    scored.sort(key=lambda x: x[0], reverse=True)
    recs = []
    for score, node, status, prereq_ok in scored[:limit]:
        recs.append({
            "id": node["id"],
            "title": node["title"],
            "lab_id": node["lab_id"],
            "level": node["level"],
            "book_name": node.get("book_name", ""),
            "status": status,
            "prerequisites_met": prereq_ok,
            "score": round(score, 4)
        })

    return {
        "ok": True,
        "user_id": user_id,
        "goal_query": goal_query,
        "recommendations": recs
    }


def _extractive_summary(texts: List[str], max_sentences: int = 6) -> str:
    sentences = []
    for t in texts:
        for s in re.split(r"(?<=[.!?])\s+", t.strip()):
            s = s.strip()
            if len(s) > 40:
                sentences.append(s)

    if not sentences:
        return ""

    # Simple TF-IDF scoring
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(sentences)
    scores = X.sum(axis=1).A1
    top_idx = scores.argsort()[::-1][:max_sentences]
    return " ".join(sentences[i] for i in top_idx)


def synthesize_answer(question: str, top_k: int = 6, use_llm: bool = True) -> Dict[str, Any]:
    hits = search_corpus(question, top_k=top_k)
    context = "\n\n".join([h["text"] for h in hits])

    answer = ""
    used_llm = False

    if use_llm:
        prompt = (
            "You are a unified curriculum tutor. Provide a concise answer, "
            "then a short combined learning path across books. Use the context below.\n\n"
            f"Question: {question}\n\nContext:\n{context}\n"
        )
        answer = _llm_generate(prompt)
        used_llm = bool(answer)

    if not answer:
        answer = _extractive_summary([h["text"] for h in hits])

    return {
        "ok": True,
        "question": question,
        "answer": answer,
        "used_llm": used_llm,
        "sources": hits
    }


# --------------------------------------------
# Flask Routes
# --------------------------------------------

def register_unified_routes(app):
    from flask import jsonify, request

    @app.route("/api/unified/index", methods=["POST"])
    def api_unified_index():
        force = bool(request.args.get("force"))
        payload = build_or_load_index(force_rebuild=force)
        return jsonify({"ok": True, "chunks": len(payload["chunks"])})

    @app.route("/api/unified/search")
    def api_unified_search():
        q = request.args.get("q", "")
        k = int(request.args.get("top_k", 5))
        return jsonify({"ok": True, "results": search_corpus(q, k)})

    @app.route("/api/unified/graph")
    def api_unified_graph():
        force = bool(request.args.get("force"))
        return jsonify(build_or_load_graph(force_rebuild=force))

    @app.route("/api/unified/path")
    def api_unified_path():
        user_id = request.args.get("user_id", "default")
        goal = request.args.get("goal", None)
        limit = int(request.args.get("limit", 10))
        return jsonify(recommend_path(user_id, goal, limit))

    @app.route("/api/unified/synthesize", methods=["POST"])
    def api_unified_synthesize():
        data = request.get_json(silent=True) or {}
        question = data.get("question", "")
        top_k = int(data.get("top_k", 6))
        use_llm = bool(data.get("use_llm", True))
        return jsonify(synthesize_answer(question, top_k, use_llm))

    return app
