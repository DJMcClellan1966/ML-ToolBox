# Would Quantum Kernel Help Improve ML Compass?

**Short answer: Yes.** The quantum kernel (semantic embeddings, similarity, find_similar) can improve three of the four Compass components: **Oracle**, **Explainers**, and **Socratic**. Theory-as-Channel stays as-is (it's math, not text).

---

## How It Would Help

### 1. Oracle: Natural-language problem input

**Current:** User must pass a structured profile dict (e.g. `{"tabular": True, "classification": True, "n_samples": "medium"}`). Exact key match only.

**With quantum_kernel:** Add a second path: user describes the problem in plain language (e.g. "I have a few hundred rows of tabular data and need to classify"). We embed this description and compare it to **embedded rule descriptions** (e.g. "tabular data, medium sample size, classification task"). Return the rule with highest similarity. So the oracle can accept **free text** and still suggest the right pattern.

**Benefit:** Better UX; no need to know profile keys. Handles paraphrasing and fuzzy descriptions.

---

### 2. Explainers: RAG over corpus

**Current:** Static dict of concepts (entropy, bias_variance, capacity) with fixed "three views" text.

**With quantum_kernel:** Embed the user's concept (and optionally their follow-up question). **Retrieve** the most relevant chunks from the RAG corpus (e.g. `corpus/ml_foundations.md`, `corpus/information_theory_for_ml.md`) using `kernel.find_similar(query, corpus_chunks, top_k=5)`. Return those chunks as additional "views" or as the main explanation. Static views can remain as fallback or summary.

**Benefit:** Richer, up-to-date explanations; can cover many more concepts than a small static dict; same corpus you built for RAG now powers Compass.

---

### 3. Socratic: Retrieve debate viewpoints

**Current:** Hardcoded debate snippet (e.g. "theory: Ensembles reduce variance (Bishop). practice: Start with 3 models.") plus one Socratic question.

**With quantum_kernel:** Tag corpus or rule chunks by **viewpoint** (theory / practice / Bishop / Goodfellow). Embed the user's **statement** (e.g. "I used an ensemble"). Use `find_similar` to retrieve 2–3 chunks that are (a) relevant to the statement and (b) from different viewpoints. Format as "View 1: ... View 2: ..." and keep the Socratic question. So the debate content becomes **retrieved** instead of fixed.

**Benefit:** Debate responses can reflect your actual book-derived content and scale to many topics.

---

### 4. Theory-as-Channel: No change

Ensemble correction and channel capacity are numerical (predictions, signal/noise). The kernel doesn't add here. Optional: if you ever add "explain capacity in words," that could use the explainers + kernel RAG.

---

## Summary

| Component        | Improves with quantum_kernel? | How |
|-----------------|-------------------------------|-----|
| **Oracle**      | Yes                           | NL problem description -> embed -> match to rule descriptions |
| **Explainers**  | Yes                           | RAG over corpus for concept explanations |
| **Socratic**    | Yes                           | Retrieve debate viewpoints by similarity to user statement |
| **Theory-as-Channel** | No                      | Stays math-only |

**Optional dependency:** The kernel is in `quantum_kernel/` and may use `sentence-transformers` for best quality. ML Compass runs **without** it; add it for the improvements above. See `quantum_enhancements.py` for an optional integration layer.
