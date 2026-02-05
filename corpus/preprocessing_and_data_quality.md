# Preprocessing and Data Quality

When and how to preprocess data, and how the toolbox supports data quality and safety.

## When to Use Advanced Preprocessing

Use **advanced preprocessing** (e.g. the toolbox’s AdvancedDataPreprocessor with semantic deduplication and optional compression) when:

- You have **text data** that may contain near-duplicates (rephrased or slightly different wording).
- You need **safety or content filtering** before feeding data to a model or to users (e.g. user-generated content).
- You want **quality scoring** or **categorization** of items (e.g. support tickets, reviews, documents).
- You need **dimensionality reduction** or **compression** of text-derived features before a downstream model.

For simple numeric tables with no text, use conventional cleaning (missing values, scaling, outliers) rather than the full advanced text pipeline.

## Semantic Deduplication

**Semantic deduplication** removes or groups items that are similar in meaning, not just identical in text. It uses embeddings (e.g. from the quantum kernel or a sentence encoder) and a similarity threshold: items whose embeddings are closer than the threshold are treated as duplicates or near-duplicates. This reduces redundancy in training data and in search indices, and can cut dataset size while keeping diversity. The toolbox supports configurable dedup thresholds (e.g. 0.85) so you can tune strictness.

## Safety Filtering

**Safety filtering** (e.g. via PocketFence or similar) flags or filters content that is unsafe or policy-violating. It is typically applied before storing data or before training or serving a model. The toolbox can integrate safety checks as part of the preprocessing pipeline so that only safe, cleaned text is passed to downstream components.

## Data Quality Metrics

Data quality can be assessed along several dimensions:

- **Completeness**: fraction of non-missing values; impact of missingness on model (e.g. via missing_value_impact in the toolbox).
- **Informativeness**: how much each feature helps predict the target (e.g. mutual information, information gain).
- **Redundancy**: how much features overlap (e.g. pairwise mutual information or correlation); high redundancy suggests some features can be dropped or combined.
- **Consistency**: whether values and formats are coherent across records (e.g. normalization, scrubbing).

The toolbox provides utilities for quality scoring and for estimating the impact of missing values and low-information features.

## Conventional vs Advanced Preprocessing

**Conventional preprocessing** in the toolbox covers: exact duplicate removal, basic normalization, keyword-based categorization, and standard cleaning (missing values, outliers, scaling). Use it when your data is already in good shape or when you do not need semantic understanding.

**Advanced preprocessing** adds: semantic deduplication, safety filtering, quality scoring, embedding-based categorization, and optional compression of embeddings. Use it when you have text and need semantic awareness, safety, or compact representations.

## Best Practices

- Run **safety and deduplication** before training or indexing to avoid training on duplicate or unsafe content.
- Use **quality scores** to filter or weight samples (e.g. train more on high-quality items) or to monitor pipeline health.
- Set **dedup_threshold** based on your domain: stricter (e.g. 0.9) keeps more distinct items; looser (e.g. 0.75) merges more aggressively.
- For RAG or search, preprocess and deduplicate the corpus once at index time; at query time only the query may need lightweight normalization.
