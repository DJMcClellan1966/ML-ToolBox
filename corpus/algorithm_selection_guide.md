# Algorithm and Preprocessing Selection Guide

Short, practical guidance on choosing algorithms and preprocessing for common problem types. Use this as a decision support reference, not as a substitute for validation on your own data.

## Problem Type: Text Classification

**Goal**: Assign a category or label to each text (e.g. sentiment, topic, intent).

**Preprocessing**: Use advanced preprocessing with semantic deduplication to reduce near-duplicate texts; enable compression if you have many samples and want smaller feature matrices. Consider safety filtering if the text is user-generated. The toolbox’s data compartment can output compressed_embeddings or similar features for the classifier.

**Algorithms**: Start with a linear model (e.g. logistic regression) or Random Forest on the preprocessed features. If you have lots of labeled data and want maximum accuracy, try gradient boosting or a small neural net. Use cross-validation and a metric that matches your cost (e.g. F1 for imbalanced classes).

## Problem Type: High-Dimensional Classification or Regression

**Goal**: Predict a target when you have many features (e.g. hundreds or thousands).

**Preprocessing**: Consider dimensionality reduction (PCA, or the toolbox’s compression) to avoid overfitting and speed up training. Optionally use mutual-information or information-gain-based feature selection to keep the most informative features. Clean missing values and scale features as needed.

**Algorithms**: Regularized linear models (Ridge, Lasso) or tree-based models (Random Forest, gradient boosting) are robust. Avoid very complex models unless you have enough samples relative to dimensions; use cross-validation to tune regularization or depth.

## Problem Type: Clustering or Unsupervised Grouping

**Goal**: Discover groups in data without labels (e.g. customer segments, topic discovery).

**Preprocessing**: For text, use the same advanced preprocessing to get embeddings; then cluster in embedding space. For numeric data, scale features so distance-based methods (e.g. k-means) are not dominated by one scale. Remove or impute missing values.

**Algorithms**: K-means for spherical, similar-sized clusters; DBSCAN for arbitrary shapes and outliers; hierarchical clustering if you want a dendrogram. The toolbox provides clustering and evaluation utilities; use internal metrics (e.g. silhouette) and domain sense to choose k or parameters.

## Problem Type: Regression with Tabular Data

**Goal**: Predict a continuous target from numeric and/or categorical features.

**Preprocessing**: Handle missing values (impute or drop); encode categoricals (one-hot or target encoding); scale if using linear or distance-based models. Use conventional preprocessing unless you have free-text fields that need semantic treatment.

**Algorithms**: Linear or Ridge regression for interpretability and when relationships are roughly linear; Random Forest or gradient boosting for nonlinearity and interactions. Evaluate with RMSE, MAE, or R²; use cross-validation for hyperparameters.

## Problem Type: Content Moderation or Safety

**Goal**: Flag or filter unsafe or policy-violating content.

**Preprocessing**: This is where the toolbox’s strength is: run safety filtering (e.g. PocketFence path) and optionally semantic deduplication so repeated similar violations are handled consistently. Quality scoring can help prioritize human review.

**Algorithms**: Often a combination of rule-based filters and a classifier trained on labeled safe/unsafe examples. Use the preprocessed (and optionally deduplicated) corpus to train the classifier; the preprocessing pipeline becomes the first stage of the moderation system.

## Problem Type: Search or Retrieval Over Documents

**Goal**: Given a query, retrieve the most relevant documents (e.g. for RAG or search UI).

**Preprocessing**: Index time: run advanced preprocessing on documents (deduplicate, optionally compress); store embeddings and metadata. Query time: optionally normalize the query; embed with the same model used for documents. Do not over-dedup at query time; deduplication is for the corpus.

**Algorithms**: Retrieval is typically nearest-neighbor search in embedding space (cosine or Euclidean). The toolbox’s BetterKnowledgeRetriever and quantum kernel support this; use the same embedding model for index and query.

## Quick Reference Table

| Problem type              | Preprocessing emphasis              | Algorithm family to try first        |
|---------------------------|------------------------------------|--------------------------------------|
| Text classification       | Advanced, dedup, optional compress | Logistic regression, Random Forest   |
| High-dimensional predict  | Reduce dims, feature selection     | Ridge/Lasso, Random Forest, boosting |
| Clustering                | Embeddings for text; scale numeric | K-means, DBSCAN                      |
| Tabular regression        | Clean, encode, scale               | Linear, Random Forest, boosting       |
| Content moderation        | Safety filter, dedup               | Rules + classifier                  |
| Search / retrieval        | Dedup at index; same embed for query| Embedding + nearest neighbor         |

Always validate on your own data and metrics; this guide gives starting points, not fixed rules.
