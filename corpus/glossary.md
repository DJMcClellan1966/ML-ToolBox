# Glossary: ML and Toolbox Terms

Short definitions for terms used in the toolbox and in ML generally. Useful for RAG and for quick lookup.

**Accuracy** — In classification, the fraction of predictions that are correct. Can be misleading when classes are imbalanced.

**AdvancedDataPreprocessor** — Toolbox component that performs semantic deduplication, safety filtering, quality scoring, and optional compression on text. Part of the data compartment.

**Bias** — In the bias–variance tradeoff, error due to the model being too simple (underfitting).

**Bias–variance tradeoff** — Tradeoff between underfitting (high bias) and overfitting (high variance); model and regularization choices balance the two.

**Boosting** — Ensemble method that trains models sequentially, each correcting errors of the previous (e.g. AdaBoost, gradient boosting).

**Compartment** — In the toolbox, a logical grouping: Data (preprocessing), Infrastructure (kernels, AI, LLM), Algorithms (evaluation, tuning, ensembles), MLOps (deployment, monitoring).

**Compressed embeddings** — Lower-dimensional representation of text or features after dimensionality reduction; used to save memory and sometimes improve generalization.

**Cross-validation** — Technique that splits data into multiple train/validation folds to estimate performance more reliably; used for model and hyperparameter selection.

**Deduplication (semantic)** — Removing or grouping items that are similar in meaning (via embeddings), not just identical in text.

**Entropy** — In information theory, a measure of uncertainty or randomness of a distribution; used in decision trees and data quality.

**F1 score** — Harmonic mean of precision and recall; common metric for classification when both false positives and false negatives matter.

**Feature selection** — Choosing a subset of features (e.g. by mutual information or importance) to reduce dimensionality and overfitting.

**Information gain** — Reduction in entropy (or impurity) when splitting on a feature; used in decision tree splitting.

**KL divergence** — Kullback–Leibler divergence; measures how one probability distribution differs from another; used in variational inference and generative models.

**Mutual information** — Measure of how much knowing one variable reduces uncertainty about another; used for feature selection and clustering evaluation.

**Overfitting** — Model fits training data too closely (including noise) and generalizes poorly to new data.

**Precision** — In classification, among predicted positives, the fraction that are actually positive.

**Quantum kernel** — In the toolbox, the component that produces semantic embeddings and similarity for text; “quantum-inspired” in name, used for semantic operations.

**RAG** — Retrieval-augmented generation: retrieve relevant documents (e.g. from this corpus) and pass them to an LLM as context for answering questions.

**Recall** — In classification, among actual positives, the fraction that were predicted positive.

**Recall (in retrieval)** — In search/RAG, the fraction of relevant documents that were retrieved.

**Regularization** — Penalty on model complexity (e.g. L1, L2) to reduce overfitting.

**Semantic deduplication** — Same as deduplication (semantic); emphasizes meaning-based similarity.

**Underfitting** — Model is too simple to capture the main structure in the data; both training and validation performance are poor.

**Variance** — In the bias–variance tradeoff, error due to the model being too sensitive to the training set (overfitting).

**Validation set** — Data held out for tuning hyperparameters and monitoring generalization; not used to fit the model.

**Test set** — Data held out for final performance evaluation only; should not influence any training or tuning decisions.
