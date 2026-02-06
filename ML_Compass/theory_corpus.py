"""
Default theory corpus for RAG explainers. Content inspired by ML-Mastery and standard ML theory.
Optional: load additional chunks from a path (e.g. Desktop ML-Mastery).
"""
from pathlib import Path
from typing import Dict, Any, List

# Curated chunks: id -> content (concepts: entropy, bias_variance, capacity, overfitting, regularization, etc.)
DEFAULT_THEORY_CHUNKS: List[Dict[str, str]] = [
    {"id": "entropy_shannon", "content": "Shannon entropy H = -sum p log p measures uncertainty in bits. Higher H means more randomness. In ML, information gain in decision trees uses entropy reduction."},
    {"id": "entropy_boltzmann", "content": "Boltzmann linked entropy to disorder. In statistical mechanics, entropy counts microstates; same logarithmic form as Shannon. Connects thermodynamics to information."},
    {"id": "entropy_ml", "content": "In ML: decision tree splitting (information gain), softmax loss, regularization to encourage smooth distributions. Cross-entropy loss is standard for classification."},
    {"id": "bias_variance_theory", "content": "Bias is error from model being too simple; variance from being too sensitive to the training set. Tradeoff between underfitting and overfitting. Bishop Ch. 3."},
    {"id": "bias_variance_practice", "content": "Use validation set and cross-validation; regularize (L1/L2) or use ensemble to reduce variance. More data often reduces variance."},
    {"id": "capacity_channel", "content": "Shannon channel capacity C = B log2(1 + S/N) bounds reliable transmission over a noisy channel. Redundancy (e.g. ensemble) helps approach capacity."},
    {"id": "capacity_ml", "content": "Model capacity: how complex a function the model can represent. Too high -> overfitting; too low -> underfitting. For teacher-student: capacity bounds how much knowledge you can compress."},
    {"id": "overfitting", "content": "Overfitting: model fits training data too closely, poor generalization. Remedies: regularization, early stopping, more data, reduce capacity, dropout (deep learning)."},
    {"id": "regularization", "content": "L1 (Lasso) and L2 (Ridge) regularization penalize large weights. L1 encourages sparsity; L2 stabilizes solutions. Both reduce overfitting by limiting model complexity."},
    {"id": "ensemble_majority", "content": "Ensemble of 3-5 models with majority vote reduces variance. Theory-as-channel: redundancy over a noisy channel improves reliable decisions. Don't use too many weak learners on small n."},
    {"id": "validation_split", "content": "Always use a held-out validation set or cross-validation for model selection. Never tune or select on the test set. Fit scalers and imputers on training fold only to avoid data leakage."},
    {"id": "problem_definition", "content": "Six top-level tasks for ML: problem definition, analyze data, prepare data, evaluate algorithms, improve results, present results. Start with problem definition and baseline."},
    {"id": "spot_check", "content": "Spot-check classification algorithms: try a few diverse models (e.g. logistic regression, tree, SVM) with default hyperparameters to see what works before tuning."},
    {"id": "imbalanced", "content": "For imbalanced classes use class_weight, oversampling (SMOTE), or adjust decision threshold. Report precision-recall and F1; accuracy is misleading when one class is rare."},
]

# Optional path to load extra chunks (e.g. ML-Mastery lesson files on Desktop)
DEFAULT_EXTRA_PATH = None  # Set to Path(r"C:\Users\...\Machine-Learning\ML-Mastery-Python") to load lessons


def get_default_corpus() -> List[Dict[str, str]]:
    """Return the default theory corpus (list of {id, content})."""
    return list(DEFAULT_THEORY_CHUNKS)


def load_corpus_from_path(path: str | Path, max_chunk_chars: int = 800) -> List[Dict[str, str]]:
    """
    Load text from .py and .txt files under path and split into chunks.
    Each chunk is {"id": "file:line", "content": "..."}. Optional for users who have ML-Mastery on disk.
    """
    path = Path(path)
    if not path.exists() or not path.is_dir():
        return []
    chunks = []
    for ext in ("*.py", "*.txt", "*.md"):
        for f in path.rglob(ext):
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            # Simple split by lines and recombine into ~max_chunk_chars blocks
            lines = text.splitlines()
            block, length = [], 0
            for i, line in enumerate(lines):
                block.append(line)
                length += len(line) + 1
                if length >= max_chunk_chars or (block and line.strip() == "" and length > 100):
                    content = "\n".join(block).strip()
                    if len(content) > 50:
                        chunks.append({"id": f"{f.name}:{i}", "content": content[:max_chunk_chars]})
                    block, length = [], 0
            if block:
                content = "\n".join(block).strip()
                if len(content) > 50:
                    chunks.append({"id": f"{f.name}:end", "content": content[:max_chunk_chars]})
    return chunks


def get_corpus(include_path: str | Path | None = None) -> List[Dict[str, str]]:
    """
    Return default corpus; if include_path is given and exists, append chunks loaded from that path.
    """
    corpus = get_default_corpus()
    if include_path:
        extra = load_corpus_from_path(include_path)
        corpus = corpus + extra
    return corpus
