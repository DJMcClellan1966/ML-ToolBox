# Research: Novel ML/AI Connections

Three potentially novel findings discovered by analyzing cross-domain mathematical
structures in the ML-ToolBox corpus. Each has a proof-of-concept implementation
with reproducible experiments.

## Findings

### 1. L⁻² Regularization (Heisenberg Regularizer)
**File:** `heisenberg_regularizer.py`

A new regularization penalty λ/Var(w) that prevents weight collapse — the exact
mathematical dual of standard L2 regularization. Where L2 penalizes weights
that are too spread out, L⁻² penalizes weights that are too concentrated.

**Key equation:** Loss = CrossEntropy + λ₂‖w‖² + λ_H / (Var(w) + ε)

**Target problem:** Representational collapse in self-supervised learning,
attention head diversity, ensemble diversity.

### 2. Dissipative Grokking Predictor
**File:** `dissipative_grokking.py`

Models neural network training dynamics as a Prigogine dissipative system to
predict when grokking (sudden generalization after memorization) will occur.
Derives a critical learning-rate-to-weight-decay ratio that triggers the
phase transition from memorization to generalization.

**Key prediction:** Grokking occurs when η/λ exceeds a critical threshold
derived from the dissipative stability condition.

### 3. Gray Code Hyperparameter Search
**File:** `gray_code_hpo.py`

Uses reflected Gray codes for hyperparameter optimization. Each evaluation
changes exactly one hyperparameter, giving perfect single-factor attribution
as a free byproduct of the search — something no existing HPO method provides.

**Key property:** Every step in the search is simultaneously an optimization
step AND a controlled experiment.

## Running

```bash
# Each file is self-contained with experiments
python research/heisenberg_regularizer.py
python research/dissipative_grokking.py
python research/gray_code_hpo.py
```

## Status

These are proof-of-concept implementations. None have been peer-reviewed
or validated at scale. Treat as exploratory research.
