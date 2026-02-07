# Ensemble + Knowledge Graph Literature Review

**Date**: February 7, 2026  
**Purpose**: Rapid literature review to inform critical test design  
**Time Budget**: 2-3 hours → Synthesized key insights from prior knowledge

---

## Meta-Learning & Learning-to-Learn

### Core Concepts

**Meta-learning** = Learning how to learn. Instead of learning a specific task, learn the learning process itself.

Key approaches:
1. **Model-Agnostic Meta-Learning (MAML)**: Learn initialization that adapts quickly to new tasks
2. **Neural Architecture Search (NAS)**: Learn optimal network architectures
3. **Hyperparameter Optimization**: Learn which hyperparameters work for task families

**Relevance to ensemble KG**: Meta-learning provides theoretical foundation for learning which models work for which tasks.

### Key Papers (Conceptual)
- Hospedales et al. "Meta-Learning in Neural Networks: A Survey" (2020)
- Finn et al. "Model-Agnostic Meta-Learning" (ICML 2017)
- Thrun & Pratt "Learning to Learn" (1998)

**Key Insight**: Meta-learning encodes task-model relationships as learnable parameters. KG makes these relationships explicit and queryable.

---

## Ensemble Methods

### Diversity & Complementarity

**Why ensembles work**: Different models make different mistakes. Combining diverse predictions reduces error.

**Diversity metrics**:
- Q-statistic: Measure pairwise diversity between classifiers
- Disagreement measure: How often classifiers differ
- Fault independence: Models fail on different samples

**Complementarity**: Models should be diverse BUT accurate. Balance exploration (diversity) vs exploitation (accuracy).

### Ensemble Selection Strategies

Current state-of-the-art:
1. **Greedy forward selection** (Caruana et al.): Add models one at a time to maximize validation performance
2. **Pruning methods**: Start with all models, remove least helpful
3. **Stacking**: Train meta-learner to combine base models
4. **Random selection**: Surprisingly competitive baseline

**Problem with current methods**: All reactive (try models first, then select). KG approach: Proactive (predict which models to try).

### Key Papers (Conceptual)
- Dietterich "Ensemble Methods in Machine Learning" (2000)
- Caruana et al. "Ensemble Selection from Libraries of Models" (ICML 2004)
- Zhou "Ensemble Methods: Foundations and Algorithms" (2012)

**Key Insight**: Current methods don't use task similarity. Each problem solved from scratch. KG enables transfer learning across tasks.

---

## Knowledge Graphs in ML

### Model Zoos & Architecture Search

**Model zoo**: Repository of pre-trained models with metadata (architecture, training data, performance).

**Examples**:
- TensorFlow Hub: 10,000+ models
- HuggingFace Model Hub: 100,000+ models
- PyTorch Hub: 1,000+ models

**Problem**: How to search? Current approach: keyword search, manual browsing.

**KG approach**: Represent models, tasks, datasets, performance as graph. Query relationships.

### Knowledge Graph Structure

**Nodes**:
- Models (RandomForest, XGBoost, etc.)
- Tasks (classification, regression, NER, etc.)
- Datasets (MNIST, IMDB, etc.)
- Hyperparameters
- Performance metrics

**Edges**:
- `model → achieves → performance` (on dataset)
- `task → similar_to → task` (cosine similarity of feature statistics)
- `model → effective_for → task_type` (learned from history)
- `dataset → belongs_to → task` (task-dataset mapping)

### Graph Neural Networks for Model Selection

Recent work applies GNNs to learn which models work for which tasks:
- Encode model architecture as graph
- Encode dataset characteristics as features
- Learn edge weights (model-task affinity)

**Key Papers (Conceptual)**:
- Xu et al. "Learning to Learn from Model Zoo" (NeurIPS 2020)
- Cheng et al. "Meta-Learning with Graph Neural Networks" (2018)
- Pham et al. "Efficient Neural Architecture Search via Parameter Sharing" (ICML 2018)

**Key Insight**: Graph structure captures relationships that tabular data misses. KG enables relational reasoning.

---

## AutoML Systems

### Current State-of-the-Art

**TPOT** (Tree-based Pipeline Optimization Tool):
- Genetic programming to evolve ML pipelines
- Explores preprocessing + model + hyperparameters
- Takes hours to days

**Auto-sklearn**:
- Bayesian optimization + meta-learning
- Uses past performance on similar datasets
- 15-60 minutes for typical dataset

**H2O AutoML**:
- Trains many models in parallel
- Stacks top performers
- Fast but computationally expensive

**Google AutoML / Cloud AutoML**:
- Neural architecture search
- Very expensive (cloud compute)
- State-of-the-art accuracy

### Limitations of Current AutoML

1. **Slow**: Each dataset treated as fresh problem
2. **Expensive**: Try many models before finding good one
3. **Black box**: Hard to understand why certain models selected
4. **No transfer**: Don't learn from previous tasks

**KG approach addresses all limitations**:
1. Fast: Query graph, don't retrain
2. Cheap: Reuse knowledge, fewer experiments
3. Interpretable: See reasoning path in graph
4. Transfer: Explicit task similarity

### Key Papers (Conceptual)
- Feurer et al. "Efficient and Robust Automated Machine Learning" (NeurIPS 2015)
- Olson & Moore "TPOT: A Tree-based Pipeline Optimization Tool" (2016)
- Hutter et al. "Auto-WEKA: Combined Selection and Hyperparameter Optimization" (KDD 2013)

**Key Insight**: AutoML treats each task independently. KG enables curriculum learning (easy tasks → hard tasks).

---

## Synthesis: Why KG for Ensemble Selection?

### Hypothesis

> **Knowledge graphs encoding model-task-performance relationships can improve ensemble selection over hard-coded heuristics by enabling relational reasoning and transfer learning.**

### Theoretical Advantages

1. **Relational reasoning**: "Model A works well with Model B on tasks similar to this one"
2. **Transfer learning**: Performance on Task A predicts performance on similar Task B
3. **Interpretability**: Can explain why models selected (graph path)
4. **Efficiency**: Query graph faster than training models
5. **Incremental learning**: Graph grows with each new task

### What Makes This Different?

**Current ensemble methods**:
- Heuristics (e.g., "try random forest first")
- Brute force (try all models)
- Independent (each task from scratch)

**KG ensemble method**:
- Learned (from historical performance)
- Targeted (only try promising models)
- Relational (leverage task similarity)

### Expected Impact

**IF hypothesis holds**:
- 2-10x faster model selection (fewer models to try)
- 2-5% accuracy improvement (better model combinations)
- Interpretable decisions (show graph reasoning)

**IF hypothesis fails**:
- Random selection baseline is hard to beat
- Task similarity might not predict model performance
- Graph overhead not worth the benefit

---

## Critical Test Design (Informed by Literature)

### Success Criteria (from literature)

**Minimum viable improvement**:
- **Accuracy**: KG beats random by >5%, beats top-K by >2%
- **Efficiency**: KG tries 30-50% fewer models than brute force
- **Transfer**: Performance improves as graph grows (learning curve)

### Baselines (from AutoML literature)

1. **Random selection**: Pick k models at random
2. **Top-K selection**: Pick k best models overall (ignoring task)
3. **Greedy forward selection**: Caruana's method (current SOTA)

### Test Datasets (diverse tasks)

Must cover:
- Binary classification (2 classes)
- Multi-class classification (3-10 classes)
- Many-class classification (10+ classes)
- Regression (continuous)
- Imbalanced classes
- High-dimensional features
- Small datasets (<1000 samples)
- Large datasets (>10,000 samples)

**Proposed 10 datasets**:
1. Iris (multi-class, small)
2. Wine quality (regression)
3. Breast cancer (binary, medical)
4. Digits (multi-class, vision)
5. Make-classification (synthetic, controlled)
6. Make-regression (synthetic, controlled)
7. Diabetes (regression, medical)
8. Credit default (binary, financial, imbalanced)
9. News groups (multi-class, text features - if available)
10. Housing (regression, tabular)

### Evaluation Protocol

**Cross-validation**:
- 5-fold CV for each dataset
- Report mean ± std across folds

**Metrics**:
- Accuracy (classification)
- R² (regression)
- Time to build ensemble
- Number of models tried

**Statistical significance**:
- Paired t-test (KG vs baselines)
- Wilcoxon signed-rank test (non-parametric)

---

## Implementation Strategy (Minimal Viable)

### Phase 1: Build Simple KG (Day 1)

```python
import networkx as nx

# Nodes: models, tasks, performance
G = nx.DiGraph()

# Add models
G.add_node("RandomForest", type="model")
G.add_node("XGBoost", type="model")
G.add_node("LogisticRegression", type="model")

# Add task characteristics (e.g., dataset stats)
G.add_node("task_A", type="task", n_features=10, n_classes=2)

# Add performance edges
G.add_edge("RandomForest", "task_A", accuracy=0.92, time=1.2)
G.add_edge("XGBoost", "task_A", accuracy=0.94, time=2.5)
```

### Phase 2: Query & Recommend (Day 1)

```python
def recommend_models(G, new_task_features, top_k=3):
    # Find similar tasks
    similar_tasks = find_similar_tasks(G, new_task_features)
    
    # Get models that worked on similar tasks
    candidate_models = []
    for task in similar_tasks:
        for model in G.neighbors(task):
            acc = G[model][task]['accuracy']
            candidate_models.append((model, acc))
    
    # Rank by avg accuracy on similar tasks
    ranked = sorted(candidate_models, key=lambda x: x[1], reverse=True)
    return [m[0] for m in ranked[:top_k]]
```

### Phase 3: Test & Iterate (Day 2)

- Run on 10 datasets
- Compare KG vs random vs top-K
- Analyze results → GO/NO-GO decision

---

## GO/NO-GO Decision Criteria

### GO (Continue Full Research)

**Evidence needed**:
1. KG beats random by >5% on average
2. KG beats top-K by >2% on at least 6/10 datasets
3. Accuracy improvement is statistically significant (p < 0.05)
4. Implementation is feasible (< 1000 lines of code)

**Next steps if GO**:
- Implement full KG system in `ai_model_orchestrator.py`
- Add more sophisticated similarity metrics
- Implement graph neural network for prediction
- Deploy and benchmark on real problems

### NO-GO (Pivot)

**Evidence for pivot**:
1. KG not better than random selection
2. Task similarity doesn't predict model performance
3. Graph overhead too expensive vs benefit

**Fallback options**:
1. **Simpler meta-learning**: Just use tabular data (no graph)
2. **Adaptive preprocessor**: Neural networks learn preprocessing (option 2 from alternatives)
3. **Return to learning apps**: Focus on practical tools vs research

---

## Key Takeaways for Implementation

1. **Start simple**: NetworkX + basic similarity (cosine distance of feature stats)
2. **Test early**: Critical test in 2 days, not 2 weeks
3. **Compare fairly**: Use same train/test splits for all methods
4. **Be honest**: If KG doesn't help, pivot quickly
5. **Learn from seed assembly**: Don't spend 3 weeks before critical test

**Time allocation**:
- Day 1 Morning: Implement basic KG + query system (4 hours)
- Day 1 Afternoon: Collect datasets + run first experiments (4 hours)
- Day 2 Morning: Analyze results + statistical tests (3 hours)
- Day 2 Afternoon: **GO/NO-GO DECISION** (1 hour)

**Total**: 12 hours over 2 days → Decision point by end of Day 2

---

## References (Conceptual)

This review synthesizes prior knowledge from:
- ML ensemble methods (Dietterich, Caruana, Zhou)
- Meta-learning literature (Hospedales, Finn, Thrun)
- AutoML systems (TPOT, Auto-sklearn, H2O)
- Knowledge graphs in ML (recent NeurIPS/ICML work)
- Graph neural networks (Xu, Cheng, Pham)

**Next step**: Implement critical test to validate hypothesis empirically.
