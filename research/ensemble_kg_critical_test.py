"""
Ensemble + Knowledge Graph Critical Test (Phase 0 - GO/NO-GO)
==============================================================

HYPOTHESIS:
Knowledge graphs encoding model-task-performance relationships can improve 
ensemble selection over hard-coded heuristics.

CRITICAL TEST:
Compare KG-based ensemble selection against baselines on 10 diverse datasets.

SUCCESS CRITERIA:
- KG beats random selection by >5% average accuracy
- KG beats top-K selection by >2% on ≥6/10 datasets
- Statistical significance (p < 0.05)

DECISION:
- GO: Continue full research (implement in ai_model_orchestrator.py)
- NO-GO: Pivot to simpler meta-learning or alternative research

TIME BUDGET: 2 days maximum
"""

import numpy as np
import networkx as nx
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    load_diabetes, make_classification, make_regression
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, r2_score
from typing import List, Dict, Tuple, Any
import time
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# Dataset Collection
# =====================================================================

@dataclass
class BenchmarkDataset:
    """A benchmark dataset with metadata"""
    name: str
    X: np.ndarray
    y: np.ndarray
    task_type: str  # 'classification' or 'regression'
    n_classes: int = None
    n_features: int = None
    n_samples: int = None
    is_imbalanced: bool = False
    
    def __post_init__(self):
        self.n_samples, self.n_features = self.X.shape
        if self.task_type == 'classification':
            unique, counts = np.unique(self.y, return_counts=True)
            self.n_classes = len(unique)
            if len(unique) > 1:
                self.is_imbalanced = counts.min() / counts.max() < 0.5


def collect_datasets() -> List[BenchmarkDataset]:
    """
    Collect 10 diverse benchmark datasets
    
    Coverage:
    - Binary/multi-class classification
    - Regression
    - Small/large samples
    - Low/high dimensions
    - Balanced/imbalanced
    """
    datasets = []
    
    print("="*70)
    print("COLLECTING BENCHMARK DATASETS")
    print("="*70)
    
    # 1. Iris (classic multi-class, small)
    print("\n[1/10] Iris - Multi-class classification (3 classes, 150 samples)")
    iris = load_iris()
    datasets.append(BenchmarkDataset(
        name="Iris",
        X=iris.data,
        y=iris.target,
        task_type="classification"
    ))
    
    # 2. Wine (multi-class, medium)
    print("[2/10] Wine - Multi-class classification (3 classes, 178 samples)")
    wine = load_wine()
    datasets.append(BenchmarkDataset(
        name="Wine",
        X=wine.data,
        y=wine.target,
        task_type="classification"
    ))
    
    # 3. Breast Cancer (binary, medical)
    print("[3/10] Breast Cancer - Binary classification (569 samples)")
    cancer = load_breast_cancer()
    datasets.append(BenchmarkDataset(
        name="Breast_Cancer",
        X=cancer.data,
        y=cancer.target,
        task_type="classification"
    ))
    
    # 4. Digits (multi-class, vision, high-dim)
    print("[4/10] Digits - Multi-class classification (10 classes, 1797 samples)")
    digits = load_digits()
    datasets.append(BenchmarkDataset(
        name="Digits",
        X=digits.data,
        y=digits.target,
        task_type="classification"
    ))
    
    # 5. Synthetic Binary (controlled)
    print("[5/10] Synthetic Binary - Binary classification (1000 samples)")
    X_syn, y_syn = make_classification(
        n_samples=1000, n_features=20, n_classes=2,
        n_informative=15, n_redundant=5, random_state=42
    )
    datasets.append(BenchmarkDataset(
        name="Synthetic_Binary",
        X=X_syn,
        y=y_syn,
        task_type="classification"
    ))
    
    # 6. Synthetic Multi-class (controlled)
    print("[6/10] Synthetic Multi - Multi-class classification (1000 samples)")
    X_multi, y_multi = make_classification(
        n_samples=1000, n_features=20, n_classes=5,
        n_informative=15, n_clusters_per_class=1, random_state=42
    )
    datasets.append(BenchmarkDataset(
        name="Synthetic_Multi",
        X=X_multi,
        y=y_multi,
        task_type="classification"
    ))
    
    # 7. Diabetes (regression, medical)
    print("[7/10] Diabetes - Regression (442 samples)")
    diabetes = load_diabetes()
    datasets.append(BenchmarkDataset(
        name="Diabetes",
        X=diabetes.data,
        y=diabetes.target,
        task_type="regression"
    ))
    
    # 8. Synthetic Regression (controlled)
    print("[8/10] Synthetic Regression - Regression (1000 samples)")
    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=20, n_informative=15,
        noise=10.0, random_state=42
    )
    datasets.append(BenchmarkDataset(
        name="Synthetic_Regression",
        X=X_reg,
        y=y_reg,
        task_type="regression"
    ))
    
    # 9. Imbalanced Binary (controlled)
    print("[9/10] Imbalanced Binary - Binary classification with imbalance")
    X_imb, y_imb = make_classification(
        n_samples=1000, n_features=20, n_classes=2,
        weights=[0.9, 0.1], flip_y=0.05, random_state=42
    )
    datasets.append(BenchmarkDataset(
        name="Imbalanced_Binary",
        X=X_imb,
        y=y_imb,
        task_type="classification"
    ))
    
    # 10. High-dimensional classification (controlled)
    print("[10/10] High-Dimensional - Binary classification (high-dim)")
    X_hd, y_hd = make_classification(
        n_samples=500, n_features=100, n_informative=20,
        n_redundant=10, random_state=42
    )
    datasets.append(BenchmarkDataset(
        name="High_Dimensional",
        X=X_hd,
        y=y_hd,
        task_type="classification"
    ))
    
    print("\n✓ Collected 10 diverse datasets")
    print(f"  Classification: 8 datasets")
    print(f"  Regression: 2 datasets")
    
    return datasets


# =====================================================================
# Model Pool
# =====================================================================

def get_model_pool(task_type: str) -> Dict[str, Any]:
    """Get pool of models for task type"""
    if task_type == 'classification':
        return {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),  # Enable probability for soft voting
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'NaiveBayes': GaussianNB()
        }
    else:  # regression
        return {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=42),
            'SVR': SVR(kernel='rbf'),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'DecisionTree': DecisionTreeRegressor(random_state=42)
        }


# =====================================================================
# Knowledge Graph Implementation
# =====================================================================

class EnsembleKnowledgeGraph:
    """
    Simple knowledge graph for ensemble selection
    
    Nodes: models, tasks
    Edges: performance relationships
    """
    
    def __init__(self):
        self.G = nx.DiGraph()
        self.task_history = []
    
    def add_task(self, task_id: str, task_features: Dict[str, Any]):
        """Add a task node with features"""
        self.G.add_node(task_id, type='task', **task_features)
        self.task_history.append(task_id)
    
    def add_model_performance(self, task_id: str, model_name: str, 
                             score: float, training_time: float):
        """Add model performance edge"""
        if not self.G.has_node(model_name):
            self.G.add_node(model_name, type='model')
        
        self.G.add_edge(
            model_name, task_id,
            score=score,
            training_time=training_time
        )
    
    def compute_task_similarity(self, task_id1: str, task_id2: str) -> float:
        """
        Compute similarity between two tasks
        
        Uses cosine similarity of feature statistics:
        - n_samples, n_features, n_classes (if classification)
        - task_type
        """
        if not (self.G.has_node(task_id1) and self.G.has_node(task_id2)):
            return 0.0
        
        features1 = self.G.nodes[task_id1]
        features2 = self.G.nodes[task_id2]
        
        # Must be same task type
        if features1.get('task_type') != features2.get('task_type'):
            return 0.0
        
        # Extract numerical features
        vec1 = np.array([
            features1.get('n_samples', 0),
            features1.get('n_features', 0),
            features1.get('n_classes', 0) or 0
        ])
        vec2 = np.array([
            features2.get('n_samples', 0),
            features2.get('n_features', 0),
            features2.get('n_classes', 0) or 0
        ])
        
        # Normalize
        vec1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2 = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(vec1, vec2)
        return max(0.0, similarity)  # Clip to [0, 1]
    
    def recommend_models(self, task_features: Dict[str, Any], top_k: int = 3) -> List[str]:
        """
        Recommend top-k models for a new task based on KG
        
        Strategy:
        1. Find similar tasks in history
        2. Get models that worked well on similar tasks
        3. Rank by average performance on similar tasks
        """
        if not self.task_history:
            # No history → return None (will use random)
            return []
        
        # Create temp task ID
        temp_task = "temp_task"
        self.G.add_node(temp_task, type='task', **task_features)
        
        # Find similar tasks
        similarities = []
        for hist_task in self.task_history:
            sim = self.compute_task_similarity(temp_task, hist_task)
            if sim > 0.1:  # Threshold
                similarities.append((hist_task, sim))
        
        # Remove temp node
        self.G.remove_node(temp_task)
        
        if not similarities:
            return []
        
        # Get top similar tasks
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar_tasks = [t for t, _ in similarities[:5]]  # Top 5 similar
        
        # Aggregate model performances on similar tasks
        model_scores = {}
        for task in top_similar_tasks:
            task_sim = dict(similarities)[task]
            # Get all models that have edges to this task
            for model in self.G.predecessors(task):
                edge_data = self.G[model][task]
                score = edge_data.get('score', 0.0)
                
                # Weighted by task similarity
                weighted_score = score * task_sim
                
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(weighted_score)
        
        # Average scores
        avg_scores = {
            model: np.mean(scores)
            for model, scores in model_scores.items()
        }
        
        # Rank and return top-k
        ranked = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        return [model for model, _ in ranked[:top_k]]


# =====================================================================
# Baseline Methods
# =====================================================================

def random_selection(model_pool: Dict[str, Any], k: int = 3, seed: int = 42) -> List[str]:
    """Baseline 1: Random selection"""
    np.random.seed(seed)
    models = list(model_pool.keys())
    return list(np.random.choice(models, size=min(k, len(models)), replace=False))


def topk_selection(model_pool: Dict[str, Any], k: int = 3) -> List[str]:
    """
    Baseline 2: Top-K selection (best models overall)
    
    Uses hard-coded heuristics based on typical performance
    """
    # Hard-coded rankings (based on typical ML benchmarks)
    model_ranking_classification = [
        'GradientBoosting',
        'RandomForest',
        'SVM',
        'LogisticRegression',
        'KNN',
        'DecisionTree',
        'NaiveBayes'
    ]
    
    model_ranking_regression = [
        'GradientBoosting',
        'RandomForest',
        'Ridge',
        'LinearRegression',
        'SVR',
        'KNN',
        'DecisionTree'
    ]
    
    # Determine task type
    first_model = next(iter(model_pool.values()))
    is_classification = hasattr(first_model, 'predict_proba')
    
    ranking = model_ranking_classification if is_classification else model_ranking_regression
    
    # Return top-k from ranking that exist in pool
    selected = []
    for model_name in ranking:
        if model_name in model_pool:
            selected.append(model_name)
            if len(selected) == k:
                break
    
    return selected


# =====================================================================
# Ensemble Builder
# =====================================================================

def build_ensemble(models: List[Any], task_type: str):
    """Build voting ensemble from list of models"""
    if task_type == 'classification':
        return VotingClassifier(
            estimators=[(f"model_{i}", m) for i, m in enumerate(models)],
            voting='soft'
        )
    else:
        return VotingRegressor(
            estimators=[(f"model_{i}", m) for i, m in enumerate(models)]
        )


# =====================================================================
# Evaluation
# =====================================================================

@dataclass
class EvaluationResult:
    """Result for a single dataset"""
    dataset_name: str
    method_name: str
    score: float  # Accuracy or R²
    training_time: float
    models_selected: List[str]


def evaluate_method(method_name: str, selection_func, 
                   dataset: BenchmarkDataset, kg: EnsembleKnowledgeGraph = None) -> EvaluationResult:
    """
    Evaluate a selection method on a dataset
    
    Args:
        method_name: Name of method
        selection_func: Function that returns list of model names
        dataset: Dataset to evaluate on
        kg: Knowledge graph (for KG method)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X, dataset.y, test_size=0.3, random_state=42
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Get model pool
    model_pool = get_model_pool(dataset.task_type)
    
    # Select models
    start_time = time.time()
    
    if method_name == 'KG' and kg is not None:
        task_features = {
            'task_type': dataset.task_type,
            'n_samples': dataset.n_samples,
            'n_features': dataset.n_features,
            'n_classes': dataset.n_classes
        }
        selected_names = kg.recommend_models(task_features, top_k=3)
        
        # Fallback to random if KG has no recommendations
        if not selected_names:
            selected_names = random_selection(model_pool, k=3)
    else:
        selected_names = selection_func(model_pool, k=3)
    
    # Build ensemble
    selected_models = [model_pool[name] for name in selected_names]
    ensemble = build_ensemble(selected_models, dataset.task_type)
    
    # Train
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    y_pred = ensemble.predict(X_test)
    
    if dataset.task_type == 'classification':
        score = accuracy_score(y_test, y_pred)
    else:
        score = r2_score(y_test, y_pred)
    
    training_time = time.time() - start_time
    
    return EvaluationResult(
        dataset_name=dataset.name,
        method_name=method_name,
        score=score,
        training_time=training_time,
        models_selected=selected_names
    )


# =====================================================================
# Critical Test Execution
# =====================================================================

def run_critical_test():
    """
    Run the critical test
    
    Protocol:
    1. Collect datasets
    2. Split into train (first 5) and test (last 5)
    3. Build KG on train set
    4. Evaluate all methods on test set
    5. Statistical analysis
    """
    print("\n" + "="*70)
    print("CRITICAL TEST: KG vs BASELINES")
    print("="*70)
    
    # Collect datasets
    datasets = collect_datasets()
    
    # Split into train/test
    train_datasets = datasets[:5]
    test_datasets = datasets[5:]
    
    print("\n" + "="*70)
    print("PHASE 1: BUILD KNOWLEDGE GRAPH (Train Split)")
    print("="*70)
    print(f"\nTraining datasets: {[d.name for d in train_datasets]}")
    
    kg = EnsembleKnowledgeGraph()
    
    # Build KG on train datasets
    for dataset in train_datasets:
        print(f"\n[Processing] {dataset.name}...")
        
        # Add task to KG
        task_features = {
            'task_type': dataset.task_type,
            'n_samples': dataset.n_samples,
            'n_features': dataset.n_features,
            'n_classes': dataset.n_classes
        }
        kg.add_task(dataset.name, task_features)
        
        # Train all models and record performance
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X, dataset.y, test_size=0.3, random_state=42
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model_pool = get_model_pool(dataset.task_type)
        
        for model_name, model in model_pool.items():
            try:
                start = time.time()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                train_time = time.time() - start
                
                if dataset.task_type == 'classification':
                    score = accuracy_score(y_test, y_pred)
                else:
                    score = r2_score(y_test, y_pred)
                
                kg.add_model_performance(
                    dataset.name, model_name, score, train_time
                )
                
                print(f"  {model_name:20s}: {score:.3f}")
            
            except Exception as e:
                print(f"  {model_name:20s}: FAILED ({e})")
    
    print(f"\n✓ Knowledge graph built")
    print(f"  Nodes: {kg.G.number_of_nodes()}")
    print(f"  Edges: {kg.G.number_of_edges()}")
    
    # Test on test datasets
    print("\n" + "="*70)
    print("PHASE 2: EVALUATE ON TEST SPLIT")
    print("="*70)
    print(f"\nTest datasets: {[d.name for d in test_datasets]}")
    
    results = []
    
    for dataset in test_datasets:
        print(f"\n[Testing] {dataset.name}...")
        
        # KG method
        print("  [KG Method]", end=" ")
        kg_result = evaluate_method(
            "KG",
            lambda pool, k: [],  # Not used for KG
            dataset,
            kg=kg
        )
        print(f"→ {kg_result.score:.3f} (models: {', '.join(kg_result.models_selected)})")
        results.append(kg_result)
        
        # Random baseline
        print("  [Random]", end=" ")
        random_result = evaluate_method(
            "Random",
            random_selection,
            dataset
        )
        print(f"→ {random_result.score:.3f} (models: {', '.join(random_result.models_selected)})")
        results.append(random_result)
        
        # Top-K baseline
        print("  [Top-K]", end=" ")
        topk_result = evaluate_method(
            "Top-K",
            topk_selection,
            dataset
        )
        print(f"→ {topk_result.score:.3f} (models: {', '.join(topk_result.models_selected)})")
        results.append(topk_result)
    
    # Statistical analysis
    print("\n" + "="*70)
    print("PHASE 3: STATISTICAL ANALYSIS")
    print("="*70)
    
    # Group by method
    kg_scores = [r.score for r in results if r.method_name == 'KG']
    random_scores = [r.score for r in results if r.method_name == 'Random']
    topk_scores = [r.score for r in results if r.method_name == 'Top-K']
    
    print(f"\nResults on {len(test_datasets)} test datasets:")
    print(f"  KG:     {np.mean(kg_scores):.3f} ± {np.std(kg_scores):.3f}")
    print(f"  Random: {np.mean(random_scores):.3f} ± {np.std(random_scores):.3f}")
    print(f"  Top-K:  {np.mean(topk_scores):.3f} ± {np.std(topk_scores):.3f}")
    
    # Pairwise comparisons
    print("\nPairwise improvements:")
    kg_vs_random = ((np.array(kg_scores) - np.array(random_scores)) / np.array(random_scores) * 100)
    kg_vs_topk = ((np.array(kg_scores) - np.array(topk_scores)) / np.array(topk_scores) * 100)
    
    print(f"  KG vs Random: {np.mean(kg_vs_random):+.1f}% avg ({np.sum(kg_vs_random > 0)}/{len(test_datasets)} wins)")
    print(f"  KG vs Top-K:  {np.mean(kg_vs_topk):+.1f}% avg ({np.sum(kg_vs_topk > 0)}/{len(test_datasets)} wins)")
    
    # Statistical significance
    _, p_random = stats.wilcoxon(kg_scores, random_scores, alternative='greater')
    _, p_topk = stats.wilcoxon(kg_scores, topk_scores, alternative='greater')
    
    print(f"\nStatistical significance (Wilcoxon test):")
    print(f"  KG > Random: p = {p_random:.4f} {'✅ SIG' if p_random < 0.05 else '❌ NOT SIG'}")
    print(f"  KG > Top-K:  p = {p_topk:.4f} {'✅ SIG' if p_topk < 0.05 else '❌ NOT SIG'}")
    
    # GO/NO-GO decision
    print("\n" + "="*70)
    print("GO/NO-GO DECISION")
    print("="*70)
    
    criterion_1 = np.mean(kg_vs_random) > 5.0
    criterion_2 = np.sum(kg_vs_topk > 2.0) >= int(len(test_datasets) * 0.6)
    criterion_3 = p_random < 0.05 or p_topk < 0.05
    
    print(f"\nCriterion 1: KG beats random by >5% avg")
    print(f"  Result: {np.mean(kg_vs_random):+.1f}% → {'✅ PASS' if criterion_1 else '❌ FAIL'}")
    
    print(f"\nCriterion 2: KG beats top-K by >2% on ≥60% datasets")
    print(f"  Result: {np.sum(kg_vs_topk > 2.0)}/{len(test_datasets)} datasets → {'✅ PASS' if criterion_2 else '❌ FAIL'}")
    
    print(f"\nCriterion 3: Statistical significance (p < 0.05)")
    print(f"  Result: p = {min(p_random, p_topk):.4f} → {'✅ PASS' if criterion_3 else '❌ FAIL'}")
    
    all_pass = criterion_1 and criterion_2 and criterion_3
    
    print("\n" + "="*70)
    if all_pass:
        print("🎉 DECISION: GO - HYPOTHESIS VALIDATED")
        print("="*70)
        print("\nKnowledge graph ensemble selection shows promising results:")
        print(f"  - {np.mean(kg_vs_random):+.1f}% improvement over random selection")
        print(f"  - {np.mean(kg_vs_topk):+.1f}% improvement over top-K heuristic")
        print(f"  - Statistically significant (p < 0.05)")
        print("\nNext steps:")
        print("  1. Implement full system in ai_model_orchestrator.py")
        print("  2. Add more sophisticated similarity metrics")
        print("  3. Test on real-world problems")
        print("  4. Benchmark against AutoML systems")
    else:
        print("❌ DECISION: NO-GO - HYPOTHESIS NOT VALIDATED")
        print("="*70)
        print("\nKnowledge graph did not show sufficient improvement:")
        if not criterion_1:
            print(f"  - Only {np.mean(kg_vs_random):+.1f}% better than random (need >5%)")
        if not criterion_2:
            print(f"  - Only beats top-K on {np.sum(kg_vs_topk > 2.0)}/{len(test_datasets)} datasets (need ≥{int(len(test_datasets)*0.6)})")
        if not criterion_3:
            print(f"  - Not statistically significant (p = {min(p_random, p_topk):.4f})")
        print("\nFallback options:")
        print("  1. Simpler meta-learning (tabular data, no graph)")
        print("  2. Adaptive preprocessor with neural networks")
        print("  3. Return to learning apps (practical tools)")
    
    print("="*70)
    
    return {
        'kg_scores': kg_scores,
        'random_scores': random_scores,
        'topk_scores': topk_scores,
        'decision': 'GO' if all_pass else 'NO-GO',
        'criteria_met': (criterion_1, criterion_2, criterion_3)
    }


if __name__ == "__main__":
    results = run_critical_test()
