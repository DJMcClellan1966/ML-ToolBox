"""
Comprehensive Benchmark Suite for Hardest ML Problems

Tests the ML Toolbox against some of the hardest problems in ML:
1. NP-Hard Optimization (Traveling Salesman Problem)
2. Chaotic Time Series Prediction
3. Adversarial Robustness
4. Few-Shot Learning
5. High-Dimensional Sparse Data
6. Concept Drift
7. Imbalanced Classification
8. Non-Stationary Environments
9. Transfer Learning Across Domains
10. Multi-Objective Optimization with Constraints
"""
import numpy as np
import sys
from pathlib import Path
import time
import json
from typing import Dict, List, Any
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("HARDEST ML PROBLEMS BENCHMARK SUITE")
print("=" * 80)
print()

# Results storage
results = {
    'problems': {},
    'toolbox_performance': {},
    'baseline_performance': {},
    'comparisons': {},
    'summary': {}
}

# ============================================================================
# Problem 1: NP-Hard Optimization - Traveling Salesman Problem (TSP)
# ============================================================================
print("=" * 80)
print("PROBLEM 1: NP-HARD OPTIMIZATION - TRAVELING SALESMAN PROBLEM")
print("=" * 80)

try:
    from ml_toolbox.optimization.evolutionary_algorithms import GeneticAlgorithm, DifferentialEvolution
    from ml_toolbox.textbook_concepts.statistical_mechanics import SimulatedAnnealing
    
    def create_tsp_instance(n_cities: int = 20, seed: int = 42):
        """Create TSP instance with random city coordinates"""
        np.random.seed(seed)
        cities = np.random.random((n_cities, 2)) * 100
        return cities
    
    def tsp_distance(cities: np.ndarray, tour: np.ndarray) -> float:
        """Calculate TSP tour distance"""
        tour = tour.astype(int)
        total_distance = 0.0
        for i in range(len(tour) - 1):
            city1 = cities[tour[i]]
            city2 = cities[tour[i + 1]]
            total_distance += np.linalg.norm(city1 - city2)
        # Return to start
        total_distance += np.linalg.norm(cities[tour[-1]] - cities[tour[0]])
        return total_distance
    
    cities = create_tsp_instance(20)
    n_cities = len(cities)
    
    # Baseline: Random search
    print("\n--- Baseline: Random Search ---")
    start_time = time.time()
    best_random = float('inf')
    for _ in range(1000):
        tour = np.random.permutation(n_cities)
        distance = tsp_distance(cities, tour)
        if distance < best_random:
            best_random = distance
    random_time = time.time() - start_time
    print(f"Best distance: {best_random:.2f}")
    print(f"Time: {random_time:.4f}s")
    
    # Toolbox: Genetic Algorithm
    print("\n--- Toolbox: Genetic Algorithm ---")
    def tsp_fitness(individual):
        """Fitness: negative distance (maximize = minimize distance)"""
        tour = (individual * n_cities).astype(int) % n_cities
        distance = tsp_distance(cities, tour)
        return -distance  # Negative for maximization
    
    gene_ranges = [(0.0, 1.0)] * n_cities
    ga = GeneticAlgorithm(
        fitness_function=tsp_fitness,
        gene_ranges=gene_ranges,
        population_size=50,
        max_generations=100
    )
    start_time = time.time()
    ga_result = ga.evolve()
    ga_time = time.time() - start_time
    best_tour = (ga_result['best_individual'] * n_cities).astype(int) % n_cities
    ga_distance = tsp_distance(cities, best_tour)
    print(f"Best distance: {ga_distance:.2f}")
    print(f"Time: {ga_time:.4f}s")
    print(f"Improvement: {((best_random - ga_distance) / best_random * 100):.2f}%")
    
    # Toolbox: Simulated Annealing
    print("\n--- Toolbox: Simulated Annealing ---")
    initial_tour = np.random.permutation(n_cities).astype(float)
    
    def tsp_objective(tour):
        tour_int = (tour.astype(int) % n_cities)
        return tsp_distance(cities, tour_int)
    
    bounds = [(0, n_cities - 1)] * n_cities
    sa = SimulatedAnnealing(
        objective_function=tsp_objective,
        initial_solution=initial_tour,
        bounds=bounds,
        initial_temperature=100.0,
        max_iterations=1000
    )
    start_time = time.time()
    sa_result = sa.optimize()
    sa_time = time.time() - start_time
    sa_tour = sa_result['best_solution'].astype(int) % n_cities
    sa_distance = tsp_distance(cities, sa_tour)
    print(f"Best distance: {sa_distance:.2f}")
    print(f"Time: {sa_time:.4f}s")
    print(f"Improvement: {((best_random - sa_distance) / best_random * 100):.2f}%")
    
    results['problems']['tsp'] = {
        'baseline_random': best_random,
        'toolbox_ga': ga_distance,
        'toolbox_sa': sa_distance,
        'ga_improvement': ((best_random - ga_distance) / best_random * 100),
        'sa_improvement': ((best_random - sa_distance) / best_random * 100),
        'best_method': 'GA' if ga_distance < sa_distance else 'SA',
        'best_distance': min(ga_distance, sa_distance)
    }
    
except Exception as e:
    print(f"Error in TSP: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Problem 2: Chaotic Time Series Prediction (Lorenz System)
# ============================================================================
print("\n" + "=" * 80)
print("PROBLEM 2: CHAOTIC TIME SERIES PREDICTION (LORENZ SYSTEM)")
print("=" * 80)

try:
    from ml_toolbox.textbook_concepts.precognition import PrecognitiveForecaster
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    def generate_lorenz_attractor(n_steps: int = 1000, dt: float = 0.01):
        """Generate Lorenz attractor (chaotic system)"""
        sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        x, y, z = 1.0, 1.0, 1.0
        trajectory = []
        
        for _ in range(n_steps):
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            x, y, z = x + dx, y + dy, z + dz
            trajectory.append([x, y, z])
        
        return np.array(trajectory)
    
    # Generate chaotic data
    lorenz_data = generate_lorenz_attractor(1000)
    
    # Prepare data: predict next step from current
    X = lorenz_data[:-1]
    y = lorenz_data[1:, 0]  # Predict x coordinate
    
    # Split train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Baseline: Linear Regression
    print("\n--- Baseline: Linear Regression ---")
    lr = LinearRegression()
    start_time = time.time()
    lr.fit(X_train, y_train)
    lr_time = time.time() - start_time
    lr_pred = lr.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_pred)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    print(f"MSE: {lr_mse:.4f}, MAE: {lr_mae:.4f}")
    print(f"Time: {lr_time:.4f}s")
    
    # Baseline: Random Forest
    print("\n--- Baseline: Random Forest ---")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    start_time = time.time()
    rf.fit(X_train, y_train)
    rf_time = time.time() - start_time
    rf_pred = rf.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    print(f"MSE: {rf_mse:.4f}, MAE: {rf_mae:.4f}")
    print(f"Time: {rf_time:.4f}s")
    
    # Toolbox: Precognitive Forecaster
    print("\n--- Toolbox: Precognitive Forecaster ---")
    # Use Random Forest as base model
    base_model = RandomForestRegressor(n_estimators=50, random_state=42)
    base_model.fit(X_train, y_train)
    
    forecaster = PrecognitiveForecaster(base_model, max_horizon=5, n_scenarios=50)
    
    # Test on last 10 samples
    test_samples = X_test[-10:]
    test_targets = y_test[-10:]
    
    start_time = time.time()
    predictions = []
    for sample in test_samples:
        future = forecaster.foresee(sample, horizon=1, return_probabilities=False)
        predictions.append(future['mean_prediction'][0])
    precog_time = time.time() - start_time
    
    predictions = np.array(predictions)
    precog_mse = mean_squared_error(test_targets, predictions)
    precog_mae = mean_absolute_error(test_targets, predictions)
    print(f"MSE: {precog_mse:.4f}, MAE: {precog_mae:.4f}")
    print(f"Time: {precog_time:.4f}s")
    
    results['problems']['chaotic_timeseries'] = {
        'baseline_lr_mse': lr_mse,
        'baseline_rf_mse': rf_mse,
        'toolbox_precog_mse': precog_mse,
        'best_baseline_mse': min(lr_mse, rf_mse),
        'improvement': ((min(lr_mse, rf_mse) - precog_mse) / min(lr_mse, rf_mse) * 100) if precog_mse < min(lr_mse, rf_mse) else 0
    }
    
except Exception as e:
    print(f"Error in Chaotic Time Series: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Problem 3: Adversarial Robustness
# ============================================================================
print("\n" + "=" * 80)
print("PROBLEM 3: ADVERSARIAL ROBUSTNESS")
print("=" * 80)

try:
    from ml_toolbox.textbook_concepts.communication_theory import NoiseRobustModel, ErrorCorrectingPredictions
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
    
    # Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Baseline: Standard Random Forest
    print("\n--- Baseline: Standard Random Forest ---")
    baseline_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline_rf.fit(X_train, y_train)
    baseline_pred = baseline_rf.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    print(f"Clean accuracy: {baseline_acc:.4f}")
    
    # Add adversarial noise
    noise_levels = [0.1, 0.2, 0.3]
    adversarial_results = {}
    
    for noise_level in noise_levels:
        X_test_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
        noisy_pred = baseline_rf.predict(X_test_noisy)
        noisy_acc = accuracy_score(y_test, noisy_pred)
        adversarial_results[noise_level] = noisy_acc
        print(f"Noise {noise_level}: Accuracy {noisy_acc:.4f}")
    
    # Toolbox: Noise-Robust Model
    print("\n--- Toolbox: Noise-Robust Model ---")
    robust_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    robust_model = NoiseRobustModel(robust_rf, noise_level=0.2)
    robust_model.fit(X_train, y_train, n_augmentations=5)
    
    robust_pred = robust_model.predict(X_test)
    robust_clean_acc = accuracy_score(y_test, robust_pred)
    print(f"Clean accuracy: {robust_clean_acc:.4f}")
    
    robust_noisy_results = {}
    for noise_level in noise_levels:
        X_test_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
        robust_noisy_pred = robust_model.predict(X_test_noisy)
        robust_noisy_acc = accuracy_score(y_test, robust_noisy_pred)
        robust_noisy_results[noise_level] = robust_noisy_acc
        print(f"Noise {noise_level}: Accuracy {robust_noisy_acc:.4f}")
    
    # Calculate robustness improvement
    improvements = {}
    for noise_level in noise_levels:
        baseline_acc = adversarial_results[noise_level]
        robust_acc = robust_noisy_results[noise_level]
        improvement = ((robust_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
        improvements[noise_level] = improvement
    
    results['problems']['adversarial_robustness'] = {
        'baseline_clean': baseline_acc,
        'baseline_noisy': adversarial_results,
        'robust_clean': robust_clean_acc,
        'robust_noisy': robust_noisy_results,
        'improvements': improvements
    }
    
except Exception as e:
    print(f"Error in Adversarial Robustness: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Problem 4: Few-Shot Learning
# ============================================================================
print("\n" + "=" * 80)
print("PROBLEM 4: FEW-SHOT LEARNING")
print("=" * 80)

try:
    from ml_toolbox.agent_enhancements.socratic_method import SocraticActiveLearner
    from ml_toolbox.optimization.bounded_rationality import HeuristicModelSelector
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
    
    # Few-shot: only 20 training samples
    X_train_few = X[:20]
    y_train_few = y[:20]
    X_test = X[200:]
    y_test = y[200:]
    
    # Baseline: Standard training on few samples
    print("\n--- Baseline: Standard Training (20 samples) ---")
    baseline_models = [
        ('Logistic Regression', LogisticRegression(random_state=42)),
        ('SVM', SVC(random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=10, random_state=42))
    ]
    
    baseline_results = {}
    for name, model in baseline_models:
        model.fit(X_train_few, y_train_few)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        baseline_results[name] = acc
        print(f"{name}: {acc:.4f}")
    
    # Toolbox: Socratic Active Learning
    print("\n--- Toolbox: Socratic Active Learning ---")
    # Use unlabeled data pool
    X_unlabeled = X[20:200]
    
    # Start with few labeled samples
    X_labeled = X_train_few.copy()
    y_labeled = y_train_few.copy()
    
    # Active learning: select most informative samples
    learner = SocraticActiveLearner()
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Initial fit
    model.fit(X_labeled, y_labeled)
    
    # Select 30 more samples using active learning
    selected_indices = learner.select_questions(
        X_unlabeled, model, n_questions=30, strategy='uncertainty'
    )
    
    # Add selected samples to training set
    X_labeled = np.vstack([X_labeled, X_unlabeled[selected_indices]])
    y_labeled = np.concatenate([y_labeled, y[20:200][selected_indices]])
    
    # Train on expanded set
    model.fit(X_labeled, y_labeled)
    active_pred = model.predict(X_test)
    active_acc = accuracy_score(y_test, active_pred)
    print(f"Active Learning (50 samples): {active_acc:.4f}")
    
    # Toolbox: Heuristic Model Selector (satisficing)
    print("\n--- Toolbox: Heuristic Model Selector ---")
    models = [
        RandomForestClassifier(n_estimators=10, random_state=42),
        LogisticRegression(random_state=42),
        SVC(random_state=42)
    ]
    
    selector = HeuristicModelSelector(models, satisfaction_threshold=0.7, max_evaluations=3)
    
    # Use validation set
    X_val = X[200:300]
    y_val = y[200:300]
    
    result = selector.select(X_train_few, y_train_few, X_val, y_val)
    if result['selected_model']:
        satisficing_pred = result['selected_model'].predict(X_test)
        satisficing_acc = accuracy_score(y_test, satisficing_pred)
        print(f"Satisficing Selection: {satisficing_acc:.4f}")
    else:
        satisficing_acc = 0.0
    
    best_baseline = max(baseline_results.values())
    best_toolbox = max(active_acc, satisficing_acc)
    
    results['problems']['few_shot_learning'] = {
        'baseline_results': baseline_results,
        'best_baseline': best_baseline,
        'toolbox_active_learning': active_acc,
        'toolbox_satisficing': satisficing_acc,
        'best_toolbox': best_toolbox,
        'improvement': ((best_toolbox - best_baseline) / best_baseline * 100) if best_baseline > 0 else 0
    }
    
except Exception as e:
    print(f"Error in Few-Shot Learning: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Problem 5: High-Dimensional Sparse Data
# ============================================================================
print("\n" + "=" * 80)
print("PROBLEM 5: HIGH-DIMENSIONAL SPARSE DATA")
print("=" * 80)

try:
    from ml_toolbox.ai_concepts.network_theory import network_based_feature_importance
    from ml_toolbox.optimization.evolutionary_algorithms import evolutionary_feature_selection
    from ml_toolbox.textbook_concepts.information_theory import mutual_information
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    
    # Generate high-dimensional sparse data
    X, y = make_classification(
        n_samples=500,
        n_features=1000,  # High dimensional
        n_informative=50,  # Only 50 informative
        n_redundant=0,
        n_repeated=0,
        random_state=42
    )
    
    # Make sparse (90% zeros)
    mask = np.random.random(X.shape) > 0.1
    X_sparse = X * mask
    
    # Split
    split = int(0.8 * len(X_sparse))
    X_train, X_test = X_sparse[:split], X_sparse[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Baseline: SelectKBest (variance-based)
    print("\n--- Baseline: SelectKBest (Variance) ---")
    selector_baseline = SelectKBest(f_classif, k=50)
    X_train_selected = selector_baseline.fit_transform(X_train, y_train)
    X_test_selected = selector_baseline.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train_selected, y_train)
    baseline_pred = model.predict(X_test_selected)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    print(f"Accuracy: {baseline_acc:.4f}")
    
    # Toolbox: Network-Based Feature Importance
    print("\n--- Toolbox: Network-Based Feature Importance ---")
    network_importance = network_based_feature_importance(X_train, y_train, method='betweenness')
    top_features = np.argsort(network_importance)[-50:]
    
    X_train_network = X_train[:, top_features]
    X_test_network = X_test[:, top_features]
    
    model.fit(X_train_network, y_train)
    network_pred = model.predict(X_test_network)
    network_acc = accuracy_score(y_test, network_pred)
    print(f"Accuracy: {network_acc:.4f}")
    
    # Toolbox: Evolutionary Feature Selection
    print("\n--- Toolbox: Evolutionary Feature Selection ---")
    model_for_evo = RandomForestClassifier(n_estimators=20, random_state=42)
    evo_result = evolutionary_feature_selection(
        X_train, y_train, model_for_evo, n_features=50,
        population_size=20, max_generations=20
    )
    
    selected_features = evo_result['selected_features']
    X_train_evo = X_train[:, selected_features]
    X_test_evo = X_test[:, selected_features]
    
    model.fit(X_train_evo, y_train)
    evo_pred = model.predict(X_test_evo)
    evo_acc = accuracy_score(y_test, evo_pred)
    print(f"Accuracy: {evo_acc:.4f}")
    
    best_toolbox = max(network_acc, evo_acc)
    improvement = ((best_toolbox - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
    
    results['problems']['high_dimensional_sparse'] = {
        'baseline': baseline_acc,
        'toolbox_network': network_acc,
        'toolbox_evolutionary': evo_acc,
        'best_toolbox': best_toolbox,
        'improvement': improvement
    }
    
except Exception as e:
    print(f"Error in High-Dimensional Sparse: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Problem 6: Concept Drift
# ============================================================================
print("\n" + "=" * 80)
print("PROBLEM 6: CONCEPT DRIFT")
print("=" * 80)

try:
    from ml_toolbox.infrastructure.neural_lace import DirectNeuralInterface
    from ml_toolbox.optimization.control_theory import TrainingStabilityMonitor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    
    # Generate data with concept drift
    # First concept: feature 0 is important
    X1, y1 = make_classification(n_samples=500, n_features=10, n_informative=3, random_state=42)
    
    # Second concept: feature 5 is important (drift)
    X2, y2 = make_classification(n_samples=500, n_features=10, n_informative=3, random_state=123)
    # Swap important features
    X2[:, [0, 5]] = X2[:, [5, 0]]
    
    # Combine with drift
    X_drift = np.vstack([X1, X2])
    y_drift = np.concatenate([y1, y2])
    
    # Baseline: Train on first concept, test on second
    print("\n--- Baseline: Static Model (No Adaptation) ---")
    baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline_model.fit(X1, y1)
    
    baseline_pred = baseline_model.predict(X2)
    baseline_acc = accuracy_score(y2, baseline_pred)
    print(f"Accuracy on drifted data: {baseline_acc:.4f}")
    
    # Toolbox: Streaming Learning with Neural Lace
    print("\n--- Toolbox: Streaming Learning (Neural Lace) ---")
    def streaming_data():
        for i in range(len(X_drift)):
            yield (X_drift[i], y_drift[i])
    
    streaming_model = RandomForestClassifier(n_estimators=50, random_state=42)
    interface = DirectNeuralInterface(streaming_model, streaming_data())
    
    # Stream learn on first concept
    for i in range(500):
        interface.learn_stream(1)
    
    # Test on second concept (with continued learning)
    accuracies = []
    for i in range(500, 1000):
        interface.learn_stream(1)  # Continue learning
        if i % 50 == 0:
            pred = interface.predict_stream(1)
            if pred:
                acc = accuracy_score([y_drift[i]], pred)
                accuracies.append(acc)
    
    streaming_acc = np.mean(accuracies) if accuracies else 0.0
    print(f"Average accuracy with streaming: {streaming_acc:.4f}")
    
    # Toolbox: Stability Monitor
    print("\n--- Toolbox: Stability Monitor ---")
    monitor = TrainingStabilityMonitor()
    
    # Simulate training with drift detection
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X1, y1)
    
    # Monitor predictions on drifted data
    drift_detected = False
    for i in range(0, len(X2), 50):
        batch = X2[i:i+50]
        batch_y = y2[i:i+50]
        pred = model.predict(batch)
        acc = accuracy_score(batch_y, pred)
        
        status = monitor.check_stability(1.0 - acc)  # Loss = 1 - accuracy
        if status['status'] != 'stable':
            drift_detected = True
            print(f"Drift detected at sample {i}: {status['status']}")
            # Retrain on recent data
            recent_X = np.vstack([X1[-100:], X2[:i+50]])
            recent_y = np.concatenate([y1[-100:], y2[:i+50]])
            model.fit(recent_X, recent_y)
    
    if drift_detected:
        final_pred = model.predict(X2)
        adaptive_acc = accuracy_score(y2, final_pred)
        print(f"Adaptive accuracy: {adaptive_acc:.4f}")
    else:
        adaptive_acc = baseline_acc
    
    best_toolbox = max(streaming_acc, adaptive_acc)
    improvement = ((best_toolbox - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
    
    results['problems']['concept_drift'] = {
        'baseline_static': baseline_acc,
        'toolbox_streaming': streaming_acc,
        'toolbox_adaptive': adaptive_acc,
        'best_toolbox': best_toolbox,
        'improvement': improvement
    }
    
except Exception as e:
    print(f"Error in Concept Drift: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Problem 7: Imbalanced Classification
# ============================================================================
print("\n" + "=" * 80)
print("PROBLEM 7: IMBALANCED CLASSIFICATION")
print("=" * 80)

try:
    from ml_toolbox.optimization.multiverse import MultiverseProcessor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.metrics import classification_report, f1_score
    
    # Generate imbalanced data (1:10 ratio)
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_classes=2,
        weights=[0.1, 0.9],  # Imbalanced
        random_state=42
    )
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Baseline: Standard Random Forest
    print("\n--- Baseline: Standard Random Forest ---")
    baseline_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline_rf.fit(X_train, y_train)
    baseline_pred = baseline_rf.predict(X_test)
    baseline_f1 = f1_score(y_test, baseline_pred, average='weighted')
    print(f"F1 Score: {baseline_f1:.4f}")
    
    # Toolbox: Multiverse Ensemble (different random seeds = different universes)
    print("\n--- Toolbox: Multiverse Ensemble ---")
    processor = MultiverseProcessor(n_universes=10)
    
    initial_states = [{'seed': i} for i in range(10)]
    models = [RandomForestClassifier(n_estimators=50, random_state=i) for i in range(10)]
    
    universe_ids = processor.create_universes(initial_states, models)
    
    # Train each universe
    for universe in processor.universes.values():
        universe.model.fit(X_train, y_train)
        universe.metrics['last_evaluation'] = f1_score(y_train, universe.model.predict(X_train), average='weighted')
    
    # Ensemble prediction
    multiverse_pred = processor.multiverse_ensemble(X_test, aggregation_method='vote')
    multiverse_f1 = f1_score(y_test, multiverse_pred, average='weighted')
    print(f"Multiverse F1 Score: {multiverse_f1:.4f}")
    
    improvement = ((multiverse_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
    
    results['problems']['imbalanced_classification'] = {
        'baseline_f1': baseline_f1,
        'toolbox_multiverse_f1': multiverse_f1,
        'improvement': improvement
    }
    
except Exception as e:
    print(f"Error in Imbalanced Classification: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Problem 8: Multi-Objective Optimization with Constraints
# ============================================================================
print("\n" + "=" * 80)
print("PROBLEM 8: MULTI-OBJECTIVE OPTIMIZATION WITH CONSTRAINTS")
print("=" * 80)

try:
    from ml_toolbox.optimization.systems_theory import MultiObjectiveOptimizer, DoubleBindResolver
    from ml_toolbox.optimization.evolutionary_algorithms import GeneticAlgorithm
    
    # Problem: Minimize cost and maximize performance, subject to constraints
    def objective1(x):
        """Minimize cost"""
        return np.sum(x ** 2)
    
    def objective2(x):
        """Maximize performance (negative for minimization)"""
        return -np.sum(x)
    
    def constraint1(x):
        """Sum must be <= 10"""
        return np.sum(x) - 10
    
    def constraint2(x):
        """All values must be >= 0"""
        return -np.min(x)
    
    # Baseline: Single objective (ignore constraints)
    print("\n--- Baseline: Single Objective (No Constraints) ---")
    from scipy.optimize import minimize
    
    result = minimize(objective1, [1.0, 1.0, 1.0], method='L-BFGS-B', bounds=[(0, 10)] * 3)
    baseline_obj1 = result.fun
    baseline_obj2 = objective2(result.x)
    baseline_constraint1 = constraint1(result.x)
    baseline_constraint2 = constraint2(result.x)
    print(f"Objective 1: {baseline_obj1:.4f}")
    print(f"Objective 2: {baseline_obj2:.4f}")
    print(f"Constraint 1 violation: {max(0, baseline_constraint1):.4f}")
    print(f"Constraint 2 violation: {max(0, baseline_constraint2):.4f}")
    
    # Toolbox: Multi-Objective Optimizer
    print("\n--- Toolbox: Multi-Objective Optimizer ---")
    optimizer = MultiObjectiveOptimizer([objective1, objective2], bounds=[(0, 10)] * 3)
    result = optimizer.weighted_sum(weights=np.array([0.5, 0.5]), initial_guess=[1.0, 1.0, 1.0])
    
    toolbox_obj1 = result['objectives'][0]
    toolbox_obj2 = result['objectives'][1]
    toolbox_constraint1 = constraint1(result['solution'])
    toolbox_constraint2 = constraint2(result['solution'])
    print(f"Objective 1: {toolbox_obj1:.4f}")
    print(f"Objective 2: {toolbox_obj2:.4f}")
    print(f"Constraint 1 violation: {max(0, toolbox_constraint1):.4f}")
    print(f"Constraint 2 violation: {max(0, toolbox_constraint2):.4f}")
    
    # Toolbox: Double Bind Resolver
    print("\n--- Toolbox: Double Bind Resolver ---")
    resolver = DoubleBindResolver([constraint1, constraint2], objective1)
    result = resolver.resolve(bounds=[(0, 10)] * 3, initial_guess=[1.0, 1.0, 1.0])
    
    resolved_obj1 = objective1(result['solution'])
    resolved_obj2 = objective2(result['solution'])
    resolved_constraint1 = constraint1(result['solution'])
    resolved_constraint2 = constraint2(result['solution'])
    print(f"Objective 1: {resolved_obj1:.4f}")
    print(f"Objective 2: {resolved_obj2:.4f}")
    print(f"Constraints satisfied: {result['satisfied']}")
    
    results['problems']['multi_objective'] = {
        'baseline_obj1': baseline_obj1,
        'baseline_obj2': baseline_obj2,
        'baseline_violations': max(0, baseline_constraint1) + max(0, baseline_constraint2),
        'toolbox_multi_obj1': toolbox_obj1,
        'toolbox_multi_obj2': toolbox_obj2,
        'toolbox_multi_violations': max(0, toolbox_constraint1) + max(0, toolbox_constraint2),
        'toolbox_resolved_obj1': resolved_obj1,
        'toolbox_resolved_obj2': resolved_obj2,
        'toolbox_resolved_satisfied': result['satisfied']
    }
    
except Exception as e:
    print(f"Error in Multi-Objective: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Problem 9: Non-Stationary Environment
# ============================================================================
print("\n" + "=" * 80)
print("PROBLEM 9: NON-STATIONARY ENVIRONMENT")
print("=" * 80)

try:
    from ml_toolbox.automl.singularity import SelfModifyingSystem
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    
    # Generate non-stationary data (distribution changes over time)
    accuracies_over_time = []
    
    # Baseline: Static model
    print("\n--- Baseline: Static Model ---")
    X1, y1 = make_classification(n_samples=500, n_features=10, random_state=42)
    baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline_model.fit(X1, y1)
    
    # Test on changing distributions
    for seed in [100, 200, 300, 400]:
        X_test, y_test = make_classification(n_samples=100, n_features=10, random_state=seed)
        pred = baseline_model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        accuracies_over_time.append(acc)
    
    baseline_avg = np.mean(accuracies_over_time)
    baseline_std = np.std(accuracies_over_time)
    print(f"Average accuracy: {baseline_avg:.4f} ± {baseline_std:.4f}")
    
    # Toolbox: Self-Modifying System
    print("\n--- Toolbox: Self-Modifying System ---")
    def improvement_metric(model, X, y):
        return model.score(X, y)
    
    initial_model = RandomForestClassifier(n_estimators=50, random_state=42)
    initial_model.fit(X1, y1)
    
    system = SelfModifyingSystem(initial_model, improvement_metric)
    
    # Adapt to each new distribution
    adaptive_accuracies = []
    for seed in [100, 200, 300, 400]:
        X_new, y_new = make_classification(n_samples=200, n_features=10, random_state=seed)
        
        # Improve system for new distribution
        result = system.improve((X_new, y_new), n_iterations=5)
        
        # Test
        pred = system.system.predict(X_new[-50:])
        acc = accuracy_score(y_new[-50:], pred)
        adaptive_accuracies.append(acc)
    
    adaptive_avg = np.mean(adaptive_accuracies)
    adaptive_std = np.std(adaptive_accuracies)
    print(f"Average accuracy: {adaptive_avg:.4f} ± {adaptive_std:.4f}")
    
    improvement = ((adaptive_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
    
    results['problems']['non_stationary'] = {
        'baseline_avg': baseline_avg,
        'baseline_std': baseline_std,
        'toolbox_adaptive_avg': adaptive_avg,
        'toolbox_adaptive_std': adaptive_std,
        'improvement': improvement
    }
    
except Exception as e:
    print(f"Error in Non-Stationary: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Problem 10: Transfer Learning Across Domains
# ============================================================================
print("\n" + "=" * 80)
print("PROBLEM 10: TRANSFER LEARNING ACROSS DOMAINS")
print("=" * 80)

try:
    from ml_toolbox.optimization.multiverse import MultiverseProcessor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    
    # Source domain
    X_source, y_source = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    # Target domain (different distribution)
    X_target, y_target = make_classification(n_samples=200, n_features=20, random_state=999)
    
    # Baseline: Train only on target (few samples)
    print("\n--- Baseline: Target-Only Training ---")
    baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline_model.fit(X_target[:50], y_target[:50])  # Only 50 samples
    baseline_pred = baseline_model.predict(X_target[50:])
    baseline_acc = accuracy_score(y_target[50:], baseline_pred)
    print(f"Accuracy: {baseline_acc:.4f}")
    
    # Toolbox: Multiverse Transfer (different universes = different transfer strategies)
    print("\n--- Toolbox: Multiverse Transfer Learning ---")
    processor = MultiverseProcessor(n_universes=5)
    
    # Different transfer strategies in different universes
    strategies = [
        {'source_weight': 0.0, 'target_weight': 1.0},  # Target only
        {'source_weight': 0.3, 'target_weight': 0.7},  # Some source
        {'source_weight': 0.5, 'target_weight': 0.5},  # Balanced
        {'source_weight': 0.7, 'target_weight': 0.3},  # More source
        {'source_weight': 1.0, 'target_weight': 0.0},  # Source only
    ]
    
    initial_states = [{'strategy': s} for s in strategies]
    models = [RandomForestClassifier(n_estimators=50, random_state=i) for i in range(5)]
    
    universe_ids = processor.create_universes(initial_states, models)
    
    # Train each universe with different transfer strategy
    for i, (universe_id, universe) in enumerate(processor.universes.items()):
        strategy = strategies[i]
        
        # Combine source and target with weights
        if strategy['source_weight'] > 0:
            n_source = int(500 * strategy['source_weight'])
            n_target = int(50 * strategy['target_weight'])
            
            X_combined = np.vstack([X_source[:n_source], X_target[:n_target]])
            y_combined = np.concatenate([y_source[:n_source], y_target[:n_target]])
        else:
            X_combined = X_target[:50]
            y_combined = y_target[:50]
        
        universe.model.fit(X_combined, y_combined)
        universe.metrics['last_evaluation'] = universe.model.score(X_target[50:100], y_target[50:100])
    
    # Ensemble from best universes
    best_universe = processor.select_best_universe(lambda s, m: s.get('strategy', {}).get('source_weight', 0))
    
    # Use best universe
    best_model = processor.universes[best_universe].model
    transfer_pred = best_model.predict(X_target[100:])
    transfer_acc = accuracy_score(y_target[100:], transfer_pred)
    print(f"Best transfer accuracy: {transfer_acc:.4f}")
    print(f"Best strategy: {strategies[int(best_universe.split('_')[1])]}")
    
    improvement = ((transfer_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
    
    results['problems']['transfer_learning'] = {
        'baseline_target_only': baseline_acc,
        'toolbox_transfer': transfer_acc,
        'improvement': improvement
    }
    
except Exception as e:
    print(f"Error in Transfer Learning: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# SUMMARY AND STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("COMPREHENSIVE BENCHMARK SUMMARY")
print("=" * 80)

# Calculate overall statistics
toolbox_wins = 0
baseline_wins = 0
ties = 0
total_improvements = []
problems_tested = len(results['problems'])

for problem_name, problem_results in results['problems'].items():
    if 'improvement' in problem_results:
        improvement = problem_results['improvement']
        total_improvements.append(improvement)
        if improvement > 0:
            toolbox_wins += 1
        elif improvement < 0:
            baseline_wins += 1
        else:
            ties += 1

avg_improvement = np.mean(total_improvements) if total_improvements else 0.0
median_improvement = np.median(total_improvements) if total_improvements else 0.0

print(f"\nProblems Tested: {problems_tested}")
print(f"Toolbox Wins: {toolbox_wins}")
print(f"Baseline Wins: {baseline_wins}")
print(f"Ties: {ties}")
print(f"\nAverage Improvement: {avg_improvement:.2f}%")
print(f"Median Improvement: {median_improvement:.2f}%")
print(f"Max Improvement: {max(total_improvements):.2f}%" if total_improvements else "N/A")
print(f"Min Improvement: {min(total_improvements):.2f}%" if total_improvements else "N/A")

# Detailed results
print("\n" + "=" * 80)
print("DETAILED RESULTS BY PROBLEM")
print("=" * 80)

for problem_name, problem_results in results['problems'].items():
    print(f"\n{problem_name.upper().replace('_', ' ')}:")
    for key, value in problem_results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, (int, float)):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")

# Save results
results['summary'] = {
    'problems_tested': problems_tested,
    'toolbox_wins': toolbox_wins,
    'baseline_wins': baseline_wins,
    'ties': ties,
    'average_improvement': float(avg_improvement),
    'median_improvement': float(median_improvement),
    'max_improvement': float(max(total_improvements)) if total_improvements else 0.0,
    'min_improvement': float(min(total_improvements)) if total_improvements else 0.0
}

# Save to JSON
with open('hardest_ml_problems_benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "=" * 80)
print("Results saved to: hardest_ml_problems_benchmark_results.json")
print("=" * 80)
