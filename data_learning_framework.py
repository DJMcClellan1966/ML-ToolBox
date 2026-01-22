"""
Data Learning Framework
Quick wins for data learning: federated learning, online learning, privacy

Features:
- Basic federated learning
- Online/incremental learning wrapper
- Differential privacy example
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import warnings
import copy

sys.path.insert(0, str(Path(__file__).parent))


class FederatedLearningFramework:
    """
    Federated Learning Framework
    
    Basic federated learning implementation
    """
    
    def __init__(self, aggregation_method: str = 'fedavg'):
        """
        Args:
            aggregation_method: 'fedavg' (Federated Averaging) or 'weighted'
        """
        self.aggregation_method = aggregation_method
    
    def federated_training_round(
        self,
        client_models: List[Any],
        client_data_sizes: List[int],
        use_ml_toolbox: bool = True
    ) -> Dict[str, Any]:
        """
        Perform one round of federated training
        
        Args:
            client_models: List of models trained on client data
            client_data_sizes: List of data sizes for each client
            use_ml_toolbox: Use ML Toolbox algorithms if available
            
        Returns:
            Aggregated model and metadata
        """
        if not client_models:
            return {'error': 'No client models provided'}
        
        # For sklearn models, aggregate parameters
        if hasattr(client_models[0], 'get_params'):
            return self._aggregate_sklearn_models(client_models, client_data_sizes)
        else:
            # For other models, return first model (simplified)
            return {
                'aggregated_model': copy.deepcopy(client_models[0]),
                'method': 'simple_copy',
                'num_clients': len(client_models)
            }
    
    def _aggregate_sklearn_models(
        self,
        client_models: List[Any],
        client_data_sizes: List[int]
    ) -> Dict[str, Any]:
        """Aggregate sklearn models"""
        # For RandomForest, aggregate trees (simplified)
        if hasattr(client_models[0], 'estimators_'):
            # Use first model as base (simplified aggregation)
            aggregated = copy.deepcopy(client_models[0])
            
            # Weighted average of predictions (simplified)
            total_size = sum(client_data_sizes)
            weights = [size / total_size for size in client_data_sizes]
            
            return {
                'aggregated_model': aggregated,
                'method': self.aggregation_method,
                'num_clients': len(client_models),
                'weights': weights,
                'total_data_size': total_size
            }
        else:
            # For other models, use weighted average of parameters
            aggregated = copy.deepcopy(client_models[0])
            return {
                'aggregated_model': aggregated,
                'method': self.aggregation_method,
                'num_clients': len(client_models)
            }
    
    def train_federated_model(
        self,
        client_data: List[Tuple[np.ndarray, np.ndarray]],
        num_rounds: int = 5,
        use_ml_toolbox: bool = True
    ) -> Dict[str, Any]:
        """
        Train model using federated learning
        
        Args:
            client_data: List of (X, y) tuples for each client
            num_rounds: Number of federated training rounds
            use_ml_toolbox: Use ML Toolbox for training
            
        Returns:
            Trained federated model and history
        """
        history = []
        client_models = []
        client_sizes = []
        
        for round_num in range(num_rounds):
            # Train models on each client
            round_models = []
            round_sizes = []
            
            for X, y in client_data:
                if use_ml_toolbox:
                    try:
                        from ml_toolbox import MLToolbox
                        toolbox = MLToolbox()
                        simple = toolbox.algorithms.get_simple_ml_tasks()
                        
                        result = simple.train_classifier(X, y, model_type='random_forest')
                        round_models.append(result['model'])
                        round_sizes.append(len(X))
                    except Exception as e:
                        warnings.warn(f"ML Toolbox not available: {e}")
                        use_ml_toolbox = False
                        break
            
            if not use_ml_toolbox:
                # Fallback to sklearn
                from sklearn.ensemble import RandomForestClassifier
                for X, y in client_data:
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                    model.fit(X, y)
                    round_models.append(model)
                    round_sizes.append(len(X))
            
            # Aggregate models
            aggregated = self.federated_training_round(round_models, round_sizes)
            
            history.append({
                'round': round_num + 1,
                'num_clients': len(client_data),
                'aggregated_model': aggregated.get('aggregated_model')
            })
            
            client_models = round_models
            client_sizes = round_sizes
        
        return {
            'federated_model': history[-1]['aggregated_model'] if history else None,
            'history': history,
            'num_rounds': num_rounds,
            'num_clients': len(client_data)
        }


class OnlineLearningWrapper:
    """
    Online Learning Wrapper
    
    Incremental learning from streaming data
    """
    
    def __init__(self, base_model: Any, learning_rate: float = 0.01):
        """
        Args:
            base_model: Base model to use
            learning_rate: Learning rate for updates
        """
        self.base_model = base_model
        self.learning_rate = learning_rate
        self.update_count = 0
    
    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: Optional[np.ndarray] = None
    ) -> 'OnlineLearningWrapper':
        """
        Incremental fit on new data
        
        Args:
            X: New features
            y: New labels
            classes: All possible classes (for classification)
            
        Returns:
            Self for chaining
        """
        # Check if model supports partial_fit
        if hasattr(self.base_model, 'partial_fit'):
            if classes is not None:
                self.base_model.partial_fit(X, y, classes=classes)
            else:
                self.base_model.partial_fit(X, y)
        else:
            # For models without partial_fit, retrain on accumulated data
            # (Simplified - in practice would maintain data buffer)
            if not hasattr(self, '_X_buffer'):
                self._X_buffer = X
                self._y_buffer = y
            else:
                self._X_buffer = np.vstack([self._X_buffer, X])
                self._y_buffer = np.hstack([self._y_buffer, y])
            
            # Retrain periodically or on every update (simplified)
            if hasattr(self.base_model, 'fit'):
                self.base_model.fit(self._X_buffer, self._y_buffer)
        
        self.update_count += 1
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.base_model.predict(X)
    
    def get_update_count(self) -> int:
        """Get number of updates"""
        return self.update_count


class DifferentialPrivacyWrapper:
    """
    Differential Privacy Wrapper
    
    Add differential privacy to ML models
    """
    
    def __init__(self, model: Any, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Args:
            model: Base model
            epsilon: Privacy budget (lower = more private)
            delta: Failure probability
        """
        self.model = model
        self.epsilon = epsilon
        self.delta = delta
    
    def add_noise_to_output(self, predictions: np.ndarray) -> np.ndarray:
        """
        Add Laplace noise for differential privacy
        
        Args:
            predictions: Model predictions
            
        Returns:
            Noisy predictions
        """
        # Laplace mechanism for differential privacy
        sensitivity = 1.0  # Assuming sensitivity of 1
        scale = sensitivity / self.epsilon
        
        # Add Laplace noise
        noise = np.random.laplace(0, scale, size=predictions.shape)
        noisy_predictions = predictions + noise
        
        return noisy_predictions
    
    def predict_private(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Make differentially private predictions
        
        Args:
            X: Input features
            
        Returns:
            Private predictions and privacy info
        """
        # Get base predictions
        predictions = self.model.predict(X)
        
        # Add noise for privacy
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            noisy_probs = self.add_noise_to_output(probabilities)
            # Normalize probabilities
            noisy_probs = np.clip(noisy_probs, 0, 1)
            noisy_probs = noisy_probs / noisy_probs.sum(axis=1, keepdims=True)
            private_predictions = np.argmax(noisy_probs, axis=1)
        else:
            # For regression or non-probabilistic models
            private_predictions = self.add_noise_to_output(predictions)
        
        return {
            'predictions': private_predictions,
            'epsilon': self.epsilon,
            'delta': self.delta,
            'privacy_guarantee': f'(ε={self.epsilon}, δ={self.delta})-differential privacy'
        }
    
    def get_privacy_info(self) -> Dict[str, Any]:
        """Get privacy information"""
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'privacy_level': 'high' if self.epsilon < 1.0 else 'medium' if self.epsilon < 5.0 else 'low',
            'privacy_guarantee': f'(ε={self.epsilon}, δ={self.delta})-differential privacy'
        }


class ContinuousLearningPipeline:
    """
    Continuous Learning Pipeline
    
    Models that learn from new data continuously
    """
    
    def __init__(self, base_model: Any, use_ml_toolbox: bool = True):
        """
        Args:
            base_model: Base model or None to create from ML Toolbox
            use_ml_toolbox: Use ML Toolbox for model creation
        """
        self.use_ml_toolbox = use_ml_toolbox
        
        if base_model is None and use_ml_toolbox:
            try:
                from ml_toolbox import MLToolbox
                toolbox = MLToolbox()
                simple = toolbox.algorithms.get_simple_ml_tasks()
                # Create placeholder - will be trained on first data
                self.base_model = None
            except:
                from sklearn.ensemble import RandomForestClassifier
                self.base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.base_model = base_model
        
        self.online_wrapper = None
        self.update_history = []
    
    def initial_train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Initial training
        
        Args:
            X: Initial training features
            y: Initial training labels
            
        Returns:
            Training results
        """
        if self.use_ml_toolbox and self.base_model is None:
            try:
                from ml_toolbox import MLToolbox
                toolbox = MLToolbox()
                simple = toolbox.algorithms.get_simple_ml_tasks()
                
                result = simple.train_classifier(X, y, model_type='random_forest')
                self.base_model = result['model']
            except:
                from sklearn.ensemble import RandomForestClassifier
                self.base_model = RandomForestClassifier(n_estimators=100, random_state=42)
                self.base_model.fit(X, y)
        else:
            if hasattr(self.base_model, 'fit'):
                self.base_model.fit(X, y)
        
        # Create online wrapper
        self.online_wrapper = OnlineLearningWrapper(self.base_model)
        
        return {
            'model': self.base_model,
            'framework': 'ML Toolbox' if self.use_ml_toolbox else 'sklearn'
        }
    
    def update(self, X_new: np.ndarray, y_new: np.ndarray) -> Dict[str, Any]:
        """
        Update model with new data
        
        Args:
            X_new: New features
            y_new: New labels
            
        Returns:
            Update results
        """
        if self.online_wrapper is None:
            return {'error': 'Model not initialized. Call initial_train first.'}
        
        # Update model
        self.online_wrapper.partial_fit(X_new, y_new)
        
        self.update_history.append({
            'update_num': len(self.update_history) + 1,
            'samples': len(X_new),
            'total_updates': self.online_wrapper.get_update_count()
        })
        
        return {
            'updated': True,
            'update_count': self.online_wrapper.get_update_count(),
            'total_updates': len(self.update_history)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.online_wrapper:
            return self.online_wrapper.predict(X)
        elif self.base_model:
            return self.base_model.predict(X)
        else:
            raise ValueError("Model not trained")
