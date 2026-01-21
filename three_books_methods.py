"""
Key Methods from Three Influential ML Books

1. Elements of Statistical Learning (Hastie, Tibshirani, Friedman)
2. Pattern Recognition and Machine Learning (Bishop)
3. Deep Learning (Goodfellow, Bengio, Courville)

This module implements key methods from these books that complement
the existing ML Toolbox capabilities.
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Dependencies tracking
REQUIRED_DEPENDENCIES = {
    'sklearn': 'scikit-learn>=1.5.0',
    'scipy': 'scipy>=1.11.0',
    'numpy': 'numpy>=1.26.0',
    'pandas': 'pandas>=2.0.0',
    'statsmodels': 'statsmodels>=0.14.0',
    'matplotlib': 'matplotlib>=3.7.0',
    'seaborn': 'seaborn>=0.12.0',
    'shap': 'shap>=0.42.0',
    'lime': 'lime>=0.2.0',
    'pgmpy': 'pgmpy>=0.1.19',
    'hmmlearn': 'hmmlearn>=0.2.7',
    'imbalanced-learn': 'imbalanced-learn>=0.11.0',
    'scikit-optimize': 'scikit-optimize>=0.9.0'
}

# Try to import required libraries
try:
    from sklearn.svm import SVC, SVR
    from sklearn.linear_model import Lasso, Ridge, ElasticNet, LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")

try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Install with: pip install pandas")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
    GAUSSIAN_PROCESS_AVAILABLE = True
except ImportError:
    GAUSSIAN_PROCESS_AVAILABLE = False
    print("Warning: sklearn Gaussian Process not available")


class ESLMethods:
    """
    Methods from Elements of Statistical Learning (Hastie, Tibshirani, Friedman)
    
    Key methods:
    - Support Vector Machines (SVM)
    - Boosting variants (AdaBoost, Gradient Boosting)
    - Lasso, Ridge, Elastic Net (already have, but can enhance)
    - Random Forests (already have)
    - Generalized Additive Models (GAM)
    """
    
    def __init__(self):
        self.dependencies = ['sklearn', 'scipy', 'numpy']
    
    def support_vector_machine(
        self,
        X: np.ndarray,
        y: np.ndarray,
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: str = 'scale',
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Support Vector Machine (ESL Chapter 12)
        
        Args:
            X: Features
            y: Labels
            kernel: 'linear', 'poly', 'rbf', 'sigmoid'
            C: Regularization parameter
            gamma: Kernel coefficient
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with fitted model and information
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        if task_type == 'classification':
            model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        else:
            model = SVR(kernel=kernel, C=C, gamma=gamma)
        
        model.fit(X, y)
        
        return {
            'model': model,
            'kernel': kernel,
            'C': C,
            'gamma': gamma,
            'task_type': task_type,
            'n_support_vectors': len(model.support_vectors_) if hasattr(model, 'support_vectors_') else 0,
            'support_vectors': model.support_vectors_.tolist() if hasattr(model, 'support_vectors_') else []
        }
    
    def adaboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        AdaBoost (ESL Chapter 10)
        
        Adaptive Boosting algorithm
        
        Args:
            X: Features
            y: Labels
            n_estimators: Number of weak learners
            learning_rate: Learning rate
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with fitted model
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
        
        base_estimator = DecisionTreeClassifier(max_depth=1) if task_type == 'classification' else DecisionTreeRegressor(max_depth=1)
        
        if task_type == 'classification':
            model = AdaBoostClassifier(
                base_estimator=base_estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate
            )
        else:
            model = AdaBoostRegressor(
                base_estimator=base_estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate
            )
        
        model.fit(X, y)
        
        return {
            'model': model,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'feature_importances': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else [],
            'estimator_weights': model.estimator_weights_.tolist() if hasattr(model, 'estimator_weights_') else []
        }
    
    def generalized_additive_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = 'regression'
    ) -> Dict[str, Any]:
        """
        Generalized Additive Model (ESL Chapter 9)
        
        GAM: g(E[Y]) = f1(X1) + f2(X2) + ... + fn(Xn)
        
        Simplified implementation using splines
        
        Args:
            X: Features
            y: Labels
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with model information
        """
        if not SKLEARN_AVAILABLE or not SCIPY_AVAILABLE:
            return {'error': 'sklearn and scipy required for GAM'}
        
        # Simplified GAM using splines
        # In practice, would use pygam library
        warnings.warn("GAM implementation is simplified. For full GAM, use pygam library.")
        
        # Use additive model approximation with splines
        from sklearn.preprocessing import SplineTransformer
        from sklearn.linear_model import LinearRegression, LogisticRegression
        
        # Transform features with splines
        spline = SplineTransformer(n_knots=5, degree=3)
        X_spline = spline.fit_transform(X)
        
        # Fit linear model on spline features
        if task_type == 'classification':
            model = LogisticRegression(max_iter=1000)
        else:
            model = LinearRegression()
        
        model.fit(X_spline, y)
        
        return {
            'model': model,
            'spline_transformer': spline,
            'task_type': task_type,
            'n_features': X.shape[1],
            'n_spline_features': X_spline.shape[1]
        }


class BishopMethods:
    """
    Methods from Pattern Recognition and Machine Learning (Bishop)
    
    Key methods:
    - Gaussian Processes
    - Variational Inference (simplified)
    - Expectation-Maximization (EM) algorithm
    - Mixture Models
    - Probabilistic PCA
    """
    
    def __init__(self):
        self.dependencies = ['sklearn', 'scipy', 'numpy']
    
    def gaussian_process(
        self,
        X: np.ndarray,
        y: np.ndarray,
        kernel: str = 'rbf',
        task_type: str = 'regression',
        alpha: float = 1e-10
    ) -> Dict[str, Any]:
        """
        Gaussian Process (Bishop Chapter 6)
        
        Non-parametric Bayesian method
        
        Args:
            X: Features
            y: Labels
            kernel: 'rbf', 'matern', 'white'
            task_type: 'classification' or 'regression'
            alpha: Noise level
            
        Returns:
            Dictionary with fitted GP model
        """
        if not GAUSSIAN_PROCESS_AVAILABLE:
            return {'error': 'sklearn Gaussian Process not available'}
        
        # Create kernel
        if kernel == 'rbf':
            kernel_obj = RBF(length_scale=1.0)
        elif kernel == 'matern':
            kernel_obj = Matern(length_scale=1.0, nu=1.5)
        else:
            kernel_obj = RBF(length_scale=1.0) + WhiteKernel(noise_level=alpha)
        
        # Create GP model
        if task_type == 'classification':
            model = GaussianProcessClassifier(kernel=kernel_obj, random_state=42)
        else:
            model = GaussianProcessRegressor(kernel=kernel_obj, alpha=alpha, random_state=42)
        
        model.fit(X, y)
        
        return {
            'model': model,
            'kernel': kernel,
            'task_type': task_type,
            'kernel_params': model.kernel_.get_params() if hasattr(model, 'kernel_') else {}
        }
    
    def gaussian_mixture_model(
        self,
        X: np.ndarray,
        n_components: int = 3,
        covariance_type: str = 'full'
    ) -> Dict[str, Any]:
        """
        Gaussian Mixture Model (Bishop Chapter 9)
        
        Clustering using mixture of Gaussians
        
        Args:
            X: Features
            n_components: Number of mixture components
            covariance_type: 'full', 'tied', 'diag', 'spherical'
            
        Returns:
            Dictionary with fitted GMM
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        from sklearn.mixture import GaussianMixture
        
        model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=42
        )
        
        model.fit(X)
        
        return {
            'model': model,
            'n_components': n_components,
            'means': model.means_.tolist(),
            'covariances': model.covariances_.tolist() if isinstance(model.covariances_, np.ndarray) else model.covariances_,
            'weights': model.weights_.tolist(),
            'aic': float(model.aic(X)),
            'bic': float(model.bic(X))
        }
    
    def expectation_maximization(
        self,
        X: np.ndarray,
        n_components: int = 3,
        max_iter: int = 100
    ) -> Dict[str, Any]:
        """
        Expectation-Maximization Algorithm (Bishop Chapter 9)
        
        For GMM fitting (uses sklearn's GMM which uses EM)
        
        Args:
            X: Features
            n_components: Number of components
            max_iter: Maximum EM iterations
            
        Returns:
            Dictionary with EM results
        """
        # GMM uses EM internally
        gmm_result = self.gaussian_mixture_model(X, n_components)
        
        if 'error' in gmm_result:
            return gmm_result
        
        return {
            'gmm': gmm_result,
            'n_iterations': gmm_result['model'].n_iter_ if hasattr(gmm_result['model'], 'n_iter_') else max_iter,
            'converged': gmm_result['model'].converged_ if hasattr(gmm_result['model'], 'converged_') else True,
            'method': 'expectation_maximization'
        }
    
    def probabilistic_pca(
        self,
        X: np.ndarray,
        n_components: int = 2
    ) -> Dict[str, Any]:
        """
        Probabilistic PCA (Bishop Chapter 12)
        
        Probabilistic formulation of PCA
        
        Args:
            X: Features
            n_components: Number of components
            
        Returns:
            Dictionary with PPCA model
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        from sklearn.decomposition import PCA
        
        # Standard PCA (probabilistic PCA is similar but with probabilistic interpretation)
        model = PCA(n_components=n_components, random_state=42)
        X_transformed = model.fit_transform(X)
        
        return {
            'model': model,
            'n_components': n_components,
            'explained_variance_ratio': model.explained_variance_ratio_.tolist(),
            'components': model.components_.tolist(),
            'transformed_data': X_transformed.tolist(),
            'mean': model.mean_.tolist()
        }


class DeepLearningMethods:
    """
    Methods from Deep Learning (Goodfellow, Bengio, Courville)
    
    Key methods:
    - Neural Network architectures (basic)
    - Backpropagation (handled by frameworks)
    - Regularization techniques (dropout, batch norm)
    - Optimization algorithms (Adam, RMSprop)
    - Batch normalization
    """
    
    def __init__(self):
        self.dependencies = ['torch', 'numpy']
    
    def neural_network_basic(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hidden_layers: List[int] = [64, 32],
        activation: str = 'relu',
        task_type: str = 'classification',
        use_pytorch: bool = True
    ) -> Dict[str, Any]:
        """
        Basic Neural Network (Deep Learning Chapter 6)
        
        Multi-layer perceptron
        
        Args:
            X: Features
            y: Labels
            hidden_layers: List of hidden layer sizes
            activation: 'relu', 'sigmoid', 'tanh'
            task_type: 'classification' or 'regression'
            use_pytorch: Whether to use PyTorch (if available)
            
        Returns:
            Dictionary with model information
        """
        if use_pytorch:
            try:
                import torch
                import torch.nn as nn
                import torch.optim as optim
                TORCH_AVAILABLE = True
            except ImportError:
                TORCH_AVAILABLE = False
                use_pytorch = False
        
        if use_pytorch and TORCH_AVAILABLE:
            return self._neural_network_pytorch(X, y, hidden_layers, activation, task_type)
        else:
            # Use sklearn MLP
            if not SKLEARN_AVAILABLE:
                return {'error': 'sklearn or torch required for neural networks'}
            
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            
            if task_type == 'classification':
                model = MLPClassifier(
                    hidden_layer_sizes=tuple(hidden_layers),
                    activation=activation,
                    max_iter=1000,
                    random_state=42
                )
            else:
                model = MLPRegressor(
                    hidden_layer_sizes=tuple(hidden_layers),
                    activation=activation,
                    max_iter=1000,
                    random_state=42
                )
            
            model.fit(X, y)
            
            return {
                'model': model,
                'hidden_layers': hidden_layers,
                'activation': activation,
                'task_type': task_type,
                'n_layers': len(hidden_layers) + 2,  # Input + hidden + output
                'framework': 'sklearn'
            }
    
    def _neural_network_pytorch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hidden_layers: List[int],
        activation: str,
        task_type: str
    ) -> Dict[str, Any]:
        """Neural network using PyTorch"""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y) if task_type == 'regression' else torch.LongTensor(y)
        
        # Define network
        layers = []
        input_size = X.shape[1]
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            input_size = hidden_size
        
        # Output layer
        output_size = 1 if task_type == 'regression' else len(np.unique(y))
        layers.append(nn.Linear(input_size, output_size))
        if task_type == 'classification':
            layers.append(nn.Softmax(dim=1))
        
        model = nn.Sequential(*layers)
        
        # Training (simplified)
        criterion = nn.MSELoss() if task_type == 'regression' else nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train for a few epochs
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            if task_type == 'regression':
                loss = criterion(outputs.squeeze(), y_tensor)
            else:
                loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        return {
            'model': model,
            'hidden_layers': hidden_layers,
            'activation': activation,
            'task_type': task_type,
            'n_layers': len(hidden_layers) + 2,
            'framework': 'pytorch',
            'final_loss': float(loss.item())
        }
    
    def dropout_regularization(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dropout_rate: float = 0.5,
        hidden_layers: List[int] = [64, 32]
    ) -> Dict[str, Any]:
        """
        Dropout Regularization (Deep Learning Chapter 7)
        
        Prevents overfitting by randomly dropping neurons
        
        Args:
            X: Features
            y: Labels
            dropout_rate: Probability of dropping a neuron
            hidden_layers: Hidden layer sizes
            
        Returns:
            Dictionary with model information
        """
        try:
            import torch
            import torch.nn as nn
            TORCH_AVAILABLE = True
        except ImportError:
            return {'error': 'PyTorch required for dropout. Install with: pip install torch'}
        
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Define network with dropout
        class DropoutNet(nn.Module):
            def __init__(self, input_size, hidden_layers, output_size, dropout_rate):
                super().__init__()
                layers = []
                prev_size = input_size
                
                for hidden_size in hidden_layers:
                    layers.append(nn.Linear(prev_size, hidden_size))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rate))
                    prev_size = hidden_size
                
                layers.append(nn.Linear(prev_size, output_size))
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y) if len(np.unique(y)) < 10 else torch.FloatTensor(y)
        
        output_size = len(np.unique(y)) if len(np.unique(y)) < 10 else 1
        model = DropoutNet(X.shape[1], hidden_layers, output_size, dropout_rate)
        
        criterion = nn.CrossEntropyLoss() if len(np.unique(y)) < 10 else nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            if len(np.unique(y)) < 10:
                loss = criterion(outputs, y_tensor)
            else:
                loss = criterion(outputs.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
        
        return {
            'model': model,
            'dropout_rate': dropout_rate,
            'hidden_layers': hidden_layers,
            'final_loss': float(loss.item()),
            'framework': 'pytorch'
        }
    
    def batch_normalization(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hidden_layers: List[int] = [64, 32]
    ) -> Dict[str, Any]:
        """
        Batch Normalization (Deep Learning Chapter 8)
        
        Normalizes activations to improve training
        
        Args:
            X: Features
            y: Labels
            hidden_layers: Hidden layer sizes
            
        Returns:
            Dictionary with model information
        """
        try:
            import torch
            import torch.nn as nn
            TORCH_AVAILABLE = True
        except ImportError:
            return {'error': 'PyTorch required for batch normalization. Install with: pip install torch'}
        
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Define network with batch normalization
        class BatchNormNet(nn.Module):
            def __init__(self, input_size, hidden_layers, output_size):
                super().__init__()
                layers = []
                prev_size = input_size
                
                for hidden_size in hidden_layers:
                    layers.append(nn.Linear(prev_size, hidden_size))
                    layers.append(nn.BatchNorm1d(hidden_size))
                    layers.append(nn.ReLU())
                    prev_size = hidden_size
                
                layers.append(nn.Linear(prev_size, output_size))
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y) if len(np.unique(y)) < 10 else torch.FloatTensor(y)
        
        output_size = len(np.unique(y)) if len(np.unique(y)) < 10 else 1
        model = BatchNormNet(X.shape[1], hidden_layers, output_size)
        
        criterion = nn.CrossEntropyLoss() if len(np.unique(y)) < 10 else nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            if len(np.unique(y)) < 10:
                loss = criterion(outputs, y_tensor)
            else:
                loss = criterion(outputs.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
        
        return {
            'model': model,
            'hidden_layers': hidden_layers,
            'final_loss': float(loss.item()),
            'framework': 'pytorch',
            'has_batch_norm': True
        }


class ThreeBooksMethods:
    """
    Unified interface for methods from all three books
    """
    
    def __init__(self):
        self.esl = ESLMethods()
        self.bishop = BishopMethods()
        self.deep_learning = DeepLearningMethods()
        
        # Track all dependencies
        self.all_dependencies = set()
        self.all_dependencies.update(self.esl.dependencies)
        self.all_dependencies.update(self.bishop.dependencies)
        self.all_dependencies.update(self.deep_learning.dependencies)
    
    def get_dependencies(self) -> Dict[str, str]:
        """
        Get all required dependencies
        
        Returns:
            Dictionary mapping dependency names to pip install commands
        """
        return {dep: REQUIRED_DEPENDENCIES.get(dep, f'{dep}>=latest') for dep in self.all_dependencies}
    
    def get_dependency_list(self) -> List[str]:
        """
        Get list of all dependencies as pip install commands
        
        Returns:
            List of pip install commands
        """
        deps = self.get_dependencies()
        return [f"pip install {package}" for package in deps.values()]


def get_all_dependencies() -> Dict[str, Dict[str, str]]:
    """
    Get all dependencies organized by book
    
    Returns:
        Dictionary with dependencies for each book
    """
    return {
        'Elements_of_Statistical_Learning': {
            'sklearn': 'scikit-learn>=1.5.0',
            'scipy': 'scipy>=1.11.0',
            'numpy': 'numpy>=1.26.0'
        },
        'Pattern_Recognition_and_Machine_Learning': {
            'sklearn': 'scikit-learn>=1.5.0',
            'scipy': 'scipy>=1.11.0',
            'numpy': 'numpy>=1.26.0'
        },
        'Deep_Learning': {
            'torch': 'torch>=2.3.0',
            'numpy': 'numpy>=1.26.0'
        },
        'All_Books_Combined': {
            'sklearn': 'scikit-learn>=1.5.0',
            'scipy': 'scipy>=1.11.0',
            'numpy': 'numpy>=1.26.0',
            'torch': 'torch>=2.3.0'
        }
    }
