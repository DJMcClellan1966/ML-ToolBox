"""
Optimized ML Tasks
Speed-optimized and accuracy-improved versions of ML tasks

Optimizations:
- Model caching
- Parallel processing
- Optimized preprocessing
- Better hyperparameter tuning
- Ensemble methods
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import warnings
import time
import hashlib
import pickle
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).parent))


class OptimizedMLTasks:
    """
    Optimized ML Tasks
    
    Speed-optimized and accuracy-improved ML tasks
    """
    
    def __init__(self, cache_dir: str = ".ml_cache"):
        """
        Initialize optimized ML tasks
        
        Args:
            cache_dir: Directory for model caching
        """
        self.dependencies = ['sklearn', 'numpy', 'joblib']
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check dependencies"""
        try:
            import sklearn
            self.sklearn_available = True
        except ImportError:
            self.sklearn_available = False
            warnings.warn("sklearn not available. Optimized ML tasks will be limited.")
        
        try:
            from joblib import Parallel, delayed
            self.joblib_available = True
        except ImportError:
            self.joblib_available = False
            warnings.warn("joblib not available. Parallel processing disabled.")
    
    def _get_cache_key(self, X: np.ndarray, y: np.ndarray, model_type: str, **kwargs) -> str:
        """Generate cache key for model"""
        # Create hash from data and parameters
        data_hash = hashlib.md5(
            (str(X.shape) + str(y.shape) + model_type + str(sorted(kwargs.items()))).encode()
        ).hexdigest()
        return f"{model_type}_{data_hash}"
    
    def _load_cached_model(self, cache_key: str):
        """Load cached model if exists"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
    
    def _save_cached_model(self, cache_key: str, model: Any):
        """Save model to cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            warnings.warn(f"Could not cache model: {e}")
    
    def train_classifier_optimized(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = 'auto',
        use_cache: bool = True,
        n_jobs: int = -1,
        tune_hyperparameters: bool = True
    ) -> Dict[str, Any]:
        """
        Train classifier with optimizations
        
        Args:
            X: Features
            y: Labels
            model_type: 'auto', 'random_forest', 'svm', 'logistic', 'knn', 'ensemble'
            use_cache: Use model caching
            n_jobs: Number of parallel jobs (-1 for all cores)
            tune_hyperparameters: Enable hyperparameter tuning
            
        Returns:
            Trained model and metrics
        """
        if not self.sklearn_available:
            return {'error': 'sklearn required'}
        
        from sklearn.model_selection import train_test_split, RandomizedSearchCV
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score, classification_report
        import numpy as np
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(X, y, model_type, tune=tune_hyperparameters)
            cached = self._load_cached_model(cache_key)
            if cached:
                return cached
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        start_time = time.time()
        
        # Select and optimize model
        if model_type == 'auto' or model_type == 'ensemble':
            # Use ensemble for better accuracy
            model = self._create_optimized_ensemble(X_train, y_train, n_jobs, tune_hyperparameters)
        elif model_type == 'random_forest':
            model = self._create_optimized_rf(X_train, y_train, n_jobs, tune_hyperparameters)
        elif model_type == 'svm':
            model = self._create_optimized_svm(X_train, y_train, n_jobs, tune_hyperparameters)
        elif model_type == 'logistic':
            model = self._create_optimized_lr(X_train, y_train, n_jobs, tune_hyperparameters)
        elif model_type == 'knn':
            model = self._create_optimized_knn(X_train, y_train, n_jobs, tune_hyperparameters)
        else:
            model = self._create_optimized_rf(X_train, y_train, n_jobs, tune_hyperparameters)
        
        training_time = time.time() - start_time
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        result = {
            'model': model,
            'accuracy': accuracy,
            'classification_report': report,
            'model_type': model_type,
            'training_time': training_time,
            'optimized': True
        }
        
        # Cache result
        if use_cache:
            self._save_cached_model(cache_key, result)
        
        return result
    
    def _create_optimized_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_jobs: int,
        tune: bool
    ):
        """Create optimized ensemble model"""
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        # Create base models
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            n_jobs=n_jobs,
            random_state=42
        )
        
        lr = LogisticRegression(
            max_iter=1000,
            n_jobs=n_jobs,
            random_state=42
        )
        
        svm = SVC(
            probability=True,
            random_state=42
        )
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('lr', lr),
                ('svm', svm)
            ],
            voting='soft',
            n_jobs=n_jobs
        )
        
        ensemble.fit(X_train, y_train)
        return ensemble
    
    def _create_optimized_rf(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_jobs: int,
        tune: bool
    ):
        """Create optimized Random Forest"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import RandomizedSearchCV
        
        if tune and len(X_train) > 100:
            # Hyperparameter tuning for better accuracy
            param_dist = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            base_model = RandomForestClassifier(n_jobs=n_jobs, random_state=42)
            search = RandomizedSearchCV(
                base_model,
                param_dist,
                n_iter=10,
                cv=3,
                n_jobs=n_jobs,
                random_state=42,
                verbose=0
            )
            search.fit(X_train, y_train)
            return search.best_estimator_
        else:
            # Fast default
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                n_jobs=n_jobs,
                random_state=42
            ).fit(X_train, y_train)
    
    def _create_optimized_svm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_jobs: int,
        tune: bool
    ):
        """Create optimized SVM"""
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        # Scale data for SVM
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        return pipeline
    
    def _create_optimized_lr(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_jobs: int,
        tune: bool
    ):
        """Create optimized Logistic Regression"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        if tune:
            from sklearn.model_selection import RandomizedSearchCV
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(random_state=42, n_jobs=n_jobs))
            ])
            
            param_dist = {
                'lr__C': [0.1, 1, 10, 100],
                'lr__penalty': ['l1', 'l2'],
                'lr__solver': ['liblinear', 'lbfgs']
            }
            
            search = RandomizedSearchCV(
                pipeline,
                param_dist,
                n_iter=5,
                cv=3,
                n_jobs=n_jobs,
                random_state=42,
                verbose=0
            )
            search.fit(X_train, y_train)
            return search.best_estimator_
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(max_iter=1000, n_jobs=n_jobs, random_state=42))
            ])
            pipeline.fit(X_train, y_train)
            return pipeline
    
    def _create_optimized_knn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_jobs: int,
        tune: bool
    ):
        """Create optimized KNN"""
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        if tune:
            from sklearn.model_selection import RandomizedSearchCV
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier(n_jobs=n_jobs))
            ])
            
            param_dist = {
                'knn__n_neighbors': [3, 5, 7, 9, 11],
                'knn__weights': ['uniform', 'distance'],
                'knn__metric': ['euclidean', 'manhattan', 'minkowski']
            }
            
            search = RandomizedSearchCV(
                pipeline,
                param_dist,
                n_iter=5,
                cv=3,
                n_jobs=n_jobs,
                random_state=42,
                verbose=0
            )
            search.fit(X_train, y_train)
            return search.best_estimator_
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier(n_neighbors=5, n_jobs=n_jobs))
            ])
            pipeline.fit(X_train, y_train)
            return pipeline
    
    def train_regressor_optimized(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = 'auto',
        use_cache: bool = True,
        n_jobs: int = -1,
        tune_hyperparameters: bool = True
    ) -> Dict[str, Any]:
        """
        Train regressor with optimizations
        
        Args:
            X: Features
            y: Target values
            model_type: 'auto', 'random_forest', 'linear', 'svr', 'knn', 'ensemble'
            use_cache: Use model caching
            n_jobs: Number of parallel jobs
            tune_hyperparameters: Enable hyperparameter tuning
            
        Returns:
            Trained model and metrics
        """
        if not self.sklearn_available:
            return {'error': 'sklearn required'}
        
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor, VotingRegressor
        from sklearn.linear_model import Ridge, LinearRegression
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(X, y, f"{model_type}_reg", tune=tune_hyperparameters)
            cached = self._load_cached_model(cache_key)
            if cached:
                return cached
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        start_time = time.time()
        
        # Select and optimize model
        if model_type == 'auto' or model_type == 'ensemble':
            model = self._create_optimized_regression_ensemble(X_train, y_train, n_jobs, tune_hyperparameters)
        elif model_type == 'random_forest':
            model = self._create_optimized_rf_regressor(X_train, y_train, n_jobs, tune_hyperparameters)
        elif model_type == 'linear':
            model = self._create_optimized_ridge(X_train, y_train, n_jobs, tune_hyperparameters)
        elif model_type == 'svr':
            model = self._create_optimized_svr(X_train, y_train, n_jobs, tune_hyperparameters)
        elif model_type == 'knn':
            model = self._create_optimized_knn_regressor(X_train, y_train, n_jobs, tune_hyperparameters)
        else:
            model = self._create_optimized_rf_regressor(X_train, y_train, n_jobs, tune_hyperparameters)
        
        training_time = time.time() - start_time
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        result = {
            'model': model,
            'r2_score': r2,
            'mse': mse,
            'mae': mae,
            'model_type': model_type,
            'training_time': training_time,
            'optimized': True
        }
        
        # Cache result
        if use_cache:
            self._save_cached_model(cache_key, result)
        
        return result
    
    def _create_optimized_regression_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_jobs: int,
        tune: bool
    ):
        """Create optimized regression ensemble"""
        from sklearn.ensemble import RandomForestRegressor, VotingRegressor
        from sklearn.linear_model import Ridge
        
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            n_jobs=n_jobs,
            random_state=42
        )
        
        ridge = Ridge(alpha=1.0, random_state=42)
        
        ensemble = VotingRegressor(
            estimators=[('rf', rf), ('ridge', ridge)],
            n_jobs=n_jobs
        )
        
        ensemble.fit(X_train, y_train)
        return ensemble
    
    def _create_optimized_rf_regressor(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_jobs: int,
        tune: bool
    ):
        """Create optimized Random Forest Regressor"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import RandomizedSearchCV
        
        if tune and len(X_train) > 100:
            param_dist = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
            
            base_model = RandomForestRegressor(n_jobs=n_jobs, random_state=42)
            search = RandomizedSearchCV(
                base_model,
                param_dist,
                n_iter=10,
                cv=3,
                n_jobs=n_jobs,
                random_state=42,
                verbose=0
            )
            search.fit(X_train, y_train)
            return search.best_estimator_
        else:
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                n_jobs=n_jobs,
                random_state=42
            ).fit(X_train, y_train)
    
    def _create_optimized_ridge(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_jobs: int,
        tune: bool
    ):
        """Create optimized Ridge regression"""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        if tune:
            from sklearn.model_selection import RandomizedSearchCV
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(random_state=42))
            ])
            
            param_dist = {
                'ridge__alpha': [0.1, 1, 10, 100, 1000]
            }
            
            search = RandomizedSearchCV(
                pipeline,
                param_dist,
                n_iter=5,
                cv=3,
                n_jobs=n_jobs,
                random_state=42,
                verbose=0
            )
            search.fit(X_train, y_train)
            return search.best_estimator_
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=1.0, random_state=42))
            ])
            pipeline.fit(X_train, y_train)
            return pipeline
    
    def _create_optimized_svr(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_jobs: int,
        tune: bool
    ):
        """Create optimized SVR"""
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR())
        ])
        
        pipeline.fit(X_train, y_train)
        return pipeline
    
    def _create_optimized_knn_regressor(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_jobs: int,
        tune: bool
    ):
        """Create optimized KNN Regressor"""
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor(n_neighbors=5, n_jobs=n_jobs))
        ])
        
        pipeline.fit(X_train, y_train)
        return pipeline
    
    def quick_train_optimized(
        self,
        X: np.ndarray,
        y: np.ndarray,
        use_cache: bool = True,
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        Quick train with optimizations (auto-detect task and train)
        
        Args:
            X: Features
            y: Labels or target values
            use_cache: Use model caching
            n_jobs: Number of parallel jobs
            
        Returns:
            Trained model and results
        """
        # Auto-detect task type
        if len(np.unique(y)) < 20 and np.all(y == y.astype(int)):
            # Classification
            return self.train_classifier_optimized(
                X, y, model_type='ensemble', use_cache=use_cache, n_jobs=n_jobs
            )
        else:
            # Regression
            return self.train_regressor_optimized(
                X, y, model_type='ensemble', use_cache=use_cache, n_jobs=n_jobs
            )
    
    def get_dependencies(self) -> Dict[str, str]:
        """Get dependencies"""
        return {
            'sklearn': 'scikit-learn>=1.3.0',
            'numpy': 'numpy>=1.26.0',
            'joblib': 'joblib>=1.3.0'
        }
