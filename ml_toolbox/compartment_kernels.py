"""
Compartment Kernels
Each compartment as a unified algorithm/kernel interface
"""
from typing import Any, Dict, Optional, List, Union, Callable
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class CompartmentKernel(ABC):
    """Base class for compartment kernels"""
    
    def __init__(self, name: str, compartment):
        self.name = name
        self.compartment = compartment
        self.config = {}
    
    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> Any:
        """Process input data through the compartment kernel"""
        pass
    
    @abstractmethod
    def fit(self, X, y=None, **kwargs) -> 'CompartmentKernel':
        """Fit the kernel to data"""
        pass
    
    @abstractmethod
    def transform(self, X, **kwargs) -> Any:
        """Transform data using the fitted kernel"""
        pass
    
    def __call__(self, input_data: Any, **kwargs) -> Any:
        """Make kernel callable"""
        return self.process(input_data, **kwargs)


class DataKernel(CompartmentKernel):
    """
    Compartment 1 as a unified data preprocessing kernel
    
    Treats the entire data compartment as a single algorithm
    that handles all preprocessing, validation, and transformation.
    """
    
    def __init__(self, compartment, config: Optional[Dict] = None):
        super().__init__("DataKernel", compartment)
        self.config = config or {
            'auto_detect': True,
            'use_advanced': True,
            'use_universal': True,
            'quality_check': True
        }
        self._fitted = False
        self._preprocessor = None
    
    def fit(self, X, y=None, **kwargs) -> 'DataKernel':
        """Fit the data kernel to training data"""
        logger.info(f"[DataKernel] Fitting to data shape: {X.shape if hasattr(X, 'shape') else len(X)}")
        
        # Auto-detect best preprocessor
        if self.config.get('use_universal') and hasattr(self.compartment, 'universal_adaptive_preprocessor'):
            try:
                from universal_adaptive_preprocessor import get_universal_preprocessor
                self._preprocessor = get_universal_preprocessor()
                self._preprocessor.fit(X, y)
                logger.info("[DataKernel] Using Universal Adaptive Preprocessor")
            except:
                pass
        
        if self._preprocessor is None and self.config.get('use_advanced'):
            # Try advanced preprocessor
            if hasattr(self.compartment, 'advanced_data_preprocessor'):
                self._preprocessor = self.compartment.advanced_data_preprocessor
            elif hasattr(self.compartment, 'get_advanced_preprocessor'):
                self._preprocessor = self.compartment.get_advanced_preprocessor()
        
        if self._preprocessor is None:
            # Fallback to conventional
            if hasattr(self.compartment, 'conventional_preprocessor'):
                self._preprocessor = self.compartment.conventional_preprocessor
            elif hasattr(self.compartment, 'get_conventional_preprocessor'):
                self._preprocessor = self.compartment.get_conventional_preprocessor()
        
        # Fit preprocessor
        if self._preprocessor and hasattr(self._preprocessor, 'fit'):
            self._preprocessor.fit(X, y)
        
        self._fitted = True
        return self
    
    def transform(self, X, **kwargs) -> Any:
        """Transform data using fitted kernel"""
        if not self._fitted:
            raise ValueError("Kernel must be fitted before transform. Call fit() first.")
        
        logger.info(f"[DataKernel] Transforming data shape: {X.shape if hasattr(X, 'shape') else len(X)}")
        
        # Use fitted preprocessor
        if self._preprocessor and hasattr(self._preprocessor, 'transform'):
            return self._preprocessor.transform(X, **kwargs)
        elif self._preprocessor and hasattr(self._preprocessor, 'fit_transform'):
            # Some preprocessors don't separate fit/transform
            return self._preprocessor.fit_transform(X, **kwargs)
        else:
            # Fallback: return as-is
            return X
    
    def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Process data through the entire data compartment
        
        Returns:
            Dictionary with:
            - processed_data: Preprocessed data
            - quality_score: Data quality score
            - metadata: Processing metadata
        """
        result = {
            'processed_data': None,
            'quality_score': None,
            'metadata': {}
        }
        
        # Fit if not already fitted
        if not self._fitted:
            self.fit(input_data)
        
        # Transform
        processed = self.transform(input_data, **kwargs)
        result['processed_data'] = processed
        
        # Quality scoring
        if self.config.get('quality_check'):
            quality_score = self._assess_quality(input_data, processed)
            result['quality_score'] = quality_score
            result['metadata']['quality_assessment'] = quality_score
        
        # Metadata
        result['metadata'].update({
            'preprocessor_type': type(self._preprocessor).__name__ if self._preprocessor else 'none',
            'input_shape': getattr(input_data, 'shape', None),
            'output_shape': getattr(processed, 'shape', None) if hasattr(processed, 'shape') else None
        })
        
        return result
    
    def _assess_quality(self, original, processed) -> float:
        """Assess data quality"""
        try:
            # Simple quality metric
            if hasattr(processed, 'shape') and hasattr(original, 'shape'):
                # Check if data is valid (no NaN, no inf)
                if isinstance(processed, np.ndarray):
                    valid_ratio = np.sum(np.isfinite(processed)) / processed.size
                    return float(valid_ratio)
            return 1.0
        except:
            return 0.5


class InfrastructureKernel(CompartmentKernel):
    """
    Compartment 2 as a unified infrastructure/AI kernel
    
    Treats the entire infrastructure compartment as a single kernel
    that handles semantic understanding, reasoning, and AI operations.
    """
    
    def __init__(self, compartment, config: Optional[Dict] = None):
        super().__init__("InfrastructureKernel", compartment)
        self.config = config or {
            'use_quantum': True,
            'use_ai_system': True,
            'use_reasoning': True
        }
        self._quantum_kernel = None
        self._ai_system = None
        self._fitted = False
    
    def fit(self, X, y=None, **kwargs) -> 'InfrastructureKernel':
        """Fit the infrastructure kernel (build knowledge base, etc.)"""
        logger.info(f"[InfrastructureKernel] Fitting to {len(X) if hasattr(X, '__len__') else 'data'}")
        
        # Initialize quantum kernel
        if self.config.get('use_quantum'):
            try:
                QuantumKernel = self.compartment.components.get('QuantumKernel')
                if QuantumKernel:
                    from quantum_kernel.kernel import KernelConfig
                    config = KernelConfig(embedding_dim=256)
                    self._quantum_kernel = QuantumKernel(config)
                    logger.info("[InfrastructureKernel] Quantum Kernel initialized")
            except Exception as e:
                logger.warning(f"[InfrastructureKernel] Quantum Kernel not available: {e}")
        
        # Initialize AI system
        if self.config.get('use_ai_system'):
            try:
                CompleteAISystem = self.compartment.components.get('CompleteAISystem')
                if CompleteAISystem:
                    self._ai_system = CompleteAISystem(use_llm=False)
                    logger.info("[InfrastructureKernel] AI System initialized")
            except Exception as e:
                logger.warning(f"[InfrastructureKernel] AI System not available: {e}")
        
        # Build knowledge base from data
        if isinstance(X, (list, tuple)) and self._ai_system:
            # Treat as documents for knowledge graph
            documents = [str(x) for x in X[:100]]  # Limit for performance
            self._ai_system.process({'documents': documents})
        
        self._fitted = True
        return self
    
    def transform(self, X, **kwargs) -> Any:
        """Transform input through infrastructure kernel"""
        if not self._fitted:
            raise ValueError("Kernel must be fitted before transform. Call fit() first.")
        
        operation = kwargs.get('operation', 'embed')
        
        if operation == 'embed' and self._quantum_kernel:
            # Semantic embedding
            if isinstance(X, str):
                return self._quantum_kernel.embed(X)
            elif isinstance(X, (list, tuple)):
                return np.array([self._quantum_kernel.embed(str(x)) for x in X])
        
        elif operation == 'understand' and self._ai_system:
            # Semantic understanding
            if isinstance(X, str):
                return self._ai_system.process({'query': X})
            elif isinstance(X, dict):
                return self._ai_system.process(X)
        
        elif operation == 'search' and self._ai_system and 'corpus' in kwargs:
            # Semantic search
            query = str(X) if not isinstance(X, str) else X
            corpus = kwargs['corpus']
            return self._ai_system.process({
                'query': query,
                'documents': corpus
            })
        
        # Default: return as-is
        return X
    
    def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Process input through infrastructure kernel
        
        Returns:
            Dictionary with:
            - embeddings: Semantic embeddings
            - understanding: Intent/meaning understanding
            - reasoning: Logical reasoning results
            - metadata: Processing metadata
        """
        result = {
            'embeddings': None,
            'understanding': None,
            'reasoning': None,
            'metadata': {}
        }
        
        # Fit if not already fitted
        if not self._fitted:
            # Use input_data as training corpus
            corpus = input_data if isinstance(input_data, (list, tuple)) else [input_data]
            self.fit(corpus)
        
        # Generate embeddings
        if self._quantum_kernel:
            embeddings = self.transform(input_data, operation='embed')
            result['embeddings'] = embeddings
        
        # Understanding
        if self._ai_system and isinstance(input_data, str):
            understanding = self.transform(input_data, operation='understand')
            result['understanding'] = understanding
        
        # Reasoning (if premises provided)
        if self._ai_system and 'premises' in kwargs:
            reasoning = self._ai_system.process({
                'premises': kwargs['premises'],
                'question': kwargs.get('question', input_data)
            })
            result['reasoning'] = reasoning
        
        result['metadata'] = {
            'quantum_kernel_available': self._quantum_kernel is not None,
            'ai_system_available': self._ai_system is not None
        }
        
        return result


class AlgorithmsKernel(CompartmentKernel):
    """
    Compartment 3 as a unified ML algorithms kernel
    
    Treats the entire algorithms compartment as a single algorithm
    that automatically selects, trains, and optimizes models.
    """
    
    def __init__(self, compartment, config: Optional[Dict] = None):
        super().__init__("AlgorithmsKernel", compartment)
        self.config = config or {
            'auto_select': True,
            'use_ensemble': True,
            'optimize_hyperparameters': True
        }
        self._model = None
        self._fitted = False
    
    def fit(self, X, y, **kwargs) -> 'AlgorithmsKernel':
        """Fit the algorithms kernel (train model)"""
        logger.info(f"[AlgorithmsKernel] Fitting to data shape: {X.shape if hasattr(X, 'shape') else len(X)}")
        
        task_type = kwargs.get('task_type', 'auto')
        
        # Auto-detect task type
        if task_type == 'auto':
            if y is None:
                task_type = 'clustering'
            elif len(np.unique(y)) == 2:
                task_type = 'classification'
            elif len(np.unique(y)) > 2:
                task_type = 'classification'
            else:
                task_type = 'regression'
        
        # Use AI Model Orchestrator if available
        if hasattr(self.compartment, 'ai_model_orchestrator') and self.compartment.ai_model_orchestrator:
            result = self.compartment.ai_model_orchestrator.fit(
                X, y,
                task_type=task_type,
                auto_select=True,
                optimize=True
            )
            self._model = result.get('model')
        else:
            # Use standard algorithm selection
            if task_type == 'classification':
                self._model = self.compartment.train_classifier(X, y, **kwargs)
            elif task_type == 'regression':
                self._model = self.compartment.train_regressor(X, y, **kwargs)
            elif task_type == 'clustering':
                self._model = self.compartment.train_clusterer(X, **kwargs)
        
        self._fitted = True
        return self
    
    def transform(self, X, **kwargs) -> Any:
        """Transform data (make predictions)"""
        if not self._fitted:
            raise ValueError("Kernel must be fitted before transform. Call fit() first.")
        
        if self._model is None:
            raise ValueError("No model available. Fit failed.")
        
        logger.info(f"[AlgorithmsKernel] Predicting on data shape: {X.shape if hasattr(X, 'shape') else len(X)}")
        
        # Make predictions
        if hasattr(self._model, 'predict'):
            return self._model.predict(X)
        elif hasattr(self._model, 'transform'):
            return self._model.transform(X)
        else:
            raise ValueError("Model does not support predict or transform")
    
    def process(self, input_data: Any, y=None, **kwargs) -> Dict[str, Any]:
        """
        Process data through algorithms kernel
        
        Returns:
            Dictionary with:
            - model: Trained model
            - predictions: Model predictions
            - metrics: Performance metrics (if y provided)
            - metadata: Model metadata
        """
        result = {
            'model': None,
            'predictions': None,
            'metrics': None,
            'metadata': {}
        }
        
        # Fit if not already fitted
        if not self._fitted:
            self.fit(input_data, y, **kwargs)
        
        result['model'] = self._model
        
        # Make predictions
        predictions = self.transform(input_data, **kwargs)
        result['predictions'] = predictions
        
        # Calculate metrics if y provided
        if y is not None and hasattr(self.compartment, 'evaluate_model'):
            try:
                metrics = self.compartment.evaluate_model(self._model, input_data, y)
                result['metrics'] = metrics
            except:
                pass
        
        result['metadata'] = {
            'model_type': type(self._model).__name__ if self._model else None,
            'task_type': kwargs.get('task_type', 'auto'),
            'fitted': self._fitted
        }
        
        return result


class MLOpsKernel(CompartmentKernel):
    """
    Compartment 4 as a unified MLOps kernel
    
    Treats the entire MLOps compartment as a single kernel
    that handles deployment, monitoring, and production operations.
    """
    
    def __init__(self, compartment, config: Optional[Dict] = None):
        super().__init__("MLOpsKernel", compartment)
        self.config = config or {
            'auto_deploy': False,
            'enable_monitoring': True,
            'track_experiments': True
        }
        self._deployment = None
        self._fitted = False
    
    def fit(self, X, y=None, **kwargs) -> 'MLOpsKernel':
        """Fit MLOps kernel (set up deployment infrastructure)"""
        logger.info("[MLOpsKernel] Setting up MLOps infrastructure")
        
        # Initialize deployment components
        if hasattr(self.compartment, 'model_server'):
            self._deployment = self.compartment.model_server
        elif hasattr(self.compartment, 'get_model_server'):
            self._deployment = self.compartment.get_model_server()
        
        self._fitted = True
        return self
    
    def transform(self, X, **kwargs) -> Any:
        """Transform (deploy/monitor model)"""
        if not self._fitted:
            raise ValueError("Kernel must be fitted before transform. Call fit() first.")
        
        operation = kwargs.get('operation', 'deploy')
        model = kwargs.get('model')
        
        if operation == 'deploy' and model and self._deployment:
            # Deploy model
            deployment_info = self._deployment.deploy_model(model, **kwargs)
            return deployment_info
        
        elif operation == 'monitor' and model:
            # Monitor model
            if hasattr(self.compartment, 'monitor_model'):
                return self.compartment.monitor_model(model, X, **kwargs)
        
        elif operation == 'track' and model:
            # Track experiment
            if hasattr(self.compartment, 'track_experiment'):
                return self.compartment.track_experiment(model, X, **kwargs)
        
        return None
    
    def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Process through MLOps kernel
        
        Returns:
            Dictionary with:
            - deployment: Deployment information
            - monitoring: Monitoring setup
            - experiment_id: Experiment tracking ID
            - metadata: MLOps metadata
        """
        result = {
            'deployment': None,
            'monitoring': None,
            'experiment_id': None,
            'metadata': {}
        }
        
        # Fit if not already fitted
        if not self._fitted:
            self.fit(input_data)
        
        model = kwargs.get('model')
        if model is None:
            result['metadata']['error'] = 'No model provided for MLOps operations'
            return result
        
        # Deploy if requested
        if self.config.get('auto_deploy') and model:
            deployment = self.transform(input_data, operation='deploy', model=model, **kwargs)
            result['deployment'] = deployment
        
        # Set up monitoring
        if self.config.get('enable_monitoring') and model:
            monitoring = self.transform(input_data, operation='monitor', model=model, **kwargs)
            result['monitoring'] = monitoring
        
        # Track experiment
        if self.config.get('track_experiments') and model:
            experiment = self.transform(input_data, operation='track', model=model, **kwargs)
            result['experiment_id'] = experiment.get('experiment_id') if experiment else None
        
        result['metadata'] = {
            'deployment_available': self._deployment is not None,
            'monitoring_enabled': self.config.get('enable_monitoring', False),
            'experiment_tracking_enabled': self.config.get('track_experiments', False)
        }
        
        return result


class UnifiedCompartmentKernel:
    """
    Unified kernel that combines all compartments into one algorithm
    
    This treats the entire ML Toolbox as a single unified kernel/algorithm
    that processes data through all compartments sequentially.
    """
    
    def __init__(self, toolbox, config: Optional[Dict] = None):
        """
        Initialize unified kernel
        
        Args:
            toolbox: MLToolbox instance
            config: Configuration for each compartment kernel
        """
        self.toolbox = toolbox
        self.config = config or {}
        
        # Initialize compartment kernels
        self.data_kernel = DataKernel(
            toolbox.data,
            config=self.config.get('data', {})
        )
        self.infrastructure_kernel = InfrastructureKernel(
            toolbox.infrastructure,
            config=self.config.get('infrastructure', {})
        )
        self.algorithms_kernel = AlgorithmsKernel(
            toolbox.algorithms,
            config=self.config.get('algorithms', {})
        )
        self.mlops_kernel = None
        if toolbox.mlops:
            self.mlops_kernel = MLOpsKernel(
                toolbox.mlops,
                config=self.config.get('mlops', {})
            )
    
    def fit(self, X, y=None, **kwargs) -> 'UnifiedCompartmentKernel':
        """Fit all compartment kernels"""
        logger.info("[UnifiedKernel] Fitting all compartment kernels")
        
        # Fit in sequence: Data -> Infrastructure -> Algorithms
        self.data_kernel.fit(X, y)
        
        # Infrastructure uses processed data
        processed_X = self.data_kernel.transform(X)
        self.infrastructure_kernel.fit(processed_X, y)
        
        # Algorithms use processed data
        self.algorithms_kernel.fit(processed_X, y, **kwargs)
        
        # MLOps setup
        if self.mlops_kernel:
            self.mlops_kernel.fit(X, y)
        
        return self
    
    def transform(self, X, **kwargs) -> Dict[str, Any]:
        """Transform data through all compartments"""
        logger.info("[UnifiedKernel] Transforming through all compartments")
        
        results = {}
        
        # Step 1: Data preprocessing
        data_result = self.data_kernel.process(X)
        processed_X = data_result['processed_data']
        results['data'] = data_result
        
        # Step 2: Infrastructure (semantic understanding)
        infra_result = self.infrastructure_kernel.process(processed_X, **kwargs)
        results['infrastructure'] = infra_result
        
        # Step 3: Algorithms (predictions)
        algo_result = self.algorithms_kernel.process(processed_X, **kwargs)
        results['algorithms'] = algo_result
        
        # Step 4: MLOps (deployment/monitoring)
        if self.mlops_kernel and 'model' in algo_result:
            mlops_result = self.mlops_kernel.process(
                processed_X,
                model=algo_result['model'],
                **kwargs
            )
            results['mlops'] = mlops_result
        
        return results
    
    def __call__(self, X, y=None, **kwargs) -> Dict[str, Any]:
        """Make unified kernel callable"""
        # Fit if not fitted
        if not hasattr(self, '_fitted') or not self._fitted:
            self.fit(X, y, **kwargs)
            self._fitted = True
        
        # Transform
        return self.transform(X, **kwargs)


def get_compartment_kernels(toolbox, config: Optional[Dict] = None) -> Dict[str, CompartmentKernel]:
    """
    Get all compartment kernels as a dictionary
    
    Returns:
        Dictionary with keys: 'data', 'infrastructure', 'algorithms', 'mlops'
    """
    kernels = {
        'data': DataKernel(toolbox.data, config=config.get('data', {}) if config else {}),
        'infrastructure': InfrastructureKernel(
            toolbox.infrastructure,
            config=config.get('infrastructure', {}) if config else {}
        ),
        'algorithms': AlgorithmsKernel(
            toolbox.algorithms,
            config=config.get('algorithms', {}) if config else {}
        )
    }
    
    if toolbox.mlops:
        kernels['mlops'] = MLOpsKernel(
            toolbox.mlops,
            config=config.get('mlops', {}) if config else {}
        )
    
    return kernels


def get_unified_kernel(toolbox, config: Optional[Dict] = None) -> UnifiedCompartmentKernel:
    """Get unified kernel that combines all compartments"""
    return UnifiedCompartmentKernel(toolbox, config=config)
