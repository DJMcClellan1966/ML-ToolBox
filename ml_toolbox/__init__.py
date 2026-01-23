"""
Machine Learning Toolbox
Organized into four compartments:
1. Data: Preprocessing, validation, transformation
2. Infrastructure: Kernels, AI components, LLM
3. Algorithms: Models, evaluation, tuning, ensembles
4. MLOps: Production deployment, monitoring, A/B testing, experiment tracking

Also includes Advanced ML Toolbox for big data and advanced features
"""
from .compartment1_data import DataCompartment
from .compartment2_infrastructure import InfrastructureCompartment
from .compartment3_algorithms import AlgorithmsCompartment

# Try to import MLOps compartment
try:
    from .compartment4_mlops import MLOpsCompartment
    MLOPS_AVAILABLE = True
except ImportError:
    MLOPS_AVAILABLE = False
    MLOpsCompartment = None

# Import advanced toolbox
try:
    from .advanced import AdvancedMLToolbox
    __all__ = [
        'DataCompartment',
        'InfrastructureCompartment',
        'AlgorithmsCompartment',
        'MLToolbox',
        'AdvancedMLToolbox'
    ]
except ImportError:
    __all__ = [
        'DataCompartment',
        'InfrastructureCompartment',
        'AlgorithmsCompartment',
        'MLToolbox'
    ]


class MLToolbox:
    """
    Complete Machine Learning Toolbox
    
    Four compartments:
    1. Data: Preprocessing, validation, transformation
    2. Infrastructure: Kernels, AI components, LLM
    3. Algorithms: Models, evaluation, tuning, ensembles
    4. MLOps: Production deployment, monitoring, A/B testing, experiment tracking
    
    Also includes:
    - Medulla Oblongata System: Automatic resource regulation
    - Virtual Quantum Computer: CPU-based quantum simulation (optional)
    """
    
    def __init__(self, include_mlops: bool = True, auto_start_optimizer: bool = True):
        """
        Initialize ML Toolbox
        
        Args:
            include_mlops: Include MLOps compartment
            auto_start_optimizer: Automatically start Medulla Toolbox Optimizer
        """
        # Initialize Medulla Toolbox Optimizer (automatic ML operation optimization)
        self.optimizer = None
        if auto_start_optimizer:
            try:
                from medulla_toolbox_optimizer import MedullaToolboxOptimizer, MLTaskType
                self.optimizer = MedullaToolboxOptimizer(
                    max_cpu_percent=85.0,
                    max_memory_percent=80.0,
                    min_cpu_reserve=15.0,
                    min_memory_reserve_mb=1024.0,
                    enable_caching=True,
                    enable_adaptive_allocation=True
                )
                self.optimizer.start_regulation()
                self.MLTaskType = MLTaskType  # Expose for use
                print("[MLToolbox] Medulla Toolbox Optimizer started (automatic ML operation optimization)")
            except ImportError as e:
                print(f"[MLToolbox] Warning: Medulla optimizer not available: {e}")
            except Exception as e:
                print(f"[MLToolbox] Warning: Could not start Medulla optimizer: {e}")
        
        # Keep legacy medulla reference for backward compatibility
        self.medulla = self.optimizer
        
        # Initialize compartments (pass medulla to infrastructure)
        self.data = DataCompartment()
        self.infrastructure = InfrastructureCompartment(medulla=self.medulla)
        self.algorithms = AlgorithmsCompartment()
        
        # MLOps compartment (optional)
        if include_mlops and MLOPS_AVAILABLE:
            self.mlops = MLOpsCompartment()
        else:
            self.mlops = None
    
    def __repr__(self):
        mlops_info = f", mlops={len(self.mlops.components)}" if self.mlops else ""
        optimizer_info = ", optimizer=active" if self.optimizer and self.optimizer.regulation_running else ""
        return f"MLToolbox(data={len(self.data.components)}, infrastructure={len(self.infrastructure.components)}, algorithms={len(self.algorithms.components)}{mlops_info}{optimizer_info})"
    
    def __enter__(self):
        """Context manager entry - Medulla already started"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - Stop optimizer if running"""
        if self.optimizer and self.optimizer.regulation_running:
            self.optimizer.stop_regulation()
    
    def get_system_status(self):
        """Get optimizer system status"""
        if self.optimizer:
            return self.optimizer.get_system_status()
        return {"status": "optimizer_not_available"}
    
    def optimize_operation(self, operation_name: str, operation_func, task_type=None, use_cache: bool = True, *args, **kwargs):
        """
        Optimize an ML operation using Medulla optimizer
        
        Args:
            operation_name: Name of the operation (for caching)
            operation_func: Function to execute
            task_type: MLTaskType (DATA_PREPROCESSING, MODEL_TRAINING, etc.)
            use_cache: Whether to use result caching
            *args, **kwargs: Arguments to pass to operation_func
        
        Returns:
            Result of operation_func
        """
        if self.optimizer:
            if task_type is None:
                # Default to MODEL_TRAINING
                task_type = self.MLTaskType.MODEL_TRAINING if hasattr(self, 'MLTaskType') else None
            
            if task_type:
                return self.optimizer.optimize_operation(
                    operation_name,
                    operation_func,
                    task_type=task_type,
                    use_cache=use_cache,
                    *args,
                    **kwargs
                )
            else:
                # Fallback if optimizer not available
                return operation_func(*args, **kwargs)
        else:
            # No optimizer, just execute
            return operation_func(*args, **kwargs)
    
    def get_optimization_stats(self):
        """Get optimization statistics"""
        if self.optimizer:
            return self.optimizer.get_optimization_stats()
        return {"status": "optimizer_not_available"}
    
    def get_ml_math_optimizer(self):
        """Get ML Math Optimizer for optimized mathematical operations"""
        try:
            from ml_math_optimizer import get_ml_math_optimizer
            return get_ml_math_optimizer()
        except ImportError:
            raise ImportError("ML Math Optimizer not available. Install required dependencies.")
    
    def get_quantum_computer(self, num_qubits: int = 8, use_architecture_optimizations: bool = True):
        """
        Get a Virtual Quantum Computer instance (optional/experimental feature)
        
        Note: Quantum simulation is resource-intensive and provides no quantum advantage
        on regular laptops. Consider using ML Math Optimizer instead for better performance.
        """
        try:
            from virtual_quantum_computer import VirtualQuantumComputer
            # Quantum computer is optional - don't allocate resources by default
            return VirtualQuantumComputer(
                num_qubits=num_qubits,
                medulla=None,  # Don't allocate resources for quantum by default
                use_architecture_optimizations=use_architecture_optimizations
            )
        except ImportError:
            raise ImportError("Virtual Quantum Computer not available. Install required dependencies.")
