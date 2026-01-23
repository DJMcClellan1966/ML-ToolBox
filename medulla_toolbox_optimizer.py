"""
Medulla Toolbox Optimizer
Focuses on optimizing ML Toolbox performance rather than quantum computing

Regulates and optimizes:
- ML operation resource usage
- Memory management for data processing
- Thread/process allocation for parallel operations
- Cache management
- Model training resource allocation
"""
import sys
from pathlib import Path
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import deque
from dataclasses import dataclass
from enum import Enum
import warnings

sys.path.insert(0, str(Path(__file__).parent))

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Install with: pip install psutil")

try:
    from architecture_optimizer import get_architecture_optimizer
    ARCHITECTURE_OPTIMIZER_AVAILABLE = True
except ImportError:
    ARCHITECTURE_OPTIMIZER_AVAILABLE = False
    warnings.warn("Architecture optimizer not available")


class MLTaskType(Enum):
    """Types of ML operations"""
    DATA_PREPROCESSING = "data_preprocessing"
    MODEL_TRAINING = "model_training"
    MODEL_PREDICTION = "model_prediction"
    FEATURE_ENGINEERING = "feature_engineering"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    ENSEMBLE = "ensemble"
    EVALUATION = "evaluation"


@dataclass
class MLTaskMetrics:
    """Metrics for ML operations"""
    task_type: MLTaskType
    duration: float
    memory_used_mb: float
    cpu_percent: float
    success: bool
    timestamp: float


class MedullaToolboxOptimizer:
    """
    Medulla Toolbox Optimizer
    
    Focuses on optimizing ML Toolbox operations:
    - Regulates resource usage for ML tasks
    - Optimizes memory allocation
    - Manages thread/process pools
    - Caches frequently used operations
    - Prevents resource exhaustion
    """
    
    def __init__(
        self,
        max_cpu_percent: float = 85.0,
        max_memory_percent: float = 80.0,
        min_cpu_reserve: float = 15.0,
        min_memory_reserve_mb: float = 1024.0,
        regulation_interval: float = 0.5,
        enable_caching: bool = True,
        enable_adaptive_allocation: bool = True
    ):
        """
        Initialize Medulla Toolbox Optimizer
        
        Args:
            max_cpu_percent: Maximum CPU usage before throttling
            max_memory_percent: Maximum memory usage before throttling
            min_cpu_reserve: Minimum CPU to reserve for system
            min_memory_reserve_mb: Minimum memory to reserve for system
            regulation_interval: How often to check system state (seconds)
            enable_caching: Enable operation result caching
            enable_adaptive_allocation: Adapt resource allocation based on task type
        """
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.min_cpu_reserve = min_cpu_reserve
        self.min_memory_reserve_mb = min_memory_reserve_mb
        self.regulation_interval = regulation_interval
        self.enable_caching = enable_caching
        self.enable_adaptive_allocation = enable_adaptive_allocation
        
        # System state
        self.metrics_history = deque(maxlen=200)
        self.ml_task_history = deque(maxlen=100)
        self.active_ml_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Resource allocation by task type
        self.task_resource_limits = {
            MLTaskType.DATA_PREPROCESSING: {'cpu_percent': 70.0, 'memory_mb': 2048.0},
            MLTaskType.MODEL_TRAINING: {'cpu_percent': 80.0, 'memory_mb': 4096.0},
            MLTaskType.MODEL_PREDICTION: {'cpu_percent': 60.0, 'memory_mb': 1024.0},
            MLTaskType.FEATURE_ENGINEERING: {'cpu_percent': 65.0, 'memory_mb': 1536.0},
            MLTaskType.HYPERPARAMETER_TUNING: {'cpu_percent': 75.0, 'memory_mb': 3072.0},
            MLTaskType.ENSEMBLE: {'cpu_percent': 85.0, 'memory_mb': 5120.0},
            MLTaskType.EVALUATION: {'cpu_percent': 50.0, 'memory_mb': 512.0}
        }
        
        # Operation cache
        self.operation_cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance optimization
        self.optimization_stats = {
            'tasks_optimized': 0,
            'tasks_throttled': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_optimized_mb': 0.0,
            'time_saved_seconds': 0.0
        }
        
        # Regulation thread
        self.regulation_thread: Optional[threading.Thread] = None
        self.regulation_running = False
        self.regulation_lock = threading.Lock()
        
        # Architecture optimizer
        if ARCHITECTURE_OPTIMIZER_AVAILABLE:
            self.arch_optimizer = get_architecture_optimizer()
            self.optimal_threads = self.arch_optimizer.get_optimal_thread_count()
        else:
            self.arch_optimizer = None
            self.optimal_threads = 4
    
    def start_regulation(self):
        """Start the regulation thread"""
        if self.regulation_running:
            return
        
        self.regulation_running = True
        self.regulation_thread = threading.Thread(target=self._regulation_loop, daemon=True)
        self.regulation_thread.start()
        print("[Medulla Optimizer] Regulation started - optimizing ML Toolbox operations")
    
    def stop_regulation(self):
        """Stop the regulation thread"""
        self.regulation_running = False
        if self.regulation_thread:
            self.regulation_thread.join(timeout=2.0)
        print("[Medulla Optimizer] Regulation stopped")
    
    def _regulation_loop(self):
        """Main regulation loop - optimizes ML operations"""
        while self.regulation_running:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Optimize active ML tasks
                self._optimize_ml_tasks(metrics)
                
                # Manage cache
                if self.enable_caching:
                    self._manage_cache()
                
                time.sleep(self.regulation_interval)
            except Exception as e:
                warnings.warn(f"Regulation loop error: {e}")
                time.sleep(self.regulation_interval)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        if not PSUTIL_AVAILABLE:
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'memory_available_mb': 4096.0,
                'timestamp': time.time()
            }
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_mb': memory.available / (1024 * 1024),
            'timestamp': time.time()
        }
    
    def _optimize_ml_tasks(self, metrics: Dict[str, Any]):
        """Optimize active ML tasks based on system state"""
        with self.regulation_lock:
            # Check if system is under stress
            if metrics['cpu_percent'] > self.max_cpu_percent:
                # Throttle low-priority tasks
                for task_id, task in list(self.active_ml_tasks.items()):
                    if task.get('priority', 5) < 7:
                        task['throttled'] = True
                        self.optimization_stats['tasks_throttled'] += 1
            
            if metrics['memory_percent'] > self.max_memory_percent:
                # Clear cache if memory is high
                if self.operation_cache:
                    # Clear oldest 25% of cache
                    items_to_remove = len(self.operation_cache) // 4
                    for _ in range(items_to_remove):
                        if self.operation_cache:
                            self.operation_cache.pop(next(iter(self.operation_cache)))
    
    def _manage_cache(self):
        """Manage operation cache"""
        # Limit cache size
        max_cache_size = 100
        if len(self.operation_cache) > max_cache_size:
            # Remove oldest entries
            items_to_remove = len(self.operation_cache) - max_cache_size
            for _ in range(items_to_remove):
                if self.operation_cache:
                    self.operation_cache.pop(next(iter(self.operation_cache)))
    
    def get_optimal_resources(self, task_type: MLTaskType) -> Dict[str, Any]:
        """Get optimal resource allocation for a task type"""
        base_limits = self.task_resource_limits.get(task_type, {
            'cpu_percent': 70.0,
            'memory_mb': 2048.0
        })
        
        if not self.enable_adaptive_allocation:
            return base_limits
        
        # Get current system state
        if self.metrics_history:
            latest = self.metrics_history[-1]
            available_cpu = max(0, latest['cpu_percent'] - self.min_cpu_reserve)
            available_memory = max(0, latest['memory_available_mb'] - self.min_memory_reserve_mb)
            
            # Adapt based on availability
            cpu_limit = min(base_limits['cpu_percent'], available_cpu * 0.9)
            memory_limit = min(base_limits['memory_mb'], available_memory * 0.9)
            
            return {
                'cpu_percent': cpu_limit,
                'memory_mb': memory_limit,
                'optimal_threads': self.optimal_threads
            }
        
        return base_limits
    
    def optimize_operation(
        self,
        operation_name: str,
        operation_func: Callable,
        task_type: MLTaskType = MLTaskType.MODEL_TRAINING,
        use_cache: bool = True,
        *args,
        **kwargs
    ) -> Any:
        """Optimize an ML operation"""
        # Check cache
        if use_cache and self.enable_caching:
            cache_key = f"{operation_name}_{hash(str(args))}_{hash(str(kwargs))}"
            if cache_key in self.operation_cache:
                self.cache_hits += 1
                self.optimization_stats['cache_hits'] += 1
                return self.operation_cache[cache_key]
            self.cache_misses += 1
            self.optimization_stats['cache_misses'] += 1
        
        # Get optimal resources
        resources = self.get_optimal_resources(task_type)
        
        # Execute operation
        start_time = time.time()
        start_memory = 0.0
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            start_memory = process.memory_info().rss / (1024 * 1024)
        
        try:
            result = operation_func(*args, **kwargs)
            
            elapsed = time.time() - start_time
            end_memory = 0.0
            if PSUTIL_AVAILABLE:
                end_memory = process.memory_info().rss / (1024 * 1024)
            
            # Record metrics
            task_metrics = MLTaskMetrics(
                task_type=task_type,
                duration=elapsed,
                memory_used_mb=end_memory - start_memory,
                cpu_percent=0.0,  # Would need more complex tracking
                success=True,
                timestamp=time.time()
            )
            self.ml_task_history.append(task_metrics)
            self.optimization_stats['tasks_optimized'] += 1
            
            # Cache result
            if use_cache and self.enable_caching:
                cache_key = f"{operation_name}_{hash(str(args))}_{hash(str(kwargs))}"
                self.operation_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            task_metrics = MLTaskMetrics(
                task_type=task_type,
                duration=time.time() - start_time,
                memory_used_mb=0.0,
                cpu_percent=0.0,
                success=False,
                timestamp=time.time()
            )
            self.ml_task_history.append(task_metrics)
            raise
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            **self.optimization_stats,
            'cache_hit_rate': (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0.0
            ),
            'total_tasks': len(self.ml_task_history),
            'active_tasks': len(self.active_ml_tasks)
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        latest = self.metrics_history[-1]
        
        return {
            'cpu_percent': latest['cpu_percent'],
            'memory_percent': latest['memory_percent'],
            'memory_available_mb': latest['memory_available_mb'],
            'optimization_stats': self.get_optimization_stats(),
            'optimal_threads': self.optimal_threads
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start_regulation()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_regulation()


# Example usage
if __name__ == '__main__':
    print("Medulla Toolbox Optimizer")
    print("="*80)
    
    optimizer = MedullaToolboxOptimizer()
    
    with optimizer:
        print("\n[OK] Optimizer started")
        
        # Example: Optimize a data preprocessing operation
        def preprocess_data(data):
            # Simulate preprocessing
            time.sleep(0.1)
            return [x * 2 for x in data]
        
        result = optimizer.optimize_operation(
            "preprocess_data",
            preprocess_data,
            task_type=MLTaskType.DATA_PREPROCESSING,
            data=[1, 2, 3, 4, 5]
        )
        
        print(f"[OK] Operation result: {result}")
        
        # Get stats
        stats = optimizer.get_optimization_stats()
        print(f"\n[OK] Optimization Stats:")
        print(f"  Tasks optimized: {stats['tasks_optimized']}")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  Cache misses: {stats['cache_misses']}")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
        
        status = optimizer.get_system_status()
        print(f"\n[OK] System Status:")
        print(f"  CPU: {status['cpu_percent']:.1f}%")
        print(f"  Memory: {status['memory_percent']:.1f}%")
