"""
ML Toolbox Resource Monitor
Track CPU and memory usage to identify bottlenecks

Features:
- Real-time CPU/memory monitoring
- Resource usage tracking
- Bottleneck identification
- Performance metrics
- Resource usage reports
- Integration with profiling system
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Callable
import time
import threading
from collections import defaultdict, deque
import json
import warnings
from datetime import datetime
from functools import wraps

sys.path.insert(0, str(Path(__file__).parent))

# Try to import psutil for system monitoring
try:
    import psutil
    import os
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Install with: pip install psutil")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class ResourceMonitor:
    """
    Resource Monitor
    
    Tracks CPU and memory usage for ML operations
    """
    
    def __init__(self, sample_interval: float = 0.1, max_samples: int = 10000):
        """
        Initialize resource monitor
        
        Args:
            sample_interval: Sampling interval in seconds
            max_samples: Maximum number of samples to keep
        """
        self.sample_interval = sample_interval
        self.max_samples = max_samples
        self.monitoring = False
        self.monitor_thread = None
        
        # Resource data
        self.cpu_samples = deque(maxlen=max_samples)
        self.memory_samples = deque(maxlen=max_samples)
        self.timestamps = deque(maxlen=max_samples)
        
        # Process-specific tracking
        self.process = None
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
        
        # Function-level tracking
        self.function_cpu = defaultdict(list)
        self.function_memory = defaultdict(list)
        self.function_timestamps = defaultdict(list)
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        if not PSUTIL_AVAILABLE:
            warnings.warn("psutil not available. Monitoring disabled.")
            return
        
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Monitoring loop"""
        while self.monitoring:
            try:
                # Get CPU usage
                cpu_percent = self.process.cpu_percent(interval=None)
                
                # Get memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
                
                # Store samples
                timestamp = time.time()
                self.cpu_samples.append(cpu_percent)
                self.memory_samples.append(memory_mb)
                self.timestamps.append(timestamp)
                
                time.sleep(self.sample_interval)
            except Exception as e:
                warnings.warn(f"Monitoring error: {e}")
                break
    
    def get_current_usage(self) -> Dict[str, float]:
        """
        Get current resource usage
        
        Returns:
            Dictionary with current CPU and memory usage
        """
        if not PSUTIL_AVAILABLE:
            return {'cpu_percent': 0.0, 'memory_mb': 0.0, 'available': False}
        
        try:
            cpu_percent = self.process.cpu_percent(interval=0.1)
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            return {
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'memory_percent': self.process.memory_percent(),
                'available': True,
                'timestamp': time.time()
            }
        except Exception as e:
            return {'error': str(e), 'available': False}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from collected samples
        
        Returns:
            Statistics dictionary
        """
        if not self.cpu_samples:
            return {'error': 'No samples collected'}
        
        cpu_array = np.array(self.cpu_samples) if NUMPY_AVAILABLE else list(self.cpu_samples)
        memory_array = np.array(self.memory_samples) if NUMPY_AVAILABLE else list(self.memory_samples)
        
        stats = {
            'cpu': {
                'mean': float(np.mean(cpu_array)) if NUMPY_AVAILABLE else sum(cpu_array) / len(cpu_array),
                'max': float(np.max(cpu_array)),
                'min': float(np.min(cpu_array)),
                'std': float(np.std(cpu_array)) if NUMPY_AVAILABLE else 0,
                'p95': float(np.percentile(cpu_array, 95)) if NUMPY_AVAILABLE else sorted(cpu_array)[int(len(cpu_array)*0.95)],
                'p99': float(np.percentile(cpu_array, 99)) if NUMPY_AVAILABLE else sorted(cpu_array)[int(len(cpu_array)*0.99)],
                'samples': len(self.cpu_samples)
            },
            'memory': {
                'mean_mb': float(np.mean(memory_array)) if NUMPY_AVAILABLE else sum(memory_array) / len(memory_array),
                'max_mb': float(np.max(memory_array)),
                'min_mb': float(np.min(memory_array)),
                'std_mb': float(np.std(memory_array)) if NUMPY_AVAILABLE else 0,
                'p95_mb': float(np.percentile(memory_array, 95)) if NUMPY_AVAILABLE else sorted(memory_array)[int(len(memory_array)*0.95)],
                'p99_mb': float(np.percentile(memory_array, 99)) if NUMPY_AVAILABLE else sorted(memory_array)[int(len(memory_array)*0.99)],
                'samples': len(self.memory_samples)
            },
            'duration': self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0
        }
        
        return stats
    
    def monitor_function(self, func: Callable) -> Callable:
        """
        Decorator to monitor function resource usage
        
        Args:
            func: Function to monitor
            
        Returns:
            Monitored function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Get baseline
            baseline = self.get_current_usage()
            baseline_memory = baseline.get('memory_mb', 0)
            baseline_cpu = baseline.get('cpu_percent', 0)
            
            # Execute function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                # Still record metrics even on error
                end_usage = self.get_current_usage()
                self._record_function_metrics(
                    func_name, start_time, time.time(),
                    baseline, end_usage
                )
                raise
            
            # Get usage after execution
            end_time = time.time()
            end_usage = self.get_current_usage()
            
            # Record metrics
            self._record_function_metrics(
                func_name, start_time, end_time,
                baseline, end_usage
            )
            
            return result
        
        return wrapper
    
    def _record_function_metrics(self, func_name: str, start_time: float, end_time: float,
                                baseline: Dict[str, float], end_usage: Dict[str, float]):
        """Record function-level metrics"""
        duration = end_time - start_time
        
        # Calculate deltas
        memory_delta = end_usage.get('memory_mb', 0) - baseline.get('memory_mb', 0)
        cpu_avg = (baseline.get('cpu_percent', 0) + end_usage.get('cpu_percent', 0)) / 2
        
        # Store metrics
        self.function_cpu[func_name].append({
            'cpu_percent': cpu_avg,
            'duration': duration,
            'timestamp': start_time
        })
        
        self.function_memory[func_name].append({
            'memory_mb': end_usage.get('memory_mb', 0),
            'memory_delta': memory_delta,
            'duration': duration,
            'timestamp': start_time
        })
    
    def get_function_statistics(self, func_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for monitored functions
        
        Args:
            func_name: Specific function name, or None for all
            
        Returns:
            Function statistics
        """
        if func_name:
            functions = [func_name]
        else:
            functions = list(set(list(self.function_cpu.keys()) + list(self.function_memory.keys())))
        
        stats = {}
        
        for func in functions:
            func_stats = {}
            
            # CPU statistics
            if func in self.function_cpu:
                cpu_data = self.function_cpu[func]
                cpu_values = [d['cpu_percent'] for d in cpu_data]
                
                if cpu_values:
                    func_stats['cpu'] = {
                        'mean_percent': float(np.mean(cpu_values)) if NUMPY_AVAILABLE else sum(cpu_values) / len(cpu_values),
                        'max_percent': float(np.max(cpu_values)),
                        'min_percent': float(np.min(cpu_values)),
                        'calls': len(cpu_data)
                    }
            
            # Memory statistics
            if func in self.function_memory:
                memory_data = self.function_memory[func]
                memory_values = [d['memory_mb'] for d in memory_data]
                memory_deltas = [d['memory_delta'] for d in memory_data]
                
                if memory_values:
                    func_stats['memory'] = {
                        'mean_mb': float(np.mean(memory_values)) if NUMPY_AVAILABLE else sum(memory_values) / len(memory_values),
                        'max_mb': float(np.max(memory_values)),
                        'mean_delta_mb': float(np.mean(memory_deltas)) if NUMPY_AVAILABLE else sum(memory_deltas) / len(memory_deltas),
                        'max_delta_mb': float(np.max(memory_deltas)),
                        'calls': len(memory_data)
                    }
            
            if func_stats:
                stats[func] = func_stats
        
        return stats
    
    def identify_resource_bottlenecks(self, cpu_threshold: float = 80.0, 
                                     memory_threshold_mb: float = 1000.0) -> List[Dict[str, Any]]:
        """
        Identify resource bottlenecks
        
        Args:
            cpu_threshold: CPU usage threshold (percent)
            memory_threshold_mb: Memory usage threshold (MB)
            
        Returns:
            List of bottlenecks
        """
        bottlenecks = []
        stats = self.get_statistics()
        func_stats = self.get_function_statistics()
        
        # System-level bottlenecks
        if 'cpu' in stats:
            cpu_stats = stats['cpu']
            if cpu_stats['mean'] > cpu_threshold or cpu_stats['p95'] > cpu_threshold:
                bottlenecks.append({
                    'type': 'cpu',
                    'level': 'system',
                    'metric': 'cpu_percent',
                    'value': cpu_stats['mean'],
                    'p95': cpu_stats['p95'],
                    'threshold': cpu_threshold,
                    'severity': 'high' if cpu_stats['mean'] > 90 else 'medium',
                    'recommendation': 'Consider parallelization, optimization, or reducing workload'
                })
        
        if 'memory' in stats:
            memory_stats = stats['memory']
            if memory_stats['mean_mb'] > memory_threshold_mb or memory_stats['p95_mb'] > memory_threshold_mb:
                bottlenecks.append({
                    'type': 'memory',
                    'level': 'system',
                    'metric': 'memory_mb',
                    'value': memory_stats['mean_mb'],
                    'p95': memory_stats['p95_mb'],
                    'threshold': memory_threshold_mb,
                    'severity': 'high' if memory_stats['mean_mb'] > memory_threshold_mb * 2 else 'medium',
                    'recommendation': 'Consider memory-efficient algorithms, data streaming, or reducing batch size'
                })
        
        # Function-level bottlenecks
        for func_name, func_stat in func_stats.items():
            if 'cpu' in func_stat:
                cpu_stat = func_stat['cpu']
                if cpu_stat['mean_percent'] > cpu_threshold:
                    bottlenecks.append({
                        'type': 'cpu',
                        'level': 'function',
                        'function': func_name,
                        'metric': 'cpu_percent',
                        'value': cpu_stat['mean_percent'],
                        'max': cpu_stat['max_percent'],
                        'calls': cpu_stat['calls'],
                        'threshold': cpu_threshold,
                        'severity': 'high' if cpu_stat['mean_percent'] > 90 else 'medium',
                        'recommendation': f'Optimize {func_name} or consider parallelization'
                    })
            
            if 'memory' in func_stat:
                memory_stat = func_stat['memory']
                if memory_stat['mean_mb'] > memory_threshold_mb:
                    bottlenecks.append({
                        'type': 'memory',
                        'level': 'function',
                        'function': func_name,
                        'metric': 'memory_mb',
                        'value': memory_stat['mean_mb'],
                        'max': memory_stat['max_mb'],
                        'delta': memory_stat['mean_delta_mb'],
                        'calls': memory_stat['calls'],
                        'threshold': memory_threshold_mb,
                        'severity': 'high' if memory_stat['mean_mb'] > memory_threshold_mb * 2 else 'medium',
                        'recommendation': f'Optimize {func_name} memory usage or use memory-efficient algorithms'
                    })
        
        # Sort by severity and value
        bottlenecks.sort(key=lambda x: (
            {'high': 0, 'medium': 1, 'low': 2}.get(x.get('severity', 'low'), 2),
            -x.get('value', 0)
        ))
        
        return bottlenecks
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate resource monitoring report
        
        Args:
            output_file: Optional file path to save report
            
        Returns:
            Report string
        """
        stats = self.get_statistics()
        func_stats = self.get_function_statistics()
        bottlenecks = self.identify_resource_bottlenecks()
        
        report = f"""
{'='*80}
ML TOOLBOX RESOURCE MONITORING REPORT
{'='*80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        # System-level statistics
        if 'cpu' in stats and 'memory' in stats:
            report += f"""
SYSTEM RESOURCE USAGE
{'-'*80}

CPU Usage:
  Mean: {stats['cpu']['mean']:.2f}%
  Max: {stats['cpu']['max']:.2f}%
  P95: {stats['cpu']['p95']:.2f}%
  Samples: {stats['cpu']['samples']:,}

Memory Usage:
  Mean: {stats['memory']['mean_mb']:.2f} MB
  Max: {stats['memory']['max_mb']:.2f} MB
  P95: {stats['memory']['p95_mb']:.2f} MB
  Samples: {stats['memory']['samples']:,}

Monitoring Duration: {stats.get('duration', 0):.2f} seconds

"""
        
        # Function-level statistics
        if func_stats:
            report += f"""
FUNCTION-LEVEL RESOURCE USAGE
{'-'*80}

"""
            for func_name, func_stat in sorted(func_stats.items(), 
                                               key=lambda x: x[1].get('cpu', {}).get('mean_percent', 0) 
                                               if 'cpu' in x[1] else 0, reverse=True)[:10]:
                report += f"{func_name}:\n"
                
                if 'cpu' in func_stat:
                    cpu = func_stat['cpu']
                    report += f"  CPU: Mean {cpu['mean_percent']:.2f}%, Max {cpu['max_percent']:.2f}% ({cpu['calls']} calls)\n"
                
                if 'memory' in func_stat:
                    mem = func_stat['memory']
                    report += f"  Memory: Mean {mem['mean_mb']:.2f} MB, Max {mem['max_mb']:.2f} MB\n"
                    report += f"  Memory Delta: {mem['mean_delta_mb']:.2f} MB per call\n"
                
                report += "\n"
        
        # Bottlenecks
        if bottlenecks:
            report += f"""
RESOURCE BOTTLENECKS IDENTIFIED
{'-'*80}

Found {len(bottlenecks)} resource bottlenecks:

"""
            for i, bottleneck in enumerate(bottlenecks[:10], 1):
                report += f"""
{i}. {bottleneck['type'].upper()} Bottleneck [{bottleneck.get('severity', 'unknown').upper()}]
   Level: {bottleneck.get('level', 'unknown')}
"""
                if bottleneck.get('level') == 'function':
                    report += f"   Function: {bottleneck.get('function', 'unknown')}\n"
                
                report += f"   Metric: {bottleneck.get('metric', 'unknown')}\n"
                report += f"   Value: {bottleneck.get('value', 0):.2f}\n"
                if 'p95' in bottleneck:
                    report += f"   P95: {bottleneck['p95']:.2f}\n"
                report += f"   Threshold: {bottleneck.get('threshold', 0):.2f}\n"
                report += f"   Recommendation: {bottleneck.get('recommendation', 'N/A')}\n"
        
        # Recommendations
        report += f"""
OPTIMIZATION RECOMMENDATIONS
{'-'*80}

"""
        
        if bottlenecks:
            high_cpu = [b for b in bottlenecks if b['type'] == 'cpu' and b.get('severity') == 'high']
            high_memory = [b for b in bottlenecks if b['type'] == 'memory' and b.get('severity') == 'high']
            
            if high_cpu:
                report += "High CPU Usage:\n"
                report += "  • Consider parallelization for CPU-intensive operations\n"
                report += "  • Optimize algorithms to reduce computational complexity\n"
                report += "  • Use batch processing to reduce overhead\n"
                report += "  • Consider using GPU acceleration if available\n\n"
            
            if high_memory:
                report += "High Memory Usage:\n"
                report += "  • Use memory-efficient data structures\n"
                report += "  • Implement data streaming for large datasets\n"
                report += "  • Reduce batch sizes\n"
                report += "  • Clear unused variables and cache\n"
                report += "  • Consider using generators instead of lists\n\n"
        
        report += f"""
{'='*80}
End of Report
{'='*80}
"""
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
    
    def export_data(self, output_file: str):
        """Export monitoring data to JSON"""
        data = {
            'statistics': self.get_statistics(),
            'function_statistics': self.get_function_statistics(),
            'bottlenecks': self.identify_resource_bottlenecks(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert numpy types
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        data = convert_types(data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def reset(self):
        """Reset all monitoring data"""
        self.cpu_samples.clear()
        self.memory_samples.clear()
        self.timestamps.clear()
        self.function_cpu.clear()
        self.function_memory.clear()
        self.function_timestamps.clear()


class MonitoredMLToolbox:
    """
    Monitored wrapper for ML Toolbox
    
    Automatically monitors all operations
    """
    
    def __init__(self, toolbox=None, enable_monitoring: bool = True):
        """
        Args:
            toolbox: MLToolbox instance
            enable_monitoring: Enable monitoring
        """
        if toolbox is None:
            try:
                from ml_toolbox import MLToolbox
                toolbox = MLToolbox()
            except ImportError:
                raise ImportError("ML Toolbox not available")
        
        self.toolbox = toolbox
        self.monitor = ResourceMonitor() if enable_monitoring else None
        
        if self.monitor:
            self.monitor.start_monitoring()
    
    def monitor_operation(self, operation_name: str, func: Callable, *args, **kwargs):
        """
        Monitor a single operation
        
        Args:
            operation_name: Name of the operation
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        if self.monitor:
            monitored_func = self.monitor.monitor_function(func)
            return monitored_func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def get_monitoring_report(self) -> str:
        """Get monitoring report"""
        if self.monitor:
            return self.monitor.generate_report()
        else:
            return "Monitoring not enabled"
    
    def get_resource_bottlenecks(self) -> List[Dict[str, Any]]:
        """Get identified resource bottlenecks"""
        if self.monitor:
            return self.monitor.identify_resource_bottlenecks()
        else:
            return []
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        if self.monitor:
            return self.monitor.get_current_usage()
        else:
            return {'available': False}
    
    def __del__(self):
        """Cleanup"""
        if self.monitor:
            self.monitor.stop_monitoring()


# Example usage
if __name__ == '__main__':
    # Create monitor
    monitor = ResourceMonitor(sample_interval=0.1)
    monitor.start_monitoring()
    
    # Monitor a function
    @monitor.monitor_function
    def example_function(n: int):
        """Example function to monitor"""
        result = 0
        for i in range(n):
            result += i ** 2
        return result
    
    # Run function
    for _ in range(10):
        example_function(10000)
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Generate report
    report = monitor.generate_report('monitoring_report.txt')
    print(report)
    
    # Get bottlenecks
    bottlenecks = monitor.identify_resource_bottlenecks()
    print("\nResource Bottlenecks:")
    for bottleneck in bottlenecks:
        print(f"  {bottleneck['type']}: {bottleneck.get('value', 0):.2f}")
