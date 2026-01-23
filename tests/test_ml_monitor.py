"""
Tests for ML Monitor
"""
import sys
from pathlib import Path
import pytest
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ml_monitor import ResourceMonitor, MonitoredMLToolbox
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    pytestmark = pytest.mark.skip("Features not available")


class TestResourceMonitor:
    """Tests for Resource Monitor"""
    
    def test_monitor_initialization(self):
        """Test monitor initialization"""
        monitor = ResourceMonitor()
        assert monitor is not None
        assert monitor.sample_interval == 0.1
    
    def test_get_current_usage(self):
        """Test getting current usage"""
        monitor = ResourceMonitor()
        usage = monitor.get_current_usage()
        
        assert 'available' in usage
        # Usage may not be available if psutil not installed
        if usage.get('available'):
            assert 'cpu_percent' in usage
            assert 'memory_mb' in usage
    
    def test_monitor_function_decorator(self):
        """Test function monitoring decorator"""
        monitor = ResourceMonitor()
        
        @monitor.monitor_function
        def test_function(n):
            time.sleep(0.01)
            return n * 2
        
        result = test_function(5)
        assert result == 10
        
        # Check monitoring data
        assert len(monitor.function_cpu) > 0 or len(monitor.function_memory) > 0
    
    def test_get_function_statistics(self):
        """Test getting function statistics"""
        monitor = ResourceMonitor()
        
        @monitor.monitor_function
        def test_function(n):
            time.sleep(0.001)
            return n
        
        # Run multiple times
        for i in range(5):
            test_function(i)
        
        stats = monitor.get_function_statistics()
        # Stats may be empty if psutil not available
        assert isinstance(stats, dict)
    
    def test_identify_resource_bottlenecks(self):
        """Test bottleneck identification"""
        monitor = ResourceMonitor()
        
        @monitor.monitor_function
        def test_function():
            time.sleep(0.01)
            return 1
        
        test_function()
        
        bottlenecks = monitor.identify_resource_bottlenecks()
        assert isinstance(bottlenecks, list)
    
    def test_generate_report(self):
        """Test report generation"""
        monitor = ResourceMonitor()
        
        @monitor.monitor_function
        def test_function():
            return 1
        
        test_function()
        
        report = monitor.generate_report()
        assert 'MONITORING REPORT' in report or 'RESOURCE MONITORING' in report
    
    def test_export_data(self):
        """Test data export"""
        import tempfile
        import os
        
        monitor = ResourceMonitor()
        
        @monitor.monitor_function
        def test_function():
            return 1
        
        test_function()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            monitor.export_data(temp_path)
            assert os.path.exists(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_reset(self):
        """Test monitor reset"""
        monitor = ResourceMonitor()
        
        @monitor.monitor_function
        def test_function():
            return 1
        
        test_function()
        
        monitor.reset()
        
        assert len(monitor.function_cpu) == 0
        assert len(monitor.function_memory) == 0


class TestMonitoredMLToolbox:
    """Tests for Monitored ML Toolbox"""
    
    def test_monitored_toolbox_initialization(self):
        """Test monitored toolbox initialization"""
        try:
            from ml_toolbox import MLToolbox
            toolbox = MLToolbox()
            monitored = MonitoredMLToolbox(toolbox)
            assert monitored.toolbox is not None
            # Monitor may be None if psutil not available
            assert monitored.monitor is not None or not hasattr(monitored, 'monitor')
        except ImportError:
            pytest.skip("ML Toolbox not available")
    
    def test_monitor_operation(self):
        """Test monitoring an operation"""
        try:
            from ml_toolbox import MLToolbox
            toolbox = MLToolbox()
            monitored = MonitoredMLToolbox(toolbox)
            
            def test_op():
                return 42
            
            result = monitored.monitor_operation('test', test_op)
            assert result == 42
        except ImportError:
            pytest.skip("ML Toolbox not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
