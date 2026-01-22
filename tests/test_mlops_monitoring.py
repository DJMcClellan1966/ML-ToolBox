"""
Tests for MLOps Monitoring Components
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from model_monitoring import (
        DataDriftDetector,
        ConceptDriftDetector,
        PerformanceMonitor,
        ModelMonitor
    )
    from sklearn.ensemble import RandomForestClassifier
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    pytestmark = pytest.mark.skip("Model monitoring not available")


@pytest.fixture
def sample_data():
    """Sample data for testing"""
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(50, 5)
    y_test = np.random.randint(0, 2, 50)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def sample_model(sample_data):
    """Sample trained model"""
    X_train, y_train, _, _ = sample_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model


class TestDataDriftDetector:
    """Tests for DataDriftDetector"""
    
    def test_initialization(self, sample_data):
        """Test detector initialization"""
        X_train, _, _, _ = sample_data
        detector = DataDriftDetector(X_train, alpha=0.05)
        assert detector.reference_data.shape == X_train.shape
        assert detector.alpha == 0.05
    
    def test_detect_drift_ks_test(self, sample_data):
        """Test drift detection with KS test"""
        X_train, _, X_test, _ = sample_data
        detector = DataDriftDetector(X_train)
        result = detector.detect_drift(X_test, method='ks_test')
        
        assert 'has_drift' in result
        assert 'drift_by_feature' in result
        assert 'method' in result
        assert result['method'] == 'ks_test'
    
    def test_detect_drift_psi(self, sample_data):
        """Test drift detection with PSI"""
        X_train, _, X_test, _ = sample_data
        detector = DataDriftDetector(X_train)
        result = detector.detect_drift(X_test, method='psi')
        
        assert 'has_drift' in result
        assert 'method' in result
        assert result['method'] == 'psi'


class TestConceptDriftDetector:
    """Tests for ConceptDriftDetector"""
    
    def test_initialization(self):
        """Test detector initialization"""
        detector = ConceptDriftDetector(baseline_performance=0.9, threshold=0.1)
        assert detector.baseline_performance == 0.9
        assert detector.threshold == 0.1
    
    def test_detect_drift(self):
        """Test concept drift detection"""
        detector = ConceptDriftDetector(baseline_performance=0.9, threshold=0.1)
        result = detector.detect_drift(current_performance=0.75)
        
        assert 'has_drift' in result
        assert 'baseline_performance' in result
        assert 'current_performance' in result
        assert result['has_drift'] == True  # 0.75 < 0.9 - 0.1
    
    def test_detect_trend(self):
        """Test trend detection"""
        detector = ConceptDriftDetector(baseline_performance=0.9)
        # Add some performance history
        for i in range(10):
            detector.detect_drift(0.9 - i * 0.01)
        
        result = detector.detect_trend(window_size=10)
        assert 'is_degrading' in result
        assert 'trend' in result


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor"""
    
    def test_initialization(self):
        """Test monitor initialization"""
        monitor = PerformanceMonitor(model_name='test_model')
        assert monitor.model_name == 'test_model'
    
    def test_record_prediction(self, sample_data):
        """Test recording predictions"""
        monitor = PerformanceMonitor()
        _, y_train, _, y_test = sample_data
        
        y_pred = np.random.randint(0, 2, len(y_test))
        result = monitor.record_prediction(y_test, y_pred, latency=0.1)
        
        assert 'metrics' in result
        assert 'latency' in result
        assert result['latency'] == 0.1
    
    def test_get_summary(self, sample_data):
        """Test getting performance summary"""
        monitor = PerformanceMonitor()
        _, y_train, _, y_test = sample_data
        
        # Record some predictions
        for _ in range(10):
            y_pred = np.random.randint(0, 2, len(y_test))
            monitor.record_prediction(y_test, y_pred, latency=0.1)
        
        summary = monitor.get_summary()
        assert 'accuracy' in summary or 'mse' in summary
        assert 'n_predictions' in summary


class TestModelMonitor:
    """Tests for ModelMonitor"""
    
    def test_initialization(self, sample_data, sample_model):
        """Test monitor initialization"""
        X_train, y_train, _, _ = sample_data
        monitor = ModelMonitor(
            sample_model,
            X_train,
            y_train,
            baseline_performance=0.9
        )
        assert monitor.model_name == 'default'
        assert monitor.data_drift_detector is not None
        assert monitor.concept_drift_detector is not None
    
    def test_monitor(self, sample_data, sample_model):
        """Test comprehensive monitoring"""
        X_train, y_train, X_test, y_test = sample_data
        monitor = ModelMonitor(
            sample_model,
            X_train,
            y_train,
            baseline_performance=0.9
        )
        
        result = monitor.monitor(
            X_test,
            y_test,
            check_data_drift=True,
            check_concept_drift=True,
            check_performance=True
        )
        
        assert 'timestamp' in result
        assert 'data_drift' in result
        assert 'concept_drift' in result
        assert 'performance' in result
    
    def test_get_alerts(self, sample_data, sample_model):
        """Test getting alerts"""
        X_train, y_train, X_test, y_test = sample_data
        monitor = ModelMonitor(
            sample_model,
            X_train,
            y_train,
            baseline_performance=0.9
        )
        
        # Trigger some alerts
        monitor.monitor(X_test, y_test)
        
        alerts = monitor.get_alerts()
        assert isinstance(alerts, list)
    
    def test_get_summary(self, sample_data, sample_model):
        """Test getting monitoring summary"""
        X_train, y_train, X_test, y_test = sample_data
        monitor = ModelMonitor(
            sample_model,
            X_train,
            y_train,
            baseline_performance=0.9
        )
        
        monitor.monitor(X_test, y_test)
        summary = monitor.get_summary()
        
        assert 'model_name' in summary
        assert 'performance_summary' in summary
        assert 'n_alerts' in summary
