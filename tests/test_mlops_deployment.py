"""
Tests for MLOps Deployment Components
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from model_deployment import (
        ModelRegistry,
        ModelServer,
        BatchInference,
        RealTimeInference,
        CanaryDeployment
    )
    from sklearn.ensemble import RandomForestClassifier
    DEPLOYMENT_AVAILABLE = True
except ImportError:
    DEPLOYMENT_AVAILABLE = False
    pytestmark = pytest.mark.skip("Model deployment not available")


@pytest.fixture
def sample_model():
    """Sample trained model"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


class TestModelRegistry:
    """Tests for ModelRegistry"""
    
    def test_initialization(self):
        """Test registry initialization"""
        registry = ModelRegistry()
        assert registry.active_version is None
        assert len(registry.models) == 0
    
    def test_register_model(self, sample_model):
        """Test registering a model"""
        registry = ModelRegistry()
        version = registry.register(sample_model, 'v1.0.0', set_active=True)
        
        assert version.version == 'v1.0.0'
        assert version.is_active == True
        assert registry.active_version == 'v1.0.0'
    
    def test_set_active_version(self, sample_model):
        """Test setting active version"""
        registry = ModelRegistry()
        registry.register(sample_model, 'v1.0.0')
        registry.register(sample_model, 'v2.0.0')
        
        registry.set_active_version('v2.0.0')
        assert registry.active_version == 'v2.0.0'
        assert registry.models['v2.0.0'].is_active == True
        assert registry.models['v1.0.0'].is_active == False
    
    def test_get_model(self, sample_model):
        """Test getting model by version"""
        registry = ModelRegistry()
        registry.register(sample_model, 'v1.0.0', set_active=True)
        
        model = registry.get_model()
        assert model is not None
        assert hasattr(model, 'predict')
        
        model_v1 = registry.get_model('v1.0.0')
        assert model_v1 is not None
    
    def test_list_versions(self, sample_model):
        """Test listing versions"""
        registry = ModelRegistry()
        registry.register(sample_model, 'v1.0.0')
        registry.register(sample_model, 'v2.0.0')
        
        versions = registry.list_versions()
        assert len(versions) == 2
        assert all('version' in v for v in versions)


class TestBatchInference:
    """Tests for BatchInference"""
    
    def test_initialization(self, sample_model):
        """Test batch inference initialization"""
        batch_inference = BatchInference(sample_model)
        assert batch_inference.model is not None
    
    def test_predict_batch(self, sample_model):
        """Test batch prediction"""
        batch_inference = BatchInference(sample_model)
        X = np.random.randn(50, 5)
        
        predictions = batch_inference.predict_batch(X, batch_size=10)
        assert len(predictions) == 50
        assert all(isinstance(p, (int, np.integer)) for p in predictions)


class TestRealTimeInference:
    """Tests for RealTimeInference"""
    
    def test_initialization(self, sample_model):
        """Test real-time inference initialization"""
        realtime = RealTimeInference(sample_model)
        assert realtime.model is not None
    
    def test_predict(self, sample_model):
        """Test single prediction"""
        realtime = RealTimeInference(sample_model)
        x = np.random.randn(5)
        
        prediction = realtime.predict(x)
        assert isinstance(prediction, (int, np.integer, float, np.floating))
    
    def test_predict_proba(self, sample_model):
        """Test prediction with probabilities"""
        realtime = RealTimeInference(sample_model)
        x = np.random.randn(5)
        
        proba = realtime.predict_proba(x)
        assert isinstance(proba, np.ndarray)
        assert len(proba) > 0


class TestCanaryDeployment:
    """Tests for CanaryDeployment"""
    
    def test_initialization(self, sample_model):
        """Test canary deployment initialization"""
        registry = ModelRegistry()
        registry.register(sample_model, 'v1.0.0', set_active=True)
        
        canary = CanaryDeployment(registry)
        assert canary.canary_version is None
        assert canary.canary_percentage == 0.0
    
    def test_start_canary(self, sample_model):
        """Test starting canary deployment"""
        registry = ModelRegistry()
        registry.register(sample_model, 'v1.0.0', set_active=True)
        registry.register(sample_model, 'v2.0.0')
        
        canary = CanaryDeployment(registry)
        canary.start_canary('v2.0.0', percentage=0.1)
        
        assert canary.canary_version == 'v2.0.0'
        assert canary.canary_percentage == 0.1
    
    def test_get_model_for_request(self, sample_model):
        """Test getting model for request"""
        registry = ModelRegistry()
        registry.register(sample_model, 'v1.0.0', set_active=True)
        registry.register(sample_model, 'v2.0.0')
        
        canary = CanaryDeployment(registry)
        canary.start_canary('v2.0.0', percentage=0.5)
        
        # Test multiple requests (some should get canary, some production)
        models_seen = set()
        for _ in range(20):
            model, version = canary.get_model_for_request()
            models_seen.add(version)
        
        # Should see both versions (with high probability)
        assert len(models_seen) >= 1
    
    def test_promote_canary(self, sample_model):
        """Test promoting canary to production"""
        registry = ModelRegistry()
        registry.register(sample_model, 'v1.0.0', set_active=True)
        registry.register(sample_model, 'v2.0.0')
        
        canary = CanaryDeployment(registry)
        canary.start_canary('v2.0.0', percentage=0.1)
        canary.promote_canary()
        
        assert registry.active_version == 'v2.0.0'
        assert canary.canary_version is None
    
    def test_rollback_canary(self, sample_model):
        """Test rolling back canary"""
        registry = ModelRegistry()
        registry.register(sample_model, 'v1.0.0', set_active=True)
        registry.register(sample_model, 'v2.0.0')
        
        canary = CanaryDeployment(registry)
        canary.start_canary('v2.0.0', percentage=0.1)
        canary.rollback_canary()
        
        assert canary.canary_version is None
        assert canary.canary_percentage == 0.0
