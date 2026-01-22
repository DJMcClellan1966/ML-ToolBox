"""
Tests for Phase 1 Focus Items
Interactive Dashboard, Model Registry, Pre-trained Hub
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from interactive_dashboard import InteractiveDashboard
    from model_registry import ModelRegistry, ModelStage, ModelVersion
    from pretrained_model_hub import PretrainedModelHub, PretrainedModel
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    pytestmark = pytest.mark.skip("Features not available")


class TestInteractiveDashboard:
    """Tests for interactive dashboard"""
    
    def test_log_experiment_with_history(self):
        """Test logging experiment with training history"""
        dashboard = InteractiveDashboard(storage_path="test_experiments.json")
        
        exp_id = dashboard.log_experiment(
            'test_exp',
            {'accuracy': 0.95, 'loss': 0.05},
            {'lr': 0.001},
            training_history={
                'loss': [0.5, 0.3, 0.1, 0.05],
                'accuracy': [0.6, 0.8, 0.9, 0.95]
            }
        )
        
        assert exp_id is not None
        
        # Cleanup
        if os.path.exists("test_experiments.json"):
            os.remove("test_experiments.json")
    
    def test_generate_dashboard(self):
        """Test dashboard generation"""
        dashboard = InteractiveDashboard(storage_path="test_experiments.json")
        dashboard.log_experiment(
            'test',
            {'accuracy': 0.9},
            {'lr': 0.001},
            training_history={'loss': [0.5, 0.3], 'accuracy': [0.6, 0.8]}
        )
        
        html = dashboard.generate_interactive_dashboard()
        assert '<html>' in html
        assert 'test' in html
        
        # Cleanup
        if os.path.exists("test_experiments.json"):
            os.remove("test_experiments.json")
    
    def test_save_dashboard(self):
        """Test saving dashboard"""
        dashboard = InteractiveDashboard(storage_path="test_experiments.json")
        dashboard.log_experiment('test', {'accuracy': 0.9}, {})
        
        path = dashboard.save_dashboard("test_dashboard.html")
        assert os.path.exists(path)
        
        # Cleanup
        if os.path.exists("test_experiments.json"):
            os.remove("test_experiments.json")
        if os.path.exists("test_dashboard.html"):
            os.remove("test_dashboard.html")


class TestModelRegistry:
    """Tests for model registry"""
    
    def test_register_model(self):
        """Test model registration"""
        registry = ModelRegistry(registry_path="test_registry")
        
        # Create dummy model
        class DummyModel:
            def __init__(self):
                self.weights = [1, 2, 3]
        
        model = DummyModel()
        version = registry.register_model(
            model,
            {'accuracy': 0.95, 'loss': 0.05},
            version='1.0.0',
            stage=ModelStage.DEV
        )
        
        assert version == '1.0.0'
        assert registry.get_model('1.0.0') is not None
        
        # Cleanup
        import shutil
        if os.path.exists("test_registry"):
            shutil.rmtree("test_registry")
    
    def test_promote_model(self):
        """Test model promotion"""
        registry = ModelRegistry(registry_path="test_registry")
        
        class DummyModel:
            pass
        
        version = registry.register_model(
            DummyModel(),
            {},
            version='1.0.0',
            stage=ModelStage.DEV
        )
        
        success = registry.promote_model(version, ModelStage.STAGING)
        assert success
        
        model = registry.get_model(version)
        assert model.stage == ModelStage.STAGING
        
        # Cleanup
        import shutil
        if os.path.exists("test_registry"):
            shutil.rmtree("test_registry")
    
    def test_list_versions(self):
        """Test listing versions"""
        registry = ModelRegistry(registry_path="test_registry")
        
        class DummyModel:
            pass
        
        registry.register_model(DummyModel(), {}, version='1.0.0', stage=ModelStage.DEV)
        registry.register_model(DummyModel(), {}, version='1.0.1', stage=ModelStage.STAGING)
        
        versions = registry.list_versions()
        assert len(versions) == 2
        
        staging_models = registry.list_versions(stage=ModelStage.STAGING)
        assert len(staging_models) == 1
        
        # Cleanup
        import shutil
        if os.path.exists("test_registry"):
            shutil.rmtree("test_registry")
    
    def test_rollback(self):
        """Test production rollback"""
        registry = ModelRegistry(registry_path="test_registry")
        
        class DummyModel:
            pass
        
        v1 = registry.register_model(DummyModel(), {}, version='1.0.0', stage=ModelStage.PRODUCTION)
        v2 = registry.register_model(DummyModel(), {}, version='1.0.1', stage=ModelStage.STAGING)
        
        # Promote v2 to production (simulating rollback)
        registry.promote_model(v2, ModelStage.PRODUCTION)
        
        # v1 should be archived
        model1 = registry.get_model(v1)
        assert model1.stage == ModelStage.ARCHIVED
        
        # Cleanup
        import shutil
        if os.path.exists("test_registry"):
            shutil.rmtree("test_registry")


class TestPretrainedModelHub:
    """Tests for pre-trained model hub"""
    
    def test_list_models(self):
        """Test listing models"""
        hub = PretrainedModelHub(hub_path="test_hub")
        
        models = hub.list_models()
        assert len(models) > 0
        
        # Cleanup
        import shutil
        if os.path.exists("test_hub"):
            shutil.rmtree("test_hub")
    
    def test_get_model_info(self):
        """Test getting model info"""
        hub = PretrainedModelHub(hub_path="test_hub")
        
        model_info = hub.get_model_info('resnet18-imagenet')
        assert model_info is not None
        assert model_info.model_id == 'resnet18-imagenet'
        
        # Cleanup
        import shutil
        if os.path.exists("test_hub"):
            shutil.rmtree("test_hub")
    
    def test_register_model(self):
        """Test registering model"""
        hub = PretrainedModelHub(hub_path="test_hub")
        
        class DummyModel:
            pass
        
        success = hub.register_model(
            'test-model',
            'Test Model',
            'A test model',
            'cnn',
            DummyModel(),
            metadata={'accuracy': 0.9}
        )
        
        assert success
        assert hub.get_model_info('test-model') is not None
        
        # Cleanup
        import shutil
        if os.path.exists("test_hub"):
            shutil.rmtree("test_hub")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
