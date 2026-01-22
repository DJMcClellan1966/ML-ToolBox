"""
Tests for Deep Learning Framework
Test CNN, RNN, Transformer, training, optimization
"""
import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from deep_learning_framework import DeepLearningFramework
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    pytestmark = pytest.mark.skip("Deep learning framework not available")


class TestDeepLearningFramework:
    """Tests for deep learning framework"""
    
    def test_create_cnn(self):
        """Test CNN creation"""
        dl = DeepLearningFramework()
        
        if dl.torch_available:
            model = dl.create_cnn(input_channels=3, num_classes=10, architecture='simple')
            assert model is not None
            assert hasattr(model, 'forward')
        else:
            result = dl.create_cnn()
            assert 'error' in result
    
    def test_create_rnn(self):
        """Test RNN creation"""
        dl = DeepLearningFramework()
        
        if dl.torch_available:
            model = dl.create_rnn(input_size=10, hidden_size=64, num_classes=2, rnn_type='LSTM')
            assert model is not None
            assert hasattr(model, 'forward')
        else:
            result = dl.create_rnn(10, 64, 1, 2)
            assert 'error' in result
    
    def test_create_transformer(self):
        """Test Transformer creation"""
        dl = DeepLearningFramework()
        
        if dl.torch_available:
            model = dl.create_transformer(
                vocab_size=1000, d_model=512, nhead=8, 
                num_layers=2, num_classes=10
            )
            assert model is not None
            assert hasattr(model, 'forward')
        else:
            result = dl.create_transformer(1000)
            assert 'error' in result
    
    def test_create_optimizer(self):
        """Test optimizer creation"""
        dl = DeepLearningFramework()
        
        if dl.torch_available:
            import torch.nn as nn
            model = nn.Linear(10, 2)
            optimizer = dl.create_optimizer(model, 'Adam', 0.001)
            assert optimizer is not None
        else:
            result = dl.create_optimizer(None)
            assert 'error' in result
    
    def test_create_lr_scheduler(self):
        """Test learning rate scheduler creation"""
        dl = DeepLearningFramework()
        
        if dl.torch_available:
            import torch.nn as nn
            import torch.optim as optim
            model = nn.Linear(10, 2)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = dl.create_lr_scheduler(optimizer, 'StepLR', step_size=10)
            assert scheduler is not None
        else:
            result = dl.create_lr_scheduler(None)
            assert 'error' in result
    
    def test_get_dependencies(self):
        """Test dependencies"""
        dl = DeepLearningFramework()
        deps = dl.get_dependencies()
        assert 'torch' in deps
        assert 'torchvision' in deps


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
