"""
Tests for Configuration and Validation
"""
import sys
from pathlib import Path
import numpy as np
import pytest
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config_manager import MLToolboxConfig, get_config
    from input_validation import InputValidator, ValidationError
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    pytestmark = pytest.mark.skip("Config and validation not available")


class TestMLToolboxConfig:
    """Tests for MLToolboxConfig"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = MLToolboxConfig()
        assert config.data_preprocessing['dedup_threshold'] == 0.85
        assert config.quantum_kernel['use_sentence_transformers'] == True
        assert config.mlops['deployment']['api_port'] == 8000
    
    def test_from_env(self, monkeypatch):
        """Test loading from environment variables"""
        monkeypatch.setenv('MLTOOLBOX_DATA_DEDUP_THRESHOLD', '0.9')
        monkeypatch.setenv('MLTOOLBOX_MLOPS_API_PORT', '9000')
        monkeypatch.setenv('MLTOOLBOX_LOGGING_LEVEL', 'DEBUG')
        
        config = MLToolboxConfig.from_env()
        assert config.data_preprocessing['dedup_threshold'] == 0.9
        assert config.mlops['deployment']['api_port'] == 9000
        assert config.logging['level'] == 'DEBUG'
    
    def test_from_file_json(self):
        """Test loading from JSON file"""
        config_data = {
            'data_preprocessing': {'dedup_threshold': 0.9},
            'mlops': {'deployment': {'api_port': 9000}}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            config = MLToolboxConfig.from_file(config_path)
            assert config.data_preprocessing['dedup_threshold'] == 0.9
            assert config.mlops['deployment']['api_port'] == 9000
        finally:
            os.unlink(config_path)
    
    def test_validate(self):
        """Test configuration validation"""
        config = MLToolboxConfig()
        validation = config.validate()
        
        assert 'valid' in validation
        assert validation['valid'] == True
    
    def test_validate_invalid(self):
        """Test validation with invalid config"""
        config = MLToolboxConfig()
        config.data_preprocessing['dedup_threshold'] = 1.5  # Invalid
        config.mlops['deployment']['api_port'] = 70000  # Invalid
        
        validation = config.validate()
        assert validation['valid'] == False
        assert len(validation['errors']) > 0
    
    def test_to_dict(self):
        """Test converting to dictionary"""
        config = MLToolboxConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'data_preprocessing' in config_dict
        assert 'mlops' in config_dict


class TestInputValidator:
    """Tests for InputValidator"""
    
    def test_validate_array(self):
        """Test array validation"""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        validated = InputValidator.validate_array(X, min_dim=2, max_dim=2, min_samples=1)
        
        assert isinstance(validated, np.ndarray)
        assert validated.shape == X.shape
    
    def test_validate_array_invalid_dim(self):
        """Test array validation with invalid dimensions"""
        X = np.array([1, 2, 3])  # 1D array
        
        with pytest.raises(ValidationError):
            InputValidator.validate_array(X, min_dim=2, max_dim=2)
    
    def test_validate_array_empty(self):
        """Test array validation with empty array"""
        X = np.array([])
        
        with pytest.raises(ValidationError):
            InputValidator.validate_array(X, allow_empty=False)
    
    def test_validate_range(self):
        """Test range validation"""
        value = InputValidator.validate_range(5.0, min_val=0.0, max_val=10.0, name='test')
        assert value == 5.0
    
    def test_validate_range_invalid(self):
        """Test range validation with invalid value"""
        with pytest.raises(ValidationError):
            InputValidator.validate_range(15.0, min_val=0.0, max_val=10.0, name='test')
    
    def test_validate_file_path(self, tmp_path):
        """Test file path validation"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        validated = InputValidator.validate_file_path(
            test_file,
            must_exist=True,
            must_be_file=True
        )
        assert validated == test_file
    
    def test_validate_file_path_not_exist(self):
        """Test file path validation with non-existent file"""
        with pytest.raises(ValidationError):
            InputValidator.validate_file_path(
                "nonexistent.txt",
                must_exist=True
            )
    
    def test_validate_model(self):
        """Test model validation"""
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        
        validated = InputValidator.validate_model(model, required_methods=['predict'])
        assert validated == model
    
    def test_validate_model_missing_method(self):
        """Test model validation with missing method"""
        class BadModel:
            pass
        
        model = BadModel()
        with pytest.raises(ValidationError):
            InputValidator.validate_model(model, required_methods=['predict'])
    
    def test_validate_config(self):
        """Test configuration validation"""
        config = {
            'required_key1': 'value1',
            'required_key2': 'value2',
            'optional_key': 'value3'
        }
        
        validated = InputValidator.validate_config(
            config,
            required_keys=['required_key1', 'required_key2'],
            optional_keys=['optional_key']
        )
        assert validated == config
    
    def test_validate_config_missing_key(self):
        """Test configuration validation with missing key"""
        config = {'key1': 'value1'}
        
        with pytest.raises(ValidationError):
            InputValidator.validate_config(
                config,
                required_keys=['key1', 'key2']
            )
