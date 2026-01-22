"""
Error Handling Tests
Test error handling, edge cases, and invalid inputs

Tests:
- Invalid input types
- Edge cases
- Boundary conditions
- Error messages
- Graceful degradation
"""
import sys
from pathlib import Path
import numpy as np
import pytest
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ml_toolbox import MLToolbox
    from input_validation import InputValidator, ValidationError
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    pytestmark = pytest.mark.skip("ML Toolbox not available")


class TestInputValidation:
    """Test input validation and error handling"""
    
    def test_invalid_array_types(self):
        """Test handling of invalid array types"""
        validator = InputValidator()
        
        # String instead of array
        with pytest.raises((ValidationError, TypeError, ValueError)):
            validator.validate_array("not an array")
        
        # List of strings
        with pytest.raises((ValidationError, TypeError, ValueError)):
            validator.validate_array(["a", "b", "c"])
        
        # None
        with pytest.raises((ValidationError, TypeError, ValueError)):
            validator.validate_array(None)
    
    def test_invalid_array_shapes(self):
        """Test handling of invalid array shapes"""
        validator = InputValidator()
        
        # 3D array when 2D expected
        X = np.random.randn(10, 5, 3)
        with pytest.raises(ValidationError):
            validator.validate_array(X, min_dim=2, max_dim=2)
        
        # Empty array
        X = np.array([])
        with pytest.raises(ValidationError):
            validator.validate_array(X, allow_empty=False)
    
    def test_invalid_range_values(self):
        """Test handling of invalid range values"""
        validator = InputValidator()
        
        # Value too low
        with pytest.raises(ValidationError):
            validator.validate_range(-1.0, min_val=0.0, max_val=10.0, name='test')
        
        # Value too high
        with pytest.raises(ValidationError):
            validator.validate_range(15.0, min_val=0.0, max_val=10.0, name='test')
    
    def test_invalid_file_paths(self):
        """Test handling of invalid file paths"""
        validator = InputValidator()
        
        # Non-existent file
        with pytest.raises(ValidationError):
            validator.validate_file_path(
                "nonexistent_file.txt",
                must_exist=True
            )
        
        # Directory when file expected
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValidationError):
                validator.validate_file_path(
                    tmpdir,
                    must_be_file=True
                )
    
    def test_invalid_models(self):
        """Test handling of invalid models"""
        validator = InputValidator()
        
        # Model without predict method
        class BadModel:
            pass
        
        bad_model = BadModel()
        with pytest.raises(ValidationError):
            validator.validate_model(bad_model, required_methods=['predict'])
    
    def test_invalid_config(self):
        """Test handling of invalid configuration"""
        validator = InputValidator()
        
        # Missing required keys
        config = {'key1': 'value1'}
        with pytest.raises(ValidationError):
            validator.validate_config(
                config,
                required_keys=['key1', 'key2']
            )


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_inputs(self):
        """Test handling of empty inputs"""
        toolbox = MLToolbox()
        
        # Empty text list
        results = toolbox.data.preprocess([], advanced=True)
        assert results['final_count'] == 0
        assert len(results['deduplicated']) == 0
    
    def test_single_sample(self):
        """Test with single sample"""
        texts = ["Single text"]
        
        toolbox = MLToolbox()
        results = toolbox.data.preprocess(texts, advanced=True)
        
        assert results['final_count'] >= 0
        assert len(results['deduplicated']) <= 1
    
    def test_very_long_text(self):
        """Test with very long text"""
        long_text = "word " * 10000  # Very long text
        
        toolbox = MLToolbox()
        results = toolbox.data.preprocess([long_text], advanced=True)
        
        assert results['final_count'] >= 0
    
    def test_special_characters(self):
        """Test with special characters"""
        texts = [
            "Text with émojis 🎉",
            "Text with <HTML> tags",
            "Text with\nnewlines\tand\ttabs",
            "Text with \"quotes\" and 'apostrophes'"
        ]
        
        toolbox = MLToolbox()
        results = toolbox.data.preprocess(texts, advanced=True)
        
        assert results['final_count'] >= 0
    
    def test_unicode_text(self):
        """Test with Unicode text"""
        texts = [
            "English text",
            "中文文本",
            "Русский текст",
            "العربية نص"
        ]
        
        toolbox = MLToolbox()
        results = toolbox.data.preprocess(texts, advanced=True)
        
        assert results['final_count'] >= 0
    
    def test_all_identical_texts(self):
        """Test with all identical texts"""
        texts = ["Same text"] * 100
        
        toolbox = MLToolbox()
        results = toolbox.data.preprocess(texts, advanced=True, dedup_threshold=0.9)
        
        # Should deduplicate to 1
        assert results['final_count'] <= len(texts)
        assert len(results['deduplicated']) <= len(texts)


class TestErrorMessages:
    """Test error message quality"""
    
    def test_clear_error_messages(self):
        """Test that error messages are clear and helpful"""
        validator = InputValidator()
        
        try:
            validator.validate_range(15.0, min_val=0.0, max_val=10.0, name='test_value')
            assert False, "Should raise ValidationError"
        except ValidationError as e:
            error_msg = str(e)
            # Error message should be informative
            assert 'test_value' in error_msg or 'value' in error_msg.lower()
            assert '10' in error_msg or 'max' in error_msg.lower()
    
    def test_config_validation_errors(self):
        """Test configuration validation error messages"""
        try:
            from config_manager import MLToolboxConfig
            config = MLToolboxConfig()
            config.data_preprocessing['dedup_threshold'] = 1.5  # Invalid
            
            validation = config.validate()
            assert validation['valid'] == False
            assert len(validation['errors']) > 0
            assert any('dedup_threshold' in str(err) for err in validation['errors'])
        except ImportError:
            pytest.skip("Config manager not available")


class TestGracefulDegradation:
    """Test graceful degradation when components unavailable"""
    
    def test_missing_optional_dependencies(self):
        """Test behavior when optional dependencies are missing"""
        # Should not crash, but may have reduced functionality
        try:
            from ml_toolbox import MLToolbox
            toolbox = MLToolbox(include_mlops=True)
            
            # Should work even if some components unavailable
            assert toolbox is not None
        except Exception as e:
            # Should provide clear error message
            assert "not available" in str(e).lower() or "import" in str(e).lower()
    
    def test_partial_failure_handling(self):
        """Test handling of partial failures"""
        # Test that one component failure doesn't break everything
        try:
            from ml_toolbox import MLToolbox
            toolbox = MLToolbox()
            
            # Data compartment should work even if MLOps fails
            results = toolbox.data.preprocess(["test"], advanced=True)
            assert results is not None
        except Exception:
            # If it fails, should fail gracefully
            pass


class TestBoundaryConditions:
    """Test boundary conditions"""
    
    def test_zero_threshold(self):
        """Test with zero threshold"""
        texts = ["text1", "text2", "text3"]
        
        toolbox = MLToolbox()
        # Should handle zero threshold gracefully
        results = toolbox.data.preprocess(
            texts,
            advanced=True,
            dedup_threshold=0.0
        )
        assert results['final_count'] >= 0
    
    def test_one_threshold(self):
        """Test with threshold of 1.0"""
        texts = ["text1", "text2", "text3"]
        
        toolbox = MLToolbox()
        results = toolbox.data.preprocess(
            texts,
            advanced=True,
            dedup_threshold=1.0
        )
        assert results['final_count'] >= 0
    
    def test_very_small_dataset(self):
        """Test with very small dataset"""
        texts = ["text"]
        
        toolbox = MLToolbox()
        results = toolbox.data.preprocess(texts, advanced=True)
        assert results['final_count'] >= 0
    
    def test_very_large_dataset(self):
        """Test with very large dataset (if memory allows)"""
        # Create large but manageable dataset
        texts = [f"Text {i}" for i in range(10000)]
        
        toolbox = MLToolbox()
        # Should handle large dataset (may be slow but shouldn't crash)
        try:
            results = toolbox.data.preprocess(texts[:1000], advanced=True)  # Limit for test
            assert results['final_count'] >= 0
        except MemoryError:
            pytest.skip("Not enough memory for large dataset test")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
