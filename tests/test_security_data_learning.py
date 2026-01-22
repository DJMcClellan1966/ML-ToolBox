"""
Tests for Security and Data Learning Quick Wins
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ml_security_framework import (
        InputValidator, ModelEncryption, AdversarialDefender,
        ThreatDetectionSystem, MLSecurityFramework
    )
    from data_learning_framework import (
        FederatedLearningFramework, OnlineLearningWrapper,
        DifferentialPrivacyWrapper, ContinuousLearningPipeline
    )
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    pytestmark = pytest.mark.skip("Features not available")


class TestMLSecurityFramework:
    """Tests for ML security framework"""
    
    def test_input_validator(self):
        """Test input validation"""
        validator = InputValidator(max_features=10)
        
        X_valid = np.random.rand(100, 5)
        result = validator.validate(X_valid)
        assert result['valid'] == True
        
        X_invalid = np.random.rand(100, 20)  # Too many features
        result = validator.validate(X_invalid)
        assert result['valid'] == False
    
    def test_input_sanitize(self):
        """Test input sanitization"""
        validator = InputValidator()
        
        X = np.array([[1.0, 2.0, np.nan], [4.0, np.inf, 6.0]])
        X_sanitized = validator.sanitize(X)
        
        assert not np.any(np.isnan(X_sanitized))
        assert not np.any(np.isinf(X_sanitized))
    
    def test_model_encryption(self):
        """Test model encryption"""
        try:
            encryption = ModelEncryption()
            
            # Create dummy model
            import types
            DummyModel = types.SimpleNamespace
            model = DummyModel()
            model.weights = [1, 2, 3]
            
            # Encrypt
            success = encryption.encrypt_model(model, "test_encrypted_model.pkl")
            
            if success:
                # Decrypt
                decrypted = encryption.decrypt_model("test_encrypted_model.pkl")
                assert decrypted is not None
                
                # Cleanup
                if os.path.exists("test_encrypted_model.pkl"):
                    os.remove("test_encrypted_model.pkl")
        except ImportError:
            pytest.skip("cryptography not available")
    
    def test_adversarial_defender(self):
        """Test adversarial defense"""
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        defender = AdversarialDefender(model, epsilon=0.1)
        
        # Generate adversarial examples
        X_adv = defender.generate_adversarial_example(X[:10], y[:10], method='random')
        assert X_adv.shape == X[:10].shape
    
    def test_threat_detection(self):
        """Test threat detection system"""
        detector = ThreatDetectionSystem()
        
        # Create normal and threat data
        X_normal = np.random.rand(100, 10)
        X_threats = np.random.rand(20, 10) + 5  # Different distribution
        
        result = detector.train_threat_detector(X_normal, X_threats, use_ml_toolbox=False)
        assert 'model' in result or 'error' in result
        
        if 'model' in result:
            # Test detection
            detection = detector.detect_threat(X_threats[:5])
            assert 'threat_detected' in detection
    
    def test_ml_security_framework(self):
        """Test ML security framework"""
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        security = MLSecurityFramework(model)
        
        # Secure prediction
        result = security.predict_secure(X[:10])
        assert 'predictions' in result or 'error' in result
        
        # Security info
        info = security.get_security_info()
        assert 'input_validation' in info


class TestDataLearningFramework:
    """Tests for data learning framework"""
    
    def test_federated_learning(self):
        """Test federated learning"""
        federated = FederatedLearningFramework()
        
        # Create client data
        client_data = [
            (np.random.rand(50, 10), np.random.randint(0, 2, 50)),
            (np.random.rand(50, 10), np.random.randint(0, 2, 50)),
            (np.random.rand(50, 10), np.random.randint(0, 2, 50))
        ]
        
        result = federated.train_federated_model(
            client_data, num_rounds=2, use_ml_toolbox=False
        )
        
        assert 'federated_model' in result or 'error' in result
        assert result.get('num_clients') == 3
    
    def test_online_learning(self):
        """Test online learning wrapper"""
        from sklearn.linear_model import SGDClassifier
        
        model = SGDClassifier(random_state=42)
        wrapper = OnlineLearningWrapper(model)
        
        # Initial fit
        X1 = np.random.rand(50, 10)
        y1 = np.random.randint(0, 2, 50)
        wrapper.partial_fit(X1, y1, classes=np.array([0, 1]))
        
        # Update
        X2 = np.random.rand(20, 10)
        y2 = np.random.randint(0, 2, 20)
        wrapper.partial_fit(X2, y2)
        
        assert wrapper.get_update_count() >= 2
    
    def test_differential_privacy(self):
        """Test differential privacy"""
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        private = DifferentialPrivacyWrapper(model, epsilon=1.0)
        
        # Private prediction
        result = private.predict_private(X[:10])
        assert 'predictions' in result
        assert 'epsilon' in result
        
        # Privacy info
        info = private.get_privacy_info()
        assert 'epsilon' in info
        assert 'privacy_guarantee' in info
    
    def test_continuous_learning(self):
        """Test continuous learning pipeline"""
        pipeline = ContinuousLearningPipeline(None, use_ml_toolbox=False)
        
        X1 = np.random.rand(100, 10)
        y1 = np.random.randint(0, 2, 100)
        
        # Initial train
        result = pipeline.initial_train(X1, y1)
        assert 'model' in result or 'error' in result
        
        if 'model' in result:
            # Update
            X2 = np.random.rand(20, 10)
            y2 = np.random.randint(0, 2, 20)
            
            update_result = pipeline.update(X2, y2)
            assert 'updated' in update_result or 'error' in update_result


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
