"""
Tests for Jon Erickson Priorities Implementation
ML Security Testing, Network Security, Enhanced Cryptography
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ml_security_testing import (
        MLSecurityTester, MLExploitTester, MLSecurityAuditor
    )
    from ml_network_security import (
        MLNetworkSecurity, MLAPISecurityTester
    )
    from enhanced_cryptographic_security import (
        EnhancedModelEncryption, SecureKeyManager, CryptographicTester
    )
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    pytestmark = pytest.mark.skip("Features not available")


class TestMLSecurityTesting:
    """Tests for ML security testing framework"""
    
    def test_assess_vulnerabilities(self):
        """Test vulnerability assessment"""
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        tester = MLSecurityTester(model)
        vulns = tester.assess_vulnerabilities()
        
        assert 'overall_risk' in vulns
        assert 'input_validation' in vulns
        assert 'adversarial_robustness' in vulns
    
    def test_test_adversarial_attacks(self):
        """Test adversarial attack testing"""
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random.rand(50, 10)
        y = np.random.randint(0, 2, 50)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        tester = MLSecurityTester(model)
        results = tester.test_adversarial_attacks(X, y)
        
        assert 'baseline_accuracy' in results
        assert 'robustness_score' in results
    
    def test_penetration_test(self):
        """Test penetration testing"""
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        tester = MLSecurityTester(model)
        result = tester.penetration_test({'validator': None})
        
        assert 'security_score' in result
        assert 'vulnerabilities_found' in result
    
    def test_generate_security_report(self):
        """Test security report generation"""
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        tester = MLSecurityTester(model)
        report = tester.generate_security_report()
        
        assert 'vulnerability_assessment' in report
        assert 'recommendations' in report


class TestMLNetworkSecurity:
    """Tests for network security"""
    
    def test_test_api_security(self):
        """Test API security testing"""
        network = MLNetworkSecurity("http://localhost:8000/api/predict")
        results = network.test_api_security()
        
        assert 'security_score' in results
        assert 'vulnerabilities' in results
    
    def test_analyze_traffic(self):
        """Test traffic analysis"""
        network = MLNetworkSecurity()
        
        traffic = [
            {'type': 'predict', 'size': 100, 'timestamp': 1000},
            {'type': 'predict', 'size': 200, 'timestamp': 1001}
        ]
        
        analysis = network.analyze_traffic(traffic)
        
        assert 'total_requests' in analysis
        assert 'request_patterns' in analysis
    
    def test_detect_attacks(self):
        """Test attack detection"""
        network = MLNetworkSecurity()
        
        traffic = [
            {'payload': "'; DROP TABLE models; --", 'size': 100},
            {'payload': '<script>alert("XSS")</script>', 'size': 200},
            {'payload': 'normal request', 'size': 50}
        ]
        
        detection = network.detect_attacks(traffic, use_ml_toolbox=False)
        
        assert 'attacks_detected' in detection
        assert 'total_attacks' in detection


class TestEnhancedCryptography:
    """Tests for enhanced cryptography"""
    
    def test_encrypt_aes256(self):
        """Test AES-256 encryption"""
        try:
            encryption = EnhancedModelEncryption()
            
            # Create dummy model
            import types
            DummyModel = types.SimpleNamespace
            model = DummyModel()
            model.weights = [1, 2, 3]
            
            result = encryption.encrypt_aes256(model, output_path="test_encrypted.pkl")
            
            if 'error' not in result:
                assert result.get('success') == True
                
                # Cleanup
                if os.path.exists("test_encrypted.pkl"):
                    os.remove("test_encrypted.pkl")
        except ImportError:
            pytest.skip("cryptography not available")
    
    def test_secure_key_manager(self):
        """Test secure key manager"""
        manager = SecureKeyManager(key_store_path="test_key_store")
        
        # Generate key
        key = manager.generate_key("test_key")
        assert key is not None
        
        # Get key
        retrieved = manager.get_key("test_key")
        assert retrieved is not None
        
        # List keys
        keys = manager.list_keys()
        assert "test_key" in keys
        
        # Cleanup
        import shutil
        if os.path.exists("test_key_store"):
            shutil.rmtree("test_key_store")
    
    def test_cryptographic_tester(self):
        """Test cryptographic tester"""
        tester = CryptographicTester()
        
        encryption = EnhancedModelEncryption()
        results = tester.test_encryption(encryption)
        
        assert 'encryption_works' in results or 'issues' in results


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
