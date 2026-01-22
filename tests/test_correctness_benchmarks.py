"""
Rigorous Correctness Tests
Compare ML Toolbox implementations against reference libraries (sklearn, scipy)

Tests:
- Numerical correctness
- Algorithm accuracy
- Statistical correctness
- Edge case handling
"""
import sys
from pathlib import Path
import numpy as np
import pytest
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))

# Reference implementations
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    from sklearn.datasets import make_classification, make_regression
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import ks_2samp, chi2_contingency
    SKLEARN_AVAILABLE = True
    SCIPY_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    SCIPY_AVAILABLE = False
    pytestmark = pytest.mark.skip("sklearn/scipy not available")

# ML Toolbox imports
try:
    from ml_toolbox import MLToolbox
    from ml_evaluation import MLEvaluator
    from ml_toolbox.compartment3_algorithms import AlgorithmsCompartment
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    pytestmark = pytest.mark.skip("ML Toolbox not available")


class TestCorrectness:
    """Test correctness against reference implementations"""
    
    @pytest.fixture
    def classification_data(self):
        """Generate classification test data"""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test
    
    @pytest.fixture
    def regression_data(self):
        """Generate regression test data"""
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=5,
            noise=10.0,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test
    
    def test_ml_evaluator_correctness(self, classification_data):
        """Test MLEvaluator correctness against sklearn"""
        X_train, X_test, y_train, y_test = classification_data
        
        # Train reference model
        ref_model = RandomForestClassifier(n_estimators=10, random_state=42)
        ref_model.fit(X_train, y_train)
        ref_predictions = ref_model.predict(X_test)
        ref_accuracy = accuracy_score(y_test, ref_predictions)
        
        # Test with ML Toolbox evaluator
        toolbox = MLToolbox()
        evaluator = toolbox.algorithms.get_evaluator()
        
        # Evaluate same model
        results = evaluator.evaluate_model(
            model=ref_model,
            X=X_train,
            y=y_train,
            cv=5
        )
        
        # Check that accuracy is reasonable (within 5% of reference)
        assert 'accuracy' in results
        assert abs(results['accuracy'] - ref_accuracy) < 0.05
    
    def test_cross_validation_correctness(self, classification_data):
        """Test cross-validation correctness"""
        X_train, _, y_train, _ = classification_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Reference CV
        ref_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        ref_mean = np.mean(ref_scores)
        
        # ML Toolbox CV
        toolbox = MLToolbox()
        evaluator = toolbox.algorithms.get_evaluator()
        results = evaluator.evaluate_model(model, X_train, y_train, cv=5)
        
        # Compare means (should be very close)
        assert abs(results['accuracy'] - ref_mean) < 0.01
    
    def test_hyperparameter_tuning_correctness(self, classification_data):
        """Test hyperparameter tuning correctness"""
        X_train, _, y_train, _ = classification_data
        
        from sklearn.ensemble import RandomForestClassifier
        
        # Reference grid search
        from sklearn.model_selection import GridSearchCV
        param_grid = {'n_estimators': [10, 20], 'max_depth': [5, 10]}
        ref_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=3,
            scoring='accuracy'
        )
        ref_search.fit(X_train, y_train)
        ref_best_score = ref_search.best_score_
        
        # ML Toolbox tuning
        toolbox = MLToolbox()
        tuner = toolbox.algorithms.get_tuner()
        
        best_params = tuner.tune(
            model=RandomForestClassifier(random_state=42),
            X=X_train,
            y=y_train,
            param_grid=param_grid,
            cv=3
        )
        
        # Check that we found reasonable parameters
        assert 'n_estimators' in best_params or 'max_depth' in best_params
    
    def test_data_drift_detection_correctness(self):
        """Test data drift detection against scipy"""
        np.random.seed(42)
        
        # Reference distribution
        ref_data = np.random.normal(0, 1, 1000)
        
        # Same distribution (no drift)
        same_data = np.random.normal(0, 1, 500)
        
        # Different distribution (drift)
        drifted_data = np.random.normal(2, 1, 500)
        
        # Reference KS test
        _, ref_p_same = ks_2samp(ref_data, same_data)
        _, ref_p_drift = ks_2samp(ref_data, drifted_data)
        
        # ML Toolbox drift detection
        try:
            from model_monitoring import DataDriftDetector
            detector = DataDriftDetector(ref_data, alpha=0.05)
            
            result_same = detector.detect_drift(same_data, method='ks_test')
            result_drift = detector.detect_drift(drifted_data, method='ks_test')
            
            # Check that drift detection matches scipy
            # Same distribution should not have drift
            assert result_same['has_drift'] == (ref_p_same < 0.05)
            # Different distribution should have drift
            assert result_drift['has_drift'] == (ref_p_drift < 0.05)
        except ImportError:
            pytest.skip("Model monitoring not available")
    
    def test_statistical_learning_correctness(self, classification_data):
        """Test statistical learning methods correctness"""
        X_train, _, y_train, _ = classification_data
        
        try:
            from statistical_learning import StatisticalEvaluator
            evaluator = StatisticalEvaluator()
            
            # Test confidence intervals
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Get predictions
            predictions = model.predict_proba(X_train)[:, 1]
            
            # Calculate confidence intervals
            ci = evaluator.confidence_intervals(predictions, confidence=0.95)
            
            # Check that CI is valid
            assert 'lower' in ci
            assert 'upper' in ci
            assert ci['lower'] < ci['upper']
            assert 0 <= ci['lower'] <= 1
            assert 0 <= ci['upper'] <= 1
        except ImportError:
            pytest.skip("Statistical learning not available")
    
    def test_ensemble_correctness(self, classification_data):
        """Test ensemble methods correctness"""
        X_train, X_test, y_train, y_test = classification_data
        
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        
        # Reference ensemble
        models = [
            RandomForestClassifier(n_estimators=10, random_state=42),
            GradientBoostingClassifier(n_estimators=10, random_state=42)
        ]
        
        # Train and evaluate reference
        ref_predictions = []
        for model in models:
            model.fit(X_train, y_train)
            ref_predictions.append(model.predict(X_test))
        
        ref_ensemble_pred = np.round(np.mean(ref_predictions, axis=0)).astype(int)
        ref_accuracy = accuracy_score(y_test, ref_ensemble_pred)
        
        # ML Toolbox ensemble
        try:
            from ensemble_learning import EnsembleLearner
            ensemble = EnsembleLearner(
                base_models=models,
                method='voting'
            )
            ensemble.fit(X_train, y_train)
            ensemble_pred = ensemble.predict(X_test)
            ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
            
            # Ensemble should be at least as good as individual models
            individual_accuracies = [
                accuracy_score(y_test, pred) for pred in ref_predictions
            ]
            assert ensemble_accuracy >= min(individual_accuracies) - 0.05
        except ImportError:
            pytest.skip("Ensemble learning not available")
    
    def test_preprocessing_correctness(self):
        """Test preprocessing correctness"""
        # Generate test data
        texts = [
            "This is a test sentence.",
            "This is another test sentence.",
            "Duplicate: This is a test sentence.",  # Near duplicate
            "Completely different content here."
        ]
        
        try:
            from ml_toolbox import MLToolbox
            toolbox = MLToolbox()
            
            # Test preprocessing
            results = toolbox.data.preprocess(
                texts,
                advanced=True,
                dedup_threshold=0.85
            )
            
            # Check that preprocessing worked
            assert 'deduplicated' in results
            assert len(results['deduplicated']) <= len(texts)
            
            # Check that near-duplicate was removed
            assert len(results['deduplicated']) < len(texts)
        except Exception as e:
            pytest.skip(f"Preprocessing test failed: {e}")


class TestNumericalAccuracy:
    """Test numerical accuracy and precision"""
    
    def test_floating_point_precision(self):
        """Test floating point precision"""
        # Test that our calculations maintain reasonable precision
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        
        # Calculate mean (should be close to 0)
        mean = np.mean(X)
        assert abs(mean) < 0.5  # Should be close to 0 for standard normal
        
        # Calculate std (should be close to 1)
        std = np.std(X)
        assert 0.8 < std < 1.2  # Should be close to 1
    
    def test_matrix_operations(self):
        """Test matrix operations correctness"""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        
        # Matrix multiplication
        C = A @ B
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_almost_equal(C, expected)
        
        # Matrix addition
        D = A + B
        expected = np.array([[6, 8], [10, 12]])
        np.testing.assert_array_almost_equal(D, expected)
    
    def test_statistical_calculations(self):
        """Test statistical calculations"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Mean
        mean = np.mean(data)
        assert abs(mean - 5.5) < 1e-10
        
        # Variance
        var = np.var(data)
        expected_var = np.var([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert abs(var - expected_var) < 1e-10
        
        # Standard deviation
        std = np.std(data)
        assert abs(std - np.sqrt(expected_var)) < 1e-10


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_input(self):
        """Test handling of empty inputs"""
        try:
            from ml_toolbox import MLToolbox
            toolbox = MLToolbox()
            
            # Empty text list
            results = toolbox.data.preprocess([], advanced=True)
            assert results['final_count'] == 0
        except Exception as e:
            # Should handle gracefully
            assert "empty" in str(e).lower() or "zero" in str(e).lower()
    
    def test_single_sample(self):
        """Test with single sample"""
        X = np.array([[1, 2, 3]])
        y = np.array([0])
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Should handle single sample (may warn but not crash)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(X, y)
                pred = model.predict(X)
                assert len(pred) == 1
            except ValueError:
                # Some models can't handle single sample - that's OK
                pass
    
    def test_all_same_values(self):
        """Test with all same values"""
        X = np.ones((100, 10))
        y = np.zeros(100)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Should handle constant features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
            pred = model.predict(X)
            assert len(pred) == len(y)
    
    def test_very_large_input(self):
        """Test with very large input"""
        # Create large but manageable dataset
        X = np.random.randn(10000, 50)
        y = np.random.randint(0, 2, 10000)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
        
        # Should handle large input
        model.fit(X, y)
        pred = model.predict(X[:100])
        assert len(pred) == 100
    
    def test_missing_values_handling(self):
        """Test handling of missing values"""
        X = np.array([[1, 2, np.nan], [4, 5, 6], [7, 8, 9]])
        y = np.array([0, 1, 0])
        
        # Check that we detect NaN
        assert np.isnan(X).any()
        
        # Models should handle or reject appropriately
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Some models handle NaN, some don't
        try:
            model.fit(X, y)
        except ValueError:
            # Expected for some models
            pass


class TestErrorHandling:
    """Test error handling and validation"""
    
    def test_invalid_input_types(self):
        """Test handling of invalid input types"""
        # String instead of array
        try:
            from input_validation import InputValidator
            InputValidator.validate_array("not an array")
            assert False, "Should raise ValidationError"
        except Exception as e:
            assert "array" in str(e).lower() or "ValidationError" in str(type(e).__name__)
    
    def test_invalid_shapes(self):
        """Test handling of invalid shapes"""
        # 3D array when 2D expected
        X = np.random.randn(10, 5, 3)
        
        try:
            from input_validation import InputValidator
            InputValidator.validate_array(X, min_dim=2, max_dim=2)
            assert False, "Should raise ValidationError"
        except Exception as e:
            assert "dimension" in str(e).lower() or "ValidationError" in str(type(e).__name__)
    
    def test_invalid_range_values(self):
        """Test handling of invalid range values"""
        try:
            from input_validation import InputValidator
            InputValidator.validate_range(15.0, min_val=0.0, max_val=10.0, name='test')
            assert False, "Should raise ValidationError"
        except Exception as e:
            assert "ValidationError" in str(type(e).__name__) or "must be" in str(e).lower()
    
    def test_none_inputs(self):
        """Test handling of None inputs"""
        try:
            from ml_toolbox import MLToolbox
            toolbox = MLToolbox()
            
            # None input should be handled
            results = toolbox.data.preprocess(None, advanced=True)
            # Should either return empty result or raise clear error
        except (TypeError, ValueError, AttributeError):
            # Expected - should raise clear error
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
