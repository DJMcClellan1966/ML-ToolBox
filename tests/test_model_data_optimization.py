"""
Tests for Model & Data Optimization
Distillation, pruning, specialized models, RAG
"""
import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from model_data_optimization import (
        KnowledgeDistillation, ModelPruning, SpecializedModelBuilder,
        RetrievalAugmentedGeneration, ModelOptimizationPipeline
    )
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    pytestmark = pytest.mark.skip("Features not available")


class TestKnowledgeDistillation:
    """Tests for knowledge distillation"""
    
    def test_distill_knowledge(self):
        """Test knowledge distillation"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create teacher and student models
        teacher = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        student = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
        
        # Generate data
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        # Train teacher
        teacher.fit(X_train, y_train)
        
        # Distill knowledge
        distillation = KnowledgeDistillation(temperature=3.0, alpha=0.5)
        result = distillation.distill_knowledge(teacher, student, X_train, y_train)
        
        assert 'success' in result
        assert result.get('success') == True


class TestModelPruning:
    """Tests for model pruning"""
    
    def test_prune_model(self):
        """Test model pruning"""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        model.fit(X_train, y_train)
        
        X_val = np.random.rand(20, 10)
        y_val = np.random.randint(0, 2, 20)
        
        pruning = ModelPruning(pruning_method='magnitude')
        result = pruning.prune_model(model, pruning_ratio=0.3, X_val=X_val, y_val=y_val)
        
        assert 'success' in result
        assert result.get('success') == True


class TestSpecializedModelBuilder:
    """Tests for specialized model builder"""
    
    def test_build_specialized_model(self):
        """Test building specialized model"""
        builder = SpecializedModelBuilder(target_size='small')
        
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        result = builder.build_specialized_model(
            'random_forest', X_train, y_train, task='classification'
        )
        
        assert 'success' in result
        assert result.get('success') == True
        assert 'model' in result


class TestRetrievalAugmentedGeneration:
    """Tests for RAG"""
    
    def test_add_documents(self):
        """Test adding documents"""
        rag = RetrievalAugmentedGeneration()
        
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing deals with text data."
        ]
        
        result = rag.add_documents(documents)
        
        assert result['success'] == True
        assert result['total_documents'] == 3
    
    def test_retrieve_relevant_context(self):
        """Test retrieving relevant context"""
        rag = RetrievalAugmentedGeneration()
        
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing deals with text data."
        ]
        
        rag.add_documents(documents)
        
        result = rag.retrieve_relevant_context("What is machine learning?", top_k=2)
        
        assert 'retrieved_documents' in result
        assert len(result['retrieved_documents']) > 0
    
    def test_augment_prompt(self):
        """Test prompt augmentation"""
        rag = RetrievalAugmentedGeneration()
        
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers."
        ]
        
        rag.add_documents(documents)
        
        base_prompt = "Answer the following question:"
        result = rag.augment_prompt("What is machine learning?", base_prompt, top_k=2)
        
        assert 'augmented_prompt' in result
        assert 'context' in result


class TestModelOptimizationPipeline:
    """Tests for model optimization pipeline"""
    
    def test_optimize_model(self):
        """Test model optimization pipeline"""
        from sklearn.ensemble import RandomForestClassifier
        
        teacher = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        teacher.fit(X_train, y_train)
        
        pipeline = ModelOptimizationPipeline()
        result = pipeline.optimize_model(
            teacher, X_train, y_train,
            optimization_steps=['distillation']
        )
        
        assert 'steps_applied' in result
        assert 'final_model' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
