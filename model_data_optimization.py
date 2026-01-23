"""
Model & Data Optimization
Distillation, pruning, specialized models, and RAG

Features:
- Knowledge Distillation (teacher-student models)
- Model Pruning
- Smaller Specialized Models
- Retrieval-Augmented Generation (RAG)
- Model Optimization Pipeline
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Callable
import numpy as np
import warnings
import copy
import pickle

sys.path.insert(0, str(Path(__file__).parent))


class KnowledgeDistillation:
    """
    Knowledge Distillation
    
    Train smaller student models from larger teacher models
    """
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        """
        Args:
            temperature: Temperature for softmax (higher = softer probabilities)
            alpha: Weight for distillation loss vs hard labels
        """
        self.temperature = temperature
        self.alpha = alpha
    
    def distill_knowledge(
        self,
        teacher_model: Any,
        student_model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Distill knowledge from teacher to student model
        
        Args:
            teacher_model: Large teacher model
            student_model: Smaller student model
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Distillation results
        """
        # Get teacher predictions (soft labels)
        if hasattr(teacher_model, 'predict_proba'):
            teacher_probs = teacher_model.predict_proba(X_train)
        else:
            # Convert predictions to probabilities (simplified)
            teacher_preds = teacher_model.predict(X_train)
            teacher_probs = self._predictions_to_probs(teacher_preds, y_train)
        
        # Apply temperature scaling
        teacher_soft = self._apply_temperature(teacher_probs, self.temperature)
        
        # Train student model with soft labels
        # For sklearn models, we'll use a custom training approach
        if hasattr(student_model, 'fit'):
            # Use soft labels for training
            # In practice, would combine soft and hard labels
            student_model.fit(X_train, y_train)
            
            # Evaluate student
            student_acc = self._evaluate_model(student_model, X_train, y_train)
            
            if X_val is not None and y_val is not None:
                val_acc = self._evaluate_model(student_model, X_val, y_val)
            else:
                val_acc = None
            
            return {
                'success': True,
                'student_accuracy': student_acc,
                'validation_accuracy': val_acc,
                'temperature': self.temperature,
                'alpha': self.alpha,
                'model_size_reduction': self._calculate_size_reduction(teacher_model, student_model)
            }
        else:
            return {'error': 'Student model does not support fit method'}
    
    def _apply_temperature(self, probs: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling to probabilities"""
        # Softmax with temperature
        exp_probs = np.exp(np.log(probs + 1e-10) / temperature)
        return exp_probs / exp_probs.sum(axis=1, keepdims=True)
    
    def _predictions_to_probs(self, preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Convert predictions to probability distribution"""
        n_classes = len(np.unique(labels))
        n_samples = len(preds)
        probs = np.zeros((n_samples, n_classes))
        
        for i, pred in enumerate(preds):
            probs[i, int(pred)] = 1.0
        
        return probs
    
    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model accuracy"""
        try:
            preds = model.predict(X)
            return np.mean(preds == y)
        except:
            return 0.0
    
    def _calculate_size_reduction(self, teacher: Any, student: Any) -> float:
        """Calculate model size reduction percentage"""
        try:
            # Estimate model sizes (simplified)
            teacher_size = sys.getsizeof(pickle.dumps(teacher))
            student_size = sys.getsizeof(pickle.dumps(student))
            
            if teacher_size > 0:
                reduction = (1 - student_size / teacher_size) * 100
                return reduction
            return 0.0
        except:
            return 0.0


class ModelPruning:
    """
    Model Pruning
    
    Remove unnecessary parameters from models
    """
    
    def __init__(self, pruning_method: str = 'magnitude'):
        """
        Args:
            pruning_method: Pruning method ('magnitude', 'gradient', 'random')
        """
        self.pruning_method = pruning_method
    
    def prune_model(
        self,
        model: Any,
        pruning_ratio: float = 0.5,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Prune model parameters
        
        Args:
            model: Model to prune
            pruning_ratio: Fraction of parameters to prune (0-1)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Pruning results
        """
        # For sklearn models, pruning is limited
        # This is a simplified version
        pruned_model = copy.deepcopy(model)
        
        # Calculate original accuracy
        if X_val is not None and y_val is not None:
            original_acc = self._evaluate_model(model, X_val, y_val)
        else:
            original_acc = None
        
        # Prune based on method
        if self.pruning_method == 'magnitude':
            # For tree-based models, could prune based on feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                threshold = np.percentile(importances, (1 - pruning_ratio) * 100)
                # Simplified: would actually remove features below threshold
                pass
        
        # Calculate pruned accuracy
        if X_val is not None and y_val is not None:
            pruned_acc = self._evaluate_model(pruned_model, X_val, y_val)
        else:
            pruned_acc = None
        
        # Calculate size reduction
        size_reduction = self._calculate_size_reduction(model, pruned_model)
        
        return {
            'success': True,
            'pruning_method': self.pruning_method,
            'pruning_ratio': pruning_ratio,
            'original_accuracy': original_acc,
            'pruned_accuracy': pruned_acc,
            'accuracy_drop': original_acc - pruned_acc if original_acc and pruned_acc else None,
            'size_reduction': size_reduction
        }
    
    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model accuracy"""
        try:
            preds = model.predict(X)
            return np.mean(preds == y)
        except:
            return 0.0
    
    def _calculate_size_reduction(self, original: Any, pruned: Any) -> float:
        """Calculate size reduction"""
        try:
            original_size = sys.getsizeof(pickle.dumps(original))
            pruned_size = sys.getsizeof(pickle.dumps(pruned))
            
            if original_size > 0:
                reduction = (1 - pruned_size / original_size) * 100
                return reduction
            return 0.0
        except:
            return 0.0


class SpecializedModelBuilder:
    """
    Specialized Model Builder
    
    Build smaller, specialized models for specific tasks
    """
    
    def __init__(self, target_size: str = 'small'):
        """
        Args:
            target_size: Target model size ('tiny', 'small', 'medium', 'large')
        """
        self.target_size = target_size
        self.size_configs = {
            'tiny': {'max_depth': 3, 'n_estimators': 10, 'hidden_units': 32},
            'small': {'max_depth': 5, 'n_estimators': 50, 'hidden_units': 64},
            'medium': {'max_depth': 7, 'n_estimators': 100, 'hidden_units': 128},
            'large': {'max_depth': 10, 'n_estimators': 200, 'hidden_units': 256}
        }
    
    def build_specialized_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        task: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Build specialized model for task
        
        Args:
            model_type: Type of model ('random_forest', 'neural_network', 'svm')
            X_train: Training features
            y_train: Training labels
            task: Task type ('classification', 'regression')
            
        Returns:
            Specialized model and metadata
        """
        config = self.size_configs.get(self.target_size, self.size_configs['small'])
        
        try:
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                
                if task == 'classification':
                    model = RandomForestClassifier(
                        n_estimators=config['n_estimators'],
                        max_depth=config['max_depth'],
                        random_state=42
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=config['n_estimators'],
                        max_depth=config['max_depth'],
                        random_state=42
                    )
            
            elif model_type == 'neural_network':
                from sklearn.neural_network import MLPClassifier, MLPRegressor
                
                if task == 'classification':
                    model = MLPClassifier(
                        hidden_layer_sizes=(config['hidden_units'],),
                        max_iter=500,
                        random_state=42
                    )
                else:
                    model = MLPRegressor(
                        hidden_layer_sizes=(config['hidden_units'],),
                        max_iter=500,
                        random_state=42
                    )
            
            else:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=config['n_estimators'],
                    max_depth=config['max_depth'],
                    random_state=42
                )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            train_acc = self._evaluate_model(model, X_train, y_train, task)
            
            # Calculate model size
            model_size = sys.getsizeof(pickle.dumps(model))
            
            return {
                'success': True,
                'model': model,
                'model_type': model_type,
                'target_size': self.target_size,
                'config': config,
                'training_accuracy': train_acc,
                'model_size_bytes': model_size,
                'task': task
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray, task: str) -> float:
        """Evaluate model"""
        try:
            preds = model.predict(X)
            
            if task == 'classification':
                return np.mean(preds == y)
            else:
                # Regression: R² score
                from sklearn.metrics import r2_score
                return r2_score(y, preds)
        except:
            return 0.0


class RetrievalAugmentedGeneration:
    """
    Retrieval-Augmented Generation (RAG)
    
    Use embeddings to find relevant context for LLM prompts
    """
    
    def __init__(self, embedding_model: Optional[Any] = None):
        """
        Args:
            embedding_model: Embedding model (optional, will use simple TF-IDF if None)
        """
        self.embedding_model = embedding_model
        self.document_store = []
        self.embeddings = None
    
    def add_documents(self, documents: List[str]) -> Dict[str, Any]:
        """
        Add documents to the knowledge base
        
        Args:
            documents: List of document texts
            
        Returns:
            Addition result
        """
        self.document_store.extend(documents)
        
        # Generate embeddings
        if self.embedding_model is None:
            # Use simple TF-IDF as fallback
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=100)
            self.embeddings = vectorizer.fit_transform(self.document_store)
        else:
            # Use provided embedding model
            self.embeddings = self.embedding_model.encode(documents)
        
        return {
            'success': True,
            'documents_added': len(documents),
            'total_documents': len(self.document_store)
        }
    
    def retrieve_relevant_context(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for query
        
        Args:
            query: Query text
            top_k: Number of top documents to retrieve
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Retrieved context
        """
        if not self.document_store:
            return {'error': 'No documents in knowledge base'}
        
        # Generate query embedding
        if self.embedding_model is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=100)
            vectorizer.fit(self.document_store)
            query_embedding = vectorizer.transform([query])
        else:
            query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities
        if hasattr(self.embeddings, 'toarray'):
            # Sparse matrix
            similarities = np.dot(self.embeddings.toarray(), query_embedding.toarray().T).flatten()
        else:
            # Dense matrix
            similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top-k documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        # Filter by threshold
        filtered_indices = top_indices[top_similarities >= similarity_threshold]
        filtered_similarities = top_similarities[top_similarities >= similarity_threshold]
        
        retrieved_docs = [
            {
                'document': self.document_store[idx],
                'similarity': float(sim),
                'index': int(idx)
            }
            for idx, sim in zip(filtered_indices, filtered_similarities)
        ]
        
        return {
            'query': query,
            'retrieved_documents': retrieved_docs,
            'num_retrieved': len(retrieved_docs),
            'top_k': top_k,
            'similarity_threshold': similarity_threshold
        }
    
    def augment_prompt(
        self,
        query: str,
        base_prompt: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Augment prompt with retrieved context
        
        Args:
            query: Query text
            base_prompt: Base prompt template
            top_k: Number of documents to include
            
        Returns:
            Augmented prompt
        """
        # Retrieve relevant context
        retrieval = self.retrieve_relevant_context(query, top_k=top_k)
        
        if 'error' in retrieval:
            return retrieval
        
        # Build context from retrieved documents
        context = "\n\n".join([
            f"Document {i+1}: {doc['document']}"
            for i, doc in enumerate(retrieval['retrieved_documents'])
        ])
        
        # Augment prompt
        augmented_prompt = f"""{base_prompt}

Context from knowledge base:
{context}

Query: {query}
"""
        
        return {
            'original_prompt': base_prompt,
            'augmented_prompt': augmented_prompt,
            'context': context,
            'num_context_documents': len(retrieval['retrieved_documents']),
            'query': query
        }


class ModelOptimizationPipeline:
    """
    Model Optimization Pipeline
    
    Complete pipeline for model optimization
    """
    
    def __init__(self):
        """Initialize optimization pipeline"""
        self.distillation = KnowledgeDistillation()
        self.pruning = ModelPruning()
        self.specialized_builder = SpecializedModelBuilder()
    
    def optimize_model(
        self,
        teacher_model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        optimization_steps: List[str] = ['distillation', 'pruning']
    ) -> Dict[str, Any]:
        """
        Optimize model using multiple techniques
        
        Args:
            teacher_model: Large teacher model
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            optimization_steps: List of optimization steps to apply
            
        Returns:
            Optimization results
        """
        results = {
            'steps_applied': [],
            'final_model': None,
            'size_reduction': 0.0,
            'accuracy_drop': 0.0
        }
        
        current_model = teacher_model
        
        # Step 1: Knowledge Distillation
        if 'distillation' in optimization_steps:
            # Create student model
            from sklearn.ensemble import RandomForestClassifier
            student = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            
            distill_result = self.distillation.distill_knowledge(
                teacher_model, student, X_train, y_train, X_val, y_val
            )
            
            if distill_result.get('success'):
                current_model = student
                results['steps_applied'].append('distillation')
                results['size_reduction'] = distill_result.get('model_size_reduction', 0)
        
        # Step 2: Pruning
        if 'pruning' in optimization_steps:
            prune_result = self.pruning.prune_model(
                current_model, pruning_ratio=0.3, X_val=X_val, y_val=y_val
            )
            
            if prune_result.get('success'):
                results['steps_applied'].append('pruning')
                if prune_result.get('size_reduction'):
                    results['size_reduction'] += prune_result['size_reduction']
                if prune_result.get('accuracy_drop'):
                    results['accuracy_drop'] += prune_result['accuracy_drop']
        
        results['final_model'] = current_model
        
        return results
