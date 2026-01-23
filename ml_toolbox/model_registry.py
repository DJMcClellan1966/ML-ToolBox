"""
Model Registry System
Versioning, staging, and deployment workflows for ML models
"""
import sys
from pathlib import Path
import json
import pickle
import hashlib
import time
from typing import Any, Optional, Dict, List, Tuple
from datetime import datetime
from enum import Enum
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))


class ModelStage(Enum):
    """Model staging levels"""
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelRegistry:
    """
    Model Registry for ML Toolbox
    
    Features:
    - Model versioning (semantic versioning)
    - Model staging (dev → staging → production)
    - Model metadata tracking
    - Model lineage
    - Deployment workflows
    - A/B testing support
    - Rollback capabilities
    """
    
    def __init__(self, registry_dir: str = ".model_registry"):
        """
        Initialize Model Registry
        
        Args:
            registry_dir: Directory for model registry storage
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        
        # Subdirectories
        self.models_dir = self.registry_dir / "models"
        self.metadata_dir = self.registry_dir / "metadata"
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # In-memory registry
        self._registry: Dict[str, Dict] = {}
        self._load_registry()
        
        self.stats = {
            'total_models': 0,
            'versions_created': 0,
            'deployments': 0,
            'rollbacks': 0
        }
    
    def _load_registry(self):
        """Load registry from disk"""
        registry_file = self.registry_dir / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    self._registry = json.load(f)
            except Exception as e:
                warnings.warn(f"Failed to load registry: {e}")
                self._registry = {}
    
    def _save_registry(self):
        """Save registry to disk"""
        registry_file = self.registry_dir / "registry.json"
        try:
            with open(registry_file, 'w') as f:
                json.dump(self._registry, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save registry: {e}")
    
    def register_model(
        self,
        model: Any,
        model_name: str,
        version: Optional[str] = None,
        metadata: Optional[Dict] = None,
        stage: ModelStage = ModelStage.DEV
    ) -> str:
        """
        Register a model in the registry
        
        Args:
            model: Model object to register
            model_name: Name of the model
            version: Version string (e.g., "1.0.0"). If None, auto-increments
            metadata: Additional metadata (metrics, parameters, etc.)
            stage: Initial stage (default: DEV)
        
        Returns:
            Full model identifier (name:version)
        """
        # Generate version if not provided
        if version is None:
            existing_versions = [
                v for k, v in self._registry.items()
                if k.startswith(f"{model_name}:")
            ]
            if existing_versions:
                # Extract version numbers and increment
                max_version = max([
                    tuple(map(int, v.split(':')[1].split('.')))
                    for v in existing_versions
                    if ':' in v and v.split(':')[1].count('.') == 2
                ], default=(0, 0, 0))
                version = f"{max_version[0]}.{max_version[1]}.{max_version[2] + 1}"
            else:
                version = "1.0.0"
        
        model_id = f"{model_name}:{version}"
        
        # Save model
        model_file = self.models_dir / f"{model_id.replace(':', '_')}.pkl"
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")
        
        # Create metadata
        model_metadata = {
            'model_id': model_id,
            'model_name': model_name,
            'version': version,
            'stage': stage.value,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'model_path': str(model_file),
            'metadata': metadata or {},
            'lineage': [],
            'deployment_history': []
        }
        
        # Save metadata
        metadata_file = self.metadata_dir / f"{model_id.replace(':', '_')}.json"
        with open(metadata_file, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Update registry
        self._registry[model_id] = model_metadata
        self._save_registry()
        
        self.stats['total_models'] = len(self._registry)
        self.stats['versions_created'] += 1
        
        return model_id
    
    def get_model(self, model_id: str) -> Tuple[Any, Dict]:
        """
        Get model and metadata by ID
        
        Args:
            model_id: Model identifier (name:version)
        
        Returns:
            Tuple of (model, metadata)
        """
        if model_id not in self._registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        metadata = self._registry[model_id]
        model_path = Path(metadata['model_path'])
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model, metadata
    
    def promote_model(self, model_id: str, target_stage: ModelStage) -> bool:
        """
        Promote model to new stage
        
        Args:
            model_id: Model identifier
            target_stage: Target stage (DEV → STAGING → PRODUCTION)
        
        Returns:
            True if successful
        """
        if model_id not in self._registry:
            raise ValueError(f"Model {model_id} not found")
        
        current_stage = ModelStage(self._registry[model_id]['stage'])
        
        # Validate promotion path
        valid_promotions = {
            ModelStage.DEV: [ModelStage.STAGING],
            ModelStage.STAGING: [ModelStage.PRODUCTION, ModelStage.DEV],
            ModelStage.PRODUCTION: [ModelStage.ARCHIVED],
            ModelStage.ARCHIVED: [ModelStage.DEV]
        }
        
        if target_stage not in valid_promotions.get(current_stage, []):
            raise ValueError(
                f"Cannot promote from {current_stage.value} to {target_stage.value}. "
                f"Valid promotions: {valid_promotions.get(current_stage, [])}"
            )
        
        # Update stage
        self._registry[model_id]['stage'] = target_stage.value
        self._registry[model_id]['updated_at'] = datetime.now().isoformat()
        
        # Update metadata file
        metadata_file = self.metadata_dir / f"{model_id.replace(':', '_')}.json"
        with open(metadata_file, 'w') as f:
            json.dump(self._registry[model_id], f, indent=2)
        
        self._save_registry()
        
        if target_stage == ModelStage.PRODUCTION:
            self.stats['deployments'] += 1
        
        return True
    
    def list_models(self, stage: Optional[ModelStage] = None, model_name: Optional[str] = None) -> List[Dict]:
        """
        List models in registry
        
        Args:
            stage: Filter by stage
            model_name: Filter by model name
        
        Returns:
            List of model metadata dictionaries
        """
        models = list(self._registry.values())
        
        if stage:
            models = [m for m in models if m['stage'] == stage.value]
        
        if model_name:
            models = [m for m in models if m['model_name'] == model_name]
        
        return sorted(models, key=lambda x: x['created_at'], reverse=True)
    
    def get_production_models(self) -> List[Dict]:
        """Get all production models"""
        return self.list_models(stage=ModelStage.PRODUCTION)
    
    def rollback_model(self, model_name: str, target_version: str) -> bool:
        """
        Rollback to previous version
        
        Args:
            model_name: Name of model
            target_version: Version to rollback to
        
        Returns:
            True if successful
        """
        model_id = f"{model_name}:{target_version}"
        
        if model_id not in self._registry:
            raise ValueError(f"Model {model_id} not found")
        
        # Demote current production model
        prod_models = self.get_production_models()
        for prod_model in prod_models:
            if prod_model['model_name'] == model_name:
                self.promote_model(prod_model['model_id'], ModelStage.ARCHIVED)
        
        # Promote target version to production
        self.promote_model(model_id, ModelStage.PRODUCTION)
        
        self.stats['rollbacks'] += 1
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            **self.stats,
            'models_by_stage': {
                stage.value: len(self.list_models(stage=stage))
                for stage in ModelStage
            }
        }


# Global registry instance
_global_registry: Optional[ModelRegistry] = None

def get_model_registry(registry_dir: str = ".model_registry") -> ModelRegistry:
    """Get global model registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry(registry_dir=registry_dir)
    return _global_registry
