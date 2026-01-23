"""
Unified Error Handler
Consistent error handling across the toolbox

Improvement: Better error messages, helpful suggestions
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import traceback
import logging

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolboxErrorHandler:
    """
    Unified Error Handler
    
    Improvement: Consistent, helpful error handling
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.error_history = []
    
    def handle_import_error(self, module: str, feature_name: str, 
                           is_optional: bool = True) -> Optional[Any]:
        """
        Handle import errors gracefully
        
        Args:
            module: Module name that failed to import
            feature_name: Name of feature that requires this module
            is_optional: Whether this is an optional dependency
        
        Returns:
            None if optional and missing, raises if required
        """
        if is_optional:
            logger.debug(f"{module} not available - {feature_name} will be disabled")
            logger.info(f"💡 To enable {feature_name}, install: pip install {module}")
            return None
        else:
            error_msg = f"Required dependency '{module}' not found. Install with: pip install {module}"
            logger.error(error_msg)
            raise ImportError(error_msg)
    
    def handle_runtime_error(self, error: Exception, context: str, 
                            suggest_fix: bool = True) -> Dict[str, Any]:
        """
        Handle runtime errors with context
        
        Args:
            error: The exception that occurred
            context: Context where error occurred
            suggest_fix: Whether to suggest fixes
        
        Returns:
            Dictionary with error info and suggestions
        """
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'suggestions': []
        }
        
        if suggest_fix:
            error_info['suggestions'] = self._suggest_fix(error, context)
        
        # Log error
        logger.error(f"Error in {context}: {error}")
        if self.verbose:
            logger.debug(traceback.format_exc())
        
        # Record in history
        self.error_history.append(error_info)
        
        return error_info
    
    def _suggest_fix(self, error: Exception, context: str) -> List[str]:
        """Suggest fixes for common errors"""
        suggestions = []
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        if error_type == 'ImportError':
            if 'sklearn' in error_msg:
                suggestions.append("Install scikit-learn: pip install scikit-learn")
            elif 'numpy' in error_msg:
                suggestions.append("Install numpy: pip install numpy")
            else:
                suggestions.append(f"Install missing dependency: {error_msg}")
        
        elif error_type == 'AttributeError':
            suggestions.append("Check if object is properly initialized")
            suggestions.append("Verify attribute name is correct")
        
        elif error_type == 'ValueError':
            if 'shape' in error_msg:
                suggestions.append("Check data shapes match (X.shape[0] == y.shape[0])")
            elif 'dimension' in error_msg:
                suggestions.append("Check data dimensions are compatible")
        
        elif error_type == 'MemoryError':
            suggestions.append("Reduce data size or use batch processing")
            suggestions.append("Enable model caching to save memory")
        
        return suggestions
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors"""
        if not self.error_history:
            return {'total_errors': 0, 'message': 'No errors recorded'}
        
        error_types = {}
        for error in self.error_history:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'error_types': error_types,
            'recent_errors': self.error_history[-5:]  # Last 5 errors
        }


# Global instance
_global_error_handler = None

def get_error_handler(verbose: bool = False) -> ToolboxErrorHandler:
    """Get global error handler"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ToolboxErrorHandler(verbose=verbose)
    return _global_error_handler
