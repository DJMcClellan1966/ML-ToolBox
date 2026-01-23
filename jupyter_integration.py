"""
Jupyter Notebook Integration for ML Toolbox & AI Agent
Magic commands and widgets for Jupyter notebooks
"""
import sys
from pathlib import Path
from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
from IPython.display import display, HTML
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    from ml_toolbox import MLToolbox
    from ml_toolbox.ai_agent import MLCodeAgent
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False


@magics_class
class MLToolboxMagics(Magics):
    """Jupyter magic commands for ML Toolbox"""
    
    def __init__(self, shell):
        super().__init__(shell)
        self.toolbox = MLToolbox() if TOOLBOX_AVAILABLE else None
        self.agent = MLCodeAgent(use_llm=False) if TOOLBOX_AVAILABLE else None
    
    @line_magic
    def toolbox_train(self, line):
        """Train ML model: %toolbox_train X y --model random_forest"""
        if not TOOLBOX_AVAILABLE:
            return "ML Toolbox not available"
        
        # Parse arguments
        args = line.split()
        if len(args) < 2:
            return "Usage: %toolbox_train X y [--model MODEL_TYPE]"
        
        X_var = args[0]
        y_var = args[1]
        model_type = 'auto'
        
        if '--model' in args:
            idx = args.index('--model')
            if idx + 1 < len(args):
                model_type = args[idx + 1]
        
        # Get variables from namespace
        X = self.shell.user_ns.get(X_var)
        y = self.shell.user_ns.get(y_var)
        
        if X is None or y is None:
            return f"Variables {X_var} or {y_var} not found"
        
        # Train model
        result = self.toolbox.fit(X, y, model_type=model_type)
        
        # Store result
        self.shell.user_ns['_last_model_result'] = result
        
        return f"Model trained! Accuracy: {result.get('accuracy', 0):.2%}"
    
    @line_magic
    def ai_generate(self, line):
        """Generate code with AI: %ai_generate \"Classify data into 3 classes\""""
        if not TOOLBOX_AVAILABLE:
            return "AI Agent not available"
        
        if not line:
            return "Usage: %ai_generate \"your task description\""
        
        # Generate code
        result = self.agent.build(line)
        
        if result.get('success'):
            code = result['code']
            # Execute code
            exec(code, self.shell.user_ns)
            return f"Code generated and executed!\n\n{code}"
        else:
            return f"Generation failed: {result.get('error', 'Unknown error')}"
    
    @line_magic
    def ai_fix(self, line):
        """Fix code error: %ai_fix"""
        if not TOOLBOX_AVAILABLE:
            return "AI Agent not available"
        
        # Get last error
        if hasattr(self.shell, '_last_error'):
            error = self.shell._last_error
            code = self.shell._last_code
            
            # Try to fix
            result = self.agent.build(f"Fix this code error: {error}\n\nCode:\n{code}")
            
            if result.get('success'):
                return f"Fixed code:\n\n{result['code']}"
            else:
                return f"Fix failed: {result.get('error', 'Unknown error')}"
        else:
            return "No error to fix"
    
    @line_magic
    def toolbox_predict(self, line):
        """Make predictions: %toolbox_predict model_id X"""
        if not TOOLBOX_AVAILABLE:
            return "ML Toolbox not available"
        
        args = line.split()
        if len(args) < 2:
            return "Usage: %toolbox_predict model_id X"
        
        model_id = args[0]
        X_var = args[1]
        
        X = self.shell.user_ns.get(X_var)
        if X is None:
            return f"Variable {X_var} not found"
        
        # Get model from registry
        model = self.toolbox.get_model(model_id)
        if model is None:
            return f"Model {model_id} not found"
        
        # Make predictions
        predictions = self.toolbox.predict(model_id, X)
        self.shell.user_ns['_last_predictions'] = predictions
        
        return f"Predictions made! Shape: {predictions.shape}"


def load_ipython_extension(ipython):
    """Load the extension"""
    ipython.register_magics(MLToolboxMagics)
    print("ML Toolbox & AI Agent magic commands loaded!")
    print("Available commands:")
    print("  %toolbox_train X y --model MODEL_TYPE")
    print("  %ai_generate \"task description\"")
    print("  %ai_fix")
    print("  %toolbox_predict model_id X")


def unload_ipython_extension(ipython):
    """Unload the extension"""
    pass


# Widget integration
try:
    from ipywidgets import interact, interactive, fixed, interact_manual
    import ipywidgets as widgets
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False


class ToolboxWidget:
    """Interactive widget for ML Toolbox"""
    
    def __init__(self):
        if not TOOLBOX_AVAILABLE:
            raise ImportError("ML Toolbox not available")
        
        self.toolbox = MLToolbox()
        self.agent = MLCodeAgent(use_llm=False)
    
    def create_train_widget(self):
        """Create training widget"""
        if not WIDGETS_AVAILABLE:
            return None
        
        task_type = widgets.Dropdown(
            options=['classification', 'regression', 'clustering'],
            value='classification',
            description='Task:'
        )
        
        model_type = widgets.Dropdown(
            options=['auto', 'random_forest', 'svm', 'logistic', 'linear'],
            value='auto',
            description='Model:'
        )
        
        button = widgets.Button(description='Train Model')
        output = widgets.Output()
        
        def on_button_click(b):
            with output:
                print("Training model...")
                # Get data from namespace (would need to be passed)
                # This is a simplified example
                print("Use %toolbox_train X y for actual training")
        
        button.on_click(on_button_click)
        
        return widgets.VBox([task_type, model_type, button, output])
    
    def show(self):
        """Show widget"""
        widget = self.create_train_widget()
        if widget:
            display(widget)


if __name__ == '__main__':
    print("Jupyter integration for ML Toolbox & AI Agent")
    print("Load in Jupyter with: %load_ext jupyter_integration")
