# IDE Integration Guide - ML Toolbox & AI Agent

## 🎯 **Overview**

Complete guide for integrating ML Toolbox and AI Agent into IDEs.

---

## ✅ **Recommended: VS Code Extension**

### **Why VS Code?**
- ✅ Most popular Python IDE
- ✅ Excellent extension system
- ✅ Built-in Jupyter support
- ✅ Free and open source
- ✅ Easy to extend

### **What's Included**

1. **VS Code Extension** (`vscode_extension/`)
   - Full extension structure
   - Toolbox integration
   - AI Agent integration
   - UI components
   - Commands

2. **Features:**
   - Quick model training
   - AI code generation
   - Error fixing
   - Visual panels
   - Command palette

---

## 🚀 **Quick Start**

### **1. Install VS Code Extension**

```bash
cd vscode_extension
npm install
npm run compile
```

### **2. Launch Extension**

Press `F5` in VS Code to launch extension in new window.

### **3. Use Commands**

- `Ctrl+Shift+P` → "Train ML Model"
- `Ctrl+Shift+P` → "AI: Generate Code"
- Right-click → "AI: Fix Error"

---

## 📋 **Extension Features**

### **1. ML Toolbox Integration**

**Commands:**
- `mlToolbox.trainModel` - Train ML model
- `mlToolbox.quickTrain` - Quick train with defaults
- `mlToolbox.showPanel` - Show ML Toolbox panel

**Usage:**
```typescript
// Generated code example
from ml_toolbox import MLToolbox
import numpy as np

toolbox = MLToolbox()
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)
result = toolbox.fit(X, y, task_type='classification')
```

### **2. AI Agent Integration**

**Commands:**
- `mlToolbox.aiGenerate` - Generate code from natural language
- `mlToolbox.aiFix` - Fix code errors automatically

**Usage:**
1. Press `Ctrl+Shift+P`
2. Type "AI: Generate Code"
3. Enter: "Classify data into 3 classes"
4. Code is generated and inserted!

### **3. Interactive Features**

**Context Menu:**
- Right-click in Python file
- Select "AI: Generate Code" or "AI: Fix Error"

**Status Bar:**
- Shows ML Toolbox status
- Quick access to features

**Output Panels:**
- ML Toolbox output
- AI Agent output
- Results visualization

---

## 🔧 **Alternative Integrations**

### **1. Jupyter Notebook Integration**

**Magic Commands:**
```python
%load_ext ml_toolbox_ide

# Toolbox commands
%toolbox_train X y --model random_forest
%toolbox_predict model_id X
%toolbox_evaluate model_id X y

# AI Agent commands
%ai_generate "Classify data into 3 classes"
%ai_fix_error
%ai_suggest_pattern
```

**Widget Integration:**
```python
from ml_toolbox_ide import ToolboxWidget

widget = ToolboxWidget()
widget.train_model(X, y)
widget.show_results()
```

### **2. PyCharm Plugin**

**External Tools:**
- Configure ML Toolbox as external tool
- Add run configurations
- Use AI Agent via scripts

### **3. Custom Web IDE**

**Features:**
- Browser-based (no installation)
- Integrated Toolbox
- Integrated AI Agent
- Visual ML pipeline builder

**Tech Stack:**
- React/Vue frontend
- Python backend
- Monaco Editor (VS Code editor)
- WebSocket for real-time

---

## 📦 **Extension Structure**

```
vscode_extension/
├── package.json          # Extension manifest
├── tsconfig.json         # TypeScript config
├── src/
│   ├── extension.ts      # Main extension file
│   ├── toolboxProvider.ts    # Toolbox integration
│   ├── aiAgentProvider.ts    # AI Agent integration
│   └── pythonExecutor.ts    # Python execution
└── README.md             # Extension docs
```

---

## 🎨 **UI Components**

### **1. Command Palette**

Access via `Ctrl+Shift+P`:
- Train ML Model
- AI: Generate Code
- AI: Fix Error
- Show ML Toolbox Panel
- Quick Train Model

### **2. Context Menu**

Right-click in Python file:
- AI: Generate Code
- AI: Fix Error
- Train Model

### **3. Sidebar Panel**

ML Toolbox panel with:
- Quick actions
- Model list
- Data management
- Results viewer

### **4. Status Bar**

Shows:
- ML Toolbox status
- AI Agent status
- Quick access buttons

---

## ⚙️ **Configuration**

### **VS Code Settings**

```json
{
  "mlToolbox.enableAI": true,
  "mlToolbox.autoSuggest": true,
  "mlToolbox.modelCache": true,
  "python.pythonPath": "/path/to/python"
}
```

### **Extension Settings**

- `mlToolbox.enableAI` - Enable AI Agent features
- `mlToolbox.autoSuggest` - Enable AI code suggestions
- `mlToolbox.modelCache` - Enable model caching

---

## 🧪 **Testing**

### **Test Extension**

1. Open VS Code
2. Press `F5` to launch extension
3. Test commands
4. Check output panels
5. Verify integration

### **Test Commands**

```bash
# Test Toolbox integration
Ctrl+Shift+P → "Train ML Model"

# Test AI Agent
Ctrl+Shift+P → "AI: Generate Code"
Enter: "Classify data"

# Test error fixing
Select code with error
Right-click → "AI: Fix Error"
```

---

## 📚 **Usage Examples**

### **Example 1: Quick Model Training**

1. Press `Ctrl+Shift+P`
2. Type "Quick Train Model"
3. Code is inserted:
```python
from ml_toolbox import MLToolbox
import numpy as np

toolbox = MLToolbox()
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)
result = toolbox.fit(X, y, task_type='classification')
```

### **Example 2: AI Code Generation**

1. Press `Ctrl+Shift+P`
2. Type "AI: Generate Code"
3. Enter: "Create a regression model"
4. AI generates code automatically!

### **Example 3: Error Fixing**

1. Write code with error
2. Select the code
3. Right-click → "AI: Fix Error"
4. Error is fixed automatically!

---

## 🚀 **Publishing**

### **Publish to VS Code Marketplace**

1. Install `vsce`: `npm install -g vsce`
2. Package extension: `vsce package`
3. Publish: `vsce publish`

### **Install from VSIX**

1. Package extension: `vsce package`
2. Install: `code --install-extension ml-toolbox-ai-agent-0.1.0.vsix`

---

## 💡 **Future Enhancements**

1. **Visual ML Pipeline Builder**
   - Drag-and-drop interface
   - Visual model selection
   - Pipeline visualization

2. **Real-time Collaboration**
   - Share models
   - Collaborative training
   - Live results

3. **Advanced Visualization**
   - Model performance charts
   - Data visualization
   - Training progress

4. **IntelliSense Integration**
   - Auto-complete for Toolbox APIs
   - AI-powered suggestions
   - Pattern-based completions

---

## ✅ **Summary**

### **What You Get:**
- ✅ VS Code extension (ready to use)
- ✅ Toolbox integration
- ✅ AI Agent integration
- ✅ UI components
- ✅ Commands and shortcuts

### **Benefits:**
- ✅ Seamless ML development
- ✅ AI-powered code generation
- ✅ Error fixing assistance
- ✅ Quick model training
- ✅ Visual interface

---

**Ready to use! Install the extension and start building ML models with AI assistance!**
