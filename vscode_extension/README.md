# ML Toolbox & AI Agent VS Code Extension

VS Code extension that integrates ML Toolbox and AI Agent for seamless ML development.

## Features

### 🚀 **ML Toolbox Integration**
- Quick model training
- One-click model creation
- Visual model selection
- Results visualization

### 🤖 **AI Agent Integration**
- Generate code from natural language
- Fix errors automatically
- Suggest patterns
- Code completion

### 📊 **Interactive Features**
- Command palette commands
- Context menu integration
- Status bar indicators
- Output panels

## Installation

1. Install VS Code
2. Install Python extension
3. Install this extension
4. Configure Python path in settings

## Usage

### Train Model
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "Train ML Model"
3. Select task type
4. Code is generated automatically!

### AI Generate Code
1. Press `Ctrl+Shift+P`
2. Type "AI: Generate Code"
3. Describe what you want
4. Code is inserted at cursor!

### Fix Error
1. Select code with error
2. Right-click → "AI: Fix Error"
3. Error is fixed automatically!

## Commands

- `mlToolbox.trainModel` - Train ML model
- `mlToolbox.aiGenerate` - Generate code with AI
- `mlToolbox.aiFix` - Fix code errors
- `mlToolbox.showPanel` - Show ML Toolbox panel
- `mlToolbox.quickTrain` - Quick train with defaults

## Configuration

```json
{
  "mlToolbox.enableAI": true,
  "mlToolbox.autoSuggest": true,
  "mlToolbox.modelCache": true
}
```

## Requirements

- VS Code 1.80.0+
- Python 3.8+
- ML Toolbox installed

## Development

```bash
npm install
npm run compile
npm run watch
```

Press F5 to launch extension in new window.

## License

MIT
