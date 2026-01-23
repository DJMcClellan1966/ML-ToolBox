/**
 * Toolbox Provider - Handles ML Toolbox integration
 */

import * as vscode from 'vscode';
import { PythonExecutor } from './pythonExecutor';

export class ToolboxProvider {
    private outputChannel: vscode.OutputChannel;
    private pythonExecutor: PythonExecutor;

    constructor(pythonExecutor: PythonExecutor) {
        this.pythonExecutor = pythonExecutor;
        this.outputChannel = vscode.window.createOutputChannel('ML Toolbox');
    }

    async trainModel(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        // Get user input for training
        const taskType = await vscode.window.showQuickPick(
            ['classification', 'regression', 'clustering'],
            { placeHolder: 'Select task type' }
        );

        if (!taskType) {
            return;
        }

        // Generate code using AI Agent
        const code = await this.generateTrainingCode(taskType);
        
        // Insert code at cursor
        const position = editor.selection.active;
        editor.edit(editBuilder => {
            editBuilder.insert(position, code);
        });

        vscode.window.showInformationMessage('Training code generated!');
    }

    async quickTrain(): Promise<void> {
        // Quick train with default settings
        const code = `from ml_toolbox import MLToolbox
import numpy as np

# Initialize toolbox
toolbox = MLToolbox()

# Generate sample data
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)

# Train model
result = toolbox.fit(X, y, task_type='classification')
print(f"Model trained! Accuracy: {result.get('accuracy', 0):.2%}")`;

        const editor = vscode.window.activeTextEditor;
        if (editor) {
            const position = editor.selection.active;
            editor.edit(editBuilder => {
                editBuilder.insert(position, code);
            });
        }
    }

    private async generateTrainingCode(taskType: string): Promise<string> {
        // Use AI Agent to generate code
        const prompt = `Generate code to train a ${taskType} model using ML Toolbox`;
        
        try {
            const code = await this.pythonExecutor.executeAI(prompt);
            return code;
        } catch (error) {
            // Fallback to template
            return this.getTrainingTemplate(taskType);
        }
    }

    private getTrainingTemplate(taskType: string): string {
        return `from ml_toolbox import MLToolbox
import numpy as np

# Initialize toolbox
toolbox = MLToolbox()

# Load your data here
# X = your_features
# y = your_labels

# Train model
result = toolbox.fit(X, y, task_type='${taskType}')
print(f"Model trained! Result: {result}")
`;
    }

    showPanel(): void {
        this.outputChannel.show();
    }
}
