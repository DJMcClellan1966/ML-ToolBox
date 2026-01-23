/**
 * AI Agent Provider - Handles AI Agent integration
 */

import * as vscode from 'vscode';
import { PythonExecutor } from './pythonExecutor';

export class AIAgentProvider {
    private outputChannel: vscode.OutputChannel;
    private pythonExecutor: PythonExecutor;

    constructor(pythonExecutor: PythonExecutor) {
        this.pythonExecutor = pythonExecutor;
        this.outputChannel = vscode.window.createOutputChannel('AI Agent');
    }

    async generateCode(): Promise<void> {
        // Get user input
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Describe what code you want to generate...',
            prompt: 'AI Code Generation'
        });

        if (!prompt) {
            return;
        }

        // Show progress
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Generating code...",
            cancellable: false
        }, async (progress) => {
            try {
                // Use AI Agent to generate code
                const code = await this.pythonExecutor.executeAI(prompt);
                
                // Insert code
                const editor = vscode.window.activeTextEditor;
                if (editor) {
                    const position = editor.selection.active;
                    await editor.edit(editBuilder => {
                        editBuilder.insert(position, code);
                    });
                    
                    vscode.window.showInformationMessage('Code generated successfully!');
                }
            } catch (error) {
                vscode.window.showErrorMessage(`Code generation failed: ${error}`);
            }
        });
    }

    async fixError(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        // Get selected code or current line
        const selection = editor.selection;
        const code = selection.isEmpty 
            ? editor.document.lineAt(selection.active.line).text
            : editor.document.getText(selection);

        if (!code.trim()) {
            vscode.window.showWarningMessage('No code selected');
            return;
        }

        // Try to fix using AI Agent
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Fixing error...",
            cancellable: false
        }, async (progress) => {
            try {
                const fixedCode = await this.pythonExecutor.fixCode(code);
                
                // Replace code
                await editor.edit(editBuilder => {
                    if (selection.isEmpty) {
                        const line = editor.document.lineAt(selection.active.line);
                        editBuilder.replace(line.range, fixedCode);
                    } else {
                        editBuilder.replace(selection, fixedCode);
                    }
                });
                
                vscode.window.showInformationMessage('Code fixed!');
            } catch (error) {
                vscode.window.showErrorMessage(`Error fixing failed: ${error}`);
            }
        });
    }
}
