/**
 * ML Toolbox & AI Agent VS Code Extension
 * Integrates ML Toolbox and AI Agent into VS Code
 */

import * as vscode from 'vscode';
import { ToolboxProvider } from './toolboxProvider';
import { AIAgentProvider } from './aiAgentProvider';
import { PythonExecutor } from './pythonExecutor';

export function activate(context: vscode.ExtensionContext) {
    console.log('ML Toolbox & AI Agent extension is now active!');

    // Initialize providers
    const pythonExecutor = new PythonExecutor();
    const toolboxProvider = new ToolboxProvider(pythonExecutor);
    const aiAgentProvider = new AIAgentProvider(pythonExecutor);

    // Register commands
    const trainModelCommand = vscode.commands.registerCommand(
        'mlToolbox.trainModel',
        async () => {
            await toolboxProvider.trainModel();
        }
    );

    const aiGenerateCommand = vscode.commands.registerCommand(
        'mlToolbox.aiGenerate',
        async () => {
            await aiAgentProvider.generateCode();
        }
    );

    const aiFixCommand = vscode.commands.registerCommand(
        'mlToolbox.aiFix',
        async () => {
            await aiAgentProvider.fixError();
        }
    );

    const showPanelCommand = vscode.commands.registerCommand(
        'mlToolbox.showPanel',
        () => {
            toolboxProvider.showPanel();
        }
    );

    const quickTrainCommand = vscode.commands.registerCommand(
        'mlToolbox.quickTrain',
        async () => {
            await toolboxProvider.quickTrain();
        }
    );

    // Register tree data provider
    const treeDataProvider = new ToolboxTreeDataProvider();
    vscode.window.createTreeView('mlToolboxPanel', {
        treeDataProvider: treeDataProvider
    });

    // Add to context
    context.subscriptions.push(
        trainModelCommand,
        aiGenerateCommand,
        aiFixCommand,
        showPanelCommand,
        quickTrainCommand
    );

    // Show welcome message
    vscode.window.showInformationMessage('ML Toolbox & AI Agent extension activated!');
}

export function deactivate() {
    console.log('ML Toolbox & AI Agent extension is deactivated');
}

/**
 * Tree data provider for ML Toolbox panel
 */
class ToolboxTreeDataProvider implements vscode.TreeDataProvider<ToolboxItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<ToolboxItem | undefined> = 
        new vscode.EventEmitter<ToolboxItem | undefined>();
    readonly onDidChangeTreeData: vscode.Event<ToolboxItem | undefined> = 
        this._onDidChangeTreeData.event;

    getTreeItem(element: ToolboxItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: ToolboxItem): Thenable<ToolboxItem[]> {
        if (!element) {
            return Promise.resolve([
                new ToolboxItem('Train Model', vscode.TreeItemCollapsibleState.None, 'train'),
                new ToolboxItem('AI Generate', vscode.TreeItemCollapsibleState.None, 'ai'),
                new ToolboxItem('Models', vscode.TreeItemCollapsibleState.Collapsed, 'models'),
                new ToolboxItem('Data', vscode.TreeItemCollapsibleState.Collapsed, 'data')
            ]);
        }
        return Promise.resolve([]);
    }
}

class ToolboxItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly command?: string
    ) {
        super(label, collapsibleState);
        this.tooltip = this.label;
        this.command = command ? {
            command: `mlToolbox.${command}`,
            title: this.label
        } : undefined;
    }
}
