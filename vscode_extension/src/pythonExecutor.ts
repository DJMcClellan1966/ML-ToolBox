/**
 * Python Executor - Executes Python code and communicates with ML Toolbox
 */

import * as vscode from 'vscode';
import { PythonShell } from 'python-shell';

export class PythonExecutor {
    private pythonPath: string;

    constructor() {
        // Get Python path from VS Code settings
        const config = vscode.workspace.getConfiguration('python');
        this.pythonPath = config.get<string>('pythonPath', 'python') || 'python';
    }

    async executeAI(prompt: string): Promise<string> {
        return new Promise((resolve, reject) => {
            const script = `
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_toolbox.ai_agent import MLCodeAgent

agent = MLCodeAgent(use_llm=False, use_pattern_composition=True)
result = agent.build("${prompt.replace(/"/g, '\\"')}")

if result.get('success'):
    print(result['code'])
else:
    print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
    sys.exit(1)
`;

            const options = {
                mode: 'text' as const,
                pythonPath: this.pythonPath,
                pythonOptions: ['-u'],
                scriptPath: __dirname
            };

            PythonShell.runString(script, options, (err, results) => {
                if (err) {
                    reject(err);
                    return;
                }

                if (results && results.length > 0) {
                    resolve(results.join('\n'));
                } else {
                    reject('No code generated');
                }
            });
        });
    }

    async fixCode(code: string): Promise<string> {
        return new Promise((resolve, reject) => {
            const script = `
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_toolbox.ai_agent import MLCodeAgent

agent = MLCodeAgent(use_llm=False, use_pattern_composition=True)

# Try to fix the code
code = """${code.replace(/"""/g, '\\"\\"\\"')}"""

result = agent.build("Fix this code: " + code)

if result.get('success'):
    print(result['code'])
else:
    print(code)  # Return original if fix fails
`;

            const options = {
                mode: 'text' as const,
                pythonPath: this.pythonPath,
                pythonOptions: ['-u'],
                scriptPath: __dirname
            };

            PythonShell.runString(script, options, (err, results) => {
                if (err) {
                    reject(err);
                    return;
                }

                if (results && results.length > 0) {
                    resolve(results.join('\n'));
                } else {
                    resolve(code);  // Return original if fix fails
                }
            });
        });
    }

    async executeToolbox(command: string, args: any[]): Promise<any> {
        // Execute ML Toolbox commands
        return new Promise((resolve, reject) => {
            const script = `
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_toolbox import MLToolbox
import json

toolbox = MLToolbox()
# Execute command
result = ${command}
print(json.dumps(result))
`;

            const options = {
                mode: 'text' as const,
                pythonPath: this.pythonPath,
                pythonOptions: ['-u'],
                scriptPath: __dirname
            };

            PythonShell.runString(script, options, (err, results) => {
                if (err) {
                    reject(err);
                    return;
                }

                try {
                    const result = JSON.parse(results.join('\n'));
                    resolve(result);
                } catch (e) {
                    reject('Failed to parse result');
                }
            });
        });
    }
}
