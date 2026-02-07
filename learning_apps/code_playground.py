"""
Interactive Code Playground for Learning Apps.
Provides safe Python code execution with sandboxing and real-time feedback.
"""
import sys
import io
import ast
import time
import traceback
import multiprocessing
from datetime import datetime
from typing import Dict, Any, List, Optional
from flask import request, jsonify
from contextlib import redirect_stdout, redirect_stderr

from learning_apps.database import get_db


# Execution limits
MAX_EXECUTION_TIME = 5  # seconds
MAX_OUTPUT_LENGTH = 10000  # characters
MAX_MEMORY_MB = 50

# Blocked imports and builtins for security
BLOCKED_IMPORTS = {
    'os', 'sys', 'subprocess', 'shutil', 'pathlib', 'importlib',
    'socket', 'urllib', 'requests', 'http', 'ftplib', 'smtplib',
    'pickle', 'shelve', 'marshal', 'ctypes', 'multiprocessing',
    'threading', 'asyncio', 'concurrent', 'signal', 'resource',
    '__builtins__', 'builtins', 'code', 'codeop', 'compile',
    'exec', 'eval', 'open', 'input', 'breakpoint'
}

BLOCKED_BUILTINS = {
    'open', 'exec', 'eval', 'compile', '__import__', 'input',
    'breakpoint', 'memoryview', 'globals', 'locals', 'vars',
    'dir', 'getattr', 'setattr', 'delattr', 'hasattr'
}

# Safe builtins for sandbox
SAFE_BUILTINS = {
    'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
    'bytearray': bytearray, 'bytes': bytes, 'callable': callable,
    'chr': chr, 'complex': complex, 'dict': dict, 'divmod': divmod,
    'enumerate': enumerate, 'filter': filter, 'float': float,
    'format': format, 'frozenset': frozenset, 'hash': hash, 'hex': hex,
    'int': int, 'isinstance': isinstance, 'issubclass': issubclass,
    'iter': iter, 'len': len, 'list': list, 'map': map, 'max': max,
    'min': min, 'next': next, 'oct': oct, 'ord': ord, 'pow': pow,
    'print': print, 'range': range, 'repr': repr, 'reversed': reversed,
    'round': round, 'set': set, 'slice': slice, 'sorted': sorted,
    'str': str, 'sum': sum, 'tuple': tuple, 'type': type, 'zip': zip,
    'True': True, 'False': False, 'None': None,
    'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
    'KeyError': KeyError, 'IndexError': IndexError, 'ZeroDivisionError': ZeroDivisionError,
}


class CodeValidator:
    """Validates code for safety before execution."""
    
    @staticmethod
    def validate(code: str) -> tuple:
        """
        Validate code for safety.
        Returns (is_valid, error_message).
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e.msg} (line {e.lineno})"
        
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = None
                if isinstance(node, ast.Import):
                    module = node.names[0].name.split('.')[0]
                elif node.module:
                    module = node.module.split('.')[0]
                
                if module and module in BLOCKED_IMPORTS:
                    return False, f"Import '{module}' is not allowed"
            
            # Check dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in BLOCKED_BUILTINS:
                        return False, f"Function '{node.func.id}' is not allowed"
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['system', 'popen', 'spawn', 'fork']:
                        return False, f"Method '{node.func.attr}' is not allowed"
            
            # Check attribute access to dangerous modules
            if isinstance(node, ast.Attribute):
                if node.attr in ['__class__', '__bases__', '__subclasses__', '__mro__']:
                    return False, f"Attribute '{node.attr}' access is not allowed"
        
        return True, None


def execute_code_safe(code: str, timeout: int = MAX_EXECUTION_TIME) -> Dict[str, Any]:
    """
    Execute code in a sandboxed environment.
    """
    # Validate first
    is_valid, error = CodeValidator.validate(code)
    if not is_valid:
        return {
            'ok': False,
            'output': '',
            'error': error,
            'execution_time_ms': 0
        }
    
    # Capture stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    # Create restricted globals
    restricted_globals = {
        '__builtins__': SAFE_BUILTINS,
        '__name__': '__main__',
    }
    
    # Add safe modules
    try:
        import math
        import random
        import json
        import re
        import collections
        import itertools
        import functools
        import statistics
        import decimal
        import fractions
        import datetime as dt
        
        restricted_globals.update({
            'math': math,
            'random': random,
            'json': json,
            're': re,
            'collections': collections,
            'itertools': itertools,
            'functools': functools,
            'statistics': statistics,
            'decimal': decimal,
            'fractions': fractions,
            'datetime': dt,
        })
        
        # Try numpy if available
        try:
            import numpy as np
            restricted_globals['np'] = np
            restricted_globals['numpy'] = np
        except ImportError:
            pass
        
    except Exception:
        pass
    
    start_time = time.time()
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Compile and execute
            compiled = compile(code, '<playground>', 'exec')
            exec(compiled, restricted_globals)
        
        execution_time = (time.time() - start_time) * 1000
        
        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()
        
        # Truncate if too long
        if len(output) > MAX_OUTPUT_LENGTH:
            output = output[:MAX_OUTPUT_LENGTH] + '\n... (output truncated)'
        
        return {
            'ok': True,
            'output': output,
            'errors': errors if errors else None,
            'execution_time_ms': round(execution_time, 2)
        }
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        
        # Format the error nicely
        error_lines = traceback.format_exception(type(e), e, e.__traceback__)
        # Remove internal frames
        error_msg = ''.join(error_lines[-2:]) if len(error_lines) > 2 else ''.join(error_lines)
        
        return {
            'ok': False,
            'output': stdout_capture.getvalue(),
            'error': error_msg.strip(),
            'execution_time_ms': round(execution_time, 2)
        }


# --- Code Challenges ---

CHALLENGES = {
    'fibonacci': {
        'id': 'fibonacci',
        'title': 'Fibonacci Sequence',
        'difficulty': 'easy',
        'description': 'Write a function `fibonacci(n)` that returns the nth Fibonacci number.',
        'starter_code': '''def fibonacci(n):
    """Return the nth Fibonacci number (0-indexed)."""
    # Your code here
    pass

# Test your function
print(fibonacci(0))  # Should print: 0
print(fibonacci(1))  # Should print: 1
print(fibonacci(10)) # Should print: 55
''',
        'test_cases': [
            {'input': 0, 'expected': 0},
            {'input': 1, 'expected': 1},
            {'input': 10, 'expected': 55},
            {'input': 20, 'expected': 6765},
        ],
        'hints': [
            'Think about the base cases: fib(0) = 0, fib(1) = 1',
            'Each Fibonacci number is the sum of the two before it',
            'You can use recursion or iteration'
        ]
    },
    'reverse_string': {
        'id': 'reverse_string',
        'title': 'Reverse a String',
        'difficulty': 'easy',
        'description': 'Write a function `reverse_string(s)` that returns the reversed string.',
        'starter_code': '''def reverse_string(s):
    """Return the reversed string."""
    # Your code here
    pass

# Test
print(reverse_string("hello"))  # Should print: olleh
print(reverse_string("Python")) # Should print: nohtyP
''',
        'test_cases': [
            {'input': 'hello', 'expected': 'olleh'},
            {'input': 'Python', 'expected': 'nohtyP'},
            {'input': '', 'expected': ''},
        ],
        'hints': ['Python has slicing with step: s[::step]', 'Try s[::-1]']
    },
    'is_palindrome': {
        'id': 'is_palindrome',
        'title': 'Palindrome Check',
        'difficulty': 'easy',
        'description': 'Write a function `is_palindrome(s)` that returns True if the string is a palindrome.',
        'starter_code': '''def is_palindrome(s):
    """Return True if s is a palindrome (ignoring case and spaces)."""
    # Your code here
    pass

# Test
print(is_palindrome("racecar"))  # True
print(is_palindrome("A man a plan a canal Panama"))  # True
print(is_palindrome("hello"))  # False
''',
        'test_cases': [
            {'input': 'racecar', 'expected': True},
            {'input': 'hello', 'expected': False},
        ],
        'hints': ['Remove spaces and convert to lowercase first', 'Compare string to its reverse']
    },
    'binary_search': {
        'id': 'binary_search',
        'title': 'Binary Search',
        'difficulty': 'medium',
        'description': 'Implement binary search that returns the index of target in a sorted list, or -1 if not found.',
        'starter_code': '''def binary_search(arr, target):
    """Find target in sorted array, return index or -1."""
    # Your code here
    pass

# Test
print(binary_search([1, 3, 5, 7, 9], 5))  # Should print: 2
print(binary_search([1, 3, 5, 7, 9], 4))  # Should print: -1
''',
        'test_cases': [
            {'input': [[1, 3, 5, 7, 9], 5], 'expected': 2},
            {'input': [[1, 3, 5, 7, 9], 4], 'expected': -1},
            {'input': [[1, 2, 3, 4, 5], 1], 'expected': 0},
        ],
        'hints': [
            'Use two pointers: left and right',
            'Calculate mid = (left + right) // 2',
            'Narrow the search range based on comparison'
        ]
    },
    'merge_sorted': {
        'id': 'merge_sorted',
        'title': 'Merge Sorted Arrays',
        'difficulty': 'medium',
        'description': 'Merge two sorted arrays into one sorted array.',
        'starter_code': '''def merge_sorted(arr1, arr2):
    """Merge two sorted arrays into one sorted array."""
    # Your code here
    pass

# Test
print(merge_sorted([1, 3, 5], [2, 4, 6]))  # [1, 2, 3, 4, 5, 6]
''',
        'test_cases': [
            {'input': [[1, 3, 5], [2, 4, 6]], 'expected': [1, 2, 3, 4, 5, 6]},
            {'input': [[], [1, 2, 3]], 'expected': [1, 2, 3]},
        ],
        'hints': ['Use two pointers, one for each array', 'Compare elements and add smaller one']
    },
}


def run_challenge(challenge_id: str, code: str, user_id: str = 'default') -> Dict[str, Any]:
    """Run code against challenge test cases."""
    if challenge_id not in CHALLENGES:
        return {'ok': False, 'error': 'Challenge not found'}
    
    challenge = CHALLENGES[challenge_id]
    
    # First execute the code to define functions
    result = execute_code_safe(code)
    
    if not result['ok']:
        return result
    
    # Now run test cases
    test_results = []
    all_passed = True
    
    for i, test in enumerate(challenge['test_cases']):
        # Build test code
        func_name = challenge_id.replace('_', '')
        if challenge_id == 'binary_search' or challenge_id == 'merge_sorted':
            test_code = code + f"\nresult = {challenge_id}(*{test['input']})"
        else:
            test_code = code + f"\nresult = {challenge_id}({repr(test['input'])})"
        test_code += "\nprint(repr(result))"
        
        test_result = execute_code_safe(test_code)
        
        if test_result['ok']:
            actual = test_result['output'].strip()
            expected = repr(test['expected'])
            passed = actual == expected
            
            test_results.append({
                'test': i + 1,
                'passed': passed,
                'expected': expected,
                'actual': actual
            })
            
            if not passed:
                all_passed = False
        else:
            test_results.append({
                'test': i + 1,
                'passed': False,
                'error': test_result.get('error', 'Execution failed')
            })
            all_passed = False
    
    # Save submission to database
    save_submission(user_id, 'playground', None, challenge_id, code, 
                   str(test_results), all_passed, result['execution_time_ms'])
    
    return {
        'ok': True,
        'all_passed': all_passed,
        'tests': test_results,
        'passed_count': sum(1 for t in test_results if t.get('passed')),
        'total_count': len(test_results)
    }


def save_submission(user_id: str, lab_id: str, topic_id: str, challenge_id: str,
                   code: str, output: str, passed: bool, execution_time: float):
    """Save a code submission to the database."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO code_submissions 
            (user_id, lab_id, topic_id, challenge_id, code, output, passed, execution_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, lab_id, topic_id, challenge_id, code, output, passed, execution_time))


def get_submissions(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent submissions for a user."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM code_submissions 
            WHERE user_id = ? 
            ORDER BY submitted_at DESC 
            LIMIT ?
        ''', (user_id, limit))
        return [dict(row) for row in cursor.fetchall()]


# --- Flask Routes ---

def register_playground_routes(app):
    """Register playground routes with a Flask app."""
    
    @app.route('/api/playground/run', methods=['POST'])
    def api_playground_run():
        """Run code in the playground."""
        data = request.get_json(silent=True) or {}
        code = data.get('code', '')
        
        if not code.strip():
            return jsonify({'ok': False, 'error': 'No code provided'})
        
        result = execute_code_safe(code)
        return jsonify(result)
    
    @app.route('/api/playground/challenges')
    def api_playground_challenges():
        """List available challenges."""
        challenges = []
        for cid, c in CHALLENGES.items():
            challenges.append({
                'id': c['id'],
                'title': c['title'],
                'difficulty': c['difficulty'],
                'description': c['description']
            })
        return jsonify({'ok': True, 'challenges': challenges})
    
    @app.route('/api/playground/challenge/<challenge_id>')
    def api_playground_challenge(challenge_id):
        """Get a specific challenge."""
        if challenge_id not in CHALLENGES:
            return jsonify({'ok': False, 'error': 'Challenge not found'}), 404
        return jsonify({'ok': True, 'challenge': CHALLENGES[challenge_id]})
    
    @app.route('/api/playground/submit', methods=['POST'])
    def api_playground_submit():
        """Submit code for a challenge."""
        data = request.get_json(silent=True) or {}
        challenge_id = data.get('challenge_id')
        code = data.get('code', '')
        user_id = data.get('user', 'default')
        
        if not challenge_id:
            return jsonify({'ok': False, 'error': 'challenge_id required'})
        
        result = run_challenge(challenge_id, code, user_id)
        return jsonify(result)
    
    @app.route('/api/playground/hint', methods=['POST'])
    def api_playground_hint():
        """Get a hint for a challenge."""
        data = request.get_json(silent=True) or {}
        challenge_id = data.get('challenge_id')
        hint_index = data.get('hint_index', 0)
        
        if challenge_id not in CHALLENGES:
            return jsonify({'ok': False, 'error': 'Challenge not found'})
        
        hints = CHALLENGES[challenge_id].get('hints', [])
        if hint_index >= len(hints):
            return jsonify({'ok': True, 'hint': 'No more hints available', 'has_more': False})
        
        return jsonify({
            'ok': True,
            'hint': hints[hint_index],
            'has_more': hint_index < len(hints) - 1
        })
    
    @app.route('/api/playground/submissions')
    def api_playground_submissions():
        """Get user's recent submissions."""
        user_id = request.args.get('user', 'default')
        limit = int(request.args.get('limit', 10))
        
        submissions = get_submissions(user_id, limit)
        return jsonify({'ok': True, 'submissions': submissions})


def get_playground_html_snippet() -> str:
    """HTML snippet for code playground UI."""
    return '''
    <style>
      .playground-btn {
        position: fixed; bottom: 140px; right: 24px; z-index: 100;
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white; border: none; padding: 12px 20px;
        border-radius: 24px; cursor: pointer; font-weight: 600;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.4);
        display: flex; align-items: center; gap: 8px;
      }
      .playground-btn:hover { transform: scale(1.05); }
      .playground-modal {
        display: none; position: fixed; inset: 0;
        background: rgba(0,0,0,0.9); z-index: 200;
        padding: 24px; overflow-y: auto;
      }
      .playground-modal.active { display: block; }
      .playground-container {
        max-width: 1000px; margin: 0 auto;
        background: var(--bg-secondary, #1e293b);
        border-radius: 16px; overflow: hidden;
      }
      .playground-header {
        display: flex; justify-content: space-between; align-items: center;
        padding: 16px 24px; border-bottom: 1px solid var(--border, #475569);
      }
      .playground-header h2 { margin: 0; }
      .playground-close {
        background: transparent; border: none; color: var(--text-primary);
        font-size: 1.5rem; cursor: pointer;
      }
      .playground-content { display: flex; gap: 0; }
      .playground-editor {
        flex: 1; padding: 16px; border-right: 1px solid var(--border, #475569);
      }
      .playground-editor textarea {
        width: 100%; height: 300px; 
        background: var(--bg-primary, #0f172a);
        border: 1px solid var(--border, #475569);
        border-radius: 8px; padding: 12px;
        font-family: 'Fira Code', Consolas, monospace;
        font-size: 14px; color: var(--text-primary);
        resize: vertical;
      }
      .playground-output {
        flex: 1; padding: 16px;
      }
      .playground-output pre {
        background: var(--bg-primary, #0f172a);
        border-radius: 8px; padding: 12px;
        height: 300px; overflow-y: auto;
        font-family: 'Fira Code', Consolas, monospace;
        font-size: 13px; margin: 0;
        white-space: pre-wrap;
      }
      .playground-actions {
        padding: 16px 24px; border-top: 1px solid var(--border);
        display: flex; gap: 12px; flex-wrap: wrap;
      }
      .playground-run {
        background: #22c55e; color: white; border: none;
        padding: 10px 24px; border-radius: 8px;
        cursor: pointer; font-weight: 600;
      }
      .playground-run:hover { background: #16a34a; }
      .challenge-select {
        background: var(--bg-tertiary); border: 1px solid var(--border);
        color: var(--text-primary); padding: 10px 16px;
        border-radius: 8px; cursor: pointer;
      }
      .test-results { margin-top: 16px; }
      .test-pass { color: #22c55e; }
      .test-fail { color: #ef4444; }
    </style>
    
    <button class="playground-btn" onclick="openPlayground()">
      💻 Code Playground
    </button>
    
    <div class="playground-modal" id="playground-modal">
      <div class="playground-container">
        <div class="playground-header">
          <h2>💻 Code Playground</h2>
          <button class="playground-close" onclick="closePlayground()">×</button>
        </div>
        <div style="padding: 12px 24px; background: var(--bg-tertiary);">
          <select id="challenge-select" class="challenge-select" onchange="loadChallenge()">
            <option value="">Free Coding</option>
          </select>
          <span id="challenge-desc" style="margin-left: 12px; color: var(--text-secondary);"></span>
        </div>
        <div class="playground-content">
          <div class="playground-editor">
            <label style="font-weight: 600; margin-bottom: 8px; display: block;">Code</label>
            <textarea id="playground-code" placeholder="# Write your Python code here..."></textarea>
          </div>
          <div class="playground-output">
            <label style="font-weight: 600; margin-bottom: 8px; display: block;">Output</label>
            <pre id="playground-output">Run code to see output...</pre>
            <div id="test-results" class="test-results"></div>
          </div>
        </div>
        <div class="playground-actions">
          <button class="playground-run" onclick="runPlayground()">▶ Run Code</button>
          <button class="challenge-select" onclick="submitChallenge()" id="submit-btn" style="display:none;">
            ✓ Submit Solution
          </button>
          <button class="challenge-select" onclick="getHint()" id="hint-btn" style="display:none;">
            💡 Get Hint
          </button>
        </div>
      </div>
    </div>
    
    <script>
      let currentChallenge = null;
      let hintIndex = 0;
      
      async function loadChallenges() {
        try {
          const resp = await fetch('/api/playground/challenges');
          const data = await resp.json();
          const select = document.getElementById('challenge-select');
          (data.challenges || []).forEach(c => {
            const opt = document.createElement('option');
            opt.value = c.id;
            opt.textContent = `[${c.difficulty}] ${c.title}`;
            select.appendChild(opt);
          });
        } catch (e) {}
      }
      
      async function loadChallenge() {
        const select = document.getElementById('challenge-select');
        const id = select.value;
        
        if (!id) {
          currentChallenge = null;
          document.getElementById('playground-code').value = '';
          document.getElementById('challenge-desc').textContent = '';
          document.getElementById('submit-btn').style.display = 'none';
          document.getElementById('hint-btn').style.display = 'none';
          return;
        }
        
        try {
          const resp = await fetch('/api/playground/challenge/' + id);
          const data = await resp.json();
          currentChallenge = data.challenge;
          hintIndex = 0;
          
          document.getElementById('playground-code').value = currentChallenge.starter_code;
          document.getElementById('challenge-desc').textContent = currentChallenge.description;
          document.getElementById('submit-btn').style.display = 'inline-block';
          document.getElementById('hint-btn').style.display = 'inline-block';
          document.getElementById('test-results').innerHTML = '';
        } catch (e) {}
      }
      
      function openPlayground() {
        document.getElementById('playground-modal').classList.add('active');
        loadChallenges();
      }
      
      function closePlayground() {
        document.getElementById('playground-modal').classList.remove('active');
      }
      
      async function runPlayground() {
        const code = document.getElementById('playground-code').value;
        const output = document.getElementById('playground-output');
        output.textContent = 'Running...';
        
        try {
          const resp = await fetch('/api/playground/run', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({code})
          });
          const data = await resp.json();
          
          if (data.ok) {
            output.textContent = data.output || '(no output)';
            output.style.color = 'var(--text-primary)';
          } else {
            output.textContent = data.error || 'Error';
            output.style.color = '#ef4444';
          }
          
          if (data.execution_time_ms) {
            output.textContent += '\\n\\n⏱ ' + data.execution_time_ms + 'ms';
          }
        } catch (e) {
          output.textContent = 'Error: ' + e.message;
          output.style.color = '#ef4444';
        }
      }
      
      async function submitChallenge() {
        if (!currentChallenge) return;
        
        const code = document.getElementById('playground-code').value;
        const results = document.getElementById('test-results');
        results.innerHTML = 'Testing...';
        
        try {
          const resp = await fetch('/api/playground/submit', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              challenge_id: currentChallenge.id,
              code: code
            })
          });
          const data = await resp.json();
          
          let html = `<p><strong>${data.passed_count}/${data.total_count} tests passed</strong></p>`;
          (data.tests || []).forEach(t => {
            const icon = t.passed ? '✅' : '❌';
            const cls = t.passed ? 'test-pass' : 'test-fail';
            html += `<p class="${cls}">${icon} Test ${t.test}: `;
            if (t.passed) {
              html += 'Passed';
            } else if (t.error) {
              html += t.error;
            } else {
              html += `Expected ${t.expected}, got ${t.actual}`;
            }
            html += '</p>';
          });
          
          if (data.all_passed) {
            html = '🎉 <strong style="color:#22c55e;">All tests passed!</strong>' + html;
          }
          
          results.innerHTML = html;
        } catch (e) {
          results.innerHTML = 'Error: ' + e.message;
        }
      }
      
      async function getHint() {
        if (!currentChallenge) return;
        
        try {
          const resp = await fetch('/api/playground/hint', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              challenge_id: currentChallenge.id,
              hint_index: hintIndex
            })
          });
          const data = await resp.json();
          
          alert('💡 Hint: ' + data.hint);
          if (data.has_more) hintIndex++;
        } catch (e) {}
      }
    </script>
    '''
