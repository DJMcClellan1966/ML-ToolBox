"""
ML Learning App - Desktop Version
Run this to start the learning app on your Windows 11 laptop
"""
import sys
from pathlib import Path
import webbrowser
import threading
import time

sys.path.insert(0, str(Path(__file__).parent))

try:
    from flask import Flask, render_template_string, request, jsonify, send_from_directory
    from ml_toolbox import MLToolbox
    from ml_toolbox.ai_agent import MLCodeAgent, ProactiveAgent
    from ml_toolbox.revolutionary_features import get_third_eye, get_self_healing_code
    from ml_toolbox.ui import create_wellness_dashboard, MetricCard
    from ml_toolbox.infrastructure import get_performance_monitor
    FLASK_AVAILABLE = True
except ImportError as e:
    FLASK_AVAILABLE = False
    print(f"Warning: {e}")
    print("Installing required packages...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "numpy"])
        print("Please restart the app after installation.")
    except:
        pass

if FLASK_AVAILABLE:
    app = Flask(__name__)
    
    # Initialize ML Toolbox
    try:
        toolbox = MLToolbox(check_dependencies=False)
        ai_agent = MLCodeAgent(use_llm=False, use_pattern_composition=True)
        proactive_agent = ProactiveAgent(enable_proactive=True)
        third_eye = get_third_eye()
        healing = get_self_healing_code()
        monitor = get_performance_monitor()
        TOOLBOX_AVAILABLE = True
    except Exception as e:
        TOOLBOX_AVAILABLE = False
        print(f"ML Toolbox initialization warning: {e}")
    
    # Student data (in-memory for demo)
    students = {}
    current_student = None
    
    # Course modules
    modules = [
        {
            'id': 1,
            'title': 'Introduction to Machine Learning',
            'lessons': [
                {'id': 1, 'title': 'What is Machine Learning?', 'content': 'ML is a subset of AI that enables computers to learn from data.'},
                {'id': 2, 'title': 'Types of ML', 'content': 'Supervised, Unsupervised, and Reinforcement Learning.'},
                {'id': 3, 'title': 'ML Workflow', 'content': 'Data → Preprocessing → Training → Evaluation → Deployment.'},
                {'id': 4, 'title': 'Your First Model', 'content': 'Let\'s train a simple classification model!'}
            ]
        },
        {
            'id': 2,
            'title': 'Data Preprocessing',
            'lessons': [
                {'id': 1, 'title': 'Data Cleaning', 'content': 'Remove missing values, outliers, and errors.'},
                {'id': 2, 'title': 'Feature Engineering', 'content': 'Create new features from existing data.'},
                {'id': 3, 'title': 'Data Transformation', 'content': 'Normalize, standardize, and scale data.'},
                {'id': 4, 'title': 'Handling Missing Values', 'content': 'Strategies: mean, median, mode, or drop.'}
            ]
        },
        {
            'id': 3,
            'title': 'Supervised Learning',
            'lessons': [
                {'id': 1, 'title': 'Classification', 'content': 'Predicting categories (e.g., spam/not spam).'},
                {'id': 2, 'title': 'Regression', 'content': 'Predicting continuous values (e.g., price).'},
                {'id': 3, 'title': 'Model Evaluation', 'content': 'Accuracy, precision, recall, F1-score.'},
                {'id': 4, 'title': 'Hyperparameter Tuning', 'content': 'Finding the best model parameters.'}
            ]
        }
    ]
    
    # Exercises
    exercises = [
        {
            'id': 1,
            'title': 'Train Your First Model',
            'module': 1,
            'instructions': 'Complete the code to train a classification model',
            'starter_code': '''from ml_toolbox import MLToolbox
import numpy as np

toolbox = MLToolbox()

# Create sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 1, 0, 1])

# TODO: Train a model
# result = toolbox.fit(???, ???, task_type='???')

# TODO: Make a prediction
# prediction = toolbox.predict(result['model'], [[5, 6]])
# print(f"Prediction: {prediction}")''',
            'solution': '''from ml_toolbox import MLToolbox
import numpy as np

toolbox = MLToolbox()

# Create sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 1, 0, 1])

# Train a model
result = toolbox.fit(X, y, task_type='classification')

# Make a prediction
prediction = toolbox.predict(result['model'], [[5, 6]])
print(f"Prediction: {prediction}")'''
        },
        {
            'id': 2,
            'title': 'Clean Your Data',
            'module': 2,
            'instructions': 'Use data cleaning utilities to clean missing values',
            'starter_code': '''from ml_toolbox.compartment1_data.data_cleaning_utilities import DataCleaningUtilities
import numpy as np

# Create data with missing values
data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])

# TODO: Clean the data
# cleaner = ???
# cleaned = cleaner.clean_missing_values(???, strategy='???')
# print(f"Cleaned data: {cleaned}")''',
            'solution': '''from ml_toolbox.compartment1_data.data_cleaning_utilities import DataCleaningUtilities
import numpy as np

# Create data with missing values
data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])

# Clean the data
cleaner = DataCleaningUtilities()
cleaned = cleaner.clean_missing_values(data, strategy='mean')
print(f"Cleaned data: {cleaned}")'''
        }
    ]
    
    # HTML Template
    HTML_TEMPLATE = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Learning App</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 { font-size: 2.5em; margin-bottom: 10px; }
            .header p { font-size: 1.2em; opacity: 0.9; }
            .nav {
                background: #f8f9fa;
                padding: 20px;
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }
            .nav button {
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                background: #667eea;
                color: white;
                cursor: pointer;
                font-size: 16px;
                transition: all 0.3s;
            }
            .nav button:hover { background: #5568d3; transform: translateY(-2px); }
            .nav button.active { background: #764ba2; }
            .content {
                padding: 30px;
            }
            .module-card {
                background: #f8f9fa;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                border-left: 4px solid #667eea;
                transition: all 0.3s;
            }
            .module-card:hover {
                transform: translateX(5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .module-card h3 {
                color: #667eea;
                margin-bottom: 10px;
                font-size: 1.5em;
            }
            .lesson-list {
                list-style: none;
                margin-top: 15px;
            }
            .lesson-list li {
                padding: 10px;
                margin: 5px 0;
                background: white;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s;
            }
            .lesson-list li:hover {
                background: #e9ecef;
                transform: translateX(5px);
            }
            .exercise-container {
                background: #f8f9fa;
                border-radius: 12px;
                padding: 20px;
                margin: 20px 0;
            }
            .code-editor {
                width: 100%;
                min-height: 300px;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                padding: 15px;
                border: 2px solid #667eea;
                border-radius: 8px;
                background: #1e1e1e;
                color: #d4d4d4;
                resize: vertical;
            }
            .button-group {
                display: flex;
                gap: 10px;
                margin: 20px 0;
            }
            .btn {
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                transition: all 0.3s;
            }
            .btn-primary {
                background: #667eea;
                color: white;
        }
            .btn-primary:hover { background: #5568d3; transform: translateY(-2px); }
            .btn-success {
                background: #28a745;
                color: white;
            }
            .btn-success:hover { background: #218838; }
            .btn-info {
                background: #17a2b8;
                color: white;
            }
            .btn-info:hover { background: #138496; }
            .result-box {
                margin-top: 20px;
                padding: 15px;
                border-radius: 8px;
                display: none;
            }
            .result-box.success {
                background: #d4edda;
                border: 2px solid #28a745;
                color: #155724;
            }
            .result-box.error {
                background: #f8d7da;
                border: 2px solid #dc3545;
                color: #721c24;
            }
            .result-box.info {
                background: #d1ecf1;
                border: 2px solid #17a2b8;
                color: #0c5460;
            }
            .progress-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 12px;
                margin: 20px 0;
            }
            .progress-card h3 { margin-bottom: 15px; }
            .progress-bar {
                background: rgba(255,255,255,0.3);
                border-radius: 10px;
                height: 30px;
                margin: 10px 0;
                overflow: hidden;
            }
            .progress-fill {
                background: white;
                height: 100%;
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #667eea;
                font-weight: bold;
                transition: width 0.5s;
            }
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .metric-card {
                background: white;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
            }
            .metric-card h4 {
                color: #667eea;
                margin-bottom: 10px;
            }
            .metric-card .value {
                font-size: 2em;
                font-weight: bold;
                color: #764ba2;
            }
            .hidden { display: none; }
            .quiz-container {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 12px;
                margin: 20px 0;
            }
            .quiz-question {
                background: white;
                padding: 20px;
                border-radius: 8px;
                margin: 15px 0;
            }
            .quiz-option {
                padding: 10px;
                margin: 5px 0;
                background: #e9ecef;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s;
            }
            .quiz-option:hover {
                background: #667eea;
                color: white;
            }
            .quiz-option.selected {
                background: #764ba2;
                color: white;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 ML Learning App</h1>
                <p>Learn Machine Learning from the Ground Up</p>
            </div>
            
            <div class="nav">
                <button onclick="showSection('modules')" class="active">📚 Modules</button>
                <button onclick="showSection('exercises')">💻 Exercises</button>
                <button onclick="showSection('quiz')">📝 Quiz</button>
                <button onclick="showSection('progress')">📊 Progress</button>
                <button onclick="showSection('ai-tutor')">🤖 AI Tutor</button>
            </div>
            
            <div class="content">
                <!-- Modules Section -->
                <div id="modules-section">
                    <h2>Course Modules</h2>
                    <div id="modules-list"></div>
                </div>
                
                <!-- Exercises Section -->
                <div id="exercises-section" class="hidden">
                    <h2>Interactive Exercises</h2>
                    <div id="exercises-list"></div>
                </div>
                
                <!-- Quiz Section -->
                <div id="quiz-section" class="hidden">
                    <h2>Take a Quiz</h2>
                    <div id="quiz-container"></div>
                </div>
                
                <!-- Progress Section -->
                <div id="progress-section" class="hidden">
                    <h2>Your Progress</h2>
                    <div id="progress-content"></div>
                </div>
                
                <!-- AI Tutor Section -->
                <div id="ai-tutor-section" class="hidden">
                    <h2>AI Tutor</h2>
                    <div class="exercise-container">
                        <h3>Ask a Question</h3>
                        <textarea id="tutor-question" style="width: 100%; min-height: 100px; padding: 15px; border-radius: 8px; border: 2px solid #667eea; font-size: 16px;" placeholder="Ask me anything about machine learning..."></textarea>
                        <button class="btn btn-primary" onclick="askTutor()" style="margin-top: 15px;">Ask AI Tutor</button>
                        <div id="tutor-response" class="result-box"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let currentStudent = null;
            let currentExercise = null;
            let quizAnswers = {};
            
            // Initialize
            window.onload = function() {
                loadModules();
                loadExercises();
                loadProgress();
                loadQuiz();
            };
            
            function showSection(section) {
                // Hide all sections
                document.querySelectorAll('[id$="-section"]').forEach(el => el.classList.add('hidden'));
                document.querySelectorAll('.nav button').forEach(btn => btn.classList.remove('active'));
                
                // Show selected section
                document.getElementById(section + '-section').classList.remove('hidden');
                event.target.classList.add('active');
            }
            
            function loadModules() {
                fetch('/api/modules')
                    .then(r => r.json())
                    .then(data => {
                        const container = document.getElementById('modules-list');
                        container.innerHTML = data.modules.map(module => `
                            <div class="module-card">
                                <h3>${module.title}</h3>
                                <ul class="lesson-list">
                                    ${module.lessons.map(lesson => `
                                        <li onclick="showLesson(${module.id}, ${lesson.id})">
                                            📖 ${lesson.title}
                                        </li>
                                    `).join('')}
                                </ul>
                            </div>
                        `).join('');
                    });
            }
            
            function showLesson(moduleId, lessonId) {
                fetch(`/api/lesson/${moduleId}/${lessonId}`)
                    .then(r => r.json())
                    .then(data => {
                        alert(`Lesson: ${data.lesson.title}\\n\\n${data.lesson.content}`);
                    });
            }
            
            function loadExercises() {
                fetch('/api/exercises')
                    .then(r => r.json())
                    .then(data => {
                        const container = document.getElementById('exercises-list');
                        container.innerHTML = data.exercises.map(ex => `
                            <div class="exercise-container">
                                <h3>${ex.title}</h3>
                                <p>${ex.instructions}</p>
                                <textarea class="code-editor" id="code-${ex.id}">${ex.starter_code}</textarea>
                                <div class="button-group">
                                    <button class="btn btn-primary" onclick="runExercise(${ex.id})">▶ Run Code</button>
                                    <button class="btn btn-success" onclick="submitExercise(${ex.id})">✓ Submit</button>
                                    <button class="btn btn-info" onclick="showHint(${ex.id})">💡 Hint</button>
                                    <button class="btn btn-info" onclick="showSolution(${ex.id})">🔍 Solution</button>
                                </div>
                                <div id="result-${ex.id}" class="result-box"></div>
                            </div>
                        `).join('');
                    });
            }
            
            function runExercise(exId) {
                const code = document.getElementById(`code-${exId}`).value;
                fetch('/api/exercise/run', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({exercise_id: exId, code: code})
                })
                .then(r => r.json())
                .then(data => {
                    const resultBox = document.getElementById(`result-${exId}`);
                    resultBox.className = 'result-box ' + (data.success ? 'success' : 'error');
                    resultBox.style.display = 'block';
                    resultBox.innerHTML = `<strong>${data.success ? '✓ Success!' : '✗ Error'}</strong><br>${data.message}`;
                });
            }
            
            function submitExercise(exId) {
                const code = document.getElementById(`code-${exId}`).value;
                fetch('/api/exercise/submit', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({exercise_id: exId, code: code})
                })
                .then(r => r.json())
                .then(data => {
                    const resultBox = document.getElementById(`result-${exId}`);
                    resultBox.className = 'result-box ' + (data.correct ? 'success' : 'info');
                    resultBox.style.display = 'block';
                    resultBox.innerHTML = `<strong>${data.correct ? '🎉 Correct! Great job!' : 'Keep trying!'}</strong><br>${data.feedback || ''}`;
                    if (data.correct) {
                        loadProgress();
                    }
                });
            }
            
            function showHint(exId) {
                fetch(`/api/exercise/${exId}/hint`)
                    .then(r => r.json())
                    .then(data => {
                        alert(`Hint: ${data.hint}`);
                    });
            }
            
            function showSolution(exId) {
                fetch(`/api/exercise/${exId}/solution`)
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById(`code-${exId}`).value = data.solution;
                    });
            }
            
            function loadProgress() {
                fetch('/api/progress')
                    .then(r => r.json())
                    .then(data => {
                        const container = document.getElementById('progress-content');
                        container.innerHTML = `
                            <div class="progress-card">
                                <h3>Your Learning Progress</h3>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: ${data.progress}%">${data.progress}%</div>
                                </div>
                            </div>
                            <div class="metric-grid">
                                <div class="metric-card">
                                    <h4>Modules Completed</h4>
                                    <div class="value">${data.modules_completed}</div>
                                </div>
                                <div class="metric-card">
                                    <h4>Exercises Solved</h4>
                                    <div class="value">${data.exercises_solved}</div>
                                </div>
                                <div class="metric-card">
                                    <h4>Average Score</h4>
                                    <div class="value">${data.average_score}%</div>
                                </div>
                                <div class="metric-card">
                                    <h4>Time Spent</h4>
                                    <div class="value">${data.time_spent}h</div>
                                </div>
                            </div>
                        `;
                    });
            }
            
            function loadQuiz() {
                fetch('/api/quiz')
                    .then(r => r.json())
                    .then(data => {
                        const container = document.getElementById('quiz-container');
                        container.innerHTML = `
                            <div class="quiz-container">
                                <h3>${data.topic} Quiz</h3>
                                ${data.questions.map((q, idx) => `
                                    <div class="quiz-question">
                                        <h4>${idx + 1}. ${q.question}</h4>
                                        ${q.options.map((opt, optIdx) => `
                                            <div class="quiz-option" onclick="selectAnswer(${idx}, ${optIdx})" id="opt-${idx}-${optIdx}">
                                                ${opt}
                                            </div>
                                        `).join('')}
                                    </div>
                                `).join('')}
                                <button class="btn btn-primary" onclick="submitQuiz()" style="margin-top: 20px;">Submit Quiz</button>
                            </div>
                        `;
                    });
            }
            
            function selectAnswer(qIdx, optIdx) {
                quizAnswers[qIdx] = optIdx;
                document.querySelectorAll(`[id^="opt-${qIdx}-"]`).forEach(el => el.classList.remove('selected'));
                document.getElementById(`opt-${qIdx}-${optIdx}`).classList.add('selected');
            }
            
            function submitQuiz() {
                fetch('/api/quiz/submit', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({answers: quizAnswers})
                })
                .then(r => r.json())
                .then(data => {
                    alert(`Quiz Score: ${data.score}%\\n\\n${data.feedback}`);
                    loadProgress();
                });
            }
            
            function askTutor() {
                const question = document.getElementById('tutor-question').value;
                if (!question.trim()) {
                    alert('Please enter a question');
                    return;
                }
                
                fetch('/api/tutor/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question})
                })
                .then(r => r.json())
                .then(data => {
                    const responseBox = document.getElementById('tutor-response');
                    responseBox.className = 'result-box info';
                    responseBox.style.display = 'block';
                    responseBox.innerHTML = `<strong>AI Tutor Response:</strong><br><br>${data.answer}`;
                });
            }
        </script>
    </body>
    </html>
    '''
    
    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)
    
    @app.route('/api/modules')
    def get_modules():
        return jsonify({'modules': modules})
    
    @app.route('/api/lesson/<int:module_id>/<int:lesson_id>')
    def get_lesson(module_id, lesson_id):
        module = modules[module_id - 1]
        lesson = module['lessons'][lesson_id - 1]
        return jsonify({
            'lesson': lesson,
            'module': module['title']
        })
    
    @app.route('/api/exercises')
    def get_exercises():
        return jsonify({'exercises': exercises})
    
    @app.route('/api/exercise/run', methods=['POST'])
    def run_exercise():
        data = request.json
        code = data.get('code', '')
        exercise_id = data.get('exercise_id', 0)
        
        try:
            # Create a safe execution environment
            exec_globals = {'__builtins__': __builtins__}
            if TOOLBOX_AVAILABLE:
                exec_globals['MLToolbox'] = MLToolbox
                exec_globals['toolbox'] = toolbox
                import numpy as np
                exec_globals['np'] = np
                exec_globals['numpy'] = np
            
            exec(code, exec_globals)
            
            return jsonify({
                'success': True,
                'message': 'Code executed successfully!'
            })
        except Exception as e:
            # Try to fix with self-healing
            if TOOLBOX_AVAILABLE and healing:
                try:
                    fixed_code = healing.fix_code(code, {'error': str(e)})
                    return jsonify({
                        'success': False,
                        'message': f'Error: {str(e)}',
                        'suggestion': 'Try this fixed code:',
                        'fixed_code': fixed_code if fixed_code else None
                    })
                except:
                    pass
            
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}'
            })
    
    @app.route('/api/exercise/submit', methods=['POST'])
    def submit_exercise():
        data = request.json
        code = data.get('code', '')
        exercise_id = data.get('exercise_id', 0)
        
        if exercise_id < len(exercises):
            exercise = exercises[exercise_id]
            solution = exercise.get('solution', '')
            
            # Simple validation (check if key parts are present)
            is_correct = (
                'toolbox.fit' in code or 'MLToolbox' in code or 'fit(' in code
            ) and (
                'predict' in code or 'toolbox.predict' in code
            )
            
            feedback = []
            if is_correct:
                feedback.append("Great job! You've successfully trained and used a model!")
            else:
                feedback.append("Keep trying! Make sure you're using toolbox.fit() and toolbox.predict()")
            
            return jsonify({
                'correct': is_correct,
                'feedback': ' '.join(feedback),
                'score': 100 if is_correct else 0
            })
        
        return jsonify({'correct': False, 'feedback': 'Exercise not found'})
    
    @app.route('/api/exercise/<int:ex_id>/hint')
    def get_hint(ex_id):
        hints = [
            "Try using toolbox.fit() to train your model",
            "Don't forget to specify task_type='classification'",
            "Use toolbox.predict() to make predictions",
            "Check the ML Toolbox documentation for examples"
        ]
        return jsonify({'hint': hints[ex_id % len(hints)]})
    
    @app.route('/api/exercise/<int:ex_id>/solution')
    def get_solution(ex_id):
        if ex_id < len(exercises):
            return jsonify({'solution': exercises[ex_id].get('solution', '')})
        return jsonify({'solution': ''})
    
    @app.route('/api/quiz')
    def get_quiz():
        questions = [
            {
                'question': 'What is Machine Learning?',
                'options': [
                    'A type of programming language',
                    'A subset of AI that enables computers to learn from data',
                    'A database system',
                    'A web framework'
                ],
                'correct': 1
            },
            {
                'question': 'What are the main types of ML?',
                'options': [
                    'Supervised, Unsupervised, Reinforcement',
                    'Python, Java, C++',
                    'Classification, Regression, Clustering',
                    'Training, Testing, Validation'
                ],
                'correct': 0
            },
            {
                'question': 'What does toolbox.fit() do?',
                'options': [
                    'Fits data into a database',
                    'Trains a machine learning model',
                    'Fixes code errors',
                    'Creates a dashboard'
                ],
                'correct': 1
            }
        ]
        return jsonify({
            'topic': 'Machine Learning Basics',
            'questions': questions
        })
    
    @app.route('/api/quiz/submit', methods=['POST'])
    def submit_quiz():
        data = request.json
        answers = data.get('answers', {})
        
        # Simple grading (in real app, would check against correct answers)
        score = min(100, len(answers) * 33)  # Simplified
        
        feedback = "Great job!" if score >= 70 else "Keep studying!"
        
        return jsonify({
            'score': score,
            'feedback': feedback
        })
    
    @app.route('/api/progress')
    def get_progress():
        # Demo progress data
        return jsonify({
            'progress': 35,
            'modules_completed': 1,
            'exercises_solved': 2,
            'average_score': 85,
            'time_spent': 5
        })
    
    @app.route('/api/tutor/ask', methods=['POST'])
    def ask_tutor():
        data = request.json
        question = data.get('question', '')
        
        if TOOLBOX_AVAILABLE and ai_agent:
            try:
                result = ai_agent.build(f"Explain this simply: {question}")
                answer = result.get('code', f"Great question! {question} is an important concept in machine learning. Let me explain...")
            except:
                answer = f"Great question about '{question}'! In machine learning, this concept is important because..."
        else:
            answer = f"Great question! '{question}' is a fundamental concept in machine learning. Here's a simple explanation..."
        
        return jsonify({'answer': answer})
    
    def open_browser():
        """Open browser after server starts"""
        time.sleep(1.5)
        webbrowser.open('http://127.0.0.1:5000')
    
    if __name__ == '__main__':
        print("="*60)
        print("🚀 ML Learning App Starting...")
        print("="*60)
        print("\nThe app will open in your browser automatically.")
        print("If it doesn't, go to: http://127.0.0.1:5000")
        print("\nPress Ctrl+C to stop the server.")
        print("="*60)
        
        # Open browser in background thread
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Run Flask app
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

else:
    print("="*60)
    print("ERROR: Flask is not installed")
    print("="*60)
    print("\nPlease install Flask:")
    print("  pip install flask")
    print("\nThen run this script again.")
    print("="*60)
