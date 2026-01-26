"""
AI Learning Companion - Web UI

Simple web interface using Flask for better UX.
Run with: python ai_learning_companion_web.py
Then open: http://localhost:5000
"""

try:
    from flask import Flask, render_template_string, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask")

from ai_learning_companion import LearningCompanion
import json


if FLASK_AVAILABLE:
    app = Flask(__name__)
    companion = LearningCompanion()
    
    # HTML Template
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Learning Companion</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 900px;
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
            .header p { opacity: 0.9; }
            .content {
                padding: 30px;
            }
            .tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
                border-bottom: 2px solid #eee;
            }
            .tab {
                padding: 15px 25px;
                cursor: pointer;
                border: none;
                background: none;
                font-size: 16px;
                color: #666;
                border-bottom: 3px solid transparent;
                transition: all 0.3s;
            }
            .tab:hover { color: #667eea; }
            .tab.active {
                color: #667eea;
                border-bottom-color: #667eea;
                font-weight: bold;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            .input-group {
                margin-bottom: 20px;
            }
            .input-group label {
                display: block;
                margin-bottom: 8px;
                color: #333;
                font-weight: 600;
            }
            .input-group input,
            .input-group textarea,
            .input-group select {
                width: 100%;
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s;
            }
            .input-group input:focus,
            .input-group textarea:focus,
            .input-group select:focus {
                outline: none;
                border-color: #667eea;
            }
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                transition: transform 0.2s;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            .result {
                margin-top: 20px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                border-left: 4px solid #667eea;
            }
            .result h3 {
                color: #667eea;
                margin-bottom: 15px;
            }
            .result-section {
                margin-bottom: 15px;
            }
            .result-section h4 {
                color: #333;
                margin-bottom: 8px;
            }
            .result-section ul {
                margin-left: 20px;
            }
            .result-section li {
                margin-bottom: 5px;
            }
            .concept-list {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 10px;
                margin-top: 10px;
            }
            .concept-item {
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s;
                border: 2px solid transparent;
            }
            .concept-item:hover {
                border-color: #667eea;
                transform: translateY(-2px);
            }
            .progress-bar {
                width: 100%;
                height: 30px;
                background: #eee;
                border-radius: 15px;
                overflow: hidden;
                margin: 10px 0;
            }
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                transition: width 0.3s;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 AI Learning Companion</h1>
                <p>Your Personal ML/AI Tutor</p>
            </div>
            <div class="content">
                <div class="tabs">
                    <button class="tab active" onclick="showTab('learn')">Learn</button>
                    <button class="tab" onclick="showTab('ask')">Ask</button>
                    <button class="tab" onclick="showTab('path')">Path</button>
                    <button class="tab" onclick="showTab('progress')">Progress</button>
                    <button class="tab" onclick="showTab('browse')">Browse</button>
                </div>
                
                <!-- Learn Tab -->
                <div id="learn" class="tab-content active">
                    <div class="input-group">
                        <label>Concept to Learn</label>
                        <input type="text" id="concept-input" placeholder="e.g., classification, neural_networks">
                    </div>
                    <button class="btn" onclick="learnConcept()">Learn Concept</button>
                    <div id="learn-result"></div>
                </div>
                
                <!-- Ask Tab -->
                <div id="ask" class="tab-content">
                    <div class="input-group">
                        <label>Your Question</label>
                        <textarea id="question-input" rows="3" placeholder="e.g., What is machine learning?"></textarea>
                    </div>
                    <button class="btn" onclick="askQuestion()">Ask Question</button>
                    <div id="ask-result"></div>
                </div>
                
                <!-- Path Tab -->
                <div id="path" class="tab-content">
                    <div class="input-group">
                        <label>Learning Path</label>
                        <select id="path-input">
                            <option value="ml_fundamentals">ML Fundamentals</option>
                            <option value="deep_learning">Deep Learning</option>
                            <option value="practical_ml">Practical ML</option>
                        </select>
                    </div>
                    <button class="btn" onclick="getPath()">Get Learning Path</button>
                    <div id="path-result"></div>
                </div>
                
                <!-- Progress Tab -->
                <div id="progress" class="tab-content">
                    <button class="btn" onclick="checkProgress()">Check Progress</button>
                    <div id="progress-result"></div>
                </div>
                
                <!-- Browse Tab -->
                <div id="browse" class="tab-content">
                    <h3>Available Concepts</h3>
                    <div class="concept-list" id="concept-list"></div>
                </div>
            </div>
        </div>
        
        <script>
            function showTab(tabName) {
                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(tab => {
                    tab.classList.remove('active');
                });
                document.querySelectorAll('.tab').forEach(btn => {
                    btn.classList.remove('active');
                });
                
                // Show selected tab
                document.getElementById(tabName).classList.add('active');
                event.target.classList.add('active');
                
                // Load browse concepts
                if (tabName === 'browse') {
                    loadConcepts();
                }
            }
            
            function learnConcept() {
                const concept = document.getElementById('concept-input').value;
                if (!concept) {
                    alert('Please enter a concept name');
                    return;
                }
                
                fetch('/api/learn', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({concept: concept})
                })
                .then(r => r.json())
                .then(data => {
                    displayLearnResult(data);
                });
            }
            
            function askQuestion() {
                const question = document.getElementById('question-input').value;
                if (!question) {
                    alert('Please enter a question');
                    return;
                }
                
                fetch('/api/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question})
                })
                .then(r => r.json())
                .then(data => {
                    displayAskResult(data);
                });
            }
            
            function getPath() {
                const path = document.getElementById('path-input').value;
                
                fetch('/api/path', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({goal: path})
                })
                .then(r => r.json())
                .then(data => {
                    displayPathResult(data);
                });
            }
            
            function checkProgress() {
                fetch('/api/progress')
                .then(r => r.json())
                .then(data => {
                    displayProgressResult(data);
                });
            }
            
            function loadConcepts() {
                const concepts = [
                    'machine_learning', 'supervised_learning', 'classification', 'regression',
                    'neural_networks', 'deep_learning', 'feature_engineering',
                    'transformer', 'reinforcement_learning'
                ];
                
                const list = document.getElementById('concept-list');
                list.innerHTML = concepts.map(c => 
                    `<div class="concept-item" onclick="learnConceptByName('${c}')">
                        ${c.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </div>`
                ).join('');
            }
            
            function learnConceptByName(concept) {
                document.getElementById('concept-input').value = concept;
                showTab('learn');
                document.querySelectorAll('.tab').forEach(btn => {
                    if (btn.textContent === 'Learn') btn.classList.add('active');
                    else btn.classList.remove('active');
                });
                learnConcept();
            }
            
            function displayLearnResult(data) {
                const result = document.getElementById('learn-result');
                if (data.error) {
                    result.innerHTML = `<div class="result"><p>${data.error}</p></div>`;
                    return;
                }
                
                let html = '<div class="result">';
                html += `<h3>${data.concept.replace('_', ' ').toUpperCase()}</h3>`;
                html += `<div class="result-section"><h4>Explanation</h4><p>${data.explanation}</p></div>`;
                
                if (data.examples && data.examples.length > 0) {
                    html += `<div class="result-section"><h4>Examples</h4><ul>`;
                    data.examples.forEach(ex => html += `<li>${ex}</li>`);
                    html += `</ul></div>`;
                }
                
                if (data.key_terms && data.key_terms.length > 0) {
                    html += `<div class="result-section"><h4>Key Terms</h4><p>${data.key_terms.join(', ')}</p></div>`;
                }
                
                if (data.learning_tips && data.learning_tips.length > 0) {
                    html += `<div class="result-section"><h4>Learning Tips</h4><ul>`;
                    data.learning_tips.forEach(tip => html += `<li>${tip}</li>`);
                    html += `</ul></div>`;
                }
                
                html += '</div>';
                result.innerHTML = html;
            }
            
            function displayAskResult(data) {
                const result = document.getElementById('ask-result');
                let html = '<div class="result">';
                html += `<h3>Answer</h3>`;
                html += `<p><strong>Question:</strong> ${data.question}</p>`;
                html += `<p><strong>Answer:</strong> ${data.answer}</p>`;
                html += '</div>';
                result.innerHTML = html;
            }
            
            function displayPathResult(data) {
                const result = document.getElementById('path-result');
                let html = '<div class="result">';
                html += `<h3>Learning Path: ${data.goal.replace('_', ' ').toUpperCase()}</h3>`;
                html += `<p><strong>Steps:</strong> ${data.steps}</p>`;
                html += `<p><strong>Estimated Time:</strong> ${data.estimated_time}</p>`;
                html += `<div class="result-section"><h4>Path</h4><ol>`;
                data.path.forEach(step => html += `<li>${step}</li>`);
                html += `</ol></div>`;
                html += '</div>';
                result.innerHTML = html;
            }
            
            function displayProgressResult(data) {
                const result = document.getElementById('progress-result');
                const progress = Math.min(data.topics_learned / 10, 1.0) * 100;
                
                let html = '<div class="result">';
                html += `<h3>Your Progress</h3>`;
                html += `<p><strong>Topics Learned:</strong> ${data.topics_learned}</p>`;
                html += `<p><strong>Learning Sessions:</strong> ${data.learning_sessions}</p>`;
                html += `<p><strong>Questions Asked:</strong> ${data.questions_asked}</p>`;
                html += `<p><strong>Current Level:</strong> ${data.current_level}</p>`;
                html += `<div class="progress-bar"><div class="progress-fill" style="width: ${progress}%">${progress.toFixed(0)}%</div></div>`;
                
                if (data.recommendations && data.recommendations.length > 0) {
                    html += `<div class="result-section"><h4>Recommendations</h4><ul>`;
                    data.recommendations.forEach(rec => html += `<li>${rec}</li>`);
                    html += `</ul></div>`;
                }
                
                html += '</div>';
                result.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """
    
    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)
    
    @app.route('/api/learn', methods=['POST'])
    def api_learn():
        data = request.json
        concept = data.get('concept', '')
        try:
            result = companion.learn_concept(concept)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})
    
    @app.route('/api/ask', methods=['POST'])
    def api_ask():
        data = request.json
        question = data.get('question', '')
        try:
            result = companion.answer_question(question)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})
    
    @app.route('/api/path', methods=['POST'])
    def api_path():
        data = request.json
        goal = data.get('goal', '')
        try:
            result = companion.suggest_learning_path(goal)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})
    
    @app.route('/api/progress')
    def api_progress():
        try:
            result = companion.assess_progress()
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})
    
    if __name__ == '__main__':
        print("\n" + "="*80)
        print("AI LEARNING COMPANION - Web UI".center(80))
        print("="*80)
        print("\nStarting web server...")
        print("Open your browser and go to: http://localhost:5000")
        print("\nPress Ctrl+C to stop the server")
        print("="*80 + "\n")
        app.run(debug=True, port=5000)

else:
    print("Flask not available. Install with: pip install flask")
    print("Or use the enhanced CLI: python ai_learning_companion_ui.py")
