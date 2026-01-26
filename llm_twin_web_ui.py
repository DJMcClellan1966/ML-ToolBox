"""
LLM Twin Learning Companion - Web UI

Simple, beautiful web interface for the LLM Twin Learning Companion.
Includes content ingestion for adding knowledge to the system.
"""

try:
    from flask import Flask, render_template_string, request, jsonify, send_from_directory
    from werkzeug.utils import secure_filename
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from llm_twin_learning_companion import LLMTwinLearningCompanion
from pathlib import Path
import json
import os
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if FLASK_AVAILABLE:
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['UPLOAD_FOLDER'] = 'uploads'
    
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize companion (singleton pattern)
    companion = None
    
    def get_companion():
        """Get or create companion instance"""
        global companion
        if companion is None:
            user_id = request.cookies.get('user_id', 'default_user')
            companion = LLMTwinLearningCompanion(user_id=user_id, personality_type="helpful_mentor")
        return companion
    
    # HTML Template
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LLM Twin Learning Companion</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
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
                display: grid;
                grid-template-columns: 250px 1fr;
                min-height: calc(100vh - 40px);
            }
            .sidebar {
                background: #2d3748;
                color: white;
                padding: 30px 20px;
            }
            .sidebar h2 {
                font-size: 1.2em;
                margin-bottom: 30px;
                color: #a0aec0;
            }
            .sidebar-menu {
                list-style: none;
            }
            .sidebar-menu li {
                margin-bottom: 10px;
            }
            .sidebar-menu a {
                color: #cbd5e0;
                text-decoration: none;
                padding: 12px 15px;
                display: block;
                border-radius: 8px;
                transition: all 0.3s;
            }
            .sidebar-menu a:hover, .sidebar-menu a.active {
                background: #4a5568;
                color: white;
            }
            .main-content {
                padding: 30px;
                overflow-y: auto;
            }
            .header {
                margin-bottom: 30px;
            }
            .header h1 {
                color: #2d3748;
                font-size: 2em;
                margin-bottom: 10px;
            }
            .header p {
                color: #718096;
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
                color: #2d3748;
                font-weight: 600;
            }
            .input-group input,
            .input-group textarea,
            .input-group select {
                width: 100%;
                padding: 12px;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s;
            }
            .input-group input:focus,
            .input-group textarea:focus {
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
            .btn-secondary {
                background: #e2e8f0;
                color: #2d3748;
            }
            .btn-secondary:hover {
                background: #cbd5e0;
            }
            .response-box {
                margin-top: 20px;
                padding: 20px;
                background: #f7fafc;
                border-radius: 10px;
                border-left: 4px solid #667eea;
            }
            .response-box h3 {
                color: #667eea;
                margin-bottom: 15px;
            }
            .response-box p {
                color: #4a5568;
                line-height: 1.6;
                white-space: pre-wrap;
            }
            .message {
                padding: 12px 20px;
                border-radius: 8px;
                margin-bottom: 15px;
            }
            .message.success {
                background: #c6f6d5;
                color: #22543d;
                border-left: 4px solid #48bb78;
            }
            .message.error {
                background: #fed7d7;
                color: #742a2a;
                border-left: 4px solid #f56565;
            }
            .message.info {
                background: #bee3f8;
                color: #2c5282;
                border-left: 4px solid #4299e1;
            }
            .conversation {
                max-height: 500px;
                overflow-y: auto;
                margin-bottom: 20px;
            }
            .conversation-item {
                margin-bottom: 15px;
                padding: 15px;
                border-radius: 8px;
            }
            .conversation-item.user {
                background: #edf2f7;
                margin-left: 20%;
            }
            .conversation-item.assistant {
                background: #e6fffa;
                margin-right: 20%;
            }
            .conversation-item .role {
                font-weight: bold;
                margin-bottom: 5px;
                color: #667eea;
            }
            .file-upload {
                border: 2px dashed #cbd5e0;
                border-radius: 8px;
                padding: 30px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
            }
            .file-upload:hover {
                border-color: #667eea;
                background: #f7fafc;
            }
            .file-upload.dragover {
                border-color: #667eea;
                background: #edf2f7;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .stat-card {
                background: #f7fafc;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
            .stat-card .value {
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
            }
            .stat-card .label {
                color: #718096;
                margin-top: 5px;
            }
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="sidebar">
                <h2>LLM Twin</h2>
                <ul class="sidebar-menu">
                    <li><a href="#" onclick="showTab('chat'); return false;" class="active" id="tab-chat">💬 Chat</a></li>
                    <li><a href="#" onclick="showTab('learn'); return false;" id="tab-learn">📚 Learn</a></li>
                    <li><a href="#" onclick="showTab('ingest'); return false;" id="tab-ingest">📥 Add Content</a></li>
                    <li><a href="#" onclick="showTab('history'); return false;" id="tab-history">📖 History</a></li>
                    <li><a href="#" onclick="showTab('profile'); return false;" id="tab-profile">👤 Profile</a></li>
                </ul>
            </div>
            <div class="main-content">
                <div class="header">
                    <h1>🧠 LLM Twin Learning Companion</h1>
                    <p>Your personal AI learning companion with persistent memory</p>
                </div>
                
                <div id="messages"></div>
                
                <!-- Chat Tab -->
                <div id="chat" class="tab-content active">
                    <div class="conversation" id="conversation"></div>
                    <div class="input-group">
                        <textarea id="chat-input" rows="3" placeholder="Ask a question or say something..."></textarea>
                    </div>
                    <button class="btn" onclick="sendMessage()">Send</button>
                </div>
                
                <!-- Learn Tab -->
                <div id="learn" class="tab-content">
                    <div class="input-group">
                        <label>Concept to Learn</label>
                        <input type="text" id="concept-input" placeholder="e.g., machine learning, neural networks">
                    </div>
                    <button class="btn" onclick="learnConcept()">Learn Concept</button>
                    <div id="learn-response"></div>
                </div>
                
                <!-- Content Ingestion Tab -->
                <div id="ingest" class="tab-content">
                    <h2>Add Content to Knowledge Base</h2>
                    <p style="color: #718096; margin-bottom: 20px;">
                        Add documents, text, or files to enhance the companion's knowledge.
                    </p>
                    
                    <div class="input-group">
                        <label>Add Text Content</label>
                        <textarea id="text-content" rows="5" placeholder="Paste text content here..."></textarea>
                    </div>
                    <div class="input-group">
                        <label>Source/Category (optional)</label>
                        <input type="text" id="content-source" placeholder="e.g., notes, articles, documentation">
                    </div>
                    <button class="btn" onclick="addTextContent()">Add Text</button>
                    
                    <hr style="margin: 30px 0; border: none; border-top: 2px solid #e2e8f0;">
                    
                    <div class="input-group">
                        <label>Upload File</label>
                        <div class="file-upload" id="file-upload" onclick="document.getElementById('file-input').click()">
                            <p>📄 Click to upload or drag and drop</p>
                            <p style="font-size: 0.9em; color: #718096;">Supports: .txt, .md, .json, .py, .js, .html</p>
                        </div>
                        <input type="file" id="file-input" style="display: none;" onchange="handleFileSelect(event)">
                    </div>
                    
                    <hr style="margin: 30px 0; border: none; border-top: 2px solid #e2e8f0;">
                    
                    <div class="input-group">
                        <label>Knowledge Base Statistics</label>
                        <button class="btn btn-secondary" onclick="loadKnowledgeStats()">Refresh Stats</button>
                        <div id="knowledge-stats" style="margin-top: 15px;"></div>
                    </div>
                    
                    <hr style="margin: 30px 0; border: none; border-top: 2px solid #e2e8f0;">
                    
                    <div class="input-group">
                        <label>Sync MindForge</label>
                        <p style="color: #718096; font-size: 0.9em; margin-bottom: 10px;">
                            Sync your MindForge knowledge base to LLM Twin
                        </p>
                        <button class="btn" onclick="syncMindForge()">Sync MindForge</button>
                        <div id="mindforge-sync-response" style="margin-top: 15px;"></div>
                    </div>
                    
                    <div id="ingest-response"></div>
                </div>
                
                <!-- History Tab -->
                <div id="history" class="tab-content">
                    <h2>Conversation History</h2>
                    <div id="history-content"></div>
                </div>
                
                <!-- Profile Tab -->
                <div id="profile" class="tab-content">
                    <h2>Your Profile</h2>
                    <div id="profile-content"></div>
                </div>
            </div>
        </div>
        
        <script>
            let currentTab = 'chat';
            
            function showTab(tab) {
                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.sidebar-menu a').forEach(a => a.classList.remove('active'));
                
                // Show selected tab
                document.getElementById(tab).classList.add('active');
                document.getElementById('tab-' + tab).classList.add('active');
                currentTab = tab;
                
                // Load tab-specific data
                if (tab === 'history') loadHistory();
                if (tab === 'profile') loadProfile();
            }
            
            function showMessage(text, type = 'info') {
                const messages = document.getElementById('messages');
                const msg = document.createElement('div');
                msg.className = `message ${type}`;
                msg.textContent = text;
                messages.appendChild(msg);
                setTimeout(() => msg.remove(), 5000);
            }
            
            async function sendMessage() {
                const input = document.getElementById('chat-input');
                const message = input.value.trim();
                if (!message) return;
                
                // Add user message to conversation
                addToConversation('You', message, 'user');
                input.value = '';
                
                // Show loading
                const loading = addToConversation('Assistant', 'Thinking...', 'assistant');
                
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: message})
                    });
                    const data = await response.json();
                    
                    // Remove loading, add response
                    loading.remove();
                    addToConversation('Assistant', data.answer || data.response, 'assistant');
                } catch (error) {
                    loading.remove();
                    addToConversation('Assistant', 'Error: ' + error.message, 'assistant');
                    showMessage('Error sending message', 'error');
                }
            }
            
            function addToConversation(role, text, type) {
                const conversation = document.getElementById('conversation');
                const item = document.createElement('div');
                item.className = `conversation-item ${type}`;
                item.innerHTML = `<div class="role">${role}:</div><div>${text}</div>`;
                conversation.appendChild(item);
                conversation.scrollTop = conversation.scrollHeight;
                return item;
            }
            
            async function learnConcept() {
                const concept = document.getElementById('concept-input').value.trim();
                if (!concept) {
                    showMessage('Please enter a concept', 'error');
                    return;
                }
                
                const responseDiv = document.getElementById('learn-response');
                responseDiv.innerHTML = '<div class="loading"></div> Learning...';
                
                try {
                    const response = await fetch('/api/learn', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({concept: concept})
                    });
                    const data = await response.json();
                    
                    responseDiv.innerHTML = `
                        <div class="response-box">
                            <h3>${data.concept || concept}</h3>
                            <p>${data.explanation || data.response}</p>
                            ${data.related_knowledge ? '<p><strong>Related:</strong> ' + data.related_knowledge.join(', ') + '</p>' : ''}
                        </div>
                    `;
                    showMessage('Concept learned!', 'success');
                } catch (error) {
                    responseDiv.innerHTML = `<div class="message error">Error: ${error.message}</div>`;
                }
            }
            
            async function addTextContent() {
                const text = document.getElementById('text-content').value.trim();
                const source = document.getElementById('content-source').value.trim() || 'user_input';
                
                if (!text) {
                    showMessage('Please enter some text', 'error');
                    return;
                }
                
                showMessage('Adding content...', 'info');
                
                try {
                    const response = await fetch('/api/ingest/text', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({content: text, source: source})
                    });
                    const data = await response.json();
                    
                    if (data.success) {
                        showMessage('Content added successfully!', 'success');
                        document.getElementById('text-content').value = '';
                        document.getElementById('content-source').value = '';
                    } else {
                        showMessage('Error: ' + (data.error || 'Unknown error'), 'error');
                    }
                } catch (error) {
                    showMessage('Error: ' + error.message, 'error');
                }
            }
            
            async function handleFileSelect(event) {
                const file = event.target.files[0];
                if (!file) return;
                
                showMessage('Uploading file...', 'info');
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/api/ingest/file', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    
                    if (data.success) {
                        showMessage(`File "${file.name}" added successfully!`, 'success');
                        event.target.value = '';
                    } else {
                        showMessage('Error: ' + (data.error || 'Unknown error'), 'error');
                    }
                } catch (error) {
                    showMessage('Error: ' + error.message, 'error');
                }
            }
            
            async function loadHistory() {
                try {
                    const response = await fetch('/api/history');
                    const data = await response.json();
                    
                    const content = document.getElementById('history-content');
                    if (data.history && data.history.length > 0) {
                        content.innerHTML = data.history.map(item => `
                            <div class="conversation-item">
                                <div class="role">${item.type}:</div>
                                <div>${item.input}</div>
                                <div style="margin-top: 10px; color: #718096; font-size: 0.9em;">
                                    ${new Date(item.timestamp).toLocaleString()}
                                </div>
                            </div>
                        `).join('');
                    } else {
                        content.innerHTML = '<p style="color: #718096;">No history yet. Start a conversation!</p>';
                    }
                } catch (error) {
                    document.getElementById('history-content').innerHTML = 
                        `<div class="message error">Error loading history: ${error.message}</div>`;
                }
            }
            
            async function loadProfile() {
                try {
                    const response = await fetch('/api/profile');
                    const data = await response.json();
                    
                    const content = document.getElementById('profile-content');
                    const stats = data.conversation_stats || {};
                    const profile = data.profile || {};
                    
                    content.innerHTML = `
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="value">${stats.total_interactions || 0}</div>
                                <div class="label">Total Interactions</div>
                            </div>
                            <div class="stat-card">
                                <div class="value">${stats.topics_learned || 0}</div>
                                <div class="label">Topics Learned</div>
                            </div>
                            <div class="stat-card">
                                <div class="value">${stats.current_session_turns || 0}</div>
                                <div class="label">This Session</div>
                            </div>
                        </div>
                        <div style="margin-top: 30px;">
                            <h3>Learning Profile</h3>
                            <p><strong>User ID:</strong> ${data.user_id || 'N/A'}</p>
                            <p><strong>Personality:</strong> ${data.personality || 'N/A'}</p>
                            <p><strong>Learning Style:</strong> ${profile.learning_style || 'N/A'}</p>
                            <p><strong>Preferred Pace:</strong> ${profile.preferred_pace || 'N/A'}</p>
                        </div>
                    `;
                } catch (error) {
                    document.getElementById('profile-content').innerHTML = 
                        `<div class="message error">Error loading profile: ${error.message}</div>`;
                }
            }
            
            // Enter key to send
            document.getElementById('chat-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
            
            // File upload drag and drop
            const fileUpload = document.getElementById('file-upload');
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                fileUpload.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            ['dragenter', 'dragover'].forEach(eventName => {
                fileUpload.addEventListener(eventName, () => fileUpload.classList.add('dragover'), false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                fileUpload.addEventListener(eventName, () => fileUpload.classList.remove('dragover'), false);
            });
            
            fileUpload.addEventListener('drop', handleDrop, false);
            
            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                if (files.length > 0) {
                    document.getElementById('file-input').files = files;
                    handleFileSelect({target: {files: files}});
                }
            }
            
            async function loadKnowledgeStats() {
                try {
                    const response = await fetch('/api/knowledge/stats');
                    const data = await response.json();
                    
                    const statsDiv = document.getElementById('knowledge-stats');
                    if (data.error) {
                        statsDiv.innerHTML = `<div class="message error">${data.error}</div>`;
                        return;
                    }
                    
                    let sourcesHtml = '';
                    if (data.sources && Object.keys(data.sources).length > 0) {
                        sourcesHtml = '<div style="margin-top: 10px;"><strong>Sources:</strong><ul style="margin-left: 20px; margin-top: 5px;">';
                        for (const [source, count] of Object.entries(data.sources)) {
                            sourcesHtml += `<li>${source}: ${count} document(s)</li>`;
                        }
                        sourcesHtml += '</ul></div>';
                    }
                    
                    statsDiv.innerHTML = `
                        <div class="response-box">
                            <h3>Knowledge Base</h3>
                            <p><strong>Total Documents:</strong> ${data.total_documents || 0}</p>
                            ${sourcesHtml}
                        </div>
                    `;
                } catch (error) {
                    document.getElementById('knowledge-stats').innerHTML = 
                        `<div class="message error">Error loading stats: ${error.message}</div>`;
                }
            }
            
            async function syncMindForge() {
                const responseDiv = document.getElementById('mindforge-sync-response');
                responseDiv.innerHTML = '<div class="loading"></div> Syncing...';
                
                try {
                    const response = await fetch('/api/ingest/mindforge', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'}
                    });
                    const data = await response.json();
                    
                    if (data.success) {
                        responseDiv.innerHTML = `
                            <div class="message success">
                                ${data.message}<br>
                                Synced: ${data.synced || 0} items
                                ${data.errors > 0 ? `<br>Errors: ${data.errors}` : ''}
                            </div>
                        `;
                        // Refresh stats
                        loadKnowledgeStats();
                    } else {
                        responseDiv.innerHTML = `
                            <div class="message error">
                                Error: ${data.error || 'Unknown error'}
                            </div>
                        `;
                    }
                } catch (error) {
                    responseDiv.innerHTML = `
                        <div class="message error">
                            Error: ${error.message}
                        </div>
                    `;
                }
            }
            
            // Load stats when ingest tab is shown
            const originalShowTabFunc = showTab;
            window.showTab = function(tab) {
                originalShowTabFunc(tab);
                if (tab === 'ingest') {
                    setTimeout(loadKnowledgeStats, 100);
                }
            };
        </script>
    </body>
    </html>
    """
    
    @app.route('/')
    def index():
        """Main page"""
        return render_template_string(HTML_TEMPLATE)
    
    @app.route('/api/chat', methods=['POST'])
    def chat():
        """Chat endpoint"""
        try:
            data = request.json
            message = data.get('message', '')
            
            if not message:
                return jsonify({'error': 'Message is required'}), 400
            
            companion = get_companion()
            result = companion.continue_conversation(message)
            
            return jsonify({
                'answer': result.get('answer') or result.get('explanation', ''),
                'response': result.get('answer') or result.get('explanation', ''),
                'rag_context': result.get('rag_context'),
                'reasoning_steps': result.get('reasoning_steps', [])
            })
        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/learn', methods=['POST'])
    def learn():
        """Learn concept endpoint"""
        try:
            data = request.json
            concept = data.get('concept', '')
            
            if not concept:
                return jsonify({'error': 'Concept is required'}), 400
            
            companion = get_companion()
            result = companion.learn_concept_twin(concept)
            
            return jsonify({
                'concept': result.get('concept', concept),
                'explanation': result.get('explanation', ''),
                'related_knowledge': result.get('related_knowledge', [])
            })
        except Exception as e:
            logger.error(f"Learn error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/ingest/text', methods=['POST'])
    def ingest_text():
        """Ingest text content"""
        try:
            data = request.json
            content = data.get('content', '')
            source = data.get('source', 'user_input')
            
            if not content:
                return jsonify({'error': 'Content is required'}), 400
            
            companion = get_companion()
            
            # Add to RAG system
            if companion.rag:
                companion.rag.add_knowledge(
                    content,
                    metadata={
                        'source': source,
                        'type': 'text',
                        'timestamp': str(Path(__file__).stat().st_mtime)
                    }
                )
                return jsonify({'success': True, 'message': 'Content added to knowledge base'})
            else:
                return jsonify({'error': 'RAG system not available'}), 500
                
        except Exception as e:
            logger.error(f"Ingest text error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/ingest/file', methods=['POST'])
    def ingest_file():
        """Ingest file content"""
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Save file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Use companion's ingest_file method
                companion = get_companion()
                result = companion.ingest_file(filepath, source=f"file_upload_{filename}")
                
                if result.get('success'):
                    return jsonify(result)
                else:
                    return jsonify(result), 500
            finally:
                # Clean up temp file
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass
                
        except Exception as e:
            logger.error(f"Ingest file error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/history', methods=['GET'])
    def get_history():
        """Get conversation history"""
        try:
            companion = get_companion()
            history = companion.memory.conversation_history
            
            # Format history
            formatted_history = []
            for item in list(history)[-50:]:  # Last 50 items
                formatted_history.append({
                    'type': 'user' if 'user_input' in item else 'assistant',
                    'input': item.get('user_input', item.get('response', '')),
                    'timestamp': item.get('timestamp', '')
                })
            
            return jsonify({'history': formatted_history})
        except Exception as e:
            logger.error(f"History error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/profile', methods=['GET'])
    def get_profile():
        """Get user profile"""
        try:
            companion = get_companion()
            profile = companion.get_user_profile()
            return jsonify(profile)
        except Exception as e:
            logger.error(f"Profile error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/save', methods=['POST'])
    def save_session():
        """Save current session"""
        try:
            companion = get_companion()
            companion.save_session()
            return jsonify({'success': True, 'message': 'Session saved'})
        except Exception as e:
            logger.error(f"Save error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/knowledge/stats', methods=['GET'])
    def get_knowledge_stats():
        """Get knowledge base statistics"""
        try:
            companion = get_companion()
            stats = companion.get_knowledge_stats()
            return jsonify(stats)
        except Exception as e:
            logger.error(f"Knowledge stats error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/ingest/mindforge', methods=['POST'])
    def ingest_mindforge():
        """Sync MindForge knowledge base"""
        try:
            data = request.json or {}
            content_types = data.get('content_types')
            
            companion = get_companion()
            result = companion.sync_mindforge(content_types=content_types)
            
            return jsonify(result)
        except Exception as e:
            logger.error(f"MindForge sync error: {e}", exc_info=True)
            return jsonify({'success': False, 'error': str(e)}), 500
    
    def main():
        """Run the web UI"""
        print("\n" + "="*80)
        print("LLM TWIN LEARNING COMPANION - WEB UI".center(80))
        print("="*80)
        print("\nStarting web server...")
        print("Open your browser to: http://localhost:5000")
        print("\nFeatures:")
        print("  • Chat with your learning companion")
        print("  • Learn new concepts")
        print("  • Add content to knowledge base")
        print("  • View conversation history")
        print("  • See your learning profile")
        print("\n" + "="*80 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
    
    if __name__ == "__main__":
        main()
else:
    print("Flask is required for the web UI.")
    print("Install with: pip install flask")
    print("\nYou can still use the companion via Python:")
    print("  from llm_twin_learning_companion import LLMTwinLearningCompanion")
    print("  companion = LLMTwinLearningCompanion()")
