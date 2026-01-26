# LLM Twin Learning Companion - Integration Guide

## 📚 Table of Contents

1. [Integration Overview](#integration-overview)
2. [Python Integration](#python-integration)
3. [Web Application Integration](#web-application-integration)
4. [API Integration](#api-integration)
5. [Database Integration](#database-integration)
6. [File System Integration](#file-system-integration)
7. [Real-World Examples](#real-world-examples)

---

## Integration Overview

The LLM Twin Learning Companion can be integrated into:
- **Python applications** - Direct import
- **Web applications** - Flask/FastAPI endpoints
- **CLI tools** - Command-line interface
- **Jupyter notebooks** - Interactive learning
- **Desktop applications** - GUI integration

---

## 1. Python Integration

### **Basic Integration**

```python
from llm_twin_learning_companion import LLMTwinLearningCompanion

class MyApplication:
    def __init__(self):
        self.companion = LLMTwinLearningCompanion(user_id="app_user")
    
    def handle_user_query(self, query: str):
        """Handle user query with companion"""
        result = self.companion.continue_conversation(query)
        return result['answer']
    
    def add_knowledge(self, content: str):
        """Add knowledge to companion"""
        return self.companion.ingest_text(content, source="app_content")

# Use it
app = MyApplication()
app.add_knowledge("Your application knowledge here...")
response = app.handle_user_query("What do you know?")
print(response)
```

### **Class-Based Integration**

```python
from llm_twin_learning_companion import LLMTwinLearningCompanion

class LearningApp:
    def __init__(self, user_id: str):
        self.companion = LLMTwinLearningCompanion(user_id=user_id)
        self.session_active = True
    
    def start_session(self):
        """Start a learning session"""
        greeting = self.companion.greet_user()
        return greeting
    
    def process_input(self, user_input: str):
        """Process user input"""
        if not self.session_active:
            return "Session not active"
        
        # Route based on input
        if user_input.startswith("learn:"):
            concept = user_input.replace("learn:", "").strip()
            result = self.companion.learn_concept_twin(concept)
            return result['explanation']
        elif user_input.startswith("add:"):
            content = user_input.replace("add:", "").strip()
            result = self.companion.ingest_text(content, source="user_input")
            return result['message']
        else:
            result = self.companion.continue_conversation(user_input)
            return result['answer']
    
    def end_session(self):
        """End session and save"""
        self.companion.save_session()
        self.session_active = False
        return "Session saved!"

# Use it
app = LearningApp("user123")
print(app.start_session())

while True:
    user_input = input("You: ")
    if user_input == "quit":
        app.end_session()
        break
    response = app.process_input(user_input)
    print(f"Companion: {response}")
```

---

## 2. Web Application Integration

### **Flask Integration**

```python
from flask import Flask, request, jsonify
from llm_twin_learning_companion import LLMTwinLearningCompanion

app = Flask(__name__)

# Global companion instance (or use session-based)
companion = LLMTwinLearningCompanion(user_id="web_user")

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint"""
    data = request.json
    message = data.get('message', '')
    
    result = companion.continue_conversation(message)
    return jsonify({
        'answer': result['answer'],
        'rag_context': result.get('rag_context', False)
    })

@app.route('/api/learn', methods=['POST'])
def learn():
    """Learn concept endpoint"""
    data = request.json
    concept = data.get('concept', '')
    
    result = companion.learn_concept_twin(concept)
    return jsonify({
        'concept': result['concept'],
        'explanation': result['explanation']
    })

@app.route('/api/ingest', methods=['POST'])
def ingest():
    """Ingest content endpoint"""
    data = request.json
    content = data.get('content', '')
    source = data.get('source', 'web_input')
    
    result = companion.ingest_text(content, source=source)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

### **FastAPI Integration**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_twin_learning_companion import LLMTwinLearningCompanion

app = FastAPI()
companion = LLMTwinLearningCompanion(user_id="api_user")

class ChatRequest(BaseModel):
    message: str

class LearnRequest(BaseModel):
    concept: str

class IngestRequest(BaseModel):
    content: str
    source: str = "api_input"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint"""
    result = companion.continue_conversation(request.message)
    return {
        'answer': result['answer'],
        'rag_context': result.get('rag_context', False)
    }

@app.post("/api/learn")
async def learn(request: LearnRequest):
    """Learn concept endpoint"""
    result = companion.learn_concept_twin(request.concept)
    return {
        'concept': result['concept'],
        'explanation': result['explanation']
    }

@app.post("/api/ingest")
async def ingest(request: IngestRequest):
    """Ingest content endpoint"""
    result = companion.ingest_text(request.content, source=request.source)
    return result
```

---

## 3. API Integration

### **REST API Wrapper**

```python
from llm_twin_learning_companion import LLMTwinLearningCompanion
from typing import Dict, Optional

class CompanionAPI:
    """RESTful API wrapper for LLM Twin"""
    
    def __init__(self, user_id: str):
        self.companion = LLMTwinLearningCompanion(user_id=user_id)
    
    def chat(self, message: str) -> Dict:
        """POST /chat"""
        result = self.companion.continue_conversation(message)
        return {
            'status': 'success',
            'answer': result['answer'],
            'metadata': {
                'rag_used': bool(result.get('rag_context')),
                'reasoning_steps': result.get('reasoning_steps', [])
            }
        }
    
    def learn(self, concept: str) -> Dict:
        """POST /learn"""
        result = self.companion.learn_concept_twin(concept)
        return {
            'status': 'success',
            'concept': result['concept'],
            'explanation': result['explanation'],
            'is_review': result.get('is_review', False)
        }
    
    def ingest_text(self, content: str, source: str = "api") -> Dict:
        """POST /ingest/text"""
        result = self.companion.ingest_text(content, source=source)
        return result
    
    def ingest_file(self, file_path: str, source: Optional[str] = None) -> Dict:
        """POST /ingest/file"""
        result = self.companion.ingest_file(file_path, source=source)
        return result
    
    def get_profile(self) -> Dict:
        """GET /profile"""
        return self.companion.get_user_profile()
    
    def get_stats(self) -> Dict:
        """GET /stats"""
        return self.companion.get_knowledge_stats()

# Use it
api = CompanionAPI("api_user")

# Chat
response = api.chat("Hello!")
print(response)

# Learn
result = api.learn("machine learning")
print(result)

# Ingest
result = api.ingest_text("Some content...", source="api")
print(result)
```

---

## 4. Database Integration

### **Save to Database**

```python
import sqlite3
from llm_twin_learning_companion import LLMTwinLearningCompanion

class DatabaseCompanion:
    """Companion with database integration"""
    
    def __init__(self, user_id: str, db_path: str = "companion.db"):
        self.companion = LLMTwinLearningCompanion(user_id=user_id)
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                type TEXT,
                input TEXT,
                output TEXT,
                timestamp TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def chat_with_logging(self, message: str):
        """Chat and log to database"""
        result = self.companion.continue_conversation(message)
        
        # Log to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO interactions (user_id, type, input, output, timestamp)
            VALUES (?, ?, ?, ?, datetime('now'))
        ''', (self.companion.user_id, 'chat', message, result['answer']))
        conn.commit()
        conn.close()
        
        return result
    
    def get_history(self, limit: int = 10):
        """Get interaction history from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT input, output, timestamp
            FROM interactions
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (self.companion.user_id, limit))
        results = cursor.fetchall()
        conn.close()
        
        return [
            {'input': r[0], 'output': r[1], 'timestamp': r[2]}
            for r in results
        ]

# Use it
db_companion = DatabaseCompanion("db_user")
result = db_companion.chat_with_logging("Hello!")
history = db_companion.get_history(5)
print(history)
```

---

## 5. File System Integration

### **Auto-Ingest from Directory**

```python
from pathlib import Path
from llm_twin_learning_companion import LLMTwinLearningCompanion
import watchdog.events
import watchdog.observers

class FileWatcherCompanion:
    """Companion that watches for new files"""
    
    def __init__(self, user_id: str, watch_directory: str):
        self.companion = LLMTwinLearningCompanion(user_id=user_id)
        self.watch_directory = Path(watch_directory)
        self.observer = watchdog.observers.Observer()
    
    def on_file_created(self, event):
        """Handle new file"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix in ['.txt', '.md', '.json']:
            print(f"New file detected: {file_path}")
            result = self.companion.ingest_file(
                str(file_path),
                source=f"watched_{file_path.name}"
            )
            print(f"Result: {result['message']}")
    
    def start_watching(self):
        """Start watching directory"""
        event_handler = FileSystemEventHandler()
        event_handler.on_created = self.on_file_created
        
        self.observer.schedule(
            event_handler,
            str(self.watch_directory),
            recursive=True
        )
        self.observer.start()
        print(f"Watching {self.watch_directory} for new files...")
    
    def stop_watching(self):
        """Stop watching"""
        self.observer.stop()
        self.observer.join()

# Use it
watcher = FileWatcherCompanion("watcher_user", "./watch_folder")
watcher.start_watching()

# Files added to ./watch_folder will be automatically ingested
```

---

## 6. Real-World Examples

### **Example 1: Jupyter Notebook Integration**

```python
# In a Jupyter notebook cell
from llm_twin_learning_companion import LLMTwinLearningCompanion

# Create companion
companion = LLMTwinLearningCompanion(user_id="notebook_user")

# Add notebook content
companion.ingest_text("""
# My Analysis Results

I found that:
- Model accuracy: 0.95
- Best features: X1, X2, X3
- Training time: 2.3 hours
""", source="notebook_analysis")

# Ask questions
result = companion.answer_question_twin("What were my analysis results?")
print(result['answer'])
```

### **Example 2: CLI Tool Integration**

```python
import argparse
from llm_twin_learning_companion import LLMTwinLearningCompanion

def main():
    parser = argparse.ArgumentParser(description='LLM Twin CLI')
    parser.add_argument('--user', default='cli_user', help='User ID')
    parser.add_argument('--command', required=True, help='Command: chat, learn, ingest')
    parser.add_argument('--input', required=True, help='Input text')
    
    args = parser.parse_args()
    
    companion = LLMTwinLearningCompanion(user_id=args.user)
    
    if args.command == 'chat':
        result = companion.continue_conversation(args.input)
        print(result['answer'])
    elif args.command == 'learn':
        result = companion.learn_concept_twin(args.input)
        print(result['explanation'])
    elif args.command == 'ingest':
        result = companion.ingest_text(args.input, source="cli")
        print(result['message'])

if __name__ == '__main__':
    main()

# Usage:
# python cli.py --command chat --input "Hello"
# python cli.py --command learn --input "machine learning"
# python cli.py --command ingest --input "Some content..."
```

### **Example 3: Desktop App Integration**

```python
import tkinter as tk
from llm_twin_learning_companion import LLMTwinLearningCompanion

class CompanionGUI:
    def __init__(self):
        self.companion = LLMTwinLearningCompanion(user_id="gui_user")
        self.setup_ui()
    
    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("LLM Twin Companion")
        
        # Chat area
        self.chat_area = tk.Text(self.root, height=20, width=60)
        self.chat_area.pack(pady=10)
        
        # Input area
        self.input_area = tk.Entry(self.root, width=60)
        self.input_area.pack(pady=5)
        self.input_area.bind('<Return>', self.send_message)
        
        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)
        
        tk.Button(button_frame, text="Send", command=self.send_message).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Learn", command=self.learn_concept).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Save", command=self.save_session).pack(side=tk.LEFT, padx=5)
    
    def send_message(self, event=None):
        message = self.input_area.get()
        if not message:
            return
        
        self.chat_area.insert(tk.END, f"You: {message}\n")
        self.input_area.delete(0, tk.END)
        
        result = self.companion.continue_conversation(message)
        self.chat_area.insert(tk.END, f"Companion: {result['answer']}\n\n")
    
    def learn_concept(self):
        concept = self.input_area.get()
        if not concept:
            return
        
        result = self.companion.learn_concept_twin(concept)
        self.chat_area.insert(tk.END, f"Learning: {concept}\n{result['explanation']}\n\n")
        self.input_area.delete(0, tk.END)
    
    def save_session(self):
        self.companion.save_session()
        self.chat_area.insert(tk.END, "Session saved!\n\n")
    
    def run(self):
        self.root.mainloop()

# Use it
app = CompanionGUI()
app.run()
```

---

## 🔧 Integration Patterns

### **Pattern 1: Singleton Pattern**

```python
class CompanionManager:
    _instance = None
    _companion = None
    
    def __new__(cls, user_id: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._companion = LLMTwinLearningCompanion(user_id=user_id)
        return cls._instance
    
    @property
    def companion(self):
        return self._companion

# Use it (same instance everywhere)
manager1 = CompanionManager("user")
manager2 = CompanionManager("user")
assert manager1.companion is manager2.companion  # Same instance
```

### **Pattern 2: Factory Pattern**

```python
class CompanionFactory:
    _companions = {}
    
    @classmethod
    def get_companion(cls, user_id: str):
        if user_id not in cls._companions:
            cls._companions[user_id] = LLMTwinLearningCompanion(user_id=user_id)
        return cls._companions[user_id]

# Use it
companion1 = CompanionFactory.get_companion("user1")
companion2 = CompanionFactory.get_companion("user2")
companion1_again = CompanionFactory.get_companion("user1")  # Same instance
```

---

## 📝 Best Practices

1. **User ID Management**: Use consistent user IDs for persistent memory
2. **Session Management**: Save sessions regularly
3. **Error Handling**: Wrap companion calls in try-except
4. **Resource Cleanup**: Save sessions before closing
5. **Content Organization**: Use meaningful source names

---

**Integrate the LLM Twin Companion into your application today!**
