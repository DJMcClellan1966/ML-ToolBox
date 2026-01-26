# LLM Twin Learning Companion

**Your personal AI learning companion with persistent memory and knowledge ingestion.**

## 🎯 What Is This?

The LLM Twin Learning Companion is an AI-powered learning assistant that:
- **Remembers you** across sessions (persistent memory)
- **Learns your preferences** and adapts to your style
- **Retrieves knowledge** from your content (RAG system)
- **Provides personalized** explanations and learning paths
- **Ingests your content** (text, files, directories)

---

## 🚀 Quick Start

### **Installation**

```bash
# Install dependencies
pip install flask  # For web UI (optional)
```

### **Basic Usage**

```python
from llm_twin_learning_companion import LLMTwinLearningCompanion

# Create companion
companion = LLMTwinLearningCompanion(user_id="your_name")

# Chat
response = companion.continue_conversation("What is machine learning?")
print(response['answer'])

# Learn a concept
result = companion.learn_concept_twin("neural networks")
print(result['explanation'])

# Add your content
companion.ingest_text("Your notes here...", source="my_notes")
```

### **Web UI**

```bash
python llm_twin_web_ui.py
# Open http://localhost:5000
```

---

## 📚 Core Features

### **1. Persistent Memory**
- Remembers conversations across sessions
- Tracks learning progress
- Stores preferences and patterns

### **2. RAG Integration**
- Retrieves relevant knowledge from your content
- Enhances responses with context
- Supports semantic search

### **3. Content Ingestion**
- Add text content
- Upload files (.txt, .md, .json, etc.)
- Index entire directories

### **4. Personalized Learning**
- Adapts to your learning style
- Suggests personalized paths
- Tracks topics learned

### **5. Chain-of-Thought Reasoning**
- Explains reasoning steps
- Provides detailed explanations
- Shows thought process

---

## 📖 Documentation

- **[Simple Examples](LLM_TWIN_EXAMPLES.md)** - Quick start examples with code
- **[Integration Guide](LLM_TWIN_INTEGRATION.md)** - How to integrate with your projects
- **[API Reference](LLM_TWIN_API.md)** - Complete API documentation
- **[Web UI Guide](LLM_TWIN_UI_GUIDE.md)** - Web interface guide

### **Quick Example Script**

Run the simple example to see it in action:

```bash
python llm_twin_simple_example.py
```

---

## 🎯 Use Cases

1. **Personal Learning Assistant**
   - Learn new concepts
   - Get explanations
   - Track progress

2. **Knowledge Base Assistant**
   - Add your documents
   - Ask questions about your content
   - Get context-aware answers

3. **Study Companion**
   - Review topics
   - Get personalized paths
   - Track what you've learned

---

## 🔧 Requirements

- Python 3.8+
- ML Toolbox (included in this project)
- Flask (optional, for web UI)

---

## 📝 License

Part of the ML Toolbox project.

---

## 🤝 Contributing

This is part of a personal ML toolbox. Contributions welcome!

---

**Start learning with your AI companion today!**
