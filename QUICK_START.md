# Quick Start Guide - LLM Twin Learning Companion

## 🚀 Get Started in 5 Minutes

### **Step 1: Run Simple Example**

```bash
python llm_twin_simple_example.py
```

This will show you:
- ✅ How to create a companion
- ✅ How to add content
- ✅ How to learn concepts
- ✅ How to ask questions
- ✅ How to get your profile

---

### **Step 2: Try the Web UI**

```bash
# Install Flask (if needed)
pip install flask

# Run web UI
python llm_twin_web_ui.py

# Open browser
# http://localhost:5000
```

**Features:**
- 💬 Chat interface
- 📚 Learn concepts
- 📥 Add content
- 📖 View history
- 👤 See profile

---

### **Step 3: Use in Your Code**

```python
from llm_twin_learning_companion import LLMTwinLearningCompanion

# Create companion
companion = LLMTwinLearningCompanion(user_id="your_name")

# Chat
response = companion.continue_conversation("Hello!")
print(response['answer'])

# Learn
result = companion.learn_concept_twin("machine learning")
print(result['explanation'])

# Add content
companion.ingest_text("Your notes here...", source="my_notes")

# Save
companion.save_session()
```

---

## 📚 Next Steps

1. **Read Examples**: `LLM_TWIN_EXAMPLES.md`
2. **Integration Guide**: `LLM_TWIN_INTEGRATION.md`
3. **API Reference**: `LLM_TWIN_API.md`
4. **Full README**: `LLM_TWIN_README.md`

---

## 🎯 Common Tasks

### **Add Your Content**
```python
# Add text
companion.ingest_text("Your content...", source="notes")

# Add file
companion.ingest_file("file.txt", source="documents")

# Add directory
companion.ingest_directory("./docs", pattern="*.md")
```

### **Learn Something**
```python
# Learn a concept
result = companion.learn_concept_twin("neural networks")
print(result['explanation'])
```

### **Ask Questions**
```python
# Ask a question
result = companion.answer_question_twin("How does it work?")
print(result['answer'])
```

### **Get Your Profile**
```python
# Get profile
profile = companion.get_user_profile()
print(f"Topics learned: {profile['conversation_stats']['topics_learned']}")
```

---

**That's it! You're ready to use the LLM Twin Learning Companion!**
