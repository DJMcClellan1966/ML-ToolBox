# LLM Twin Quick Reference Card

## 🚀 Quick Start

```python
from llm_twin_learning_companion import LLMTwinLearningCompanion

companion = LLMTwinLearningCompanion(user_id="your_name")
```

---

## 📋 All Available Methods

### **Content Ingestion**
```python
# Add text
companion.ingest_text("Your content...", source="notes")

# Add file
companion.ingest_file("file.txt", source="documents")

# Add directory
companion.ingest_directory("./docs", pattern="*.md")
```

### **Learning**
```python
# Learn a concept
result = companion.learn_concept_twin("machine learning")

# Ask questions
result = companion.answer_question_twin("What is ML?")
print(result['answer'])
if 'sources' in result:
    print("Sources:", result['sources'])
```

### **MindForge Sync**
```python
# Sync FROM MindForge (full)
companion.sync_mindforge()

# Sync FROM MindForge (incremental - faster)
companion.sync_mindforge(incremental=True)

# Sync TO MindForge (two-way sync)
companion.sync_to_mindforge()
```

### **Analytics & Stats**
```python
# Learning analytics
analytics = companion.get_learning_analytics()
print(analytics['learning_velocity'])
print(analytics['recommendations'])

# Knowledge stats
stats = companion.get_knowledge_stats()
print(stats['total_documents'])
```

### **Export & Backup**
```python
# Export knowledge base
companion.export_knowledge_base(format='json')  # or 'txt', 'csv'

# Backup session
companion.backup_session()
```

### **Conversation**
```python
# Chat
response = companion.continue_conversation("Hello!")
print(response['answer'])

# Get profile
profile = companion.get_user_profile()
```

### **Session Management**
```python
# Save session
companion.save_session()

# Load session (automatic on init)
# companion = LLMTwinLearningCompanion(user_id="user")  # Auto-loads
```

---

## 🎯 Common Patterns

### **Daily Learning Workflow**
```python
# Morning: Sync new content
companion.sync_mindforge(incremental=True)

# Learn
companion.learn_concept_twin("new topic")

# Evening: Sync learnings back
companion.sync_to_mindforge()
```

### **Research Assistant**
```python
# Add research
companion.ingest_text("Findings...", source="paper")
companion.ingest_file("notes.pdf")

# Query with sources
result = companion.answer_question_twin("What did we find?")
print(result['answer'])
print(result.get('sources', []))
```

### **Knowledge Base Builder**
```python
# Sync everything
companion.sync_mindforge()

# Add content
companion.ingest_directory("./docs")

# Export
companion.export_knowledge_base(format='json')
```

---

## 📊 Return Values

### **Answer Result**
```python
{
    'answer': '...',
    'sources': [...],  # NEW: Source attribution
    'rag_used': True,
    'rag_context': True
}
```

### **Sync Result**
```python
{
    'success': True,
    'synced': 10,
    'errors': 0,
    'message': '...'
}
```

### **Analytics Result**
```python
{
    'topics_learned': [...],
    'learning_velocity': 2.5,
    'knowledge_gaps': [...],
    'recommendations': [...],
    'topics_by_category': {...}
}
```

---

## 🔧 Configuration

### **Better RAG (Recommended)**
```bash
pip install sentence-transformers
```
Automatically used if available!

### **MindForge Sync**
```bash
pip install sqlalchemy
```

---

## 🎯 Tips

1. **Use incremental sync** for large KBs
2. **Check sources** to verify answers
3. **Review analytics** weekly
4. **Backup regularly**
5. **Export for sharing**

---

**See `WHAT_NEXT.md` for detailed next steps!**
