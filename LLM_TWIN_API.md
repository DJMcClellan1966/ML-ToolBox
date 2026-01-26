# LLM Twin Learning Companion - API Reference

## 📚 Complete API Documentation

---

## Class: `LLMTwinLearningCompanion`

### **Initialization**

```python
LLMTwinLearningCompanion(
    user_id: str = "default_user",
    personality_type: str = "helpful_mentor"
)
```

**Parameters:**
- `user_id` (str): Unique identifier for the user
- `personality_type` (str): Personality type (e.g., "helpful_mentor", "socratic_teacher")

**Returns:** `LLMTwinLearningCompanion` instance

---

## Core Methods

### **1. `continue_conversation(user_input: str) -> Dict`**

Continue a conversation with the companion.

**Parameters:**
- `user_input` (str): User's message or question

**Returns:**
```python
{
    'answer': str,              # Companion's response
    'rag_context': bool,         # Whether RAG context was used
    'reasoning_steps': List[str] # Chain-of-thought steps
}
```

**Example:**
```python
result = companion.continue_conversation("What is machine learning?")
print(result['answer'])
```

---

### **2. `learn_concept_twin(concept: str) -> Dict`**

Learn a new concept with LLM Twin features.

**Parameters:**
- `concept` (str): Concept to learn

**Returns:**
```python
{
    'concept': str,                    # The concept
    'explanation': str,                # Detailed explanation
    'is_review': bool,                 # Whether already learned
    'related_knowledge': List[str],    # Related topics
    'rag_context': bool,               # Whether RAG was used
    'reasoning_steps': List[str]       # Reasoning steps
}
```

**Example:**
```python
result = companion.learn_concept_twin("neural networks")
print(result['explanation'])
```

---

### **3. `answer_question_twin(question: str) -> Dict`**

Answer a question with enhanced context.

**Parameters:**
- `question` (str): Question to answer

**Returns:**
```python
{
    'answer': str,              # Answer to question
    'rag_context': bool,         # Whether RAG context was used
    'reasoning_steps': List[str] # Reasoning steps
}
```

**Example:**
```python
result = companion.answer_question_twin("How does backpropagation work?")
print(result['answer'])
```

---

## Content Ingestion Methods

### **4. `ingest_text(text: str, source: str = "user_input", metadata: Optional[Dict] = None) -> Dict`**

Add text content to the knowledge base.

**Parameters:**
- `text` (str): Text content to add
- `source` (str): Source identifier (default: "user_input")
- `metadata` (Dict, optional): Additional metadata

**Returns:**
```python
{
    'success': bool,           # Whether ingestion succeeded
    'message': str,            # Status message
    'characters': int,         # Number of characters added
    'doc_id': str             # Document ID
}
```

**Example:**
```python
result = companion.ingest_text(
    "Machine learning is...",
    source="my_notes"
)
print(result['message'])
```

---

### **5. `ingest_file(file_path: str, source: Optional[str] = None) -> Dict`**

Add file content to the knowledge base.

**Parameters:**
- `file_path` (str): Path to file
- `source` (str, optional): Source identifier

**Returns:**
```python
{
    'success': bool,           # Whether ingestion succeeded
    'message': str,            # Status message
    'filename': str,           # Name of file
    'characters': int,         # Number of characters added
    'doc_id': str             # Document ID
}
```

**Example:**
```python
result = companion.ingest_file("notes.txt", source="documents")
print(result['message'])
```

---

### **6. `ingest_directory(directory_path: str, pattern: str = "*.md", source: Optional[str] = None) -> Dict`**

Add all matching files from a directory.

**Parameters:**
- `directory_path` (str): Path to directory
- `pattern` (str): File pattern (default: "*.md")
- `source` (str, optional): Source identifier

**Returns:**
```python
{
    'success': bool,           # Whether ingestion succeeded
    'message': str,            # Status message
    'ingested': int,           # Number of files ingested
    'errors': int,             # Number of errors
    'total': int              # Total files found
}
```

**Example:**
```python
result = companion.ingest_directory(
    "./docs",
    pattern="*.md",
    source="documentation"
)
print(f"Ingested {result['ingested']} files")
```

---

## Profile and Statistics Methods

### **7. `get_user_profile() -> Dict`**

Get comprehensive user profile.

**Returns:**
```python
{
    'user_id': str,                    # User ID
    'profile': Dict,                   # User profile
    'preferences': Dict,               # User preferences
    'learning_patterns': Dict,         # Learning patterns
    'conversation_stats': {
        'total_interactions': int,     # Total interactions
        'topics_learned': int,         # Topics learned
        'current_session_turns': int   # Current session turns
    },
    'personality': str                 # Personality type
}
```

**Example:**
```python
profile = companion.get_user_profile()
print(f"Topics learned: {profile['conversation_stats']['topics_learned']}")
```

---

### **8. `get_knowledge_stats() -> Dict`**

Get knowledge base statistics.

**Returns:**
```python
{
    'total_documents': int,    # Total documents in knowledge base
    'sources': Dict,           # Sources and their counts
    'rag_available': bool,     # Whether RAG is available
    'user_id': str            # User ID
}
```

**Example:**
```python
stats = companion.get_knowledge_stats()
print(f"Total documents: {stats['total_documents']}")
for source, count in stats['sources'].items():
    print(f"{source}: {count}")
```

---

## Learning Path Methods

### **9. `get_personalized_learning_path(goal: str) -> Dict`**

Get personalized learning path.

**Parameters:**
- `goal` (str): Learning goal

**Returns:**
```python
{
    'path': List[str],                 # Learning path
    'personalized_path': List[str],   # Filtered path
    'topics_already_learned': List[str], # Already learned
    'estimated_time_adjusted': str     # Estimated time
}
```

**Example:**
```python
path = companion.get_personalized_learning_path("become a data scientist")
print(f"Path: {' → '.join(path['personalized_path'])}")
```

---

### **10. `update_preference(preference_key: str, preference_value: Any) -> Dict`**

Update user preference.

**Parameters:**
- `preference_key` (str): Preference key
- `preference_value` (Any): Preference value

**Returns:**
```python
{
    'success': bool,          # Whether update succeeded
    'message': str,           # Status message
    'preferences': Dict       # Updated preferences
}
```

**Example:**
```python
result = companion.update_preference("preferred_pace", "fast")
print(result['message'])
```

---

## Session Management

### **11. `save_session() -> None`**

Save current session.

**Example:**
```python
companion.save_session()
print("Session saved!")
```

---

### **12. `greet_user() -> str`**

Get personalized greeting.

**Returns:**
- `str`: Greeting message

**Example:**
```python
greeting = companion.greet_user()
print(greeting)
```

---

## Inherited Methods

The `LLMTwinLearningCompanion` inherits from `AdvancedLearningCompanion`, which provides additional methods:

### **From AdvancedLearningCompanion:**

- `learn_concept(concept: str) -> Dict` - Basic concept learning
- `answer_question(question: str) -> Dict` - Basic question answering
- `suggest_personalized_path(goal: str) -> Dict` - Basic path suggestion
- `get_learning_progress() -> Dict` - Get learning progress

---

## Error Handling

All methods may raise exceptions. Wrap calls in try-except:

```python
try:
    result = companion.learn_concept_twin("machine learning")
    print(result['explanation'])
except Exception as e:
    print(f"Error: {e}")
```

---

## Return Value Patterns

### **Success Response:**
```python
{
    'success': True,
    'message': 'Operation completed',
    # ... other fields
}
```

### **Error Response:**
```python
{
    'success': False,
    'error': 'Error message',
    # ... other fields
}
```

---

## Type Hints

For better IDE support, the companion uses type hints:

```python
from typing import Dict, Optional, List, Any
```

---

## Constants

### **Personality Types:**
- `"helpful_mentor"` - Helpful and encouraging
- `"socratic_teacher"` - Asks questions to guide learning
- `"technical_expert"` - Focuses on technical details

---

## Notes

1. **Memory Persistence**: User data is saved automatically when `save_session()` is called
2. **RAG Integration**: RAG context is used automatically when available
3. **Thread Safety**: Not thread-safe; use one instance per thread
4. **Resource Management**: Save sessions before closing

---

**Complete API reference for LLM Twin Learning Companion**
