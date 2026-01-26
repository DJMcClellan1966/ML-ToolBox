# LLM Twin Learning Companion - Simple Examples

## 📚 Table of Contents

1. [Basic Chat](#basic-chat)
2. [Learning Concepts](#learning-concepts)
3. [Content Ingestion](#content-ingestion)
4. [Personalized Learning](#personalized-learning)
5. [Complete Workflow](#complete-workflow)

---

## 1. Basic Chat

### **Simple Conversation**

```python
from llm_twin_learning_companion import LLMTwinLearningCompanion

# Create companion
companion = LLMTwinLearningCompanion(user_id="alice")

# Start conversation
response = companion.continue_conversation("Hello! Can you help me learn Python?")
print(response['answer'])

# Continue conversation
response = companion.continue_conversation("What about data structures?")
print(response['answer'])

# The companion remembers the context!
```

### **Ask Questions**

```python
companion = LLMTwinLearningCompanion(user_id="bob")

# Ask a question
result = companion.answer_question_twin("How does gradient descent work?")
print(f"Answer: {result['answer']}")

# Check if RAG context was used
if result.get('rag_context'):
    print("Used knowledge from your content!")
```

---

## 2. Learning Concepts

### **Learn a Single Concept**

```python
companion = LLMTwinLearningCompanion(user_id="charlie")

# Learn a concept
result = companion.learn_concept_twin("machine learning")

print(f"Concept: {result['concept']}")
print(f"Explanation: {result['explanation']}")

# Check if it's a review
if result.get('is_review'):
    print("You've learned this before!")

# See related knowledge
if result.get('related_knowledge'):
    print(f"Related: {', '.join(result['related_knowledge'])}")
```

### **Learn Multiple Concepts**

```python
companion = LLMTwinLearningCompanion(user_id="diana")

concepts = ["neural networks", "deep learning", "backpropagation"]

for concept in concepts:
    result = companion.learn_concept_twin(concept)
    print(f"\n✅ Learned: {concept}")
    print(f"   {result['explanation'][:100]}...")
```

### **Check Learning Progress**

```python
companion = LLMTwinLearningCompanion(user_id="eve")

# Get user profile
profile = companion.get_user_profile()

print(f"Topics learned: {profile['conversation_stats']['topics_learned']}")
print(f"Total interactions: {profile['conversation_stats']['total_interactions']}")
```

---

## 3. Content Ingestion

### **Add Text Content**

```python
companion = LLMTwinLearningCompanion(user_id="frank")

# Add a single text
notes = """
Machine learning is a subset of artificial intelligence that focuses
on algorithms that can learn from data without being explicitly programmed.
"""

result = companion.ingest_text(notes, source="my_notes")
print(result['message'])  # "Content added to knowledge base (123 characters)"
```

### **Add Multiple Texts**

```python
companion = LLMTwinLearningCompanion(user_id="grace")

texts = [
    ("Python is a high-level programming language.", "python_basics"),
    ("NumPy is a library for numerical computing.", "python_libraries"),
    ("Pandas is used for data manipulation.", "python_libraries")
]

for text, source in texts:
    result = companion.ingest_text(text, source=source)
    print(f"✅ Added: {source}")
```

### **Add a File**

```python
companion = LLMTwinLearningCompanion(user_id="henry")

# Add a single file
result = companion.ingest_file("notes.txt", source="documents")
print(result['message'])  # "File 'notes.txt' added to knowledge base"
```

### **Add All Files from Directory**

```python
companion = LLMTwinLearningCompanion(user_id="iris")

# Add all markdown files
result = companion.ingest_directory(
    directory_path="./docs",
    pattern="*.md",
    source="documentation"
)

print(f"✅ Ingested {result['ingested']} files")
print(f"❌ Errors: {result['errors']}")
```

### **Get Knowledge Statistics**

```python
companion = LLMTwinLearningCompanion(user_id="jack")

# Get stats
stats = companion.get_knowledge_stats()

print(f"Total documents: {stats['total_documents']}")
print(f"Sources:")
for source, count in stats['sources'].items():
    print(f"  {source}: {count} documents")
```

---

## 4. Personalized Learning

### **Get Learning Path**

```python
companion = LLMTwinLearningCompanion(user_id="kate")

# Get personalized path
path = companion.get_personalized_learning_path("become a data scientist")

print(f"Path: {' → '.join(path['personalized_path'])}")
print(f"Estimated time: {path['estimated_time_adjusted']}")
print(f"Already learned: {path['topics_already_learned']}")
```

### **Update Preferences**

```python
companion = LLMTwinLearningCompanion(user_id="liam")

# Update learning pace
result = companion.update_preference("preferred_pace", "fast")
print(result['message'])

# Update learning style
result = companion.update_preference("learning_style", "visual")
print(result['message'])
```

### **Save Session**

```python
companion = LLMTwinLearningCompanion(user_id="mia")

# Do some learning
companion.learn_concept_twin("reinforcement learning")
companion.continue_conversation("Tell me more about Q-learning")

# Save session (remembers everything!)
companion.save_session()
print("Session saved! The companion will remember you next time.")
```

---

## 5. Complete Workflow

### **Full Learning Session**

```python
from llm_twin_learning_companion import LLMTwinLearningCompanion

# Initialize
companion = LLMTwinLearningCompanion(
    user_id="student_001",
    personality_type="helpful_mentor"
)

# Step 1: Add your content
print("📥 Adding content...")
companion.ingest_text(
    "Machine learning uses algorithms to find patterns in data.",
    source="textbook"
)
companion.ingest_file("notes.md", source="my_notes")

# Step 2: Learn concepts
print("\n📚 Learning concepts...")
concepts = ["supervised learning", "unsupervised learning", "reinforcement learning"]
for concept in concepts:
    result = companion.learn_concept_twin(concept)
    print(f"  ✅ {concept}")

# Step 3: Ask questions
print("\n❓ Asking questions...")
questions = [
    "What's the difference between supervised and unsupervised learning?",
    "When would I use reinforcement learning?",
    "Can you give me examples?"
]

for question in questions:
    result = companion.answer_question_twin(question)
    print(f"\nQ: {question}")
    print(f"A: {result['answer'][:200]}...")

# Step 4: Get personalized path
print("\n🗺️ Getting learning path...")
path = companion.get_personalized_learning_path("master machine learning")
print(f"Path: {' → '.join(path['personalized_path'][:5])}...")

# Step 5: Check progress
print("\n📊 Progress:")
profile = companion.get_user_profile()
print(f"  Topics learned: {profile['conversation_stats']['topics_learned']}")
print(f"  Interactions: {profile['conversation_stats']['total_interactions']}")

# Step 6: Save session
print("\n💾 Saving session...")
companion.save_session()
print("Done! Your companion will remember everything next time.")
```

### **Content-First Workflow**

```python
companion = LLMTwinLearningCompanion(user_id="researcher")

# 1. Add all your content first
print("Adding content...")
companion.ingest_directory("./research_papers", pattern="*.md", source="papers")
companion.ingest_directory("./notes", pattern="*.txt", source="notes")

# 2. Check what you have
stats = companion.get_knowledge_stats()
print(f"\nKnowledge base: {stats['total_documents']} documents")

# 3. Ask questions about your content
print("\nAsking questions about your content...")
questions = [
    "What are the main findings in my research papers?",
    "Summarize my notes on neural networks",
    "What did I write about transformers?"
]

for question in questions:
    result = companion.answer_question_twin(question)
    print(f"\nQ: {question}")
    print(f"A: {result['answer'][:300]}...")
    
    # Check if RAG found relevant content
    if result.get('rag_context'):
        print("  [Used content from your knowledge base]")
```

---

## 🎯 Common Patterns

### **Pattern 1: Daily Learning Routine**

```python
companion = LLMTwinLearningCompanion(user_id="daily_learner")

# Morning: Learn new concept
companion.learn_concept_twin("attention mechanisms")

# Afternoon: Review and ask questions
companion.continue_conversation("Can you explain attention in simpler terms?")

# Evening: Save progress
companion.save_session()
```

### **Pattern 2: Research Assistant**

```python
companion = LLMTwinLearningCompanion(user_id="researcher")

# Add research content
companion.ingest_directory("./research", pattern="*.md")

# Ask research questions
result = companion.answer_question_twin("What are the key themes in my research?")
print(result['answer'])
```

### **Pattern 3: Study Buddy**

```python
companion = LLMTwinLearningCompanion(user_id="student")

# Add study materials
companion.ingest_file("textbook_chapter_1.md", source="textbook")
companion.ingest_file("lecture_notes.md", source="lectures")

# Study session
topics = ["topic1", "topic2", "topic3"]
for topic in topics:
    companion.learn_concept_twin(topic)
    companion.continue_conversation(f"Give me practice questions about {topic}")

# Save study session
companion.save_session()
```

---

## 💡 Tips

1. **Use consistent source names** - Makes it easier to track content
2. **Save sessions regularly** - Preserves your learning progress
3. **Add content before asking questions** - Better RAG results
4. **Use personalized paths** - Get tailored learning recommendations
5. **Check knowledge stats** - See what content you've added

---

**Start with the basic examples and build from there!**
