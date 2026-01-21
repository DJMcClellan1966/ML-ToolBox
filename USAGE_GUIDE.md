# Using This as Your Own LLM System

## What You Have

A complete, self-contained AI system that includes:
- **Semantic Understanding** - Understand text meaning and context
- **Knowledge Graph** - Build and query knowledge relationships
- **Intelligent Search** - Semantic search capabilities
- **LLM Generation** - Text generation with progressive learning
- **Conversational AI** - Context-aware conversations
- **No API Keys Required** - Fully local, private, and free

---

## Quick Start: Basic LLM Usage

### Simple Text Generation

```python
from quantum_kernel import get_kernel, KernelConfig
from llm.quantum_llm_standalone import StandaloneQuantumLLM

# Initialize
kernel = get_kernel(KernelConfig(use_sentence_transformers=True))
llm = StandaloneQuantumLLM(kernel=kernel)

# Generate text
prompt = "Explain how machine learning works"
response = llm.generate_grounded(
    prompt=prompt,
    max_length=200,
    temperature=0.7
)

print(response['text'])
```

---

## Use Cases for Your Projects

### 1. Document Understanding & Q&A

```python
from ai import CompleteAISystem

# Initialize complete AI system
ai = CompleteAISystem(use_llm=True)

# Add your documents
documents = [
    "Project Alpha uses machine learning for data analysis",
    "Project Beta focuses on web development with React",
    "Project Gamma implements quantum computing research"
]

# Add to knowledge base
for doc in documents:
    ai.knowledge_graph.add_document(doc)

# Ask questions
question = "What does Project Alpha do?"
results = ai.search.semantic_search(question, top_k=3)
print(results)
```

### 2. Code Documentation Generator

```python
from llm.quantum_llm_standalone import StandaloneQuantumLLM

llm = StandaloneQuantumLLM(kernel=get_kernel())

# Generate code documentation
code = "def calculate_total(items): return sum(item.price for item in items)"

prompt = f"Generate clear documentation for this Python function: {code}"
docs = llm.generate_grounded(prompt, max_length=150)

print(docs['text'])
```

### 3. Semantic Code Search

```python
from quantum_kernel import get_kernel

kernel = get_kernel()

# Your code snippets
code_snippets = [
    "def add(a, b): return a + b",
    "function multiply(x, y) { return x * y }",
    "const divide = (num, den) => num / den"
]

# Search semantically
query = "code that adds two numbers together"
similar = kernel.find_similar(query, code_snippets, top_k=2)

for snippet, score in similar:
    print(f"[{score:.3f}] {snippet}")
```

### 4. Project Knowledge Base

```python
from ai import CompleteAISystem

ai = CompleteAISystem(use_llm=True)

# Build knowledge base from your project notes
project_notes = [
    "The API uses FastAPI framework",
    "Database is PostgreSQL with SQLAlchemy ORM",
    "Authentication uses JWT tokens",
    "Frontend is React with TypeScript"
]

# Add all notes
for note in project_notes:
    ai.knowledge_graph.add_document(note)

# Discover relationships
relationships = ai.knowledge_graph.build_graph(project_notes)
print("Related concepts:")
for concept, related in relationships.items():
    print(f"{concept} -> {[r[0][:30] for r in related[:3]]}")
```

### 5. Text Summarization

```python
from llm.quantum_llm_standalone import StandaloneQuantumLLM

llm = StandaloneQuantumLLM(kernel=get_kernel())

long_text = """
Machine learning is a subset of artificial intelligence that enables 
systems to learn and improve from experience without being explicitly 
programmed. It focuses on developing algorithms that can analyze data, 
identify patterns, and make predictions or decisions. Deep learning, 
a subset of machine learning, uses neural networks with multiple layers 
to model and understand complex patterns in data.
"""

prompt = f"Summarize this text in 2-3 sentences: {long_text}"
summary = llm.generate_grounded(prompt, max_length=100)

print(summary['text'])
```

### 6. Content Generation for Projects

```python
from llm.quantum_llm_standalone import StandaloneQuantumLLM

llm = StandaloneQuantumLLM(kernel=get_kernel())

# Generate README sections
prompt = "Write a concise installation section for a Python project"
install_section = llm.generate_grounded(prompt, max_length=150)

# Generate API documentation
prompt = "Write documentation for a REST API endpoint that returns user data"
api_docs = llm.generate_grounded(prompt, max_length=200)

print(install_section['text'])
print("\n" + "="*50 + "\n")
print(api_docs['text'])
```

---

## Complete AI System Integration

### Full-Featured AI Assistant

```python
from ai import CompleteAISystem

# Initialize with LLM support
ai = CompleteAISystem(use_llm=True)

# Train on your project context
project_context = """
This project is a web application for task management.
Technologies: Python, FastAPI, React, PostgreSQL
Features: User authentication, task CRUD, notifications
"""

ai.knowledge_graph.add_document(project_context)

# Conversational AI
response = ai.conversation.chat(
    "How does user authentication work in this project?",
    context=[project_context]
)

print(response)
```

---

## Advantages of Your Own LLM

### ✅ **Privacy**
- All processing happens locally
- No data sent to external APIs
- Complete control over your data

### ✅ **No Costs**
- No API usage fees
- No rate limits
- Unlimited usage

### ✅ **Customizable**
- Train on your specific domain
- Adapt to your use cases
- Full source code access

### ✅ **Offline Capable**
- Works without internet
- Self-contained system
- No dependencies on external services

---

## When to Use This vs. External LLMs

### Use Your LLM When:
- ✅ Privacy is important
- ✅ Working with sensitive/confidential data
- ✅ Need unlimited usage without costs
- ✅ Want to customize/train specifically
- ✅ Need offline capabilities
- ✅ Building internal tools

### Use External LLMs (GPT, Claude) When:
- ✅ Need highest quality generation
- ✅ Need large context windows
- ✅ Working with general knowledge
- ✅ Need latest information
- ✅ Quality > privacy

---

## Project Examples

### Example 1: Documentation Generator Tool

```python
# doc_generator.py
from llm.quantum_llm_standalone import StandaloneQuantumLLM
from quantum_kernel import get_kernel

class DocGenerator:
    def __init__(self):
        self.llm = StandaloneQuantumLLM(kernel=get_kernel())
    
    def generate_docs(self, code, doc_type="function"):
        prompt = f"Generate {doc_type} documentation: {code}"
        return self.llm.generate_grounded(prompt, max_length=200)
    
    def explain_code(self, code):
        prompt = f"Explain what this code does: {code}"
        return self.llm.generate_grounded(prompt, max_length=300)

# Use it
gen = DocGenerator()
docs = gen.generate_docs("def process(data): return data * 2")
print(docs['text'])
```

### Example 2: Semantic Search for Codebase

```python
# code_search.py
from quantum_kernel import get_kernel

class CodeSearch:
    def __init__(self):
        self.kernel = get_kernel()
        self.code_index = []
    
    def index_code(self, file_path, code):
        self.code_index.append((file_path, code))
    
    def search(self, query, top_k=5):
        codes = [code for _, code in self.code_index]
        results = self.kernel.find_similar(query, codes, top_k=top_k)
        return [(self.code_index[i][0], code, score) 
                for code, score in results]

# Use it
search = CodeSearch()
search.index_code("utils.py", "def calculate_total(items): return sum(items)")
search.index_code("api.py", "def get_user(id): return db.query(User).filter(id=id)")

results = search.search("function that gets data from database")
for file, code, score in results:
    print(f"{file}: {code} [{score:.3f}]")
```

### Example 3: Project Knowledge Assistant

```python
# project_assistant.py
from ai import CompleteAISystem

class ProjectAssistant:
    def __init__(self):
        self.ai = CompleteAISystem(use_llm=True)
        self.documents = []
    
    def add_project_info(self, info):
        """Add project information"""
        self.ai.knowledge_graph.add_document(info)
        self.documents.append(info)
    
    def ask(self, question):
        """Ask questions about your project"""
        results = self.ai.search.semantic_search(question, top_k=3)
        return results
    
    def chat(self, message):
        """Conversational interface"""
        context = " ".join(self.documents[-5:])  # Recent context
        return self.ai.conversation.chat(message, context=[context])

# Use it
assistant = ProjectAssistant()
assistant.add_project_info("This project uses FastAPI for the backend")
assistant.add_project_info("Authentication is handled via JWT tokens")

answer = assistant.ask("What framework is used for the backend?")
print(answer)
```

---

## API Structure

### Simple API Wrapper

```python
# simple_api.py
from fastapi import FastAPI
from llm.quantum_llm_standalone import StandaloneQuantumLLM
from quantum_kernel import get_kernel

app = FastAPI()
llm = StandaloneQuantumLLM(kernel=get_kernel())

@app.post("/generate")
def generate_text(prompt: str, max_length: int = 200):
    result = llm.generate_grounded(prompt, max_length=max_length)
    return {"text": result['text']}

@app.post("/search")
def search(query: str, documents: list):
    kernel = get_kernel()
    results = kernel.find_similar(query, documents, top_k=5)
    return {"results": results}

# Run: uvicorn simple_api:app --port 8000
```

---

## Next Steps

1. **Choose a use case** from above
2. **Adapt the examples** to your needs
3. **Train on your data** by adding documents to knowledge graph
4. **Iterate** - improve prompts and parameters

---

## Tips for Best Results

1. **Provide context** - Add relevant documents to knowledge base
2. **Use specific prompts** - More specific = better results
3. **Iterate on prompts** - Try different phrasings
4. **Combine components** - Use AI system + LLM together
5. **Train progressively** - Add documents over time for better results

---

**You have a fully functional LLM system that's yours to customize and use!**
