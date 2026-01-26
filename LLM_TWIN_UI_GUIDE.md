# LLM Twin Learning Companion - UI & Content Ingestion Guide

## ✅ What's Been Created

### **1. Web UI** ✅
**File**: `llm_twin_web_ui.py`

**Features**:
- ✅ Beautiful, modern web interface
- ✅ Chat with your learning companion
- ✅ Learn new concepts
- ✅ Content ingestion (text and files)
- ✅ Conversation history
- ✅ User profile
- ✅ Responsive design

### **2. Content Ingestion Methods** ✅
**Added to**: `llm_twin_learning_companion.py`

**New Methods**:
- ✅ `ingest_text()` - Add text content
- ✅ `ingest_file()` - Add file content
- ✅ `ingest_directory()` - Add all files from directory
- ✅ `get_knowledge_stats()` - Get knowledge base statistics

---

## 🚀 Quick Start

### **Step 1: Install Flask** (if not installed)
```bash
pip install flask
```

### **Step 2: Run the Web UI**
```bash
python llm_twin_web_ui.py
```

### **Step 3: Open Browser**
```
http://localhost:5000
```

---

## 📥 Content Ingestion

### **Method 1: Via Web UI**

1. **Add Text Content**:
   - Go to "Add Content" tab
   - Paste text in the text area
   - Optionally add a source/category
   - Click "Add Text"

2. **Upload File**:
   - Go to "Add Content" tab
   - Click the upload area or drag & drop
   - Select file (.txt, .md, .json, etc.)
   - File is automatically added to knowledge base

### **Method 2: Via Python API**

```python
from llm_twin_learning_companion import LLMTwinLearningCompanion

companion = LLMTwinLearningCompanion(user_id="your_name")

# Add text content
result = companion.ingest_text(
    text="Your content here...",
    source="my_notes",
    metadata={"category": "personal"}
)
print(result)

# Add a file
result = companion.ingest_file(
    file_path="path/to/file.txt",
    source="documentation"
)
print(result)

# Add all files from a directory
result = companion.ingest_directory(
    directory_path="path/to/documents",
    pattern="*.md",  # Only markdown files
    source="documentation"
)
print(result)

# Get knowledge stats
stats = companion.get_knowledge_stats()
print(f"Total documents: {stats['total_documents']}")
print(f"Sources: {stats['sources']}")
```

---

## 💡 Usage Examples

### **Example 1: Add Your Notes**
```python
companion = LLMTwinLearningCompanion(user_id="john")

# Add personal notes
notes = """
Machine learning is a subset of AI that focuses on algorithms
that can learn from data without being explicitly programmed.
"""

companion.ingest_text(notes, source="my_notes")
```

### **Example 2: Add Documentation**
```python
# Add all markdown files from a docs directory
companion.ingest_directory(
    directory_path="./docs",
    pattern="*.md",
    source="documentation"
)
```

### **Example 3: Add Multiple Files**
```python
files = ["notes1.txt", "notes2.txt", "article.md"]

for file_path in files:
    result = companion.ingest_file(file_path, source="articles")
    if result['success']:
        print(f"✅ Added {file_path}")
    else:
        print(f"❌ Error: {result['error']}")
```

---

## 🎨 Web UI Features

### **Chat Tab**
- Real-time conversation
- Context-aware responses
- RAG-enhanced answers
- Chain-of-thought reasoning

### **Learn Tab**
- Learn new concepts
- Get explanations
- See related knowledge
- Track learning progress

### **Add Content Tab**
- Add text directly
- Upload files
- Drag & drop support
- Multiple file formats

### **History Tab**
- View conversation history
- See past interactions
- Track learning journey

### **Profile Tab**
- View learning statistics
- See topics learned
- Check interaction count
- View preferences

---

## 📊 Knowledge Base Statistics

Get statistics about your knowledge base:

```python
stats = companion.get_knowledge_stats()

print(f"Total Documents: {stats['total_documents']}")
print(f"Sources:")
for source, count in stats['sources'].items():
    print(f"  {source}: {count} documents")
```

---

## 🔧 Supported File Types

### **Text Files**
- `.txt` - Plain text
- `.md` - Markdown
- `.py` - Python code
- `.js` - JavaScript
- `.html` - HTML
- `.css` - CSS

### **Data Files**
- `.json` - JSON data

### **Other**
- Any text-readable file (with error handling)

---

## 🎯 Best Practices

### **1. Organize by Source**
```python
# Use consistent source names
companion.ingest_text(content, source="my_notes")
companion.ingest_text(content, source="articles")
companion.ingest_text(content, source="documentation")
```

### **2. Add Metadata**
```python
companion.ingest_text(
    content,
    source="articles",
    metadata={
        "category": "machine_learning",
        "date": "2024-01-26",
        "author": "John Doe"
    }
)
```

### **3. Batch Ingest**
```python
# Use directory ingestion for multiple files
companion.ingest_directory(
    directory_path="./knowledge_base",
    pattern="*.md",
    source="knowledge_base"
)
```

---

## 🐛 Troubleshooting

### **Issue: RAG system not available**
**Solution**: Make sure `ml_toolbox.llm_engineering.rag_system` is importable.

### **Issue: File upload fails**
**Solution**: 
- Check file size (max 16MB)
- Ensure file is text-readable
- Check file encoding

### **Issue: Content not appearing in responses**
**Solution**:
- Verify content was added (check stats)
- Try more specific queries
- Check RAG retrieval is working

---

## 📝 API Endpoints

### **Chat**
- `POST /api/chat` - Send message
  ```json
  {"message": "Your question"}
  ```

### **Learn**
- `POST /api/learn` - Learn concept
  ```json
  {"concept": "machine learning"}
  ```

### **Ingest Text**
- `POST /api/ingest/text` - Add text content
  ```json
  {"content": "Your text...", "source": "my_notes"}
  ```

### **Ingest File**
- `POST /api/ingest/file` - Upload file
  - Form data with `file` field

### **History**
- `GET /api/history` - Get conversation history

### **Profile**
- `GET /api/profile` - Get user profile

### **Knowledge Stats**
- `GET /api/knowledge/stats` - Get knowledge base statistics

### **Save Session**
- `POST /api/save` - Save current session

---

## ✅ Success Checklist

- [ ] Flask installed
- [ ] Web UI runs without errors
- [ ] Can chat with companion
- [ ] Can learn concepts
- [ ] Can add text content
- [ ] Can upload files
- [ ] Can view history
- [ ] Can view profile
- [ ] Session saves correctly

---

## 🎉 You're Done!

**What you have:**
- ✅ Beautiful web UI for LLM Twin
- ✅ Content ingestion (text, files, directories)
- ✅ Knowledge base statistics
- ✅ Full API for programmatic access

**Start using it:**
```bash
python llm_twin_web_ui.py
```

Then open http://localhost:5000 in your browser!

---

**Your LLM Twin Learning Companion now has a UI and content ingestion!**
