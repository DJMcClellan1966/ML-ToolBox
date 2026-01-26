# MindForge Integration Guide

## 🎯 Overview

This guide shows you how to connect MindForge to LLM Twin Learning Companion and easily add content.

---

## ✅ What's Available

### **1. MindForge Connector** ✅
**File**: `mindforge_connector.py`

**Features:**
- ✅ Connect to MindForge database
- ✅ Read knowledge items
- ✅ Search knowledge items
- ✅ Sync to LLM Twin
- ✅ Auto-detect MindForge database

### **2. Easy Content Ingestion** ✅
**File**: `easy_content_ingestion.py`

**Features:**
- ✅ Simple CLI tool
- ✅ Add text, files, directories
- ✅ Sync MindForge
- ✅ Batch operations
- ✅ Clipboard support

---

## 🚀 Quick Start

### **Method 1: Using Python**

```python
from llm_twin_learning_companion import LLMTwinLearningCompanion
from mindforge_connector import sync_mindforge_to_llm_twin

# Create companion
companion = LLMTwinLearningCompanion(user_id="your_name")

# Sync MindForge (auto-detects database)
result = companion.sync_mindforge()
print(f"Synced {result['synced']} items from MindForge")
```

### **Method 2: Using CLI**

```bash
# Sync MindForge
python easy_content_ingestion.py mindforge

# Add a file
python easy_content_ingestion.py file notes.txt --source documents

# Add a directory
python easy_content_ingestion.py dir ./docs --pattern "*.md"

# Add text
python easy_content_ingestion.py text "Your content here..." --source notes

# Get stats
python easy_content_ingestion.py stats
```

---

## 📚 Detailed Usage

### **1. MindForge Connector**

#### **Basic Connection**

```python
from mindforge_connector import MindForgeConnector

# Auto-detect database
connector = MindForgeConnector()

# Or specify path
connector = MindForgeConnector(mindforge_db_path="/path/to/mindforge.db")
```

#### **Get Knowledge Items**

```python
# Get all items
items = connector.get_all_knowledge_items(limit=10)

# Get by type
notes = connector.get_knowledge_items_by_type("note")
articles = connector.get_knowledge_items_by_type("article")

# Search
results = connector.search_knowledge_items("machine learning", limit=5)
```

#### **Sync to LLM Twin**

```python
from llm_twin_learning_companion import LLMTwinLearningCompanion

companion = LLMTwinLearningCompanion(user_id="your_name")
connector = MindForgeConnector()

# Sync all items
result = connector.sync_to_llm_twin(companion)
print(f"Synced {result['synced']} items")

# Sync specific types
result = connector.sync_to_llm_twin(
    companion,
    content_types=["note", "article"]
)
```

#### **Get Statistics**

```python
stats = connector.get_stats()
print(f"Total items: {stats['total_items']}")
print(f"By type: {stats['by_type']}")
```

---

### **2. Easy Content Ingestion**

#### **Using the Helper Class**

```python
from easy_content_ingestion import EasyIngestion

ingestion = EasyIngestion(user_id="your_name")

# Add text
result = ingestion.add_text("Your content...", source="notes")

# Add file
result = ingestion.add_file("file.txt", source="documents")

# Add directory
result = ingestion.add_directory("./docs", pattern="*.md")

# Sync MindForge
result = ingestion.sync_mindforge()

# Get stats
stats = ingestion.get_stats()

# Save
ingestion.save()
```

#### **Using CLI**

```bash
# Add text
python easy_content_ingestion.py text "Your content..." --source notes

# Add file
python easy_content_ingestion.py file notes.txt --source documents

# Add directory
python easy_content_ingestion.py dir ./docs --pattern "*.md" --source docs

# Sync MindForge
python easy_content_ingestion.py mindforge

# Sync specific types
python easy_content_ingestion.py mindforge --types note article

# Add from clipboard
python easy_content_ingestion.py clipboard --source notes

# Batch add from JSON
python easy_content_ingestion.py batch items.json

# Get stats
python easy_content_ingestion.py stats
```

#### **Batch JSON Format**

```json
[
  {
    "type": "text",
    "content": "Your text content...",
    "source": "notes"
  },
  {
    "type": "file",
    "path": "file.txt",
    "source": "documents"
  },
  {
    "type": "directory",
    "path": "./docs",
    "pattern": "*.md",
    "source": "documentation"
  }
]
```

---

## 🔧 Integration Examples

### **Example 1: Daily Sync**

```python
from llm_twin_learning_companion import LLMTwinLearningCompanion
from easy_content_ingestion import EasyIngestion

# Daily sync script
companion = LLMTwinLearningCompanion(user_id="daily_user")

# Sync MindForge
result = companion.sync_mindforge()
print(f"Synced {result['synced']} items from MindForge")

# Add new files
ingestion = EasyIngestion(user_id="daily_user")
ingestion.add_directory("./daily_notes", pattern="*.md", source="daily")
ingestion.save()
```

### **Example 2: Selective Sync**

```python
from mindforge_connector import MindForgeConnector
from llm_twin_learning_companion import LLMTwinLearningCompanion

companion = LLMTwinLearningCompanion(user_id="selective_user")
connector = MindForgeConnector()

# Get only articles
articles = connector.get_knowledge_items_by_type("article")

# Sync only articles
for article in articles:
    content = f"{article['title']}\n\n{article['content']}"
    companion.ingest_text(
        content,
        source=f"mindforge_article_{article['id']}",
        metadata={'mindforge_id': article['id']}
    )

companion.save_session()
```

### **Example 3: Search and Sync**

```python
from mindforge_connector import MindForgeConnector
from llm_twin_learning_companion import LLMTwinLearningCompanion

companion = LLMTwinLearningCompanion(user_id="search_user")
connector = MindForgeConnector()

# Search for specific topics
results = connector.search_knowledge_items("machine learning", limit=10)

# Sync search results
for item in results:
    content = f"{item['title']}\n\n{item['content']}"
    companion.ingest_text(
        content,
        source=f"mindforge_search_{item['id']}",
        metadata={'mindforge_id': item['id'], 'tags': item.get('tags', [])}
    )

companion.save_session()
```

---

## 📋 Auto-Detection

The MindForge connector automatically tries to find your MindForge database in these locations:

1. `~/OneDrive/Desktop/mindforge/mindforge.db`
2. `~/OneDrive/Desktop/mindforge/data/mindforge.db`
3. `~/Desktop/mindforge/mindforge.db`
4. `~/Desktop/mindforge/data/mindforge.db`
5. `./mindforge.db`
6. `./data/mindforge.db`

If not found, specify the path:

```python
connector = MindForgeConnector(mindforge_db_path="/path/to/mindforge.db")
```

---

## 🎯 Best Practices

### **1. Regular Syncing**

```python
# Sync MindForge daily
companion.sync_mindforge()
```

### **2. Organize by Source**

```python
# Use consistent source names
companion.ingest_text(content, source="mindforge_notes")
companion.ingest_text(content, source="mindforge_articles")
```

### **3. Filter by Type**

```python
# Sync only specific types
companion.sync_mindforge(content_types=["note", "article"])
```

### **4. Save Sessions**

```python
# Always save after syncing
companion.save_session()
```

---

## 🐛 Troubleshooting

### **Issue: Database not found**
**Solution**: Specify the path explicitly:
```python
connector = MindForgeConnector(mindforge_db_path="/path/to/mindforge.db")
```

### **Issue: SQLAlchemy not available**
**Solution**: Install SQLAlchemy:
```bash
pip install sqlalchemy
```

### **Issue: Sync fails**
**Solution**: Check database permissions and path. Ensure MindForge database exists.

---

## 📊 Statistics

### **Get MindForge Stats**

```python
connector = MindForgeConnector()
stats = connector.get_stats()
print(f"Total items: {stats['total_items']}")
print(f"By type: {stats['by_type']}")
```

### **Get LLM Twin Stats**

```python
companion = LLMTwinLearningCompanion(user_id="your_name")
stats = companion.get_knowledge_stats()
print(f"Total documents: {stats['total_documents']}")
print(f"Sources: {stats['sources']}")
```

---

## ✅ Checklist

- [ ] SQLAlchemy installed (`pip install sqlalchemy`)
- [ ] MindForge database accessible
- [ ] Can connect to MindForge
- [ ] Can sync knowledge items
- [ ] Can use easy ingestion CLI
- [ ] Content appears in LLM Twin

---

## 🎉 You're Done!

**What you have:**
- ✅ MindForge connector
- ✅ Easy content ingestion
- ✅ CLI tools
- ✅ Python API
- ✅ Auto-detection

**Start using it:**
```bash
python easy_content_ingestion.py mindforge
```

Then ask your companion questions about your MindForge content!

---

**Your MindForge knowledge is now connected to LLM Twin!**
