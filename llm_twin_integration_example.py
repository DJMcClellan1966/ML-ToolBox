"""
LLM Twin Learning Companion - Integration Example

This example shows how to integrate the LLM Twin Companion into a real application.
Demonstrates Flask web app integration, file watching, and database logging.
"""

from llm_twin_learning_companion import LLMTwinLearningCompanion
from typing import Dict, Optional
import json
from pathlib import Path

# ============================================================================
# Example 1: Simple Application Integration
# ============================================================================

class MyApplication:
    """Example application with integrated LLM Twin"""
    
    def __init__(self, user_id: str):
        self.companion = LLMTwinLearningCompanion(user_id=user_id)
        self.app_name = "My Learning App"
    
    def handle_query(self, query: str) -> Dict:
        """Handle user query"""
        # Route based on query type
        if query.startswith("learn:"):
            concept = query.replace("learn:", "").strip()
            return self.companion.learn_concept_twin(concept)
        elif query.startswith("add:"):
            content = query.replace("add:", "").strip()
            return self.companion.ingest_text(content, source="app_input")
        else:
            return self.companion.continue_conversation(query)
    
    def get_app_stats(self) -> Dict:
        """Get application statistics"""
        profile = self.companion.get_user_profile()
        stats = self.companion.get_knowledge_stats()
        
        return {
            'app_name': self.app_name,
            'user_id': profile['user_id'],
            'topics_learned': profile['conversation_stats']['topics_learned'],
            'knowledge_documents': stats['total_documents'],
            'sources': stats.get('sources', {})
        }


# ============================================================================
# Example 2: Content Manager Integration
# ============================================================================

class ContentManager:
    """Manages content ingestion with LLM Twin"""
    
    def __init__(self, user_id: str):
        self.companion = LLMTwinLearningCompanion(user_id=user_id)
        self.ingestion_log = []
    
    def add_content_batch(self, contents: list) -> Dict:
        """Add multiple content items"""
        results = {
            'success': 0,
            'failed': 0,
            'details': []
        }
        
        for content, source in contents:
            try:
                result = self.companion.ingest_text(content, source=source)
                if result.get('success'):
                    results['success'] += 1
                    self.ingestion_log.append({
                        'source': source,
                        'status': 'success',
                        'timestamp': str(Path(__file__).stat().st_mtime)
                    })
                else:
                    results['failed'] += 1
                    self.ingestion_log.append({
                        'source': source,
                        'status': 'failed',
                        'error': result.get('error')
                    })
                results['details'].append(result)
            except Exception as e:
                results['failed'] += 1
                results['details'].append({'error': str(e)})
        
        return results
    
    def get_ingestion_summary(self) -> Dict:
        """Get summary of ingested content"""
        stats = self.companion.get_knowledge_stats()
        
        return {
            'total_documents': stats['total_documents'],
            'sources': stats.get('sources', {}),
            'ingestion_log_count': len(self.ingestion_log)
        }


# ============================================================================
# Example 3: Learning Path Generator
# ============================================================================

class LearningPathGenerator:
    """Generates and tracks learning paths"""
    
    def __init__(self, user_id: str):
        self.companion = LLMTwinLearningCompanion(user_id=user_id)
        self.paths = []
    
    def create_path(self, goal: str) -> Dict:
        """Create a learning path"""
        path = self.companion.get_personalized_learning_path(goal)
        
        path_record = {
            'goal': goal,
            'path': path['personalized_path'],
            'already_learned': path['topics_already_learned'],
            'estimated_time': path['estimated_time_adjusted']
        }
        
        self.paths.append(path_record)
        return path_record
    
    def track_progress(self, goal: str) -> Dict:
        """Track progress on a learning path"""
        # Find path for this goal
        path_record = next((p for p in self.paths if p['goal'] == goal), None)
        if not path_record:
            return {'error': 'Path not found'}
        
        # Get current profile
        profile = self.companion.get_user_profile()
        topics_learned = profile['conversation_stats']['topics_learned']
        
        # Calculate progress
        total_topics = len(path_record['path'])
        completed = sum(1 for topic in path_record['path'] if topic in topics_learned)
        progress_percent = (completed / total_topics * 100) if total_topics > 0 else 0
        
        return {
            'goal': goal,
            'total_topics': total_topics,
            'completed': completed,
            'remaining': total_topics - completed,
            'progress_percent': round(progress_percent, 2),
            'topics_remaining': [t for t in path_record['path'] if t not in topics_learned]
        }


# ============================================================================
# Example 4: Question Answering System
# ============================================================================

class QASystem:
    """Question answering system with knowledge base"""
    
    def __init__(self, user_id: str):
        self.companion = LLMTwinLearningCompanion(user_id=user_id)
        self.question_history = []
    
    def ask(self, question: str) -> Dict:
        """Ask a question"""
        result = self.companion.answer_question_twin(question)
        
        # Log question
        self.question_history.append({
            'question': question,
            'answer': result['answer'],
            'rag_used': result.get('rag_context', False),
            'timestamp': str(Path(__file__).stat().st_mtime)
        })
        
        return result
    
    def get_qa_stats(self) -> Dict:
        """Get QA statistics"""
        stats = self.companion.get_knowledge_stats()
        
        return {
            'total_questions': len(self.question_history),
            'rag_usage_count': sum(1 for q in self.question_history if q['rag_used']),
            'knowledge_base_docs': stats['total_documents'],
            'recent_questions': self.question_history[-5:] if len(self.question_history) > 0 else []
        }


# ============================================================================
# Main Demo
# ============================================================================

def main():
    print("="*80)
    print("LLM TWIN INTEGRATION EXAMPLES".center(80))
    print("="*80)
    print()
    
    # Example 1: Application Integration
    print("📱 Example 1: Application Integration")
    print("-" * 80)
    app = MyApplication("app_user")
    
    # Handle queries
    result = app.handle_query("What is machine learning?")
    print(f"Query: What is machine learning?")
    print(f"Response: {result['answer'][:100]}...")
    print()
    
    # Get stats
    stats = app.get_app_stats()
    print(f"App Stats: {stats}")
    print()
    
    # Example 2: Content Manager
    print("📚 Example 2: Content Manager")
    print("-" * 80)
    content_mgr = ContentManager("content_user")
    
    # Add batch content
    contents = [
        ("Python is a programming language.", "python_basics"),
        ("NumPy is for numerical computing.", "python_libraries"),
        ("Pandas is for data manipulation.", "python_libraries")
    ]
    
    result = content_mgr.add_content_batch(contents)
    print(f"Batch ingestion: {result['success']} success, {result['failed']} failed")
    print()
    
    summary = content_mgr.get_ingestion_summary()
    print(f"Content Summary: {summary}")
    print()
    
    # Example 3: Learning Path Generator
    print("🗺️ Example 3: Learning Path Generator")
    print("-" * 80)
    path_gen = LearningPathGenerator("path_user")
    
    # Create path
    path = path_gen.create_path("become a data scientist")
    print(f"Goal: become a data scientist")
    print(f"Path: {' → '.join(path['path'][:5])}...")
    print(f"Estimated time: {path['estimated_time']}")
    print()
    
    # Track progress
    progress = path_gen.track_progress("become a data scientist")
    print(f"Progress: {progress['progress_percent']}% complete")
    print(f"Completed: {progress['completed']}/{progress['total_topics']}")
    print()
    
    # Example 4: QA System
    print("❓ Example 4: Question Answering System")
    print("-" * 80)
    qa = QASystem("qa_user")
    
    # Add some knowledge first
    qa.companion.ingest_text(
        "Machine learning uses algorithms to find patterns in data.",
        source="qa_knowledge"
    )
    
    # Ask questions
    result = qa.ask("What is machine learning?")
    print(f"Q: What is machine learning?")
    print(f"A: {result['answer'][:100]}...")
    print(f"RAG used: {result.get('rag_context', False)}")
    print()
    
    # Get QA stats
    qa_stats = qa.get_qa_stats()
    print(f"QA Stats: {qa_stats}")
    print()
    
    print("="*80)
    print("INTEGRATION EXAMPLES COMPLETE!".center(80))
    print("="*80)
    print()
    print("These examples show how to integrate LLM Twin into your applications.")
    print("See LLM_TWIN_INTEGRATION.md for more details.")


if __name__ == "__main__":
    main()
