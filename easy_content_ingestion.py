"""
Easy Content Ingestion for LLM Twin Learning Companion

Simple CLI and helper functions for easy content ingestion.
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict
import argparse
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_twin_learning_companion import LLMTwinLearningCompanion
from mindforge_connector import MindForgeConnector, sync_mindforge_to_llm_twin

class EasyIngestion:
    """Easy content ingestion helper"""
    
    def __init__(self, user_id: str = "default_user"):
        """Initialize with companion"""
        self.companion = LLMTwinLearningCompanion(user_id=user_id)
        self.user_id = user_id
    
    def add_text(self, text: str, source: str = "cli_input") -> Dict:
        """Add text content"""
        return self.companion.ingest_text(text, source=source)
    
    def add_file(self, file_path: str, source: Optional[str] = None) -> Dict:
        """Add file content"""
        return self.companion.ingest_file(file_path, source=source)
    
    def add_directory(self, directory_path: str, pattern: str = "*.md", source: Optional[str] = None) -> Dict:
        """Add all files from directory"""
        return self.companion.ingest_directory(directory_path, pattern=pattern, source=source)
    
    def sync_mindforge(self, mindforge_db_path: Optional[str] = None, content_types: Optional[List[str]] = None) -> Dict:
        """Sync MindForge knowledge to LLM Twin"""
        try:
            return sync_mindforge_to_llm_twin(
                self.companion,
                mindforge_db_path=mindforge_db_path,
                content_types=content_types
            )
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Failed to sync MindForge: {e}'
            }
    
    def add_from_clipboard(self, source: str = "clipboard") -> Dict:
        """Add content from clipboard (if available)"""
        try:
            import pyperclip
            text = pyperclip.paste()
            if text:
                return self.companion.ingest_text(text, source=source)
            else:
                return {'success': False, 'error': 'Clipboard is empty'}
        except ImportError:
            return {'success': False, 'error': 'pyperclip not installed. Install with: pip install pyperclip'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def batch_add(self, items: List[Dict]) -> Dict:
        """
        Batch add multiple items
        
        Args:
            items: List of dicts with 'type', 'content'/'path', and optional 'source'
        
        Returns:
            Dict with batch results
        """
        results = {
            'success': 0,
            'failed': 0,
            'details': []
        }
        
        for item in items:
            item_type = item.get('type', 'text')
            source = item.get('source', 'batch')
            
            try:
                if item_type == 'text':
                    result = self.companion.ingest_text(item['content'], source=source)
                elif item_type == 'file':
                    result = self.companion.ingest_file(item['path'], source=source)
                elif item_type == 'directory':
                    pattern = item.get('pattern', '*.md')
                    result = self.companion.ingest_directory(item['path'], pattern=pattern, source=source)
                else:
                    result = {'success': False, 'error': f'Unknown type: {item_type}'}
                
                if result.get('success'):
                    results['success'] += 1
                else:
                    results['failed'] += 1
                
                results['details'].append(result)
            except Exception as e:
                results['failed'] += 1
                results['details'].append({'success': False, 'error': str(e)})
        
        return results
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        return self.companion.get_knowledge_stats()
    
    def save(self):
        """Save session"""
        self.companion.save_session()


def main():
    """CLI for easy content ingestion"""
    parser = argparse.ArgumentParser(description='Easy Content Ingestion for LLM Twin')
    parser.add_argument('--user', default='default_user', help='User ID')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Add text
    text_parser = subparsers.add_parser('text', help='Add text content')
    text_parser.add_argument('content', help='Text content to add')
    text_parser.add_argument('--source', default='cli_input', help='Source identifier')
    
    # Add file
    file_parser = subparsers.add_parser('file', help='Add file content')
    file_parser.add_argument('path', help='Path to file')
    file_parser.add_argument('--source', help='Source identifier')
    
    # Add directory
    dir_parser = subparsers.add_parser('dir', help='Add directory content')
    dir_parser.add_argument('path', help='Path to directory')
    dir_parser.add_argument('--pattern', default='*.md', help='File pattern (default: *.md)')
    dir_parser.add_argument('--source', help='Source identifier')
    
    # Sync MindForge
    mindforge_parser = subparsers.add_parser('mindforge', help='Sync MindForge knowledge')
    mindforge_parser.add_argument('--db-path', help='Path to MindForge database')
    mindforge_parser.add_argument('--types', nargs='+', help='Content types to sync (e.g., note article)')
    
    # Add from clipboard
    clipboard_parser = subparsers.add_parser('clipboard', help='Add content from clipboard')
    clipboard_parser.add_argument('--source', default='clipboard', help='Source identifier')
    
    # Batch add
    batch_parser = subparsers.add_parser('batch', help='Batch add from JSON file')
    batch_parser.add_argument('json_file', help='Path to JSON file with items')
    
    # Stats
    stats_parser = subparsers.add_parser('stats', help='Get knowledge base statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize
    ingestion = EasyIngestion(user_id=args.user)
    
    # Execute command
    if args.command == 'text':
        result = ingestion.add_text(args.content, source=args.source)
        print(f"Result: {result.get('message', 'Done')}")
    
    elif args.command == 'file':
        result = ingestion.add_file(args.path, source=args.source)
        print(f"Result: {result.get('message', 'Done')}")
    
    elif args.command == 'dir':
        result = ingestion.add_directory(args.path, pattern=args.pattern, source=args.source)
        print(f"Result: {result.get('message', 'Done')}")
        print(f"Ingested: {result.get('ingested', 0)} files")
    
    elif args.command == 'mindforge':
        result = ingestion.sync_mindforge(
            mindforge_db_path=args.db_path,
            content_types=args.types
        )
        print(f"Result: {result.get('message', 'Done')}")
        print(f"Synced: {result.get('synced', 0)} items")
        if result.get('errors', 0) > 0:
            print(f"Errors: {result.get('errors', 0)}")
    
    elif args.command == 'clipboard':
        result = ingestion.add_from_clipboard(source=args.source)
        if result.get('success'):
            print(f"Added {result.get('characters', 0)} characters from clipboard")
        else:
            print(f"Error: {result.get('error')}")
    
    elif args.command == 'batch':
        with open(args.json_file, 'r') as f:
            items = json.load(f)
        result = ingestion.batch_add(items)
        print(f"Success: {result['success']}, Failed: {result['failed']}")
    
    elif args.command == 'stats':
        stats = ingestion.get_stats()
        print(f"Total documents: {stats.get('total_documents', 0)}")
        if stats.get('sources'):
            print("Sources:")
            for source, count in stats['sources'].items():
                print(f"  {source}: {count}")
    
    # Save session
    ingestion.save()
    print("Session saved!")


if __name__ == "__main__":
    main()
