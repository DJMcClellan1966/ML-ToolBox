"""
MindForge Connector for LLM Twin Learning Companion

Connects to MindForge knowledge base and syncs content to LLM Twin.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker, Session
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger.warning("SQLAlchemy not available. Install with: pip install sqlalchemy")


class MindForgeConnector:
    """
    Connector to MindForge knowledge base
    
    Reads knowledge items from MindForge database and syncs to LLM Twin.
    """
    
    def __init__(self, mindforge_db_path: Optional[str] = None):
        """
        Initialize MindForge connector
        
        Args:
            mindforge_db_path: Path to MindForge database file
                             If None, tries to find it automatically
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy is required. Install with: pip install sqlalchemy")
        
        # Try to find MindForge database
        if mindforge_db_path is None:
            mindforge_db_path = self._find_mindforge_db()
        
        if mindforge_db_path is None:
            raise FileNotFoundError(
                "MindForge database not found. "
                "Please provide path to MindForge database file."
            )
        
        self.db_path = Path(mindforge_db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"MindForge database not found: {mindforge_db_path}")
        
        # Create database connection
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            connect_args={"check_same_thread": False}
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        logger.info(f"Connected to MindForge database: {self.db_path}")
    
    def _find_mindforge_db(self) -> Optional[str]:
        """Try to find MindForge database automatically"""
        # Common locations
        possible_paths = [
            Path.home() / "OneDrive" / "Desktop" / "mindforge" / "mindforge.db",
            Path.home() / "OneDrive" / "Desktop" / "mindforge" / "data" / "mindforge.db",
            Path.home() / "Desktop" / "mindforge" / "mindforge.db",
            Path.home() / "Desktop" / "mindforge" / "data" / "mindforge.db",
            Path(".") / "mindforge.db",
            Path(".") / "data" / "mindforge.db",
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def get_all_knowledge_items(self, user_id: Optional[int] = None, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all knowledge items from MindForge
        
        Args:
            user_id: Optional user ID to filter by
            limit: Optional limit on number of items
        
        Returns:
            List of knowledge item dictionaries
        """
        db = self.SessionLocal()
        try:
            # Query knowledge items
            query = db.execute(text("""
                SELECT id, user_id, title, content, content_type, 
                       source_url, tags, created_at, updated_at
                FROM knowledge_items
                WHERE is_archived = 0
            """))
            
            items = []
            for row in query:
                item = {
                    'id': row[0],
                    'user_id': row[1],
                    'title': row[2],
                    'content': row[3],
                    'content_type': row[4] or 'note',
                    'source_url': row[5],
                    'tags': row[6] if row[6] else [],
                    'created_at': row[7],
                    'updated_at': row[8]
                }
                
                # Filter by user_id if provided
                if user_id is None or item['user_id'] == user_id:
                    items.append(item)
                
                # Apply limit
                if limit and len(items) >= limit:
                    break
            
            return items
        except Exception as e:
            logger.error(f"Error getting knowledge items: {e}", exc_info=True)
            return []
        finally:
            db.close()
    
    def get_knowledge_items_by_type(self, content_type: str, user_id: Optional[int] = None) -> List[Dict]:
        """
        Get knowledge items by content type
        
        Args:
            content_type: Type of content (note, article, pdf, etc.)
            user_id: Optional user ID to filter by
        
        Returns:
            List of knowledge item dictionaries
        """
        db = self.SessionLocal()
        try:
            query = db.execute(text("""
                SELECT id, user_id, title, content, content_type, 
                       source_url, tags, created_at, updated_at
                FROM knowledge_items
                WHERE is_archived = 0 AND content_type = :content_type
            """), {"content_type": content_type})
            
            items = []
            for row in query:
                item = {
                    'id': row[0],
                    'user_id': row[1],
                    'title': row[2],
                    'content': row[3],
                    'content_type': row[4] or 'note',
                    'source_url': row[5],
                    'tags': row[6] if row[6] else [],
                    'created_at': row[7],
                    'updated_at': row[8]
                }
                
                if user_id is None or item['user_id'] == user_id:
                    items.append(item)
            
            return items
        except Exception as e:
            logger.error(f"Error getting knowledge items by type: {e}", exc_info=True)
            return []
        finally:
            db.close()
    
    def search_knowledge_items(self, query: str, user_id: Optional[int] = None, limit: int = 20) -> List[Dict]:
        """
        Search knowledge items by title or content
        
        Args:
            query: Search query
            user_id: Optional user ID to filter by
            limit: Maximum number of results
        
        Returns:
            List of matching knowledge item dictionaries
        """
        db = self.SessionLocal()
        try:
            search_term = f"%{query}%"
            sql = """
                SELECT id, user_id, title, content, content_type, 
                       source_url, tags, created_at, updated_at
                FROM knowledge_items
                WHERE is_archived = 0 
                  AND (title LIKE :search_term OR content LIKE :search_term)
            """
            
            params = {"search_term": search_term}
            if user_id is not None:
                sql += " AND user_id = :user_id"
                params["user_id"] = user_id
            
            sql += " LIMIT :limit"
            params["limit"] = limit
            
            query_result = db.execute(text(sql), params)
            
            items = []
            for row in query_result:
                item = {
                    'id': row[0],
                    'user_id': row[1],
                    'title': row[2],
                    'content': row[3],
                    'content_type': row[4] or 'note',
                    'source_url': row[5],
                    'tags': row[6] if row[6] else [],
                    'created_at': row[7],
                    'updated_at': row[8]
                }
                items.append(item)
            
            return items
        except Exception as e:
            logger.error(f"Error searching knowledge items: {e}", exc_info=True)
            return []
        finally:
            db.close()
    
    def sync_to_llm_twin(self, companion, user_id: Optional[int] = None, content_types: Optional[List[str]] = None) -> Dict:
        """
        Sync all MindForge knowledge items to LLM Twin companion
        
        Args:
            companion: LLMTwinLearningCompanion instance
            user_id: Optional user ID to filter by
            content_types: Optional list of content types to sync
        
        Returns:
            Dict with sync results
        """
        # Get all items
        if content_types:
            items = []
            for content_type in content_types:
                items.extend(self.get_knowledge_items_by_type(content_type, user_id))
        else:
            items = self.get_all_knowledge_items(user_id)
        
        if not items:
            return {
                'success': True,
                'message': 'No knowledge items found to sync',
                'synced': 0,
                'errors': 0
            }
        
        # Sync each item
        synced = 0
        errors = 0
        
        for item in items:
            try:
                # Combine title and content
                content = f"{item['title']}\n\n{item['content']}"
                
                # Create source identifier
                source = f"mindforge_{item['content_type']}"
                if item.get('tags'):
                    source += f"_{'_'.join(str(t) for t in item['tags'][:2])}"
                
                # Add metadata
                metadata = {
                    'mindforge_id': item['id'],
                    'content_type': item['content_type'],
                    'source_url': item.get('source_url'),
                    'tags': item.get('tags', []),
                    'created_at': str(item.get('created_at', ''))
                }
                
                # Ingest into LLM Twin
                result = companion.ingest_text(content, source=source, metadata=metadata)
                
                if result.get('success'):
                    synced += 1
                else:
                    errors += 1
                    logger.warning(f"Failed to sync item {item['id']}: {result.get('error')}")
            
            except Exception as e:
                errors += 1
                logger.error(f"Error syncing item {item['id']}: {e}", exc_info=True)
        
        return {
            'success': True,
            'message': f'Synced {synced} knowledge items from MindForge',
            'synced': synced,
            'errors': errors,
            'total': len(items)
        }
    
    def get_stats(self, user_id: Optional[int] = None) -> Dict:
        """
        Get statistics about MindForge knowledge base
        
        Args:
            user_id: Optional user ID to filter by
        
        Returns:
            Dict with statistics
        """
        db = self.SessionLocal()
        try:
            # Count total items
            sql = "SELECT COUNT(*) FROM knowledge_items WHERE is_archived = 0"
            params = {}
            if user_id is not None:
                sql += " AND user_id = :user_id"
                params["user_id"] = user_id
            
            result = db.execute(text(sql), params)
            total_items = result.fetchone()[0]
            
            # Count by type
            sql = """
                SELECT content_type, COUNT(*) 
                FROM knowledge_items 
                WHERE is_archived = 0
            """
            params = {}
            if user_id is not None:
                sql += " AND user_id = :user_id"
                params["user_id"] = user_id
            sql += " GROUP BY content_type"
            
            result = db.execute(text(sql), params)
            by_type = {row[0] or 'note': row[1] for row in result}
            
            return {
                'total_items': total_items,
                'by_type': by_type,
                'user_id': user_id
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            return {'error': str(e)}
        finally:
            db.close()
    
    def create_knowledge_item(
        self,
        user_id: int,
        title: str,
        content: str,
        content_type: str = "note",
        source_url: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict:
        """
        Create a new knowledge item in MindForge
        
        Args:
            user_id: User ID
            title: Item title
            content: Item content
            content_type: Type of content (note, article, learning, etc.)
            source_url: Optional source URL
            tags: Optional list of tags
        
        Returns:
            Dict with created item info or error
        """
        db = self.SessionLocal()
        try:
            now = datetime.now().isoformat()
            tags_str = ",".join(tags) if tags else None
            
            sql = """
                INSERT INTO knowledge_items 
                (user_id, title, content, content_type, source_url, tags, created_at, updated_at, is_archived)
                VALUES (:user_id, :title, :content, :content_type, :source_url, :tags, :created_at, :updated_at, 0)
            """
            
            params = {
                'user_id': user_id,
                'title': title,
                'content': content,
                'content_type': content_type,
                'source_url': source_url,
                'tags': tags_str,
                'created_at': now,
                'updated_at': now
            }
            
            result = db.execute(text(sql), params)
            db.commit()
            
            # Get the inserted ID
            item_id = result.lastrowid
            
            return {
                'success': True,
                'id': item_id,
                'message': f'Created knowledge item: {title}',
                'item': {
                    'id': item_id,
                    'user_id': user_id,
                    'title': title,
                    'content': content,
                    'content_type': content_type
                }
            }
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating knowledge item: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'message': f'Failed to create knowledge item: {e}'
            }
        finally:
            db.close()
    
    def sync_from_llm_twin(self, companion, user_id: Optional[int] = None) -> Dict:
        """
        Sync LLM Twin learnings back to MindForge (two-way sync)
        
        Args:
            companion: LLMTwinLearningCompanion instance
            user_id: Optional user ID (defaults to companion's user_id as int)
        
        Returns:
            Dict with sync results
        """
        try:
            # Get user profile to extract learned topics
            profile = companion.get_user_profile()
            topics_learned = profile.get('conversation_stats', {}).get('topics_learned', [])
            
            if not topics_learned:
                return {
                    'success': True,
                    'message': 'No topics learned to sync',
                    'synced': 0
                }
            
            # Get user_id
            if user_id is None:
                # Try to convert companion user_id to int, or use default
                try:
                    user_id = int(companion.user_id) if companion.user_id.isdigit() else 1
                except:
                    user_id = 1
            
            synced = 0
            errors = 0
            
            # Sync each learned topic
            for topic in topics_learned:
                try:
                    # Check if topic already exists
                    existing = self.search_knowledge_items(
                        f"Learned: {topic}",
                        user_id=user_id,
                        limit=1
                    )
                    
                    if existing:
                        # Already exists, skip
                        continue
                    
                    # Create new item
                    result = self.create_knowledge_item(
                        user_id=user_id,
                        title=f"Learned: {topic}",
                        content=f"Topic learned from LLM Twin Learning Companion.\n\nTopic: {topic}\n\nSynced from LLM Twin on {datetime.now().isoformat()}",
                        content_type="learning",
                        tags=["llm_twin", "learned", topic.lower().replace(" ", "_")]
                    )
                    
                    if result.get('success'):
                        synced += 1
                    else:
                        errors += 1
                        logger.warning(f"Failed to sync topic '{topic}': {result.get('error')}")
                
                except Exception as e:
                    errors += 1
                    logger.error(f"Error syncing topic '{topic}': {e}", exc_info=True)
            
            return {
                'success': True,
                'message': f'Synced {synced} learned topics to MindForge',
                'synced': synced,
                'errors': errors,
                'total': len(topics_learned)
            }
        
        except Exception as e:
            logger.error(f"Error syncing from LLM Twin: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'message': f'Failed to sync from LLM Twin: {e}'
            }
    
    def get_items_after(self, timestamp: str, user_id: Optional[int] = None) -> List[Dict]:
        """
        Get knowledge items added/updated after a timestamp (for incremental sync)
        
        Args:
            timestamp: ISO format timestamp
            user_id: Optional user ID to filter by
        
        Returns:
            List of knowledge items
        """
        db = self.SessionLocal()
        try:
            sql = """
                SELECT id, user_id, title, content, content_type, 
                       source_url, tags, created_at, updated_at
                FROM knowledge_items
                WHERE is_archived = 0 
                  AND (created_at > :timestamp OR updated_at > :timestamp)
            """
            
            params = {"timestamp": timestamp}
            if user_id is not None:
                sql += " AND user_id = :user_id"
                params["user_id"] = user_id
            
            sql += " ORDER BY updated_at DESC"
            
            query = db.execute(text(sql), params)
            
            items = []
            for row in query:
                item = {
                    'id': row[0],
                    'user_id': row[1],
                    'title': row[2],
                    'content': row[3],
                    'content_type': row[4] or 'note',
                    'source_url': row[5],
                    'tags': row[6] if row[6] else [],
                    'created_at': row[7],
                    'updated_at': row[8]
                }
                items.append(item)
            
            return items
        except Exception as e:
            logger.error(f"Error getting items after timestamp: {e}", exc_info=True)
            return []
        finally:
            db.close()
    
    def incremental_sync_to_llm_twin(
        self,
        companion,
        last_sync_time: Optional[str] = None,
        user_id: Optional[int] = None,
        content_types: Optional[List[str]] = None
    ) -> Dict:
        """
        Incremental sync: Only sync items added/updated since last sync
        
        Args:
            companion: LLMTwinLearningCompanion instance
            last_sync_time: ISO format timestamp of last sync (None = full sync)
            user_id: Optional user ID to filter by
            content_types: Optional list of content types to sync
        
        Returns:
            Dict with sync results
        """
        try:
            if last_sync_time is None:
                # No last sync time, do full sync
                return self.sync_to_llm_twin(companion, user_id, content_types)
            
            # Get only items after last sync
            items = self.get_items_after(last_sync_time, user_id)
            
            if content_types:
                items = [item for item in items if item.get('content_type') in content_types]
            
            if not items:
                return {
                    'success': True,
                    'message': 'No new items to sync',
                    'synced': 0,
                    'errors': 0
                }
            
            # Sync each item
            synced = 0
            errors = 0
            
            for item in items:
                try:
                    content = f"{item['title']}\n\n{item['content']}"
                    source = f"mindforge_{item['content_type']}"
                    if item.get('tags'):
                        source += f"_{'_'.join(str(t) for t in item['tags'][:2])}"
                    
                    metadata = {
                        'mindforge_id': item['id'],
                        'content_type': item['content_type'],
                        'source_url': item.get('source_url'),
                        'tags': item.get('tags', []),
                        'created_at': str(item.get('created_at', '')),
                        'updated_at': str(item.get('updated_at', ''))
                    }
                    
                    result = companion.ingest_text(content, source=source, metadata=metadata)
                    
                    if result.get('success'):
                        synced += 1
                    else:
                        errors += 1
                        logger.warning(f"Failed to sync item {item['id']}: {result.get('error')}")
                
                except Exception as e:
                    errors += 1
                    logger.error(f"Error syncing item {item['id']}: {e}", exc_info=True)
            
            return {
                'success': True,
                'message': f'Incremental sync: {synced} new/updated items',
                'synced': synced,
                'errors': errors,
                'total': len(items),
                'last_sync_time': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error in incremental sync: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'message': f'Failed incremental sync: {e}'
            }


def sync_mindforge_to_llm_twin(
    companion,
    mindforge_db_path: Optional[str] = None,
    user_id: Optional[int] = None,
    content_types: Optional[List[str]] = None
) -> Dict:
    """
    Convenience function to sync MindForge to LLM Twin
    
    Args:
        companion: LLMTwinLearningCompanion instance
        mindforge_db_path: Path to MindForge database
        user_id: Optional user ID to filter by
        content_types: Optional list of content types to sync
    
    Returns:
        Dict with sync results
    """
    connector = MindForgeConnector(mindforge_db_path)
    return connector.sync_to_llm_twin(companion, user_id, content_types)


if __name__ == "__main__":
    # Test connector
    print("Testing MindForge Connector...")
    try:
        connector = MindForgeConnector()
        stats = connector.get_stats()
        print(f"MindForge Stats: {stats}")
        
        items = connector.get_all_knowledge_items(limit=5)
        print(f"Found {len(items)} knowledge items")
        for item in items[:3]:
            print(f"  - {item['title']} ({item['content_type']})")
    except Exception as e:
        print(f"Error: {e}")
