"""
SQLite Database Backend for Learning Apps.
Provides persistent storage for users, progress, spaced repetition, and code submissions.
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
import hashlib
import secrets

DB_PATH = Path(__file__).parent / "learning_apps.db"


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize the database with all required tables."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                email TEXT,
                password_hash TEXT,
                display_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                preferences TEXT DEFAULT '{}'
            )
        ''')
        
        # Progress tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                lab_id TEXT NOT NULL,
                topic_id TEXT NOT NULL,
                status TEXT DEFAULT 'not-started',
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                demo_runs INTEGER DEFAULT 0,
                time_spent_seconds INTEGER DEFAULT 0,
                UNIQUE(user_id, lab_id, topic_id)
            )
        ''')
        
        # Spaced repetition cards
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS srs_cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                lab_id TEXT NOT NULL,
                topic_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                ease_factor REAL DEFAULT 2.5,
                interval_days INTEGER DEFAULT 1,
                repetitions INTEGER DEFAULT 0,
                next_review TIMESTAMP,
                last_reviewed TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, lab_id, topic_id, question)
            )
        ''')
        
        # Code submissions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS code_submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                lab_id TEXT NOT NULL,
                topic_id TEXT,
                challenge_id TEXT,
                code TEXT NOT NULL,
                output TEXT,
                passed BOOLEAN DEFAULT 0,
                execution_time_ms INTEGER,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Badges earned
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS badges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                badge_id TEXT NOT NULL,
                earned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, badge_id)
            )
        ''')
        
        # Sessions for authentication
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_token TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_progress_user ON progress(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_srs_user_next ON srs_cards(user_id, next_review)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_submissions_user ON code_submissions(user_id)')
        
        conn.commit()


# --- User Management ---

def get_or_create_user(user_id: str, display_name: str = None) -> Dict[str, Any]:
    """Get existing user or create new one."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        
        if row:
            # Update last active
            cursor.execute('UPDATE users SET last_active = ? WHERE user_id = ?',
                          (datetime.now(), user_id))
            return dict(row)
        
        # Create new user
        cursor.execute('''
            INSERT INTO users (user_id, display_name) VALUES (?, ?)
        ''', (user_id, display_name or user_id))
        
        return {
            'user_id': user_id,
            'display_name': display_name or user_id,
            'created_at': datetime.now().isoformat()
        }


def update_user_preferences(user_id: str, preferences: Dict[str, Any]) -> bool:
    """Update user preferences."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users SET preferences = ? WHERE user_id = ?
        ''', (json.dumps(preferences), user_id))
        return cursor.rowcount > 0


# --- Progress Tracking ---

def get_progress(user_id: str, lab_id: str = None) -> Dict[str, Any]:
    """Get user progress, optionally filtered by lab."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        if lab_id:
            cursor.execute('''
                SELECT * FROM progress WHERE user_id = ? AND lab_id = ?
            ''', (user_id, lab_id))
        else:
            cursor.execute('SELECT * FROM progress WHERE user_id = ?', (user_id,))
        
        rows = cursor.fetchall()
        
        progress = {}
        stats = {'completed': 0, 'in_progress': 0, 'demo_runs': 0, 'time_spent': 0}
        
        for row in rows:
            row_dict = dict(row)
            lab = row_dict['lab_id']
            topic = row_dict['topic_id']
            
            if lab not in progress:
                progress[lab] = {'topics': {}}
            
            progress[lab]['topics'][topic] = {
                'status': row_dict['status'],
                'started_at': row_dict['started_at'],
                'completed_at': row_dict['completed_at'],
                'demo_runs': row_dict['demo_runs']
            }
            
            if row_dict['status'] == 'completed':
                stats['completed'] += 1
            elif row_dict['status'] == 'in-progress':
                stats['in_progress'] += 1
            stats['demo_runs'] += row_dict['demo_runs'] or 0
            stats['time_spent'] += row_dict['time_spent_seconds'] or 0
        
        return {'ok': True, 'labs': progress, 'stats': stats}


def mark_topic_started(user_id: str, lab_id: str, topic_id: str) -> Dict[str, Any]:
    """Mark a topic as started."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO progress (user_id, lab_id, topic_id, status, started_at)
            VALUES (?, ?, ?, 'in-progress', ?)
            ON CONFLICT(user_id, lab_id, topic_id) 
            DO UPDATE SET status = 'in-progress', started_at = COALESCE(started_at, ?)
        ''', (user_id, lab_id, topic_id, datetime.now(), datetime.now()))
        return {'ok': True, 'status': 'in-progress'}


def mark_topic_completed(user_id: str, lab_id: str, topic_id: str) -> Dict[str, Any]:
    """Mark a topic as completed."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO progress (user_id, lab_id, topic_id, status, completed_at)
            VALUES (?, ?, ?, 'completed', ?)
            ON CONFLICT(user_id, lab_id, topic_id) 
            DO UPDATE SET status = 'completed', completed_at = ?
        ''', (user_id, lab_id, topic_id, datetime.now(), datetime.now()))
        return {'ok': True, 'status': 'completed'}


def record_demo_run(user_id: str, lab_id: str, topic_id: str) -> Dict[str, Any]:
    """Record that a demo was run."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO progress (user_id, lab_id, topic_id, demo_runs)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(user_id, lab_id, topic_id) 
            DO UPDATE SET demo_runs = demo_runs + 1
        ''', (user_id, lab_id, topic_id))
        return {'ok': True}


def reset_progress(user_id: str, lab_id: str = None) -> Dict[str, Any]:
    """Reset progress for a user, optionally for a specific lab."""
    with get_db() as conn:
        cursor = conn.cursor()
        if lab_id:
            cursor.execute('DELETE FROM progress WHERE user_id = ? AND lab_id = ?',
                          (user_id, lab_id))
        else:
            cursor.execute('DELETE FROM progress WHERE user_id = ?', (user_id,))
        return {'ok': True, 'deleted': cursor.rowcount}


# --- Badges ---

def award_badge(user_id: str, badge_id: str) -> bool:
    """Award a badge to a user."""
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO badges (user_id, badge_id) VALUES (?, ?)
            ''', (user_id, badge_id))
            return True
        except sqlite3.IntegrityError:
            return False  # Already has badge


def get_user_badges(user_id: str) -> List[str]:
    """Get all badges for a user."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT badge_id FROM badges WHERE user_id = ?', (user_id,))
        return [row['badge_id'] for row in cursor.fetchall()]


# Initialize database on import
init_db()
