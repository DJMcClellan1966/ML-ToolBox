"""
Curriculum: Python practice — comprehensive Python programming covering fundamentals, OOP, functional programming,
data processing (NumPy/Pandas), and best practices (testing, debugging, optimization).
"""
from typing import Dict, Any, List
import json
from pathlib import Path

LEVELS = ["basics", "intermediate", "advanced"]

BOOKS = [
    {"id": "fundamentals", "name": "Python Fundamentals", "short": "Fundamentals", "color": "#6366f1"},
    {"id": "oop", "name": "Object-Oriented Programming", "short": "OOP", "color": "#14b8a6"},
    {"id": "functional", "name": "Functional Programming", "short": "Functional", "color": "#f97316"},
    {"id": "data_processing", "name": "Data Processing (NumPy/Pandas)", "short": "Data", "color": "#8b5cf6"},
    {"id": "best_practices", "name": "Testing & Best Practices", "short": "Best Practices", "color": "#ec4899"},
]

# Load curriculum from enriched JSON
_enriched_file = Path(__file__).parent.parent.parent / '.cache' / 'python_practice_enriched.json'
if _enriched_file.exists():
    with open(_enriched_file, 'r', encoding='utf-8') as f:
        CURRICULUM = json.load(f)
else:
    # Fallback minimal curriculum
    CURRICULUM: List[Dict[str, Any]] = [
        {"id": "py_basics", "book_id": "fundamentals", "level": "basics", "title": "Python Basics",
         "learn": "Variables, types, control flow, loops",
         "try_code": "x = 42; print(x)",
         "try_demo": None, "prerequisites": []},
    ]


def get_curriculum(): return list(CURRICULUM)
def get_books(): return list(BOOKS)
def get_levels(): return list(LEVELS)
def get_by_book(book_id: str): return [c for c in CURRICULUM if c["book_id"] == book_id]
def get_by_level(level: str): return [c for c in CURRICULUM if c["level"] == level]
def get_item(item_id: str):
    for c in CURRICULUM:
        if c["id"] == item_id: return c
    return None
