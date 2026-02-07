"""
Spaced Repetition System (SRS) for Learning Apps.
Implements SM-2 algorithm for optimal learning retention.
"""
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from flask import request, jsonify

from learning_apps.database import get_db


# SM-2 Algorithm Constants
MIN_EASE_FACTOR = 1.3
DEFAULT_EASE_FACTOR = 2.5


def calculate_next_review(quality: int, ease_factor: float, interval: int, repetitions: int) -> tuple:
    """
    SM-2 Algorithm implementation.
    
    Args:
        quality: User's rating of recall (0-5)
            0 - Complete blackout
            1 - Incorrect, but remembered upon seeing answer
            2 - Incorrect, but answer seemed easy to recall
            3 - Correct with serious difficulty
            4 - Correct with some hesitation
            5 - Perfect response
        ease_factor: Current ease factor
        interval: Current interval in days
        repetitions: Number of successful repetitions
        
    Returns:
        (new_ease_factor, new_interval, new_repetitions)
    """
    # Update ease factor
    new_ease = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_ease = max(MIN_EASE_FACTOR, new_ease)
    
    if quality < 3:
        # Failed recall - reset
        new_interval = 1
        new_repetitions = 0
    else:
        # Successful recall
        new_repetitions = repetitions + 1
        if new_repetitions == 1:
            new_interval = 1
        elif new_repetitions == 2:
            new_interval = 6
        else:
            new_interval = round(interval * new_ease)
    
    return new_ease, new_interval, new_repetitions


# --- Card Management ---

def create_card(user_id: str, lab_id: str, topic_id: str, question: str, answer: str) -> Dict[str, Any]:
    """Create a new flashcard for spaced repetition."""
    with get_db() as conn:
        cursor = conn.cursor()
        next_review = datetime.now()
        
        try:
            cursor.execute('''
                INSERT INTO srs_cards (user_id, lab_id, topic_id, question, answer, next_review)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, lab_id, topic_id, question, answer, next_review))
            
            return {
                'ok': True,
                'card_id': cursor.lastrowid,
                'message': 'Card created successfully'
            }
        except Exception as e:
            return {'ok': False, 'error': str(e)}


def get_due_cards(user_id: str, lab_id: str = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Get cards due for review."""
    with get_db() as conn:
        cursor = conn.cursor()
        now = datetime.now()
        
        if lab_id:
            cursor.execute('''
                SELECT * FROM srs_cards 
                WHERE user_id = ? AND lab_id = ? AND next_review <= ?
                ORDER BY next_review ASC
                LIMIT ?
            ''', (user_id, lab_id, now, limit))
        else:
            cursor.execute('''
                SELECT * FROM srs_cards 
                WHERE user_id = ? AND next_review <= ?
                ORDER BY next_review ASC
                LIMIT ?
            ''', (user_id, now, limit))
        
        return [dict(row) for row in cursor.fetchall()]


def review_card(card_id: int, quality: int) -> Dict[str, Any]:
    """
    Review a card and update its schedule.
    
    Args:
        card_id: ID of the card
        quality: User's rating (0-5)
    """
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get current card state
        cursor.execute('SELECT * FROM srs_cards WHERE id = ?', (card_id,))
        card = cursor.fetchone()
        
        if not card:
            return {'ok': False, 'error': 'Card not found'}
        
        card = dict(card)
        
        # Calculate next review using SM-2
        new_ease, new_interval, new_reps = calculate_next_review(
            quality,
            card['ease_factor'],
            card['interval_days'],
            card['repetitions']
        )
        
        next_review = datetime.now() + timedelta(days=new_interval)
        
        # Update card
        cursor.execute('''
            UPDATE srs_cards 
            SET ease_factor = ?, interval_days = ?, repetitions = ?,
                next_review = ?, last_reviewed = ?
            WHERE id = ?
        ''', (new_ease, new_interval, new_reps, next_review, datetime.now(), card_id))
        
        return {
            'ok': True,
            'card_id': card_id,
            'new_interval_days': new_interval,
            'next_review': next_review.isoformat(),
            'ease_factor': round(new_ease, 2)
        }


def get_srs_stats(user_id: str) -> Dict[str, Any]:
    """Get SRS statistics for a user."""
    with get_db() as conn:
        cursor = conn.cursor()
        now = datetime.now()
        
        # Total cards
        cursor.execute('SELECT COUNT(*) FROM srs_cards WHERE user_id = ?', (user_id,))
        total = cursor.fetchone()[0]
        
        # Due today
        cursor.execute('''
            SELECT COUNT(*) FROM srs_cards 
            WHERE user_id = ? AND next_review <= ?
        ''', (user_id, now))
        due_today = cursor.fetchone()[0]
        
        # Learning (interval < 7 days)
        cursor.execute('''
            SELECT COUNT(*) FROM srs_cards 
            WHERE user_id = ? AND interval_days < 7
        ''', (user_id,))
        learning = cursor.fetchone()[0]
        
        # Mature (interval >= 21 days)
        cursor.execute('''
            SELECT COUNT(*) FROM srs_cards 
            WHERE user_id = ? AND interval_days >= 21
        ''', (user_id,))
        mature = cursor.fetchone()[0]
        
        # Average ease factor
        cursor.execute('''
            SELECT AVG(ease_factor) FROM srs_cards WHERE user_id = ?
        ''', (user_id,))
        avg_ease = cursor.fetchone()[0] or 2.5
        
        return {
            'ok': True,
            'total_cards': total,
            'due_today': due_today,
            'learning': learning,
            'mature': mature,
            'young': total - learning - mature,
            'average_ease': round(avg_ease, 2)
        }


def auto_generate_cards(user_id: str, lab_id: str, topic_id: str, title: str, learn: str) -> List[Dict[str, Any]]:
    """Auto-generate flashcards from a topic."""
    cards = []
    
    # Basic concept card
    cards.append({
        'question': f"What is {title}?",
        'answer': learn[:500] if len(learn) > 500 else learn
    })
    
    # "Why" card
    cards.append({
        'question': f"Why is {title} important in ML/CS?",
        'answer': f"Understanding {title} helps with: {learn[:300]}..."
    })
    
    created = []
    for card in cards:
        result = create_card(user_id, lab_id, topic_id, card['question'], card['answer'])
        if result.get('ok'):
            created.append(result)
    
    return created


# --- Flask Routes ---

def register_srs_routes(app):
    """Register SRS routes with a Flask app."""
    
    @app.route('/api/srs/cards', methods=['GET'])
    def api_srs_cards():
        """Get due cards for review."""
        user_id = request.args.get('user', 'default')
        lab_id = request.args.get('lab')
        limit = int(request.args.get('limit', 20))
        
        cards = get_due_cards(user_id, lab_id, limit)
        return jsonify({'ok': True, 'cards': cards, 'count': len(cards)})
    
    @app.route('/api/srs/review', methods=['POST'])
    def api_srs_review():
        """Review a card."""
        data = request.get_json(silent=True) or {}
        card_id = data.get('card_id')
        quality = data.get('quality', 3)
        
        if not card_id:
            return jsonify({'ok': False, 'error': 'card_id required'}), 400
        
        if not 0 <= quality <= 5:
            return jsonify({'ok': False, 'error': 'quality must be 0-5'}), 400
        
        result = review_card(card_id, quality)
        return jsonify(result)
    
    @app.route('/api/srs/create', methods=['POST'])
    def api_srs_create():
        """Create a new card."""
        data = request.get_json(silent=True) or {}
        user_id = data.get('user', 'default')
        lab_id = data.get('lab_id')
        topic_id = data.get('topic_id', 'custom')
        question = data.get('question')
        answer = data.get('answer')
        
        if not all([lab_id, question, answer]):
            return jsonify({'ok': False, 'error': 'lab_id, question, answer required'}), 400
        
        result = create_card(user_id, lab_id, topic_id, question, answer)
        return jsonify(result)
    
    @app.route('/api/srs/auto-generate', methods=['POST'])
    def api_srs_auto_generate():
        """Auto-generate cards from a topic."""
        data = request.get_json(silent=True) or {}
        user_id = data.get('user', 'default')
        lab_id = data.get('lab_id')
        topic_id = data.get('topic_id')
        title = data.get('title')
        learn = data.get('learn')
        
        if not all([lab_id, topic_id, title, learn]):
            return jsonify({'ok': False, 'error': 'lab_id, topic_id, title, learn required'}), 400
        
        created = auto_generate_cards(user_id, lab_id, topic_id, title, learn)
        return jsonify({'ok': True, 'created': len(created), 'cards': created})
    
    @app.route('/api/srs/stats')
    def api_srs_stats():
        """Get SRS statistics."""
        user_id = request.args.get('user', 'default')
        return jsonify(get_srs_stats(user_id))


def get_srs_html_snippet() -> str:
    """HTML snippet for SRS review UI."""
    return '''
    <style>
      .srs-container { position: fixed; bottom: 80px; right: 24px; z-index: 100; }
      .srs-btn { 
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        color: white; border: none; padding: 12px 20px; 
        border-radius: 24px; cursor: pointer; font-weight: 600;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
        display: flex; align-items: center; gap: 8px;
      }
      .srs-btn:hover { transform: scale(1.05); }
      .srs-badge { 
        background: #ef4444; color: white; font-size: 0.75rem;
        padding: 2px 8px; border-radius: 12px; 
      }
      .srs-modal {
        display: none; position: fixed; inset: 0;
        background: rgba(0,0,0,0.8); z-index: 200;
        justify-content: center; align-items: center;
      }
      .srs-modal.active { display: flex; }
      .srs-card {
        background: var(--bg-secondary, #1e293b);
        border-radius: 16px; padding: 32px;
        max-width: 500px; width: 90%;
        text-align: center;
      }
      .srs-question { font-size: 1.3rem; margin-bottom: 24px; }
      .srs-answer { 
        background: var(--bg-tertiary, #334155);
        padding: 20px; border-radius: 12px;
        margin: 16px 0; display: none;
      }
      .srs-answer.revealed { display: block; }
      .srs-ratings { display: flex; gap: 8px; justify-content: center; margin-top: 20px; }
      .srs-rating {
        padding: 10px 16px; border-radius: 8px; border: none;
        cursor: pointer; font-weight: 600; color: white;
      }
      .srs-rating.again { background: #ef4444; }
      .srs-rating.hard { background: #f59e0b; }
      .srs-rating.good { background: #22c55e; }
      .srs-rating.easy { background: #3b82f6; }
      .srs-reveal { 
        background: var(--accent, #6366f1); color: white;
        border: none; padding: 12px 32px; border-radius: 8px;
        cursor: pointer; font-size: 1rem;
      }
    </style>
    
    <div class="srs-container">
      <button class="srs-btn" onclick="openSRS()">
        📚 Review <span class="srs-badge" id="srs-due-count">0</span>
      </button>
    </div>
    
    <div class="srs-modal" id="srs-modal">
      <div class="srs-card">
        <div id="srs-content">
          <p id="srs-question" class="srs-question">Loading...</p>
          <div id="srs-answer" class="srs-answer"></div>
          <button class="srs-reveal" id="srs-reveal-btn" onclick="revealAnswer()">Show Answer</button>
          <div class="srs-ratings" id="srs-ratings" style="display:none;">
            <button class="srs-rating again" onclick="rateSRS(1)">Again</button>
            <button class="srs-rating hard" onclick="rateSRS(3)">Hard</button>
            <button class="srs-rating good" onclick="rateSRS(4)">Good</button>
            <button class="srs-rating easy" onclick="rateSRS(5)">Easy</button>
          </div>
        </div>
        <p id="srs-empty" style="display:none;">🎉 All caught up! No cards due.</p>
        <button style="margin-top:20px;background:transparent;border:1px solid var(--border);color:var(--text-primary);padding:8px 16px;border-radius:8px;cursor:pointer;" onclick="closeSRS()">Close</button>
      </div>
    </div>
    
    <script>
      let srsCards = [];
      let currentCardIndex = 0;
      
      async function loadSRSCards() {
        try {
          const resp = await fetch('/api/srs/cards?user=default&limit=50');
          const data = await resp.json();
          srsCards = data.cards || [];
          document.getElementById('srs-due-count').textContent = srsCards.length;
        } catch (e) {
          console.log('SRS not available');
        }
      }
      
      function openSRS() {
        document.getElementById('srs-modal').classList.add('active');
        currentCardIndex = 0;
        showCard();
      }
      
      function closeSRS() {
        document.getElementById('srs-modal').classList.remove('active');
      }
      
      function showCard() {
        const content = document.getElementById('srs-content');
        const empty = document.getElementById('srs-empty');
        
        if (currentCardIndex >= srsCards.length) {
          content.style.display = 'none';
          empty.style.display = 'block';
          return;
        }
        
        content.style.display = 'block';
        empty.style.display = 'none';
        
        const card = srsCards[currentCardIndex];
        document.getElementById('srs-question').textContent = card.question;
        document.getElementById('srs-answer').textContent = card.answer;
        document.getElementById('srs-answer').classList.remove('revealed');
        document.getElementById('srs-reveal-btn').style.display = 'inline-block';
        document.getElementById('srs-ratings').style.display = 'none';
      }
      
      function revealAnswer() {
        document.getElementById('srs-answer').classList.add('revealed');
        document.getElementById('srs-reveal-btn').style.display = 'none';
        document.getElementById('srs-ratings').style.display = 'flex';
      }
      
      async function rateSRS(quality) {
        const card = srsCards[currentCardIndex];
        try {
          await fetch('/api/srs/review', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({card_id: card.id, quality: quality})
          });
        } catch (e) {}
        
        currentCardIndex++;
        document.getElementById('srs-due-count').textContent = Math.max(0, srsCards.length - currentCardIndex);
        showCard();
      }
      
      // Load cards on page load
      setTimeout(loadSRSCards, 1000);
    </script>
    '''
