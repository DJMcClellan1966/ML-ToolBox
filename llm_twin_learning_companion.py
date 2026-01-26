"""
LLM Twin Learning Companion

Upgraded with LLM Twin concepts from LLM Engineer's Handbook:
- Persistent Memory (Long-term user memory)
- Context Management (Multi-turn conversations)
- Personalization (Learns your preferences)
- Conversation Continuity (Remembers past sessions)
- Personality Consistency (Maintains character)
- Adaptive Learning (Learns from interactions)
- Memory Compression (Efficient context)
- User Modeling (Understands you deeply)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from typing import Any, Dict, List, Optional, Callable, Tuple
import numpy as np
import logging
from collections import defaultdict, deque
import time
import json
import pickle
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Import advanced components
try:
    from ml_toolbox.agent_brain.cognitive_architecture import CognitiveArchitecture
    from ml_toolbox.agent_brain.working_memory import WorkingMemory
    from ml_toolbox.agent_brain.episodic_memory import EpisodicMemory
    from ml_toolbox.agent_brain.semantic_memory import SemanticMemory
    from ml_toolbox.agent_brain.attention_mechanism import AttentionMechanism
    from ml_toolbox.agent_brain.metacognition import Metacognition
    BRAIN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Brain components not available: {e}")
    BRAIN_AVAILABLE = False

try:
    from ml_toolbox.llm_engineering.prompt_engineering import PromptEngineer
    from ml_toolbox.llm_engineering.chain_of_thought import ChainOfThoughtReasoner
    from ml_toolbox.llm_engineering.few_shot_learning import FewShotLearner
    LLM_ENGINEERING_AVAILABLE = True
except ImportError:
    LLM_ENGINEERING_AVAILABLE = False

# Try to use Better RAG System (sentence-transformers), fallback to simple RAG
try:
    from better_rag_system import BetterRAGSystem
    BETTER_RAG_AVAILABLE = True
    logger.info("Better RAG System (sentence-transformers) available")
except ImportError:
    BETTER_RAG_AVAILABLE = False
    try:
        from ml_toolbox.llm_engineering.rag_system import RAGSystem
        logger.warning("Better RAG not available, using simple RAG. Install sentence-transformers for better quality.")
    except ImportError:
        RAGSystem = None
        logger.warning("No RAG system available")

try:
    from ml_toolbox.agent_enhancements.socratic_method import SocraticQuestioner
    SOCRATIC_AVAILABLE = True
except ImportError:
    SOCRATIC_AVAILABLE = False

from advanced_learning_companion import AdvancedLearningCompanion


class LLMTwinMemory:
    """
    LLM Twin Memory System
    
    Implements:
    - Long-term user memory
    - Conversation history
    - User preferences
    - Learning patterns
    - Context compression
    """
    
    def __init__(self, user_id: str = "default", memory_file: Optional[str] = None):
        """
        Initialize LLM Twin Memory
        
        Args:
            user_id: Unique user identifier
            memory_file: Path to persistent memory file
        """
        self.user_id = user_id
        self.memory_file = memory_file or f"llm_twin_memory_{user_id}.pkl"
        
        # Long-term memory
        self.user_profile = {
            'name': None,
            'learning_style': 'adaptive',
            'preferred_pace': 'moderate',
            'topics_learned': [],
            'topics_mastered': [],
            'weak_areas': [],
            'strong_areas': [],
            'preferences': {},
            'goals': [],
            'created_at': datetime.now().isoformat(),
            'last_interaction': None
        }
        
        # Conversation history (with compression)
        self.conversation_history = deque(maxlen=100)  # Last 100 interactions
        self.compressed_memories = []  # Compressed important memories
        
        # Learning patterns
        self.learning_patterns = {
            'best_time_to_learn': None,
            'concept_difficulty_map': {},
            'learning_velocity': {},
            'retention_rate': {},
            'preferred_explanation_style': 'detailed'
        }
        
        # User preferences
        self.user_preferences = {
            'explanation_depth': 'detailed',  # brief, moderate, detailed
            'example_preference': 'real_world',  # abstract, real_world, code
            'interaction_style': 'conversational',  # formal, conversational, friendly
            'feedback_frequency': 'moderate'  # low, moderate, high
        }
        
        # Context window management
        self.context_window = deque(maxlen=20)  # Last 20 turns
        self.important_context = []  # Important context to keep
        
        # Load existing memory
        self._load_memory()
    
    def _load_memory(self):
        """Load persistent memory from file"""
        try:
            if Path(self.memory_file).exists():
                with open(self.memory_file, 'rb') as f:
                    data = pickle.load(f)
                    self.user_profile = data.get('user_profile', self.user_profile)
                    self.conversation_history = deque(data.get('conversation_history', []), maxlen=100)
                    self.compressed_memories = data.get('compressed_memories', [])
                    self.learning_patterns = data.get('learning_patterns', self.learning_patterns)
                    self.user_preferences = data.get('user_preferences', self.user_preferences)
                    logger.info(f"Loaded memory for user {self.user_id}")
        except Exception as e:
            logger.warning(f"Could not load memory: {e}")
    
    def save_memory(self):
        """Save memory to file"""
        try:
            data = {
                'user_profile': self.user_profile,
                'conversation_history': list(self.conversation_history),
                'compressed_memories': self.compressed_memories,
                'learning_patterns': self.learning_patterns,
                'user_preferences': self.user_preferences
            }
            with open(self.memory_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved memory for user {self.user_id}")
        except Exception as e:
            logger.warning(f"Could not save memory: {e}")
    
    def remember_interaction(self, user_input: str, response: str, context: Dict = None):
        """Remember an interaction"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'response': response,
            'context': context or {}
        }
        self.conversation_history.append(interaction)
        self.context_window.append(interaction)
        self.user_profile['last_interaction'] = datetime.now().isoformat()
    
    def get_recent_context(self, n: int = 5) -> List[Dict]:
        """Get recent conversation context"""
        return list(self.context_window)[-n:]
    
    def compress_important_memories(self):
        """Compress important memories to save space"""
        # Find important interactions (learning milestones, preferences, etc.)
        important = []
        for interaction in self.conversation_history:
            # Check if interaction contains important information
            if any(keyword in interaction['user_input'].lower() for keyword in 
                   ['prefer', 'like', 'dislike', 'goal', 'want to learn', 'mastered']):
                important.append({
                    'summary': self._summarize_interaction(interaction),
                    'timestamp': interaction['timestamp'],
                    'importance': 0.8
                })
        
        # Keep only most important
        self.compressed_memories = sorted(important, key=lambda x: x['importance'], reverse=True)[:50]
    
    def _summarize_interaction(self, interaction: Dict) -> str:
        """Summarize an interaction for compression"""
        # Simple summarization (can be enhanced with LLM)
        user_input = interaction['user_input'][:100]
        return f"User: {user_input}... (at {interaction['timestamp']})"
    
    def update_user_preference(self, key: str, value: Any):
        """Update user preference"""
        self.user_preferences[key] = value
        self.save_memory()
    
    def update_learning_pattern(self, pattern: str, data: Any):
        """Update learning pattern"""
        self.learning_patterns[pattern] = data
        self.save_memory()
    
    def get_user_context(self) -> Dict:
        """Get comprehensive user context"""
        return {
            'profile': self.user_profile,
            'preferences': self.user_preferences,
            'learning_patterns': self.learning_patterns,
            'recent_context': self.get_recent_context(5),
            'compressed_memories': self.compressed_memories[:10]  # Top 10
        }


class LLMTwinPersonality:
    """
    LLM Twin Personality System
    
    Maintains consistent personality and tone
    """
    
    def __init__(self, personality_type: str = "helpful_mentor"):
        """
        Initialize personality
        
        Args:
            personality_type: Type of personality (helpful_mentor, friendly_tutor, expert_guide)
        """
        self.personality_type = personality_type
        self.personality_traits = self._get_personality_traits(personality_type)
        self.conversation_style = self._get_conversation_style(personality_type)
    
    def _get_personality_traits(self, personality_type: str) -> Dict:
        """Get personality traits"""
        traits = {
            'helpful_mentor': {
                'encouraging': 0.9,
                'patient': 0.9,
                'supportive': 0.9,
                'knowledgeable': 0.8,
                'adaptable': 0.8
            },
            'friendly_tutor': {
                'friendly': 0.9,
                'enthusiastic': 0.8,
                'approachable': 0.9,
                'clear': 0.8,
                'engaging': 0.9
            },
            'expert_guide': {
                'precise': 0.9,
                'thorough': 0.9,
                'professional': 0.8,
                'detailed': 0.9,
                'authoritative': 0.8
            }
        }
        return traits.get(personality_type, traits['helpful_mentor'])
    
    def _get_conversation_style(self, personality_type: str) -> Dict:
        """Get conversation style"""
        styles = {
            'helpful_mentor': {
                'greeting': "Hello! I'm here to help you learn. What would you like to explore today?",
                'encouragement': "Great question! Let's dive into that.",
                'clarification': "Let me clarify that for you.",
                'closing': "Keep up the great work! Feel free to ask more questions."
            },
            'friendly_tutor': {
                'greeting': "Hey there! Ready to learn something awesome?",
                'encouragement': "Awesome! I love that question!",
                'clarification': "Sure thing! Here's what that means...",
                'closing': "That was fun! Come back anytime!"
            },
            'expert_guide': {
                'greeting': "Welcome. I'm here to guide your learning journey.",
                'encouragement': "Excellent question. Let's examine this systematically.",
                'clarification': "To clarify, this means...",
                'closing': "Good progress. Continue your studies."
            }
        }
        return styles.get(personality_type, styles['helpful_mentor'])
    
    def personalize_response(self, response: str, user_context: Dict) -> str:
        """Personalize response based on personality and user context"""
        # Adjust tone based on personality
        if self.personality_type == 'friendly_tutor':
            response = response.replace("Let's", "Let's").replace("I'll", "I'll")
        
        # Adjust based on user preferences
        if user_context.get('preferences', {}).get('explanation_depth') == 'brief':
            # Make response more concise
            sentences = response.split('.')
            response = '. '.join(sentences[:3]) + '.'
        
        return response


class LLMTwinLearningCompanion(AdvancedLearningCompanion):
    """
    LLM Twin Learning Companion
    
    Upgraded with LLM Twin concepts:
    - Persistent memory across sessions
    - Context management
    - Personalization
    - Conversation continuity
    - Personality consistency
    """
    
    def __init__(self, user_id: str = "default", personality_type: str = "helpful_mentor"):
        """
        Initialize LLM Twin Learning Companion
        
        Args:
            user_id: Unique user identifier
            personality_type: Personality type (helpful_mentor, friendly_tutor, expert_guide)
        """
        # Initialize base companion
        super().__init__()
        
        # LLM Twin components
        self.user_id = user_id
        self.memory = LLMTwinMemory(user_id=user_id)
        self.personality = LLMTwinPersonality(personality_type=personality_type)
        
        # LLM Engineering components
        if LLM_ENGINEERING_AVAILABLE:
            self.prompt_engineer = PromptEngineer()
            self.chain_of_thought = ChainOfThoughtReasoner()
            self.few_shot_learner = FewShotLearner()
        else:
            self.prompt_engineer = None
            self.chain_of_thought = None
            self.few_shot_learner = None
        
        # Initialize RAG system (prefer Better RAG, fallback to simple RAG)
        if BETTER_RAG_AVAILABLE:
            try:
                self.rag = BetterRAGSystem()
                logger.info("Using Better RAG System with sentence-transformers")
            except Exception as e:
                logger.warning(f"Failed to initialize Better RAG: {e}. Falling back to simple RAG.")
                if RAGSystem is not None:
                    self.rag = RAGSystem()
                else:
                    self.rag = None
        elif RAGSystem is not None:
            self.rag = RAGSystem()
            logger.info("Using simple RAG System")
        else:
            self.rag = None
            logger.warning("No RAG system available")
        
        # Conversation state
        self.conversation_state = {
            'turn_count': 0,
            'current_topic': None,
            'learning_session_active': False,
            'last_concept_learned': None,
            'last_mindforge_sync': None,  # Track last sync time for incremental sync
            'last_mindforge_sync_to': None  # Track last sync TO MindForge
        }
        
        # Context management
        self.context_window = deque(maxlen=20)  # Last 20 interactions for context
        self.relevant_context_cache = {}  # Cache relevant past context
        
        # Load user knowledge into RAG
        self._initialize_rag_with_knowledge()
        
        logger.info(f"LLM Twin Learning Companion initialized for user {user_id}")
    
    def _initialize_rag_with_knowledge(self):
        """Initialize RAG with knowledge base"""
        if self.rag:
            # Add all concepts to RAG
            for level, concepts in self.knowledge_base.items():
                for concept_name, concept_data in concepts.items():
                    # Create document from concept
                    doc_content = f"{concept_data['concept']}\n\n{concept_data.get('detailed_explanation', '')}"
                    self.rag.add_knowledge(concept_name, doc_content)
    
    def greet_user(self) -> str:
        """Greet user with personalized greeting"""
        user_context = self.memory.get_user_context()
        name = user_context['profile'].get('name')
        
        if name:
            greeting = f"Hello {name}! {self.personality.conversation_style['greeting']}"
        else:
            greeting = self.personality.conversation_style['greeting']
        
        # Add context about last session
        if user_context['profile'].get('last_interaction'):
            last_interaction = datetime.fromisoformat(user_context['profile']['last_interaction'])
            days_ago = (datetime.now() - last_interaction).days
            if days_ago > 0:
                greeting += f"\n\nWelcome back! It's been {days_ago} day(s) since we last talked."
        
        # Add progress context
        topics_learned = len(user_context['profile'].get('topics_learned', []))
        if topics_learned > 0:
            greeting += f"\n\nYou've learned {topics_learned} concept(s) so far. Great progress!"
        
        return greeting
    
    def learn_concept_twin(self, concept_name: str, use_socratic: bool = True) -> Dict:
        """
        Learn concept with LLM Twin features
        
        Features:
        - Remembers if you've learned this before
        - Adapts to your learning style
        - Uses RAG for better context
        - Personalizes explanation
        """
        # Get user context
        user_context = self.memory.get_user_context()
        
        # Check if already learned
        if concept_name in user_context['profile'].get('topics_learned', []):
            # Already learned - provide review
            result = super().learn_concept_advanced(concept_name, use_socratic)
            result['is_review'] = True
            result['message'] = f"You've learned {concept_name} before. Here's a review to reinforce your understanding."
        else:
            # New concept
            result = super().learn_concept_advanced(concept_name, use_socratic)
            result['is_review'] = False
        
        # Enhance with RAG
        if self.rag:
            # Retrieve related knowledge
            related_knowledge = self.rag.retriever.retrieve(concept_name, top_k=3)
            if related_knowledge:
                result['related_knowledge'] = [k['content'][:200] for k in related_knowledge]
        
        # Personalize based on user preferences
        if user_context['preferences'].get('explanation_depth') == 'brief':
            result['explanation'] = result['explanation'][:500] + "..."
        
        # Use Chain-of-Thought for complex concepts
        if result.get('difficulty', 3) >= 4 and self.chain_of_thought:
            reasoning = self.chain_of_thought.reason(concept_name, result['explanation'])
            result['reasoning_steps'] = reasoning.get('steps', [])
        
        # Remember interaction
        self.memory.remember_interaction(
            f"learn {concept_name}",
            f"Learned: {concept_name}",
            {'concept': concept_name, 'difficulty': result.get('difficulty', 3)}
        )
        
        # Update learning patterns
        if concept_name not in user_context['profile']['topics_learned']:
            self.memory.user_profile['topics_learned'].append(concept_name)
            self.memory.save_memory()
        
        # Update conversation state
        self.conversation_state['current_topic'] = concept_name
        self.conversation_state['last_concept_learned'] = concept_name
        self.conversation_state['turn_count'] += 1
        
        # Personalize response
        result['personalized_response'] = self.personality.personalize_response(
            result.get('explanation', ''),
            user_context
        )
        
        return result
    
    def answer_question_twin(self, question: str) -> Dict:
        """
        Answer question with LLM Twin features
        
        Features:
        - Uses conversation context
        - RAG for knowledge retrieval
        - Chain-of-thought reasoning
        - Personalized responses
        """
        # Get user context and recent conversation
        user_context = self.memory.get_user_context()
        recent_context = user_context['recent_context']
        
        # Better context window management - get relevant past context
        relevant_past_context = self._get_relevant_context(question, recent_context)
        
        # Build context-aware query
        if relevant_past_context:
            context_summary = "\n".join([
                f"Previous: {c['user_input'][:50]}..." 
                for c in relevant_past_context[-3:]
            ])
            enhanced_question = f"{context_summary}\n\nCurrent question: {question}"
        else:
            enhanced_question = question
        
        # Update context window
        self.context_window.append({
            'question': question,
            'timestamp': datetime.now().isoformat()
        })
        
        # Check if this is actually a knowledge question or just conversational
        question_lower = question.lower()
        is_knowledge_question = any(word in question_lower for word in [
            'what is', 'what are', 'how does', 'how do', 'why does', 'explain',
            'define', 'tell me about', 'learn about', 'understand'
        ])
        
        # Get base answer first (this generates a proper conversational response)
        result = super().answer_question_advanced(question)
        base_answer = result.get('answer', '')
        
        # Use RAG for knowledge retrieval (but only for actual knowledge questions)
        rag_used = False
        sources = []
        if self.rag and is_knowledge_question:
            retrieved_knowledge = self.rag.retriever.retrieve(enhanced_question, top_k=3)
            if retrieved_knowledge and retrieved_knowledge[0].get('score', 0) > 0.3:  # Only use if relevant
                rag_used = True
                # Get the most relevant piece
                top_knowledge = retrieved_knowledge[0]['content'][:300]
                
                # Extract sources from retrieved knowledge
                for doc in retrieved_knowledge:
                    source_info = {
                        'doc_id': doc.get('doc_id', 'unknown'),
                        'score': doc.get('score', 0.0),
                        'metadata': doc.get('metadata', {})
                    }
                    # Try to get source from metadata
                    if 'source' in source_info['metadata']:
                        source_info['source'] = source_info['metadata']['source']
                    elif 'source' in doc:
                        source_info['source'] = doc['source']
                    sources.append(source_info)
                
                # If we have a base answer, enhance it naturally
                if base_answer and len(base_answer) > 50:
                    # Add RAG context as additional information, not replacement
                    result['answer'] = f"{base_answer}\n\n[Additional context from knowledge base] {top_knowledge}"
                else:
                    # No good base answer, use RAG but make it conversational
                    result['answer'] = f"Based on what I know: {top_knowledge}"
                
                # Add sources to result
                result['sources'] = sources
                result['rag_used'] = True
        elif not is_knowledge_question:
            # Not a knowledge question - use base answer as-is (it should be conversational)
            result['answer'] = base_answer if base_answer else "I'm here to help you learn! Could you rephrase that as a question about a topic you'd like to learn about?"
        
        result['rag_context'] = rag_used
        
        # Use Chain-of-Thought for complex questions (but keep it concise)
        if self.chain_of_thought and len(question.split()) > 10:
            reasoning = self.chain_of_thought.reason(question, result['answer'])
            reasoning_steps = reasoning.get('steps', [])
            if reasoning_steps:
                result['reasoning_steps'] = reasoning_steps[:3]  # Limit to top 3
                # Don't add reasoning steps to answer - keep it clean
        
        # Personalize response
        result['personalized_answer'] = self.personality.personalize_response(
            result['answer'],
            user_context
        )
        
        # Use personalized answer as the main answer
        if result.get('personalized_answer'):
            result['answer'] = result['personalized_answer']
        
        # Remember interaction
        self.memory.remember_interaction(question, result['answer'], {'type': 'question'})
        self.conversation_state['turn_count'] += 1
        
        # Update context window
        self.memory.context_window.append({
            'user_input': question,
            'response': result['answer'],
            'timestamp': datetime.now().isoformat()
        })
        
        return result
    
    def continue_conversation(self, user_input: str) -> Dict:
        """
        Continue conversation with context awareness
        
        Features:
        - Maintains conversation context
        - Understands follow-up questions
        - Remembers what was discussed
        - Generates conversational responses (not just RAG dumps)
        """
        user_input_lower = user_input.lower().strip()
        
        # Handle greetings and simple conversational inputs
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(user_input_lower.startswith(g) for g in greetings):
            response = self._generate_conversational_response(user_input, is_greeting=True)
            self.memory.remember_interaction(user_input, response['answer'], {'type': 'greeting'})
            self.conversation_state['turn_count'] += 1
            return response
        
        # Get recent context to understand follow-ups
        recent_context = self.memory.get_recent_context(3)
        last_response = recent_context[-1]['response'] if recent_context else ""
        
        # Handle follow-up to MindForge explanation
        if 'mindforge' in last_response.lower() or 'sync mindforge' in last_response.lower():
            if any(word in user_input_lower for word in ['yes', 'ok', 'sure', 'do it', 'sync', 'go ahead']):
                # User wants to sync MindForge
                try:
                    result = self.sync_mindforge()
                    if result.get('success'):
                        response_text = f"Great! I've synced {result.get('synced', 0)} items from your MindForge knowledge base. "
                        if result.get('synced', 0) > 0:
                            response_text += "Now I can answer questions using your MindForge content! Try asking me something about what you've saved."
                        else:
                            response_text += "It looks like your MindForge database is empty or I couldn't find any items. You can add content to MindForge first, then sync again."
                    else:
                        response_text = f"I had trouble syncing: {result.get('error', 'Unknown error')}. Make sure your MindForge database is accessible."
                except Exception as e:
                    response_text = f"I encountered an error: {str(e)}. Make sure SQLAlchemy is installed (pip install sqlalchemy)."
                
                response = {
                    'answer': response_text,
                    'rag_context': False,
                    'reasoning_steps': []
                }
                self.memory.remember_interaction(user_input, response['answer'], {'type': 'action', 'action': 'sync_mindforge'})
                self.conversation_state['turn_count'] += 1
                return response
        
        # Handle questions about adding/getting knowledge
        if any(phrase in user_input_lower for phrase in ['how do i get', 'how to add', 'how can i add', 'add knowledge', 'get more knowledge', 'add content']):
            response = self._explain_adding_knowledge()
            self.memory.remember_interaction(user_input, response['answer'], {'type': 'guidance'})
            self.conversation_state['turn_count'] += 1
            return response
        
        # Handle "where do I go from here" / "what's next" type questions
        if any(phrase in user_input_lower for phrase in ['where do i go', 'what next', 'what should i do', 'where from here', 'next step']):
            response = self._suggest_next_steps()
            self.memory.remember_interaction(user_input, response['answer'], {'type': 'guidance'})
            self.conversation_state['turn_count'] += 1
            return response
        
        # Handle questions about MindForge specifically
        if 'mindforge' in user_input_lower and any(word in user_input_lower for word in ['what', 'do', 'is', 'does']):
            response = self._explain_mindforge()
            self.memory.remember_interaction(user_input, response['answer'], {'type': 'system_info'})
            self.conversation_state['turn_count'] += 1
            return response
        
        # Handle conversational questions (not knowledge questions)
        conversational_keywords = [
            'can you see', 'can you access', 'do you have', 'are you able', 
            'hello', 'hi', 'hey', 'thanks', 'thank you',
            'how do i use', 'how do i', 'how can i', 'what can you do',
            'what are you', 'who are you', 'help me', 'show me how'
        ]
        if any(keyword in user_input_lower for keyword in conversational_keywords):
            # Conversational question - respond conversationally
            response = self._generate_conversational_response(user_input, is_greeting=False)
            self.memory.remember_interaction(user_input, response['answer'], {'type': 'conversation'})
            self.conversation_state['turn_count'] += 1
            return response
        
        # Handle simple conversational inputs (not questions)
        if not any(c in user_input for c in ['?', 'what', 'how', 'why', 'when', 'where', 'who', 'explain', 'tell me']):
            # Check if it's a simple yes/no to previous question
            if user_input_lower in ['yes', 'yep', 'yeah', 'ok', 'okay', 'sure', 'alright']:
                # Get last response to understand context
                recent = self.memory.get_recent_context(1)
                if recent and 'sync' in recent[0].get('response', '').lower():
                    # User said yes to syncing - handle it
                    try:
                        result = self.sync_mindforge()
                        if result.get('success'):
                            response_text = f"Great! Synced {result.get('synced', 0)} items from MindForge. "
                            response_text += "Now I can use your MindForge content to answer questions!"
                        else:
                            response_text = f"Couldn't sync: {result.get('error', 'Unknown error')}"
                    except Exception as e:
                        response_text = f"Error: {str(e)}"
                    
                    response = {
                        'answer': response_text,
                        'rag_context': False,
                        'reasoning_steps': []
                    }
                    self.memory.remember_interaction(user_input, response['answer'], {'type': 'action'})
                    self.conversation_state['turn_count'] += 1
                    return response
            
            # Simple statement or comment - respond conversationally
            response = self._generate_conversational_response(user_input, is_greeting=False)
            self.memory.remember_interaction(user_input, response['answer'], {'type': 'conversation'})
            self.conversation_state['turn_count'] += 1
            return response
        
        # Get recent context
        recent_context = self.memory.get_recent_context(5)
        
        # Detect conversation type
        if user_input_lower.startswith('learn '):
            concept = user_input.replace('learn ', '').strip()
            return self.learn_concept_twin(concept)
        elif user_input_lower.startswith('ask ') or '?' in user_input:
            question = user_input.replace('ask ', '').strip()
            return self.answer_question_twin(question)
        else:
            # General conversation - use context
            if recent_context:
                # Check if it's a follow-up
                last_topic = self.conversation_state.get('current_topic')
                if last_topic and any(word in user_input_lower for word in ['more', 'explain', 'tell me', 'what about']):
                    # Follow-up question
                    enhanced_input = f"Regarding {last_topic}: {user_input}"
                    return self.answer_question_twin(enhanced_input)
            
            # Question - use answer_question_twin but ensure it's conversational
            return self.answer_question_twin(user_input)
    
    def _generate_conversational_response(self, user_input: str, is_greeting: bool = False) -> Dict:
        """
        Generate a conversational response (not just RAG content)
        
        Args:
            user_input: User's input
            is_greeting: Whether this is a greeting
        
        Returns:
            Dict with conversational answer
        """
        user_context = self.memory.get_user_context()
        recent_context = self.memory.get_recent_context(3)
        
        if is_greeting:
            # Personalized greeting
            name = user_context['profile'].get('name') or 'there'
            topics_count = len(user_context['profile'].get('topics_learned', []))
            
            if topics_count > 0:
                greeting = f"Hello {name}! Great to see you again. You've learned {topics_count} topics so far. What would you like to explore today?"
            else:
                if name == 'there':
                    greeting = "Hello! I'm your learning companion. I'm here to help you learn and grow. What would you like to start with?"
                else:
                    greeting = f"Hello {name}! I'm your learning companion. I'm here to help you learn and grow. What would you like to start with?"
            
            return {
                'answer': greeting,
                'rag_context': False,
                'reasoning_steps': []
            }
        
        # For non-greeting conversational inputs, generate contextual response
        # Check if we can use RAG for context, but don't just dump it
        rag_context = None
        if self.rag:
            # Get relevant context (but don't just return it)
            relevant = self.rag.retriever.retrieve(user_input, top_k=1)
            if relevant:
                rag_context = relevant[0]['content'][:200]  # Just a snippet for context
        
        # Generate conversational response based on input
        user_input_lower = user_input.lower()
        
        if 'desktop' in user_input_lower or ('see' in user_input_lower and 'my' in user_input_lower):
            response = "I can't directly access your desktop, but I can help you with learning! If you'd like to add content from your desktop to my knowledge base, you can use the 'Add Content' feature. What would you like to learn about?"
        elif 'how do i use' in user_input_lower or 'how can i use' in user_input_lower or 'use you better' in user_input_lower:
            response = """I'm your LLM Twin Learning Companion! Here's how to use me better:

**To Learn:**
- Ask me questions about topics you want to learn
- Say "learn [concept]" to get detailed explanations
- I'll remember what you've learned

**To Add Your Knowledge:**
- Use the "Add Content" tab in the web UI
- Or run: python easy_content_ingestion.py file notes.txt
- Sync your MindForge: python easy_content_ingestion.py mindforge

**To Chat:**
- Just ask me questions naturally
- I'll use your knowledge base to answer
- I remember our conversations

**Best Practices:**
- Be specific with questions ("What is machine learning?" vs "tell me stuff")
- Add your content first for better answers
- Use the web UI for the best experience

What would you like to try first?"""
        elif 'what can you do' in user_input_lower or 'what are you' in user_input_lower or 'who are you' in user_input_lower:
            response = """I'm your LLM Twin Learning Companion! I can:

✅ **Help you learn** - Explain concepts, answer questions
✅ **Remember you** - Track what you've learned across sessions
✅ **Use your knowledge** - Answer questions using your MindForge content
✅ **Personalize** - Adapt to your learning style
✅ **Guide you** - Suggest learning paths

I'm connected to your MindForge knowledge base, so I can answer questions using YOUR notes and content!

What would you like to explore?"""
        elif 'help' in user_input_lower or 'show me how' in user_input_lower:
            response = """I'm here to help you learn! Here's what I can do:

**Learning:**
- Ask me questions: "What is machine learning?"
- Learn concepts: "learn neural networks"
- Get explanations with examples

**Using Your Knowledge:**
- Sync MindForge: Use the "Add Content" tab, then click "Sync MindForge"
- Add files: Upload or use CLI
- Ask about your content: "What did I learn about Python?"

**Getting Started:**
1. Sync your MindForge (if you have one)
2. Start chatting or learning
3. Add more content as you go

What would you like to do?"""
        elif 'thank' in user_input_lower:
            response = "You're welcome! I'm glad I could help. Feel free to ask me anything else!"
        elif any(word in user_input_lower for word in ['how are you', 'how\'s it going']):
            response = "I'm doing great! I'm here and ready to help you learn. How can I assist you today?"
        else:
            # Generic conversational response
            if rag_context:
                response = f"I understand you're mentioning something about that. Let me help you explore that topic. Would you like to learn more about it, or do you have a specific question?"
            else:
                response = "I'm here to help you learn! Could you tell me more about what you'd like to explore, or ask me a specific question?"
        
        return {
            'answer': response,
            'rag_context': rag_context is not None,
            'reasoning_steps': []
        }
    
    def _explain_mindforge(self) -> Dict:
        """
        Explain what MindForge is and how it connects to LLM Twin
        
        Returns:
            Dict with explanation
        """
        explanation = """MindForge is your Personal Knowledge Intelligence Platform - it's like a "second brain" for your knowledge!

**What MindForge Does:**
- **Captures Knowledge** - Save notes, articles, PDFs, web pages, emails
- **Semantic Search** - Find by meaning, not just keywords
- **AI-Powered Insights** - Discovers patterns and connections in your knowledge
- **Writing Assistant** - Helps with research and writing
- **Learning Companion** - Adaptive learning with concept mapping
- **Conversational Knowledge** - Ask questions, get synthesized answers

**How It Connects to LLM Twin:**
I (LLM Twin) can access your MindForge knowledge base! When you sync MindForge:
- All your MindForge notes become part of my knowledge
- I can answer questions using YOUR content
- Your personal insights enhance my responses
- Everything you've saved is searchable

**To Sync:**
- Web UI: Go to "Add Content" tab, then click "Sync MindForge"
- CLI: python easy_content_ingestion.py mindforge
- Python: companion.sync_mindforge()

Think of it this way: MindForge stores your knowledge, and I help you learn from it!

Would you like to sync your MindForge knowledge now?"""
        
        return {
            'answer': explanation,
            'rag_context': False,
            'reasoning_steps': []
        }
    
    def _explain_adding_knowledge(self) -> Dict:
        """Explain how to add knowledge to the system"""
        explanation = """Here are several ways to add knowledge to me:

**1. Sync MindForge (if you have it):**
- Web UI: "Add Content" tab, then click "Sync MindForge" button
- CLI: python easy_content_ingestion.py mindforge
- Python: companion.sync_mindforge()

**2. Add Text Content:**
- Web UI: "Add Content" tab, paste text, then click "Add Text"
- CLI: python easy_content_ingestion.py text "Your content..." --source notes
- Python: companion.ingest_text("Your content...", source="notes")

**3. Add Files:**
- Web UI: "Add Content" tab, upload file (drag & drop works!)
- CLI: python easy_content_ingestion.py file notes.txt
- Python: companion.ingest_file("notes.txt", source="documents")

**4. Add Directories:**
- CLI: python easy_content_ingestion.py dir ./docs --pattern "*.md"
- Python: companion.ingest_directory("./docs", pattern="*.md")

**5. From Clipboard:**
- CLI: python easy_content_ingestion.py clipboard

Once you add content, I can use it to answer your questions!

Which method would you like to try?"""
        
        return {
            'answer': explanation,
            'rag_context': False,
            'reasoning_steps': []
        }
    
    def _suggest_next_steps(self, concept: str, data: Dict) -> List[Dict]:
        """
        Suggest next learning steps (override parent method)
        
        Args:
            concept: Concept name
            data: Concept data dictionary
        
        Returns:
            List of next step dictionaries
        """
        # Call parent method first
        next_steps = super()._suggest_next_steps(concept, data)
        
        # Add LLM Twin specific suggestions
        user_context = self.memory.get_user_context()
        stats = self.get_knowledge_stats()
        topics_learned = len(user_context['profile'].get('topics_learned', []))
        total_docs = stats.get('total_documents', 0)
        
        # Add suggestions based on knowledge base
        if total_docs > 0:
            next_steps.append({
                'type': 'action',
                'action': 'ask_question',
                'reason': f'You have {total_docs} documents in your knowledge base. Ask questions about your content!',
                'priority': 'medium'
            })
        
        if topics_learned > 0:
            next_steps.append({
                'type': 'action',
                'action': 'review_topics',
                'reason': f'Review your {topics_learned} learned topics',
                'priority': 'medium'
            })
        
        # Add MindForge sync suggestion if not synced recently
        if not self.conversation_state.get('last_mindforge_sync'):
            next_steps.append({
                'type': 'action',
                'action': 'sync_mindforge',
                'reason': 'Sync your MindForge knowledge base to get more content',
                'priority': 'high'
            })
        
        return next_steps
    
    def get_personalized_learning_path(self, goal: str) -> Dict:
        """
        Get personalized learning path based on user history
        
        Features:
        - Considers what you've already learned
        - Adapts to your pace
        - Uses your preferences
        """
        user_context = self.memory.get_user_context()
        
        # Get base path
        result = super().suggest_personalized_path(goal)
        
        # Personalize based on learned topics
        topics_learned = user_context['profile'].get('topics_learned', [])
        filtered_path = [t for t in result['path'] if t not in topics_learned]
        
        # Add learned topics as prerequisites (already mastered)
        result['personalized_path'] = filtered_path
        result['topics_already_learned'] = [t for t in result['path'] if t in topics_learned]
        result['estimated_time_adjusted'] = f"{len(filtered_path) * 2} hours"  # Rough estimate
        
        # Adapt to learning pace
        if user_context['preferences'].get('preferred_pace') == 'slow':
            result['estimated_time_adjusted'] = f"{len(filtered_path) * 3} hours"
        elif user_context['preferences'].get('preferred_pace') == 'fast':
            result['estimated_time_adjusted'] = f"{len(filtered_path) * 1.5} hours"
        
        return result
    
    def update_preference(self, preference_key: str, preference_value: Any):
        """Update user preference"""
        self.memory.update_user_preference(preference_key, preference_value)
        return {
            'success': True,
            'message': f"Updated preference: {preference_key} = {preference_value}",
            'preferences': self.memory.user_preferences
        }
    
    def get_user_profile(self) -> Dict:
        """Get comprehensive user profile"""
        user_context = self.memory.get_user_context()
        return {
            'user_id': self.user_id,
            'profile': user_context['profile'],
            'preferences': user_context['preferences'],
            'learning_patterns': user_context['learning_patterns'],
            'conversation_stats': {
                'total_interactions': len(self.memory.conversation_history),
                'topics_learned': len(user_context['profile'].get('topics_learned', [])),
                'current_session_turns': self.conversation_state['turn_count']
            },
            'personality': self.personality.personality_type
        }
    
    def save_session(self):
        """Save current session"""
        self.memory.compress_important_memories()
        self.memory.save_memory()
        logger.info(f"Session saved for user {self.user_id}")
    
    def ingest_text(self, text: str, source: str = "user_input", metadata: Optional[Dict] = None) -> Dict:
        """
        Ingest text content into knowledge base
        
        Args:
            text: Text content to add
            source: Source identifier
            metadata: Additional metadata
            
        Returns:
            Dict with success status
        """
        if not self.rag:
            return {
                'success': False,
                'error': 'RAG system not available',
                'message': 'Knowledge base is not initialized. Cannot add text content.',
                'suggestion': 'Please ensure RAG system is properly initialized. If using Better RAG, install sentence-transformers: pip install sentence-transformers'
            }
        
        try:
            import hashlib
            from datetime import datetime
            
            # Generate unique doc_id
            doc_id = f"{source}_{hashlib.md5(text.encode()).hexdigest()[:8]}_{datetime.now().timestamp()}"
            
            # Add to RAG (RAG.add_knowledge expects doc_id, content, embedding)
            self.rag.add_knowledge(doc_id, text)
            
            # Store metadata separately if needed (RAG doesn't store metadata directly)
            # We can extend this later if needed
            
            # Remember this ingestion
            self.memory.remember_interaction(
                f"ingest content from {source}",
                f"Added {len(text)} characters to knowledge base",
                {'type': 'content_ingestion', 'source': source, 'doc_id': doc_id}
            )
            
            return {
                'success': True,
                'message': f'Content added to knowledge base ({len(text)} characters)',
                'characters': len(text),
                'doc_id': doc_id
            }
        except Exception as e:
            logger.error(f"Error ingesting text: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def ingest_file(self, file_path: str, source: Optional[str] = None) -> Dict:
        """
        Ingest file content into knowledge base
        
        Args:
            file_path: Path to file
            source: Optional source identifier
            
        Returns:
            Dict with success status
        """
        if not self.rag:
            return {
                'success': False,
                'error': 'RAG system not available',
                'message': 'Knowledge base is not initialized. Cannot ingest file.',
                'suggestion': 'Please ensure RAG system is properly initialized. If using Better RAG, install sentence-transformers: pip install sentence-transformers'
            }
        
        try:
            from pathlib import Path
            import json
            
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return {'success': False, 'error': f'File not found: {file_path}'}
            
            # Read file based on extension
            if file_path_obj.suffix in ['.txt', '.md', '.py', '.js', '.html', '.css']:
                content = file_path_obj.read_text(encoding='utf-8', errors='ignore')
            elif file_path_obj.suffix == '.json':
                data = json.loads(file_path_obj.read_text(encoding='utf-8'))
                content = json.dumps(data, indent=2)
            else:
                # Try to read as text
                content = file_path_obj.read_text(encoding='utf-8', errors='ignore')
            
            source = source or f"file_{file_path_obj.name}"
            
            # Generate unique doc_id
            import hashlib
            from datetime import datetime
            doc_id = f"{source}_{hashlib.md5(content.encode()).hexdigest()[:8]}_{datetime.now().timestamp()}"
            
            # Add to RAG (RAG.add_knowledge expects doc_id, content, embedding)
            self.rag.add_knowledge(doc_id, content)
            
            # Remember this ingestion
            self.memory.remember_interaction(
                f"ingest file {file_path_obj.name}",
                f"Added file to knowledge base ({len(content)} characters)",
                {'type': 'file_ingestion', 'filename': file_path_obj.name, 'doc_id': doc_id}
            )
            
            return {
                'success': True,
                'message': f'File "{file_path_obj.name}" added to knowledge base',
                'filename': file_path_obj.name,
                'characters': len(content),
                'doc_id': doc_id
            }
        except Exception as e:
            logger.error(f"Error ingesting file: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def ingest_directory(self, directory_path: str, pattern: str = "*.md", source: Optional[str] = None) -> Dict:
        """
        Ingest all matching files from a directory
        
        Args:
            directory_path: Path to directory
            pattern: File pattern (e.g., "*.md", "*.txt")
            source: Optional source identifier
            
        Returns:
            Dict with success status and count
        """
        if not self.rag:
            return {
                'success': False,
                'error': 'RAG system not available',
                'message': 'Knowledge base is not initialized. Cannot ingest directory.',
                'suggestion': 'Please ensure RAG system is properly initialized. If using Better RAG, install sentence-transformers: pip install sentence-transformers'
            }
        
        try:
            from pathlib import Path
            
            dir_path = Path(directory_path)
            if not dir_path.exists() or not dir_path.is_dir():
                return {'success': False, 'error': f'Directory not found: {directory_path}'}
            
            files = list(dir_path.rglob(pattern))
            if not files:
                return {'success': False, 'error': f'No files found matching {pattern}'}
            
            source = source or f"directory_{dir_path.name}"
            ingested = 0
            errors = 0
            
            for file_path in files:
                result = self.ingest_file(str(file_path), source=f"{source}/{file_path.name}")
                if result.get('success'):
                    ingested += 1
                else:
                    errors += 1
                    logger.warning(f"Failed to ingest {file_path}: {result.get('error')}")
            
            return {
                'success': True,
                'message': f'Ingested {ingested} files from {directory_path}',
                'ingested': ingested,
                'errors': errors,
                'total': len(files)
            }
        except Exception as e:
            logger.error(f"Error ingesting directory: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def sync_mindforge(self, mindforge_db_path: Optional[str] = None, content_types: Optional[List[str]] = None, incremental: bool = False) -> Dict:
        """
        Sync MindForge knowledge base to LLM Twin
        
        Args:
            mindforge_db_path: Path to MindForge database (auto-detected if None)
            content_types: Optional list of content types to sync
            incremental: If True, only sync new/updated items since last sync
        
        Returns:
            Dict with sync results
        """
        try:
            from mindforge_connector import MindForgeConnector
            
            connector = MindForgeConnector(mindforge_db_path)
            
            if incremental:
                # Incremental sync
                last_sync = self.conversation_state.get('last_mindforge_sync')
                result = connector.incremental_sync_to_llm_twin(
                    self,
                    last_sync_time=last_sync,
                    content_types=content_types
                )
            else:
                # Full sync
                result = connector.sync_to_llm_twin(
                    self,
                    content_types=content_types
                )
            
            # Update last sync time
            if result.get('success'):
                self.conversation_state['last_mindforge_sync'] = datetime.now().isoformat()
            
            return result
        except ImportError:
            return {
                'success': False,
                'error': 'MindForge connector not available. Install SQLAlchemy: pip install sqlalchemy'
            }
        except Exception as e:
            logger.error(f"Error syncing MindForge: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def sync_to_mindforge(self, mindforge_db_path: Optional[str] = None, user_id: Optional[int] = None) -> Dict:
        """
        Sync LLM Twin learnings back to MindForge (two-way sync)
        
        Args:
            mindforge_db_path: Path to MindForge database (auto-detected if None)
            user_id: Optional user ID (defaults to companion's user_id)
        
        Returns:
            Dict with sync results
        """
        try:
            from mindforge_connector import MindForgeConnector
            
            connector = MindForgeConnector(mindforge_db_path)
            result = connector.sync_from_llm_twin(self, user_id=user_id)
            
            # Update last sync time
            if result.get('success'):
                self.conversation_state['last_mindforge_sync_to'] = datetime.now().isoformat()
            
            return result
        except ImportError:
            return {
                'success': False,
                'error': 'MindForge connector not available. Install SQLAlchemy: pip install sqlalchemy'
            }
        except Exception as e:
            logger.error(f"Error syncing to MindForge: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_knowledge_stats(self) -> Dict:
        """
        Get statistics about ingested knowledge
        
        Returns:
            Dict with knowledge base statistics
        """
        if not self.rag:
            return {
                'error': 'RAG system not available',
                'message': 'Knowledge base is not initialized. Please add content first.',
                'suggestion': 'Use ingest_text(), ingest_file(), or sync_mindforge() to add knowledge.'
            }
        
        try:
            # Count documents in RAG (works for both BetterRAGSystem and simple RAGSystem)
            if hasattr(self.rag, 'retriever') and hasattr(self.rag.retriever, 'documents'):
                doc_count = len(self.rag.retriever.documents)
            else:
                # Fallback: try to get from stats
                stats = self.rag.get_retrieval_stats() if hasattr(self.rag, 'get_retrieval_stats') else {}
                doc_count = stats.get('total_documents', 0)
            
            # Get sources from memory (we track ingestion in memory)
            sources = {}
            for item in self.memory.conversation_history:
                if item.get('metadata', {}).get('type') in ['content_ingestion', 'file_ingestion']:
                    source = item.get('metadata', {}).get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
            
            return {
                'total_documents': doc_count,
                'sources': sources,
                'rag_available': True,
                'user_id': self.user_id
            }
        except Exception as e:
            logger.error(f"Error getting knowledge stats: {e}", exc_info=True)
            return {'error': str(e)}
    
    def export_knowledge_base(self, format: str = 'json', filepath: Optional[str] = None) -> Dict:
        """
        Export knowledge base to file
        
        Args:
            format: Export format ('json', 'txt', 'csv')
            filepath: Optional file path (auto-generated if not provided)
        
        Returns:
            Dict with export information
        """
        if not self.rag:
            return {
                'error': 'RAG system not available',
                'message': 'Knowledge base is not initialized. Cannot export.',
                'suggestion': 'Please add content first using ingest_text(), ingest_file(), or sync_mindforge(). If RAG is not initializing, install sentence-transformers: pip install sentence-transformers'
            }
        
        try:
            # Get all documents from RAG
            if hasattr(self.rag, 'retriever') and hasattr(self.rag.retriever, 'documents'):
                documents = []
                for doc in self.rag.retriever.documents:
                    doc_data = {
                        'id': doc.get('id', doc.get('doc_id', 'unknown')),
                        'content': doc.get('content', ''),
                        'metadata': doc.get('metadata', {})
                    }
                    documents.append(doc_data)
            else:
                return {
                    'error': 'Could not access documents from RAG system.',
                    'suggestion': 'The RAG system may not be properly initialized.'
                }
            
            # Generate filepath if not provided
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"llm_twin_knowledge_export_{self.user_id}_{timestamp}.{format}"
            
            # Export based on format
            if format == 'json':
                import json
                export_data = {
                    'user_id': self.user_id,
                    'export_date': datetime.now().isoformat(),
                    'total_documents': len(documents),
                    'documents': documents
                }
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            elif format == 'txt':
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"LLM Twin Knowledge Base Export\n")
                    f.write(f"User: {self.user_id}\n")
                    f.write(f"Date: {datetime.now().isoformat()}\n")
                    f.write(f"Total Documents: {len(documents)}\n")
                    f.write("=" * 80 + "\n\n")
                    for i, doc in enumerate(documents, 1):
                        f.write(f"Document {i}: {doc['id']}\n")
                        f.write(f"Metadata: {doc.get('metadata', {})}\n")
                        f.write(f"Content:\n{doc['content']}\n")
                        f.write("-" * 80 + "\n\n")
            
            elif format == 'csv':
                import csv
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['id', 'content', 'metadata', 'source'])
                    for doc in documents:
                        metadata = doc.get('metadata', {})
                        source = metadata.get('source', 'unknown')
                        writer.writerow([
                            doc['id'],
                            doc['content'],
                            str(metadata),
                            source
                        ])
            else:
                return {
                    'error': f'Unsupported format: {format}',
                    'supported_formats': ['json', 'txt', 'csv'],
                    'suggestion': f'Please use one of the supported formats.'
                }
            
            return {
                'success': True,
                'filepath': filepath,
                'format': format,
                'total_documents': len(documents),
                'export_date': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error exporting knowledge base: {e}", exc_info=True)
            return {
                'error': f'Failed to export knowledge base: {str(e)}',
                'suggestion': 'Check file permissions and ensure RAG system is properly initialized.'
            }
    
    def backup_session(self, backup_dir: Optional[str] = None) -> Dict:
        """
        Backup current session (memory + knowledge base)
        
        Args:
            backup_dir: Optional backup directory (defaults to ./backups)
        
        Returns:
            Dict with backup information
        """
        try:
            if backup_dir is None:
                backup_dir = Path("./backups")
            else:
                backup_dir = Path(backup_dir)
            
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Backup memory
            memory_backup = backup_dir / f"memory_{self.user_id}_{timestamp}.pkl"
            self.memory.save_memory(str(memory_backup))
            
            # Backup knowledge base
            kb_backup = self.export_knowledge_base(
                format='json',
                filepath=str(backup_dir / f"knowledge_{self.user_id}_{timestamp}.json")
            )
            
            return {
                'success': True,
                'backup_dir': str(backup_dir),
                'memory_backup': str(memory_backup),
                'knowledge_backup': kb_backup.get('filepath') if kb_backup.get('success') else None,
                'timestamp': timestamp,
                'user_id': self.user_id
            }
        
        except Exception as e:
            logger.error(f"Error backing up session: {e}", exc_info=True)
            return {
                'error': f'Failed to backup session: {str(e)}',
                'suggestion': 'Check directory permissions and ensure all systems are properly initialized.'
            }
    
    def _get_relevant_context(self, current_question: str, recent_context: List[Dict], max_context: int = 3) -> List[Dict]:
        """
        Get relevant past context for current question (better context window)
        
        Uses semantic similarity to find relevant past conversations
        
        Args:
            current_question: Current question
            recent_context: Recent conversation history
            max_context: Maximum number of relevant contexts to return
        
        Returns:
            List of relevant past contexts
        """
        if not recent_context or len(recent_context) <= max_context:
            return recent_context[-max_context:] if recent_context else []
        
        # If RAG is available, use it to find semantically similar past questions
        if self.rag and hasattr(self.rag, 'retriever'):
            try:
                # Create a "document" from past conversations
                past_questions = [c.get('user_input', '') for c in recent_context]
                past_questions_text = " ".join(past_questions)
                
                # Use RAG to find similar past questions
                if past_questions_text:
                    similar = self.rag.retriever.retrieve(current_question, top_k=max_context * 2)
                    
                    # Match similar results back to original contexts
                    relevant_indices = set()
                    for result in similar[:max_context]:
                        result_text = result.get('content', '')
                        for i, ctx in enumerate(recent_context):
                            if ctx.get('user_input', '')[:50] in result_text[:100]:
                                relevant_indices.add(i)
                    
                    # Return relevant contexts
                    relevant = [recent_context[i] for i in sorted(relevant_indices)[-max_context:]]
                    if relevant:
                        return relevant
            except Exception as e:
                logger.debug(f"Error in semantic context retrieval: {e}")
        
        # Fallback: return most recent contexts
        return recent_context[-max_context:]
    
    def get_learning_analytics(self) -> Dict:
        """
        Get detailed learning analytics
        
        Returns:
            Dict with learning analytics including:
            - Topics by category
            - Learning velocity
            - Knowledge gaps
            - Recommendations
        """
        try:
            profile = self.get_user_profile()
            user_context = self.memory.get_user_context()
            
            # Topics learned
            topics_learned = profile.get('conversation_stats', {}).get('topics_learned', [])
            topics_mastered = user_context['profile'].get('topics_mastered', [])
            
            # Learning velocity (topics per session)
            total_sessions = len([h for h in self.memory.conversation_history if h.get('metadata', {}).get('type') == 'session_start'])
            learning_velocity = len(topics_learned) / max(total_sessions, 1)
            
            # Topics by category (if we can infer from knowledge base)
            topics_by_category = {}
            for topic in topics_learned:
                # Simple categorization based on keywords
                category = 'general'
                topic_lower = topic.lower()
                if any(word in topic_lower for word in ['machine learning', 'ml', 'ai', 'neural', 'deep learning']):
                    category = 'ai_ml'
                elif any(word in topic_lower for word in ['python', 'programming', 'code', 'algorithm']):
                    category = 'programming'
                elif any(word in topic_lower for word in ['data', 'analysis', 'statistics']):
                    category = 'data_science'
                elif any(word in topic_lower for word in ['math', 'mathematics', 'calculus', 'linear']):
                    category = 'mathematics'
                
                topics_by_category[category] = topics_by_category.get(category, []) + [topic]
            
            # Knowledge gaps (topics learned but not mastered)
            knowledge_gaps = [t for t in topics_learned if t not in topics_mastered]
            
            # Recommendations
            recommendations = []
            if learning_velocity < 0.5:
                recommendations.append("Consider learning more topics per session to increase learning velocity")
            if len(knowledge_gaps) > 5:
                recommendations.append(f"You have {len(knowledge_gaps)} topics to master. Consider reviewing them.")
            if not topics_learned:
                recommendations.append("Start learning topics to build your knowledge base")
            
            # RAG stats
            rag_stats = {}
            if self.rag:
                try:
                    rag_stats = self.rag.get_retrieval_stats() if hasattr(self.rag, 'get_retrieval_stats') else {}
                except:
                    pass
            
            return {
                'topics_learned': topics_learned,
                'topics_mastered': topics_mastered,
                'topics_by_category': topics_by_category,
                'learning_velocity': round(learning_velocity, 2),
                'knowledge_gaps': knowledge_gaps,
                'recommendations': recommendations,
                'total_interactions': len(self.memory.conversation_history),
                'total_sessions': total_sessions,
                'rag_stats': rag_stats,
                'context_window_size': len(self.context_window),
                'user_id': self.user_id
            }
        
        except Exception as e:
            logger.error(f"Error getting learning analytics: {e}", exc_info=True)
            return {
                'error': str(e),
                'message': 'Failed to generate learning analytics'
            }


def main():
    """Main function"""
    print("\n" + "="*80)
    print("LLM TWIN LEARNING COMPANION".center(80))
    print("="*80)
    print("\nUpgraded with LLM Twin concepts:")
    print("  • Persistent Memory (Remembers you across sessions)")
    print("  • Context Management (Multi-turn conversations)")
    print("  • Personalization (Learns your preferences)")
    print("  • Conversation Continuity (Remembers past sessions)")
    print("  • Personality Consistency (Maintains character)")
    print("  • RAG Integration (Better knowledge retrieval)")
    print("  • Chain-of-Thought (Better reasoning)")
    print("\n" + "="*80 + "\n")
    
    # Create LLM Twin companion
    companion = LLMTwinLearningCompanion(user_id="demo_user", personality_type="helpful_mentor")
    
    # Greet user
    greeting = companion.greet_user()
    print(greeting)
    
    # Demo: Learn a concept
    print("\n" + "="*80)
    print("DEMO: Learning 'classification' with LLM Twin features".center(80))
    print("="*80 + "\n")
    
    result = companion.learn_concept_twin('classification')
    
    print(f"Concept: {result['concept']}")
    print(f"Is Review: {result.get('is_review', False)}")
    print(f"\nExplanation: {result['explanation'][:200]}...")
    
    if result.get('rag_context'):
        print(f"\n[RAG Context Retrieved]")
    
    if result.get('reasoning_steps'):
        print(f"\n[Chain-of-Thought Reasoning]")
        for i, step in enumerate(result['reasoning_steps'][:3], 1):
            print(f"  {i}. {step}")
    
    # Demo: Ask a question with context
    print("\n" + "="*80)
    print("DEMO: Asking question with conversation context".center(80))
    print("="*80 + "\n")
    
    result = companion.answer_question_twin("Can you explain more about decision boundaries?")
    print(f"Question: Can you explain more about decision boundaries?")
    print(f"\nAnswer: {result['answer'][:300]}...")
    
    # Get user profile
    print("\n" + "="*80)
    print("USER PROFILE".center(80))
    print("="*80 + "\n")
    
    profile = companion.get_user_profile()
    print(f"User ID: {profile['user_id']}")
    print(f"Topics Learned: {profile['conversation_stats']['topics_learned']}")
    print(f"Total Interactions: {profile['conversation_stats']['total_interactions']}")
    print(f"Personality: {profile['personality']}")
    
    # Save session
    companion.save_session()
    print("\n" + "="*80)
    print("Session saved! The companion will remember you next time!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
