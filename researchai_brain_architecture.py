"""
ResearchAI with Brain Topology Architecture

Demonstrates how ResearchAI can be refined using brain-inspired cognitive architecture.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import brain components
try:
    from ml_toolbox.agent_brain.cognitive_architecture import CognitiveArchitecture
    from ml_toolbox.agent_brain.working_memory import WorkingMemory
    from ml_toolbox.agent_brain.episodic_memory import EpisodicMemory
    from ml_toolbox.agent_brain.semantic_memory import SemanticMemory
    from ml_toolbox.agent_brain.attention_mechanism import AttentionMechanism
    from ml_toolbox.agent_brain.metacognition import Metacognition
    from ml_toolbox.multi_agent_design.divine_omniscience import OmniscientCoordinator
    from ml_toolbox.llm_engineering.rag_system import KnowledgeRetriever
    from ml_toolbox.ai_agents.knowledge_graph_agent import KnowledgeGraph
    from ml_toolbox.agent_enhancements.socratic_method import SocraticQuestioner
    BRAIN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Brain components not available: {e}")
    BRAIN_AVAILABLE = False


class BrainBasedResearchAI:
    """
    ResearchAI with Brain Topology Architecture
    
    Uses cognitive architecture inspired by human brain:
    - Working Memory: Active research state
    - Episodic Memory: Research history
    - Semantic Memory: Knowledge base
    - Attention: Focus mechanism
    - Metacognition: Self-awareness
    """
    
    def __init__(self):
        """Initialize brain-based ResearchAI"""
        logger.info("Initializing Brain-Based ResearchAI...")
        
        # Cognitive Architecture (Brain)
        self.brain = CognitiveArchitecture(working_memory_capacity=7)
        
        # Research Components (Atomic Services - can be extracted later)
        self.retriever = KnowledgeRetriever()
        self.knowledge_graph = KnowledgeGraph()
        self.socratic = SocraticQuestioner()
        self.coordinator = OmniscientCoordinator(enable_preemptive_responses=True)
        
        # Load sample documents
        self._load_sample_documents()
        
        logger.info("Brain-Based ResearchAI initialized!")
    
    def _load_sample_documents(self):
        """Load sample research documents"""
        sample_docs = [
            {
                'id': 'doc1',
                'title': 'Transformer Architectures in Deep Learning',
                'content': 'Transformer architectures revolutionized natural language processing. The attention mechanism allows models to focus on relevant parts of input sequences.',
                'domain': 'cs'
            },
            {
                'id': 'doc2',
                'title': 'BERT: Bidirectional Encoder Representations',
                'content': 'BERT uses bidirectional context to understand language. It pre-trains on large corpora and fine-tunes for specific tasks.',
                'domain': 'cs'
            },
            {
                'id': 'doc3',
                'title': 'Ethical Considerations in AI',
                'content': 'AI systems must be developed with ethical principles. Key concerns include bias, privacy, transparency, and accountability.',
                'domain': 'ethics'
            }
        ]
        
        for doc in sample_docs:
            self.retriever.add_document(doc['id'], doc['content'])
            # Store in semantic memory
            self.brain.semantic_memory.add_fact(
                doc['content'],
                context=doc['domain'],
                source=doc['id']
            )
    
    def research(self, query: str) -> Dict[str, Any]:
        """
        Conduct research using brain topology
        
        Process:
        1. Attention: Focus on query
        2. Working Memory: Add query to active processing
        3. Episodic Memory: Check past research
        4. Semantic Memory: Retrieve knowledge
        5. Metacognition: Assess confidence
        6. Process: Conduct research
        7. Remember: Store in episodic memory
        """
        logger.info(f"Research query: {query}")
        
        # 1. Brain processes query through cognitive architecture
        brain_result = self.brain.process(query, task=f"Research: {query}")
        
        # 2. Attention: Focus on relevant information
        # (Brain's attention mechanism filters input)
        
        # 3. Working Memory: Query is now in working memory
        working_memory_items = len(self.brain.working_memory.chunks)
        
        # 4. Episodic Memory: Check past research
        past_research = self.brain.episodic_memory.search_events("Research")
        
        # 5. Semantic Memory: Retrieve knowledge
        semantic_knowledge = self.brain.semantic_memory.search(query)
        
        # 6. Conduct research (using atomic services)
        search_results = self.retriever.retrieve(query, top_k=3)
        
        # 7. Socratic questioning (if needed)
        socratic_questions = []
        if len(search_results) < 2:  # If few results, refine question
            socratic_questions = [
                self.socratic.generate_question(query, question_type='clarification')
            ]
        
        # 8. Metacognition: Assess research quality
        metacognition = self.brain.metacognition.get_self_report()
        
        # 9. Remember this research session
        event_id = self.brain.episodic_memory.remember_event(
            what=f"Researched: {query}",
            where="researchai",
            importance=0.8
        )
        
        # 10. Check if working memory is overloaded
        cognitive_load = self.brain.working_memory.cognitive_load.current_load
        is_overloaded = self.brain.working_memory.cognitive_load.is_overloaded()
        
        return {
            'query': query,
            'brain_processing': {
                'working_memory_items': working_memory_items,
                'cognitive_load': cognitive_load,
                'is_overloaded': is_overloaded,
                'past_research_count': len(past_research),
                'semantic_knowledge_count': len(semantic_knowledge)
            },
            'search_results': [
                {
                    'id': r.get('doc_id', ''),
                    'content': r.get('content', '')[:100] + '...',
                    'score': r.get('score', 0.0)
                }
                for r in search_results
            ],
            'socratic_questions': socratic_questions,
            'metacognition': {
                'confidence': metacognition.get('confidence', 0.0),
                'knowledge_gaps': metacognition.get('knowledge_gaps', [])
            },
            'episodic_memory': {
                'event_id': event_id,
                'stored': True
            }
        }
    
    def get_brain_state(self) -> Dict[str, Any]:
        """Get current brain state"""
        return {
            'working_memory': {
                'items': len(self.brain.working_memory.chunks),
                'capacity': self.brain.working_memory.capacity,
                'cognitive_load': self.brain.working_memory.cognitive_load.current_load
            },
            'episodic_memory': {
                'events': len(self.brain.episodic_memory.events)
            },
            'semantic_memory': {
                'facts': len(self.brain.semantic_memory.facts)
            },
            'attention': {
                'focus_items': len(self.brain.attention.focus_history) if hasattr(self.brain.attention, 'focus_history') else 0
            }
        }


def print_brain_results(results: Dict[str, Any]):
    """Pretty print brain-based research results"""
    print("\n" + "="*80)
    print("BRAIN-BASED RESEARCH RESULTS".center(80))
    print("="*80)
    
    print(f"\nQuery: {results['query']}")
    
    # Brain Processing
    print("\n--- Brain Processing ---")
    brain = results['brain_processing']
    print(f"Working Memory Items: {brain['working_memory_items']}")
    print(f"Cognitive Load: {brain['cognitive_load']:.2f}")
    print(f"Overloaded: {'Yes' if brain['is_overloaded'] else 'No'}")
    print(f"Past Research Sessions: {brain['past_research_count']}")
    print(f"Semantic Knowledge Retrieved: {brain['semantic_knowledge_count']}")
    
    # Search Results
    print("\n--- Search Results ---")
    for i, result in enumerate(results['search_results'], 1):
        print(f"\n  {i}. ID: {result['id']}")
        print(f"     Score: {result['score']:.3f}")
        print(f"     Content: {result['content']}")
    
    # Socratic Questions
    if results['socratic_questions']:
        print("\n--- Socratic Questions ---")
        for i, q in enumerate(results['socratic_questions'], 1):
            print(f"  {i}. {q}")
    
    # Metacognition
    print("\n--- Metacognition (Self-Awareness) ---")
    meta = results['metacognition']
    print(f"Confidence: {meta['confidence']:.2f}")
    if meta['knowledge_gaps']:
        print(f"Knowledge Gaps: {', '.join(meta['knowledge_gaps'])}")
    
    # Episodic Memory
    print("\n--- Episodic Memory ---")
    episodic = results['episodic_memory']
    print(f"Event Stored: {episodic['stored']}")
    print(f"Event ID: {episodic['event_id']}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main demo function"""
    print("\n" + "="*80)
    print("ResearchAI with Brain Topology Architecture".center(80))
    print("="*80)
    print("\nThis demo shows how ResearchAI can be refined using brain-inspired")
    print("cognitive architecture with:")
    print("  • Working Memory (active processing)")
    print("  • Episodic Memory (research history)")
    print("  • Semantic Memory (knowledge base)")
    print("  • Attention (focus mechanism)")
    print("  • Metacognition (self-awareness)")
    print("\n" + "="*80 + "\n")
    
    if not BRAIN_AVAILABLE:
        print("ERROR: Brain components not available.")
        return
    
    # Initialize
    research_ai = BrainBasedResearchAI()
    
    # Demo queries
    queries = [
        "transformer architectures",
        "ethical AI",
        "machine learning optimization"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"DEMO {i}: {query.upper()}")
        print('='*80)
        
        results = research_ai.research(query)
        print_brain_results(results)
        
        # Show brain state
        if i == len(queries):
            print("\n--- Final Brain State ---")
            brain_state = research_ai.get_brain_state()
            print(f"Working Memory: {brain_state['working_memory']['items']}/{brain_state['working_memory']['capacity']} items")
            print(f"Cognitive Load: {brain_state['working_memory']['cognitive_load']:.2f}")
            print(f"Episodic Events: {brain_state['episodic_memory']['events']}")
            print(f"Semantic Facts: {brain_state['semantic_memory']['facts']}")
    
    print("\n" + "="*80)
    print("Brain topology provides:")
    print("  ✓ Natural cognitive flow")
    print("  ✓ Context maintenance (working memory)")
    print("  ✓ Learning from experience (episodic memory)")
    print("  ✓ Knowledge retrieval (semantic memory)")
    print("  ✓ Self-awareness (metacognition)")
    print("  ✓ Focus mechanism (attention)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
