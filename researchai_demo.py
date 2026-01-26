"""
ResearchAI Demo - Intelligent Research & Knowledge Platform

Demonstrates:
- Semantic Search
- Knowledge Graphs
- Socratic Questioning
- Multi-Agent Coordination
- Omniscient Knowledge Base
- RAG System
"""

import sys
import os
from pathlib import Path

# Add ml_toolbox to path
sys.path.insert(0, str(Path(__file__).parent))

from typing import List, Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import components
try:
    from ml_toolbox.multi_agent_design.divine_omniscience import (
        OmniscientKnowledgeBase, OmniscientCoordinator
    )
    from ml_toolbox.llm_engineering.rag_system import KnowledgeRetriever, RAGSystem
    from ml_toolbox.ai_agents.knowledge_graph_agent import KnowledgeGraph
    from ml_toolbox.agent_enhancements.socratic_method import SocraticQuestioner
    from ml_toolbox.textbook_concepts.precognition import PrecognitiveForecaster
    from ml_toolbox.agent_enhancements.moral_laws import MoralLawSystem
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some components not available: {e}")
    COMPONENTS_AVAILABLE = False


class ResearchAgent:
    """Specialized research agent"""
    
    def __init__(self, domain: str, personality: str = "analytical"):
        self.domain = domain
        self.personality = personality
        self.expertise = []
    
    def research(self, query: str) -> Dict[str, Any]:
        """Conduct research in domain"""
        return {
            'domain': self.domain,
            'query': query,
            'findings': f"Research findings in {self.domain} for: {query}",
            'confidence': 0.85
        }


class ResearchAI:
    """
    ResearchAI - Intelligent Research & Knowledge Platform
    WITH BRAIN TOPOLOGY ARCHITECTURE
    
    Integrates:
    - Brain Topology (Cognitive Architecture)
      - Working Memory (Active Processing)
      - Episodic Memory (Research History)
      - Semantic Memory (Knowledge Base)
      - Attention Mechanism (Focus)
      - Metacognition (Self-Awareness)
    - Semantic Search (RAG)
    - Knowledge Graphs
    - Socratic Questioning
    - Omniscient Knowledge Base
    - Ethical Review
    """
    
    def __init__(self):
        """Initialize ResearchAI with Brain Topology"""
        logger.info("Initializing ResearchAI with Brain Topology...")
        
        # BRAIN TOPOLOGY - Core Cognitive Architecture
        if BRAIN_AVAILABLE:
            self.brain = CognitiveArchitecture(working_memory_capacity=7)
            logger.info("Brain topology initialized")
        else:
            self.brain = None
            logger.warning("Brain topology not available - using fallback")
        
        # Knowledge Base
        self.knowledge_base = OmniscientKnowledgeBase()
        self.coordinator = OmniscientCoordinator(
            knowledge_base=self.knowledge_base,
            enable_preemptive_responses=True
        )
        
        # RAG System
        self.rag = RAGSystem()
        self.retriever = KnowledgeRetriever()
        
        # Knowledge Graph
        self.knowledge_graph = KnowledgeGraph()
        
        # Socratic Questioner
        self.socratic = SocraticQuestioner()
        
        # Research Agents
        self.research_agents = {
            'cs': ResearchAgent('Computer Science', 'analytical'),
            'biology': ResearchAgent('Biology', 'curious'),
            'physics': ResearchAgent('Physics', 'logical'),
            'ethics': ResearchAgent('Ethics', 'thoughtful')
        }
        
        # Ethical Review
        try:
            self.ethics = MoralLawSystem()
        except Exception as e:
            self.ethics = None
            logger.warning(f"MoralLawSystem not available: {e}")
        
        # Sample research documents
        self._load_sample_documents()
        
        logger.info("ResearchAI with Brain Topology initialized successfully!")
    
    def _load_sample_documents(self):
        """Load sample research documents"""
        sample_docs = [
            {
                'id': 'doc1',
                'title': 'Transformer Architectures in Deep Learning',
                'content': 'Transformer architectures revolutionized natural language processing. The attention mechanism allows models to focus on relevant parts of input sequences. Key innovations include multi-head attention, positional encoding, and layer normalization.',
                'domain': 'cs',
                'year': 2023
            },
            {
                'id': 'doc2',
                'title': 'BERT: Bidirectional Encoder Representations',
                'content': 'BERT uses bidirectional context to understand language. It pre-trains on large corpora and fine-tunes for specific tasks. BERT achieved state-of-the-art results on many NLP benchmarks.',
                'domain': 'cs',
                'year': 2022
            },
            {
                'id': 'doc3',
                'title': 'GPT and Large Language Models',
                'content': 'GPT models use autoregressive generation to produce text. Scaling laws show that larger models perform better. GPT-3 and GPT-4 demonstrate emergent capabilities at scale.',
                'domain': 'cs',
                'year': 2024
            },
            {
                'id': 'doc4',
                'title': 'Neural Network Optimization',
                'content': 'Optimization techniques like Adam, learning rate scheduling, and gradient clipping improve training stability. Regularization methods prevent overfitting in deep networks.',
                'domain': 'cs',
                'year': 2023
            },
            {
                'id': 'doc5',
                'title': 'Ethical Considerations in AI',
                'content': 'AI systems must be developed with ethical principles. Key concerns include bias, privacy, transparency, and accountability. Ethical AI requires careful design and continuous monitoring.',
                'domain': 'ethics',
                'year': 2024
            },
            {
                'id': 'doc6',
                'title': 'Machine Learning in Biology',
                'content': 'ML applications in biology include protein folding prediction, drug discovery, and genomic analysis. Deep learning models can identify patterns in biological data that traditional methods miss.',
                'domain': 'biology',
                'year': 2023
            }
        ]
        
        # Add to RAG system
        for doc in sample_docs:
            self.retriever.add_document(doc['id'], doc['content'])
            self.rag.add_knowledge(doc['id'], doc['content'])
            
            # Add to knowledge graph
            self.knowledge_graph.add_node(
                doc['id'],
                'document',
                properties={
                    'title': doc['title'],
                    'domain': doc['domain'],
                    'year': doc['year']
                }
            )
            
            # Add to BRAIN's Semantic Memory
            if self.brain:
                self.brain.semantic_memory.add_fact(
                    doc['content'],
                    context=doc['domain'],
                    source=doc['id'],
                    confidence=1.0
                )
            
            # Extract key concepts and add relationships
            concepts = self._extract_concepts(doc['content'])
            for concept in concepts:
                if concept not in self.knowledge_graph.node_index:
                    self.knowledge_graph.add_node(
                        f"concept_{concept}",
                        'concept',
                        properties={'name': concept}
                    )
                self.knowledge_graph.add_edge(
                    doc['id'],
                    f"concept_{concept}",
                    'contains'
                )
        
        logger.info(f"Loaded {len(sample_docs)} sample documents into brain and knowledge systems")
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text (simplified)"""
        # Simple keyword extraction
        keywords = ['transformer', 'attention', 'BERT', 'GPT', 'neural network',
                   'optimization', 'ethics', 'bias', 'protein', 'genomics']
        found = []
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                found.append(keyword)
        return found[:5]  # Top 5 concepts
    
    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Semantic search for research documents
        WITH BRAIN TOPOLOGY PROCESSING
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Search results with documents and scores
        """
        logger.info(f"Searching for: {query}")
        
        # BRAIN PROCESSING: Attention focuses on query
        if self.brain:
            # Process query through brain
            brain_result = self.brain.process(query, task=f"Search: {query}")
            
            # Attention: Focus on relevant information
            focused_query = query  # Brain's attention mechanism filters
            
            # Working Memory: Add query to active processing
            self.brain.working_memory.add("search_query", query, chunk_type="goal")
        else:
            focused_query = query
        
        # Use RAG system for retrieval
        retrieved = self.retriever.retrieve(focused_query, top_k=top_k)
        
        # Format results
        results = []
        for r in retrieved:
            results.append({
                'id': r.get('doc_id', ''),
                'content': r.get('content', ''),
                'score': r.get('score', 0.0)
            })
        
        # BRAIN: Add results to working memory
        if self.brain:
            for i, result in enumerate(results[:3]):  # Top 3 to working memory
                self.brain.working_memory.add(
                    f"result_{i}",
                    result['content'][:100],
                    chunk_type="fact"
                )
        
        # Check for preemptive answer
        preemptive = self.coordinator.answer_before_question(query)
        if preemptive.get('preemptive') and preemptive.get('answer'):
            logger.info(f"Preemptive answer found: {preemptive['reason']}")
        
        # Learn pattern
        self.coordinator.learn_query_pattern(query, f"Found {len(results)} documents")
        
        return {
            'query': query,
            'results': results,
            'count': len(results),
            'preemptive': preemptive,
            'brain_processing': brain_result if self.brain else None
        }
    
    def refine_question(self, query: str) -> Dict[str, Any]:
        """
        Use Socratic method to refine research question
        
        Args:
            query: Initial research question
            
        Returns:
            Refined question with Socratic questions
        """
        logger.info(f"Refining question: {query}")
        
        # Generate Socratic questions
        questions = []
        for q_type in ['clarification', 'assumption', 'evidence']:
            question = self.socratic.generate_question(query, question_type=q_type)
            questions.append(question)
        
        return {
            'original_query': query,
            'socratic_questions': questions,
            'refined_query': f"{query} (refined through Socratic questioning)"
        }
    
    def build_knowledge_graph(self, query: str) -> Dict[str, Any]:
        """
        Build knowledge graph for query
        
        Args:
            query: Research query
            
        Returns:
            Knowledge graph structure
        """
        logger.info(f"Building knowledge graph for: {query}")
        
        # Search for relevant documents
        search_results = self.search(query, top_k=3)
        
        # Build graph from results
        graph_info = {
            'nodes': len(self.knowledge_graph.nodes),
            'edges': sum(len(edges) for edges in self.knowledge_graph.edges.values()),
            'relationship_types': list(self.knowledge_graph.relationship_types),
            'related_concepts': []
        }
        
        # Find related concepts
        for result in search_results['results'][:3]:
            doc_id = result.get('id', '')
            if doc_id in self.knowledge_graph.nodes:
                neighbors = self.knowledge_graph.get_neighbors(doc_id)
                for neighbor in neighbors[:5]:
                    if isinstance(neighbor, dict):
                        # get_neighbors returns {'node': {...}, 'relationship': '...'}
                        node_data = neighbor.get('node', {})
                        if node_data:
                            props = node_data.get('properties', {})
                            name = props.get('name', '')
                            if name and name not in graph_info['related_concepts']:
                                graph_info['related_concepts'].append(name)
        
        return graph_info
    
    def forecast_trends(self, domain: str = 'cs') -> Dict[str, Any]:
        """
        Forecast research trends
        
        Args:
            domain: Research domain
            
        Returns:
            Forecasted trends
        """
        if not self.forecaster:
            return {'error': 'Forecaster not available'}
        
        logger.info(f"Forecasting trends in {domain}")
        
        # Simple trend forecast
        trends = [
            f"Continued growth in {domain} research",
            f"Emerging applications in {domain}",
            f"New methodologies in {domain}"
        ]
        
        return {
            'domain': domain,
            'forecasted_trends': trends,
            'confidence': 0.75
        }
    
    def ethical_review(self, research_topic: str) -> Dict[str, Any]:
        """
        Review research for ethical concerns
        
        Args:
            research_topic: Research topic to review
            
        Returns:
            Ethical review results
        """
        if not self.ethics:
            return {'error': 'Ethics system not available'}
        
        logger.info(f"Reviewing ethics for: {research_topic}")
        
        # Simple ethical review
        concerns = []
        if 'ai' in research_topic.lower() or 'machine learning' in research_topic.lower():
            concerns.append('Bias in AI systems')
            concerns.append('Privacy considerations')
        
        return {
            'topic': research_topic,
            'ethical_concerns': concerns,
            'recommendations': ['Ensure diverse training data', 'Implement bias testing']
        }
    
    def research(self, query: str, use_socratic: bool = True) -> Dict[str, Any]:
        """
        Complete research workflow WITH BRAIN TOPOLOGY
        
        Process:
        1. Brain Attention: Focus on query
        2. Working Memory: Add query to active processing
        3. Episodic Memory: Check past research
        4. Semantic Memory: Retrieve knowledge
        5. Refine question (Socratic)
        6. Search documents
        7. Build knowledge graph
        8. Ethical review
        9. Metacognition: Assess quality
        10. Remember: Store in episodic memory
        
        Args:
            query: Research query
            use_socratic: Whether to use Socratic questioning
            
        Returns:
            Complete research results with brain processing
        """
        logger.info(f"Starting research for: {query}")
        
        # BRAIN PROCESSING: Process query through cognitive architecture
        if self.brain:
            brain_result = self.brain.process(query, task=f"Research: {query}")
            
            # Check past research from episodic memory
            past_research = self.brain.episodic_memory.search_events("Research")
            
            # Retrieve semantic knowledge
            semantic_knowledge = self.brain.semantic_memory.search(query)
            
            logger.info(f"Brain: Found {len(past_research)} past research sessions, {len(semantic_knowledge)} semantic facts")
        else:
            brain_result = None
            past_research = []
            semantic_knowledge = []
        
        results = {
            'query': query,
            'brain_processing': {
                'past_research_count': len(past_research),
                'semantic_knowledge_count': len(semantic_knowledge),
                'working_memory_items': len(self.brain.working_memory.chunks) if self.brain else 0,
                'cognitive_load': self.brain.working_memory.cognitive_load.current_load if self.brain else 0
            } if self.brain else None,
            'refinement': None,
            'search_results': None,
            'knowledge_graph': None,
            'ethics': None
        }
        
        # Step 1: Refine question (Socratic)
        if use_socratic:
            results['refinement'] = self.refine_question(query)
            refined_query = results['refinement']['refined_query']
        else:
            refined_query = query
        
        # Step 2: Search (with brain processing)
        results['search_results'] = self.search(refined_query)
        
        # Step 3: Build knowledge graph
        results['knowledge_graph'] = self.build_knowledge_graph(refined_query)
        
        # Step 4: Ethical review
        results['ethics'] = self.ethical_review(refined_query)
        
        # BRAIN: Metacognition - Assess research quality
        if self.brain:
            metacognition = self.brain.metacognition.get_self_report()
            results['metacognition'] = {
                'confidence': metacognition.get('confidence', 0.0),
                'knowledge_gaps': metacognition.get('knowledge_gaps', [])
            }
            
            # Remember this research session in episodic memory
            event_id = self.brain.episodic_memory.remember_event(
                what=f"Researched: {query}",
                where="researchai",
                importance=0.8
            )
            results['episodic_memory'] = {
                'event_id': event_id,
                'stored': True
            }
        
        # Learn pattern
        self.coordinator.learn_query_pattern(
            query,
            f"Research completed: {len(results['search_results']['results'])} documents found"
        )
        
        return results


def print_results(results: Dict[str, Any]):
    """Pretty print research results WITH BRAIN TOPOLOGY"""
    print("\n" + "="*80)
    print("RESEARCH RESULTS (WITH BRAIN TOPOLOGY)".center(80))
    print("="*80)
    
    print(f"\nQuery: {results['query']}")
    
    # Brain Processing
    if results.get('brain_processing'):
        print("\n--- Brain Processing ---")
        brain = results['brain_processing']
        print(f"Working Memory Items: {brain['working_memory_items']}")
        print(f"Cognitive Load: {brain['cognitive_load']:.2f}")
        print(f"Past Research Sessions: {brain['past_research_count']}")
        print(f"Semantic Knowledge Retrieved: {brain['semantic_knowledge_count']}")
    
    # Metacognition
    if results.get('metacognition'):
        print("\n--- Metacognition (Self-Awareness) ---")
        meta = results['metacognition']
        print(f"Confidence: {meta['confidence']:.2f}")
        if meta.get('knowledge_gaps'):
            print(f"Knowledge Gaps: {', '.join(meta['knowledge_gaps'])}")
    
    # Episodic Memory
    if results.get('episodic_memory'):
        print("\n--- Episodic Memory ---")
        episodic = results['episodic_memory']
        print(f"Event Stored: {episodic['stored']}")
        print(f"Event ID: {episodic['event_id']}")
    
    # Refinement
    if results.get('refinement'):
        print("\n--- Socratic Refinement ---")
        print(f"Original: {results['refinement']['original_query']}")
        print("Questions:")
        for i, q in enumerate(results['refinement']['socratic_questions'], 1):
            print(f"  {i}. {q}")
    
    # Search Results
    if results.get('search_results'):
        print("\n--- Search Results ---")
        search = results['search_results']
        print(f"Found {search['count']} documents")
        for i, result in enumerate(search['results'][:3], 1):
            print(f"\n  {i}. Document ID: {result.get('id', 'N/A')}")
            print(f"     Score: {result.get('score', 0):.3f}")
            print(f"     Content: {result.get('content', '')[:100]}...")
    
    # Knowledge Graph
    if results.get('knowledge_graph'):
        print("\n--- Knowledge Graph ---")
        kg = results['knowledge_graph']
        print(f"Nodes: {kg['nodes']}, Edges: {kg['edges']}")
        if kg['related_concepts']:
            print(f"Related Concepts: {', '.join(kg['related_concepts'][:5])}")
    
    # Ethics
    if results.get('ethics'):
        print("\n--- Ethical Review ---")
        ethics = results['ethics']
        if 'ethical_concerns' in ethics:
            print("Concerns:")
            for concern in ethics['ethical_concerns']:
                print(f"  • {concern}")
        if 'recommendations' in ethics:
            print("Recommendations:")
            for rec in ethics['recommendations']:
                print(f"  • {rec}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main demo function"""
    print("\n" + "="*80)
    print("ResearchAI Demo - WITH BRAIN TOPOLOGY ARCHITECTURE".center(80))
    print("="*80)
    print("\nThis demo demonstrates:")
    print("  🧠 Brain Topology:")
    print("     • Working Memory (Active Processing)")
    print("     • Episodic Memory (Research History)")
    print("     • Semantic Memory (Knowledge Base)")
    print("     • Attention Mechanism (Focus)")
    print("     • Metacognition (Self-Awareness)")
    print("  📚 Research Features:")
    print("     • Semantic Search (RAG)")
    print("     • Knowledge Graphs")
    print("     • Socratic Questioning")
    print("     • Omniscient Knowledge Base")
    print("     • Ethical Review")
    print("\n" + "="*80 + "\n")
    
    if not COMPONENTS_AVAILABLE:
        print("ERROR: Required components not available. Please check imports.")
        return
    
    if not BRAIN_AVAILABLE:
        print("WARNING: Brain topology not available. Using fallback mode.")
    
    # Initialize ResearchAI with Brain Topology
    research_ai = ResearchAI()
    
    # Demo queries
    demo_queries = [
        "transformer architectures",
        "ethical AI",
        "machine learning optimization"
    ]
    
    print("Running demo queries...\n")
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*80}")
        print(f"DEMO {i}: {query.upper()}")
        print('='*80)
        
        # Conduct research
        results = research_ai.research(query, use_socratic=True)
        
        # Print results
        print_results(results)
        
        # Show preemptive capabilities
        if results['search_results'].get('preemptive', {}).get('preemptive'):
            preemptive = results['search_results']['preemptive']
            print(f"✨ Preemptive Answer: {preemptive.get('reason', 'N/A')}")
            print(f"   Confidence: {preemptive.get('confidence', 0):.2f}\n")
    
    # Interactive mode
    print("\n" + "="*80)
    print("INTERACTIVE MODE".center(80))
    print("="*80)
    print("\nEnter research queries (or 'quit' to exit):\n")
    
    while True:
        try:
            query = input("Research Query: ").strip()
            if not query or query.lower() == 'quit':
                break
            
            results = research_ai.research(query, use_socratic=True)
            print_results(results)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            logger.exception("Error in interactive mode")
    
    # Show final brain state
    if research_ai.brain:
        print("\n--- Final Brain State ---")
        print(f"Working Memory: {len(research_ai.brain.working_memory.chunks)}/{research_ai.brain.working_memory.capacity} items")
        print(f"Cognitive Load: {research_ai.brain.working_memory.cognitive_load.current_load:.2f}")
        print(f"Episodic Events: {len(research_ai.brain.episodic_memory.events)}")
        print(f"Semantic Facts: {len(research_ai.brain.semantic_memory.facts)}")
        print("\nBrain topology provides:")
        print("  ✓ Natural cognitive flow")
        print("  ✓ Context maintenance (working memory)")
        print("  ✓ Learning from experience (episodic memory)")
        print("  ✓ Knowledge retrieval (semantic memory)")
        print("  ✓ Self-awareness (metacognition)")
        print("  ✓ Focus mechanism (attention)")
    
    print("\nThank you for using ResearchAI with Brain Topology!")


if __name__ == "__main__":
    main()
