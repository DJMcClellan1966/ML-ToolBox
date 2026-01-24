"""
Framework Integration Patterns

From "Complete Agentic AI Engineering Course" and framework tutorials:
- LangGraph integration (graph-based agents)
- CrewAI integration (crew/team patterns)
- LlamaIndex workflows (RAG-heavy)
- AutoGen patterns (conversational)
"""
try:
    from .langgraph_patterns import LangGraphAgent, StateGraph, GraphNode
    from .crewai_patterns import CrewAgent, Crew, Task, Agent
    from .llamaindex_patterns import LlamaIndexWorkflow, RAGWorkflow
    from .autogen_patterns import AutoGenAgent, GroupChat
    __all__ = [
        'LangGraphAgent',
        'StateGraph',
        'GraphNode',
        'CrewAgent',
        'Crew',
        'Task',
        'Agent',
        'LlamaIndexWorkflow',
        'RAGWorkflow',
        'AutoGenAgent',
        'GroupChat'
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Framework Integration not available: {e}")
    __all__ = []
