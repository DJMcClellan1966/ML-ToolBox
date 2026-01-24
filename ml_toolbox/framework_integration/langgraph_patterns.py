"""
LangGraph Integration Patterns

From LangGraph docs - Graph-based agent execution
Aligns with pattern_graph.py concepts
"""
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GraphNode:
    """
    Graph Node - Represents a node in agent graph
    
    Similar to LangGraph's StateGraph nodes
    """
    
    def __init__(self, name: str, handler: Callable, node_type: str = "agent"):
        """
        Initialize graph node
        
        Parameters
        ----------
        name : str
            Node name
        handler : callable
            Node handler function
        node_type : str
            Node type ('agent', 'tool', 'condition')
        """
        self.name = name
        self.handler = handler
        self.node_type = node_type
        self.edges: List[str] = []  # Connected nodes
    
    def add_edge(self, target_node: str, condition: Optional[Callable] = None):
        """Add edge to another node"""
        self.edges.append({
            'target': target_node,
            'condition': condition
        })
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node handler"""
        try:
            result = self.handler(state)
            return {
                'node': self.name,
                'result': result,
                'success': True
            }
        except Exception as e:
            return {
                'node': self.name,
                'error': str(e),
                'success': False
            }


class StateGraph:
    """
    State Graph - LangGraph-style state machine
    
    Graph-based agent execution with state management
    """
    
    def __init__(self, initial_state: Optional[Dict] = None):
        """
        Initialize state graph
        
        Parameters
        ----------
        initial_state : dict, optional
            Initial graph state
        """
        self.nodes: Dict[str, GraphNode] = {}
        self.entry_point: Optional[str] = None
        self.state = initial_state or {}
        self.execution_history: List[Dict] = []
    
    def add_node(self, name: str, handler: Callable, node_type: str = "agent") -> GraphNode:
        """Add node to graph"""
        node = GraphNode(name, handler, node_type)
        self.nodes[name] = node
        
        if self.entry_point is None:
            self.entry_point = name
        
        logger.info(f"[StateGraph] Added node: {name}")
        return node
    
    def add_edge(self, from_node: str, to_node: str, condition: Optional[Callable] = None):
        """Add edge between nodes"""
        if from_node in self.nodes:
            self.nodes[from_node].add_edge(to_node, condition)
        else:
            raise ValueError(f"Node not found: {from_node}")
    
    def set_entry_point(self, node_name: str):
        """Set graph entry point"""
        if node_name in self.nodes:
            self.entry_point = node_name
        else:
            raise ValueError(f"Node not found: {node_name}")
    
    def execute(self, input_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute graph
        
        Parameters
        ----------
        input_data : dict, optional
            Input data
            
        Returns
        -------
        result : dict
            Graph execution result
        """
        if self.entry_point is None:
            return {'error': 'No entry point set', 'success': False}
        
        # Initialize state
        if input_data:
            self.state.update(input_data)
        
        current_node = self.entry_point
        visited = set()
        max_iterations = 100
        
        for iteration in range(max_iterations):
            if current_node in visited:
                break  # Cycle detection
            
            visited.add(current_node)
            
            if current_node not in self.nodes:
                break
            
            node = self.nodes[current_node]
            
            # Execute node
            result = node.execute(self.state)
            self.execution_history.append({
                'iteration': iteration,
                'node': current_node,
                'result': result
            })
            
            # Update state
            if result.get('success') and 'result' in result:
                self.state.update(result['result'])
            
            # Determine next node
            next_node = self._get_next_node(node, result)
            if next_node is None:
                break  # End of graph
            
            current_node = next_node
        
        return {
            'final_state': self.state,
            'execution_history': self.execution_history,
            'iterations': len(self.execution_history),
            'success': True
        }
    
    def _get_next_node(self, current_node: GraphNode, result: Dict) -> Optional[str]:
        """Get next node based on edges and conditions"""
        if not current_node.edges:
            return None  # End of graph
        
        # Check conditions
        for edge in current_node.edges:
            condition = edge.get('condition')
            if condition is None:
                return edge['target']  # Unconditional edge
            elif condition(result):
                return edge['target']  # Condition met
        
        # Default: first edge
        return current_node.edges[0]['target'] if current_node.edges else None


class LangGraphAgent:
    """
    LangGraph-style Agent
    
    Wrapper for StateGraph with agent-specific patterns
    """
    
    def __init__(self, name: str = "LangGraphAgent"):
        """
        Initialize LangGraph agent
        
        Parameters
        ----------
        name : str
            Agent name
        """
        self.name = name
        self.graph = StateGraph()
        self._setup_default_graph()
    
    def _setup_default_graph(self):
        """Setup default agent graph"""
        # Entry: Start
        def start_handler(state):
            return {'status': 'started', 'message': 'Agent started'}
        
        # Think
        def think_handler(state):
            return {'thought': f"Thinking about: {state.get('task', 'task')}"}
        
        # Act
        def act_handler(state):
            return {'action': 'executed', 'result': state.get('thought', '')}
        
        # End
        def end_handler(state):
            return {'status': 'completed'}
        
        # Add nodes
        self.graph.add_node('start', start_handler)
        self.graph.add_node('think', think_handler)
        self.graph.add_node('act', act_handler)
        self.graph.add_node('end', end_handler)
        
        # Add edges
        self.graph.add_edge('start', 'think')
        self.graph.add_edge('think', 'act')
        self.graph.add_edge('act', 'end')
        
        self.graph.set_entry_point('start')
    
    def run(self, task: str) -> Dict[str, Any]:
        """Run agent with task"""
        return self.graph.execute({'task': task})
