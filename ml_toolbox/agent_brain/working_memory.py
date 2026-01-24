"""
Working Memory - Active Problem-Solving Memory

Brain-like working memory for active cognitive processing
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemoryItem:
    """Item in working memory"""
    content: Any
    activation: float = 1.0  # Activation level (decays over time)
    timestamp: float = field(default_factory=time.time)
    chunk_type: str = "fact"  # fact, goal, rule, etc.
    associations: List[str] = field(default_factory=list)


class CognitiveLoad:
    """
    Cognitive Load Management
    
    Tracks and manages cognitive load (like brain's limited capacity)
    """
    
    def __init__(self, max_capacity: int = 7):
        """
        Initialize cognitive load
        
        Parameters
        ----------
        max_capacity : int
            Maximum items (Miller's 7±2 rule)
        """
        self.max_capacity = max_capacity
        self.current_load = 0
        self.load_history: List[float] = []
    
    def add_load(self, amount: float = 1.0):
        """Add cognitive load"""
        self.current_load = min(self.current_load + amount, self.max_capacity)
        self.load_history.append(self.current_load)
    
    def reduce_load(self, amount: float = 1.0):
        """Reduce cognitive load"""
        self.current_load = max(0.0, self.current_load - amount)
    
    def is_overloaded(self) -> bool:
        """Check if overloaded"""
        return self.current_load >= self.max_capacity * 0.9
    
    def get_capacity_remaining(self) -> float:
        """Get remaining capacity"""
        return max(0.0, self.max_capacity - self.current_load)


class WorkingMemory:
    """
    Working Memory
    
    Active problem-solving memory (like human working memory)
    Limited capacity, high activation, decays over time
    """
    
    def __init__(self, capacity: int = 7, decay_rate: float = 0.1):
        """
        Initialize working memory
        
        Parameters
        ----------
        capacity : int
            Maximum items (Miller's 7±2)
        decay_rate : float
            Activation decay rate per second
        """
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.chunks: Dict[str, WorkingMemoryItem] = {}
        self.cognitive_load = CognitiveLoad(max_capacity=capacity)
        self.retrieval_history: List[str] = []
    
    def add(self, chunk_id: str, content: Any, chunk_type: str = "fact",
           activation: float = 1.0, associations: Optional[List[str]] = None):
        """
        Add chunk to working memory
        
        Parameters
        ----------
        chunk_id : str
            Chunk identifier
        content : any
            Chunk content
        chunk_type : str
            Type (fact, goal, rule, etc.)
        activation : float
            Initial activation
        associations : list, optional
            Associated chunk IDs
        """
        # Check capacity
        if len(self.chunks) >= self.capacity:
            # Remove least activated
            self._remove_least_activated()
        
        chunk = WorkingMemoryItem(
            content=content,
            activation=activation,
            timestamp=time.time(),
            chunk_type=chunk_type,
            associations=associations or []
        )
        
        self.chunks[chunk_id] = chunk
        self.cognitive_load.add_load()
        logger.debug(f"[WorkingMemory] Added chunk: {chunk_id} (load: {self.cognitive_load.current_load})")
    
    def retrieve(self, chunk_id: str, boost_activation: bool = True) -> Optional[Any]:
        """
        Retrieve chunk from working memory
        
        Parameters
        ----------
        chunk_id : str
            Chunk identifier
        boost_activation : bool
            Boost activation on retrieval (rehearsal)
            
        Returns
        -------
        content : any
            Chunk content if found and activated enough
        """
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return None
        
        # Check activation threshold
        if chunk.activation < 0.1:
            return None  # Too low activation
        
        # Boost activation on retrieval (rehearsal effect)
        if boost_activation:
            chunk.activation = min(1.0, chunk.activation + 0.2)
            chunk.timestamp = time.time()
        
        self.retrieval_history.append(chunk_id)
        return chunk.content
    
    def update_activations(self):
        """Update activations (decay over time)"""
        current_time = time.time()
        
        for chunk_id, chunk in list(self.chunks.items()):
            # Decay activation
            time_diff = current_time - chunk.timestamp
            chunk.activation = max(0.0, chunk.activation - self.decay_rate * time_diff)
            
            # Remove if activation too low
            if chunk.activation < 0.05:
                del self.chunks[chunk_id]
                self.cognitive_load.reduce_load()
                logger.debug(f"[WorkingMemory] Decayed chunk: {chunk_id}")
    
    def _remove_least_activated(self):
        """Remove least activated chunk"""
        if not self.chunks:
            return
        
        least_activated = min(self.chunks.items(), key=lambda x: x[1].activation)
        chunk_id = least_activated[0]
        del self.chunks[chunk_id]
        self.cognitive_load.reduce_load()
        logger.debug(f"[WorkingMemory] Removed least activated: {chunk_id}")
    
    def get_active_chunks(self, min_activation: float = 0.3) -> List[tuple]:
        """Get active chunks above threshold"""
        self.update_activations()
        return [
            (chunk_id, chunk.content)
            for chunk_id, chunk in self.chunks.items()
            if chunk.activation >= min_activation
        ]
    
    def get_goals(self) -> List[Any]:
        """Get goal chunks"""
        return [
            chunk.content
            for chunk in self.chunks.values()
            if chunk.chunk_type == "goal"
        ]
    
    def clear(self):
        """Clear working memory"""
        self.chunks.clear()
        self.cognitive_load.current_load = 0
