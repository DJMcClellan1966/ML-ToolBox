"""
Episodic Memory - Event-Based Memory

Brain-like episodic memory for remembering events and experiences
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class EpisodicEvent:
    """Episodic memory event"""
    event_id: str
    what: str  # What happened
    when: float  # When it happened
    where: str = ""  # Context/location
    who: str = ""  # Agent/user involved
    outcome: Any = None  # Result/outcome
    emotions: List[str] = field(default_factory=list)  # Emotional tags
    importance: float = 1.0  # Importance score
    associations: List[str] = field(default_factory=list)  # Related events


class MemoryConsolidation:
    """
    Memory Consolidation
    
    Moves important memories from episodic to semantic (like brain)
    """
    
    def __init__(self, importance_threshold: float = 0.7):
        """
        Initialize consolidation
        
        Parameters
        ----------
        importance_threshold : float
            Threshold for consolidation
        """
        self.importance_threshold = importance_threshold
        self.consolidated_count = 0
    
    def should_consolidate(self, event: EpisodicEvent) -> bool:
        """Check if event should be consolidated"""
        return event.importance >= self.importance_threshold
    
    def consolidate(self, event: EpisodicEvent) -> Dict[str, Any]:
        """Consolidate event to semantic knowledge"""
        self.consolidated_count += 1
        return {
            'fact': event.what,
            'context': event.where,
            'outcome': event.outcome,
            'learned_from': event.event_id
        }


class EpisodicMemory:
    """
    Episodic Memory
    
    Event-based memory system (like human episodic memory)
    """
    
    def __init__(self, max_events: int = 1000):
        """
        Initialize episodic memory
        
        Parameters
        ----------
        max_events : int
            Maximum events to store
        """
        self.events: Dict[str, EpisodicEvent] = {}
        self.max_events = max_events
        self.consolidation = MemoryConsolidation()
        self.event_counter = 0
    
    def remember_event(self, what: str, where: str = "", who: str = "",
                      outcome: Any = None, importance: float = 1.0,
                      emotions: Optional[List[str]] = None) -> str:
        """
        Remember an event
        
        Parameters
        ----------
        what : str
            What happened
        where : str
            Context/location
        who : str
            Agent/user
        outcome : any
            Result
        importance : float
            Importance score
        emotions : list, optional
            Emotional tags
            
        Returns
        -------
        event_id : str
            Event identifier
        """
        self.event_counter += 1
        event_id = f"event_{self.event_counter}_{int(time.time())}"
        
        event = EpisodicEvent(
            event_id=event_id,
            what=what,
            when=time.time(),
            where=where,
            who=who,
            outcome=outcome,
            emotions=emotions or [],
            importance=importance
        )
        
        self.events[event_id] = event
        
        # Check if should consolidate
        if self.consolidation.should_consolidate(event):
            consolidated = self.consolidation.consolidate(event)
            logger.info(f"[EpisodicMemory] Consolidated: {event_id}")
            return event_id, consolidated
        
        # Manage capacity
        if len(self.events) > self.max_events:
            self._remove_oldest()
        
        logger.debug(f"[EpisodicMemory] Remembered: {event_id}")
        return event_id
    
    def recall_event(self, event_id: str) -> Optional[EpisodicEvent]:
        """Recall specific event"""
        return self.events.get(event_id)
    
    def search_events(self, query: str, time_range: Optional[tuple] = None) -> List[EpisodicEvent]:
        """
        Search events by query
        
        Parameters
        ----------
        query : str
            Search query
        time_range : tuple, optional
            (start_time, end_time)
            
        Returns
        -------
        events : list
            Matching events
        """
        results = []
        query_lower = query.lower()
        
        for event in self.events.values():
            # Time filter
            if time_range:
                start, end = time_range
                if not (start <= event.when <= end):
                    continue
            
            # Content match
            if (query_lower in event.what.lower() or
                query_lower in event.where.lower() or
                any(query_lower in e.lower() for e in event.emotions)):
                results.append(event)
        
        # Sort by importance and recency
        results.sort(key=lambda e: (e.importance, e.when), reverse=True)
        return results
    
    def get_recent_events(self, n: int = 10) -> List[EpisodicEvent]:
        """Get recent events"""
        events = list(self.events.values())
        events.sort(key=lambda e: e.when, reverse=True)
        return events[:n]
    
    def get_important_events(self, min_importance: float = 0.7) -> List[EpisodicEvent]:
        """Get important events"""
        return [
            event for event in self.events.values()
            if event.importance >= min_importance
        ]
    
    def _remove_oldest(self):
        """Remove oldest events"""
        if not self.events:
            return
        
        oldest = min(self.events.items(), key=lambda x: x[1].when)
        del self.events[oldest[0]]
        logger.debug(f"[EpisodicMemory] Removed oldest: {oldest[0]}")
