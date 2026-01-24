"""
Metacognition - Thinking About Thinking

Brain-like self-awareness and meta-reasoning
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class SelfAssessment:
    """Self-assessment result"""
    capability: str
    confidence: float
    reasoning: str
    evidence: List[str] = field(default_factory=list)


class SelfAwareness:
    """
    Self-Awareness
    
    Agent's awareness of its own capabilities and limitations
    """
    
    def __init__(self):
        self.capabilities: Dict[str, float] = {}  # capability -> confidence
        self.limitations: List[str] = []
        self.performance_history: Dict[str, List[float]] = {}
    
    def assess_capability(self, capability: str, task_result: Optional[Dict] = None) -> SelfAssessment:
        """
        Assess own capability
        
        Parameters
        ----------
        capability : str
            Capability to assess
        task_result : dict, optional
            Recent task result
            
        Returns
        -------
        assessment : SelfAssessment
            Self-assessment
        """
        # Get historical performance
        performance = self.performance_history.get(capability, [])
        avg_performance = sum(performance) / len(performance) if performance else 0.5
        
        # Update based on recent result
        if task_result:
            success = task_result.get('success', False)
            if success:
                confidence = min(1.0, avg_performance + 0.1)
            else:
                confidence = max(0.0, avg_performance - 0.1)
        else:
            confidence = avg_performance
        
        self.capabilities[capability] = confidence
        
        reasoning = f"Based on {len(performance)} past attempts, average performance: {avg_performance:.2f}"
        evidence = [f"Performance: {p:.2f}" for p in performance[-5:]]  # Last 5
        
        return SelfAssessment(
            capability=capability,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence
        )
    
    def record_performance(self, capability: str, performance: float):
        """Record performance for capability"""
        if capability not in self.performance_history:
            self.performance_history[capability] = []
        self.performance_history[capability].append(performance)
    
    def get_limitations(self) -> List[str]:
        """Get known limitations"""
        return self.limitations
    
    def add_limitation(self, limitation: str):
        """Add known limitation"""
        if limitation not in self.limitations:
            self.limitations.append(limitation)
            logger.info(f"[SelfAwareness] Added limitation: {limitation}")


class Metacognition:
    """
    Metacognition
    
    Thinking about thinking - self-monitoring and regulation
    """
    
    def __init__(self):
        self.self_awareness = SelfAwareness()
        self.monitoring_history: List[Dict] = []
    
    def monitor_thinking(self, task: str, thinking_process: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor own thinking process
        
        Parameters
        ----------
        task : str
            Current task
        thinking_process : dict
            Thinking process data
            
        Returns
        -------
        monitoring : dict
            Monitoring results
        """
        monitoring = {
            'task': task,
            'timestamp': time.time(),
            'self_assessment': {},
            'recommendations': []
        }
        
        # Assess relevant capabilities
        if 'capability_used' in thinking_process:
            capability = thinking_process['capability_used']
            assessment = self.self_awareness.assess_capability(capability, thinking_process.get('result'))
            monitoring['self_assessment'][capability] = {
                'confidence': assessment.confidence,
                'reasoning': assessment.reasoning
            }
        
        # Generate recommendations
        if thinking_process.get('success'):
            monitoring['recommendations'].append("Continue current approach")
        else:
            monitoring['recommendations'].append("Consider alternative approach")
            monitoring['recommendations'].append("Review limitations")
        
        self.monitoring_history.append(monitoring)
        return monitoring
    
    def should_delegate(self, task: str, required_capabilities: List[str]) -> bool:
        """
        Decide if should delegate task
        
        Parameters
        ----------
        task : str
            Task description
        required_capabilities : list
            Required capabilities
            
        Returns
        -------
        should_delegate : bool
            Whether to delegate
        """
        # Check if have all capabilities
        for capability in required_capabilities:
            confidence = self.self_awareness.capabilities.get(capability, 0.5)
            if confidence < 0.5:
                logger.info(f"[Metacognition] Low confidence in {capability}, should delegate")
                return True
        
        return False
    
    def get_self_report(self) -> Dict[str, Any]:
        """Get self-awareness report"""
        return {
            'capabilities': dict(self.self_awareness.capabilities),
            'limitations': self.self_awareness.limitations,
            'monitoring_events': len(self.monitoring_history)
        }
