"""
Agent Loops - Lesson 4-6 from Microsoft's Course

ReAct, Plan-Act, and other agent execution loops
"""
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)

from .agent_basics import SimpleAgent, AgentState


class LoopType(Enum):
    """Agent loop types"""
    REACT = "react"  # Reasoning and Acting
    PLAN_ACT = "plan_act"  # Plan then Act
    REFLEX = "reflex"  # Direct action
    OBSERVE_THINK_ACT = "observe_think_act"  # Full cycle


class AgentLoop:
    """
    Base Agent Loop
    
    Defines execution loop pattern
    """
    
    def __init__(self, agent: SimpleAgent, max_iterations: int = 10):
        """
        Initialize agent loop
        
        Parameters
        ----------
        agent : SimpleAgent
            Agent to execute
        max_iterations : int
            Maximum loop iterations
        """
        self.agent = agent
        self.max_iterations = max_iterations
        self.iteration = 0
        self.history: List[Dict] = []
    
    def run(self, task: str) -> Dict[str, Any]:
        """
        Run agent loop
        
        Parameters
        ----------
        task : str
            Task to execute
            
        Returns
        -------
        result : dict
            Loop execution result
        """
        raise NotImplementedError("Subclasses must implement run()")
    
    def _check_stop_condition(self) -> bool:
        """Check if loop should stop"""
        return self.iteration >= self.max_iterations


class ReActLoop(AgentLoop):
    """
    ReAct Loop - Reasoning and Acting
    
    Lesson 4: Think → Act → Observe → Repeat
    """
    
    def run(self, task: str) -> Dict[str, Any]:
        """Execute ReAct loop"""
        self.iteration = 0
        self.history = []
        
        current_task = task
        observations = []
        
        while not self._check_stop_condition():
            self.iteration += 1
            
            # Think
            self.agent.state.update(AgentState.THINKING, {'iteration': self.iteration})
            thought = self._think(current_task, observations)
            
            # Act
            self.agent.state.update(AgentState.ACTING, {'thought': thought})
            action_result = self._act(thought, current_task)
            
            # Observe
            self.agent.state.update(AgentState.OBSERVING, {'action': action_result})
            observation = self._observe(action_result)
            observations.append(observation)
            
            # Record step
            step = {
                'iteration': self.iteration,
                'thought': thought,
                'action': action_result,
                'observation': observation
            }
            self.history.append(step)
            
            # Check if done
            if self._is_done(observation):
                break
            
            # Update task based on observation
            current_task = self._update_task(current_task, observation)
        
        return {
            'task': task,
            'iterations': self.iteration,
            'history': self.history,
            'final_observation': observations[-1] if observations else None,
            'success': self.iteration < self.max_iterations
        }
    
    def _think(self, task: str, observations: List[str]) -> str:
        """Generate thought"""
        # Simple: analyze task and observations
        if observations:
            return f"Based on observations: {observations[-1]}, I need to continue working on: {task}"
        return f"I need to: {task}"
    
    def _act(self, thought: str, task: str) -> Dict[str, Any]:
        """Execute action"""
        # Use agent's tools or execute task
        return self.agent.execute(task)
    
    def _observe(self, action_result: Dict[str, Any]) -> str:
        """Observe action result"""
        if action_result.get('success'):
            return f"Action succeeded: {action_result.get('result', '')}"
        return f"Action failed: {action_result.get('error', '')}"
    
    def _is_done(self, observation: str) -> bool:
        """Check if task is complete"""
        return 'succeeded' in observation.lower() or 'complete' in observation.lower()
    
    def _update_task(self, current_task: str, observation: str) -> str:
        """Update task based on observation"""
        # Simple: refine task if needed
        return current_task


class PlanActLoop(AgentLoop):
    """
    Plan-Act Loop
    
    Lesson 5: Plan → Execute → Evaluate
    """
    
    def run(self, task: str) -> Dict[str, Any]:
        """Execute Plan-Act loop"""
        self.iteration = 0
        self.history = []
        
        # Plan
        plan = self._plan(task)
        
        # Execute plan steps
        results = []
        for step in plan:
            self.iteration += 1
            if self._check_stop_condition():
                break
            
            self.agent.state.update(AgentState.ACTING, {'step': step})
            result = self.agent.execute(step)
            results.append(result)
            
            self.history.append({
                'step': step,
                'result': result
            })
        
        return {
            'task': task,
            'plan': plan,
            'results': results,
            'iterations': self.iteration,
            'success': all(r.get('success', False) for r in results)
        }
    
    def _plan(self, task: str) -> List[str]:
        """Create execution plan"""
        # Simple: decompose task into steps
        steps = []
        
        if 'classify' in task.lower():
            steps = ['Load data', 'Preprocess', 'Train classifier', 'Evaluate']
        elif 'predict' in task.lower():
            steps = ['Load data', 'Preprocess', 'Train model', 'Make predictions']
        else:
            steps = [f'Step 1: {task}', 'Step 2: Complete task']
        
        return steps
