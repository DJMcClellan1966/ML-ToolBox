"""
CrewAI Integration Patterns

From CrewAI tutorials - Multi-agent crew/team coordination
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """CrewAI-style Agent"""
    role: str
    goal: str
    backstory: str = ""
    tools: List[str] = None
    verbose: bool = False
    
    def __post_init__(self):
        if self.tools is None:
            self.tools = []


@dataclass
class Task:
    """CrewAI-style Task"""
    description: str
    agent: Optional[Agent] = None
    expected_output: str = ""
    tools: List[str] = None
    
    def __post_init__(self):
        if self.tools is None:
            self.tools = []


class CrewAgent:
    """
    CrewAI-style Agent Wrapper
    
    Individual agent in a crew
    """
    
    def __init__(self, agent: Agent):
        """
        Initialize crew agent
        
        Parameters
        ----------
        agent : Agent
            Agent definition
        """
        self.agent = agent
        self.completed_tasks: List[Task] = []
    
    def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute task"""
        logger.info(f"[{self.agent.role}] Executing: {task.description}")
        
        result = {
            'agent': self.agent.role,
            'task': task.description,
            'output': f"Completed: {task.description}",
            'success': True
        }
        
        self.completed_tasks.append(task)
        return result


class Crew:
    """
    CrewAI-style Crew
    
    Team of agents working together
    """
    
    def __init__(self, agents: List[Agent], tasks: List[Task], verbose: bool = False):
        """
        Initialize crew
        
        Parameters
        ----------
        agents : list
            List of agents
        tasks : list
            List of tasks
        verbose : bool
            Verbose output
        """
        self.agents = agents
        self.tasks = tasks
        self.verbose = verbose
        self.crew_agents = [CrewAgent(agent) for agent in agents]
        self.execution_history: List[Dict] = []
    
    def kickoff(self, inputs: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Kickoff crew execution
        
        Parameters
        ----------
        inputs : dict, optional
            Input data
            
        Returns
        -------
        result : dict
            Crew execution result
        """
        results = []
        
        for task in self.tasks:
            # Find agent for task
            agent = self._find_agent_for_task(task)
            
            if agent:
                crew_agent = next(ca for ca in self.crew_agents if ca.agent == agent)
                result = crew_agent.execute_task(task)
                results.append(result)
                self.execution_history.append(result)
            else:
                # No agent assigned, use first available
                if self.crew_agents:
                    result = self.crew_agents[0].execute_task(task)
                    results.append(result)
                    self.execution_history.append(result)
        
        return {
            'crew_size': len(self.agents),
            'tasks_completed': len(results),
            'results': results,
            'success': all(r.get('success', False) for r in results)
        }
    
    def _find_agent_for_task(self, task: Task) -> Optional[Agent]:
        """Find appropriate agent for task"""
        # If task has assigned agent
        if task.agent:
            return task.agent
        
        # Find agent by role matching
        task_lower = task.description.lower()
        for agent in self.agents:
            if agent.role.lower() in task_lower or task_lower in agent.role.lower():
                return agent
        
        # Default: first agent
        return self.agents[0] if self.agents else None
