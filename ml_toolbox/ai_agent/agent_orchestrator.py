"""
Agent Orchestrator - Coordinates Multiple Specialist Agents

Provides workflow orchestration, task distribution, and agent coordination
for the Super Power Tool.
"""
import numpy as np
from typing import Any, Dict, Optional, Union, List, Tuple
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """Types of agent workflows"""
    SEQUENTIAL = "sequential"  # Agents run one after another
    PARALLEL = "parallel"  # Agents run simultaneously
    PIPELINE = "pipeline"  # Agents form a pipeline
    ADAPTIVE = "adaptive"  # Workflow adapts based on results


class AgentOrchestrator:
    """
    Orchestrates multiple specialist agents
    
    Provides:
    - Workflow orchestration
    - Task distribution
    - Agent coordination
    - Result aggregation
    """
    
    def __init__(self, specialist_agents: Dict):
        """
        Initialize agent orchestrator
        
        Parameters
        ----------
        specialist_agents : dict
            Dictionary of specialist agents
        """
        self.agents = specialist_agents
        self.workflow_history = []
        
        logger.info("[AgentOrchestrator] Initialized")
    
    def execute_workflow(self, workflow_type: WorkflowType, 
                       task_description: str, data: np.ndarray,
                       target: Optional[np.ndarray] = None,
                       agent_sequence: Optional[List[str]] = None,
                       **kwargs) -> Dict:
        """
        Execute a multi-agent workflow
        
        Parameters
        ----------
        workflow_type : WorkflowType
            Type of workflow to execute
        task_description : str
            Description of the task
        data : array-like
            Input data
        target : array-like, optional
            Target labels
        agent_sequence : list of str, optional
            Sequence of agents to use (default: auto-detect)
        **kwargs
            Additional parameters
            
        Returns
        -------
        result : dict
            Aggregated results from all agents
        """
        logger.info(f"[AgentOrchestrator] Executing {workflow_type.value} workflow")
        
        # Auto-detect agent sequence if not provided
        if agent_sequence is None:
            agent_sequence = self._detect_agent_sequence(task_description)
        
        # Execute based on workflow type
        if workflow_type == WorkflowType.SEQUENTIAL:
            return self._execute_sequential(agent_sequence, data, target, **kwargs)
        elif workflow_type == WorkflowType.PARALLEL:
            return self._execute_parallel(agent_sequence, data, target, **kwargs)
        elif workflow_type == WorkflowType.PIPELINE:
            return self._execute_pipeline(agent_sequence, data, target, **kwargs)
        elif workflow_type == WorkflowType.ADAPTIVE:
            return self._execute_adaptive(agent_sequence, data, target, **kwargs)
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
    
    def _detect_agent_sequence(self, task_description: str) -> List[str]:
        """Auto-detect which agents to use based on task"""
        task_lower = task_description.lower()
        sequence = []
        
        # Always start with data analysis
        if 'data' in self.agents:
            sequence.append('data')
        
        # Feature engineering
        if any(word in task_lower for word in ['feature', 'engineer', 'transform']):
            if 'feature' in self.agents:
                sequence.append('feature')
        
        # Model selection/training
        if any(word in task_lower for word in ['model', 'train', 'predict', 'classify']):
            if 'model' in self.agents:
                sequence.append('model')
        
        # Hyperparameter tuning
        if any(word in task_lower for word in ['tune', 'optimize', 'hyperparameter']):
            if 'tuning' in self.agents:
                sequence.append('tuning')
        
        # Deployment
        if any(word in task_lower for word in ['deploy', 'serve', 'production']):
            if 'deploy' in self.agents:
                sequence.append('deploy')
        
        # Always end with insights
        if 'insight' in self.agents:
            sequence.append('insight')
        
        return sequence if sequence else ['data', 'model', 'insight']
    
    def _execute_sequential(self, agent_sequence: List[str], data: np.ndarray,
                           target: Optional[np.ndarray], **kwargs) -> Dict:
        """Execute agents sequentially"""
        results = {}
        current_data = data
        current_target = target
        
        for agent_name in agent_sequence:
            if agent_name not in self.agents:
                logger.warning(f"Agent {agent_name} not available")
                continue
            
            agent = self.agents[agent_name]
            logger.info(f"[AgentOrchestrator] Executing {agent_name} agent")
            
            try:
                if agent_name == 'data':
                    result = agent.analyze(current_data)
                    results['data_analysis'] = result
                elif agent_name == 'feature':
                    result = agent.suggest_features(current_data, current_target)
                    results['feature_suggestions'] = result
                elif agent_name == 'model':
                    result = agent.recommend_model(current_data, current_target)
                    results['model_recommendation'] = result
                elif agent_name == 'tuning':
                    result = agent.suggest_search_space(kwargs.get('model_type', 'rf'))
                    results['tuning_suggestions'] = result
                elif agent_name == 'deploy':
                    result = agent.prepare_deployment(kwargs.get('model'), kwargs.get('metadata'))
                    results['deployment_info'] = result
                elif agent_name == 'insight':
                    result = agent.suggest_improvements(kwargs.get('metrics', {}))
                    results['insights'] = result
            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")
                results[f'{agent_name}_error'] = str(e)
        
        results['workflow_type'] = 'sequential'
        results['agents_used'] = agent_sequence
        results['status'] = 'success'
        
        return results
    
    def _execute_parallel(self, agent_sequence: List[str], data: np.ndarray,
                          target: Optional[np.ndarray], **kwargs) -> Dict:
        """Execute agents in parallel"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {}
        
        def execute_agent(agent_name: str):
            if agent_name not in self.agents:
                return agent_name, None
            
            agent = self.agents[agent_name]
            try:
                if agent_name == 'data':
                    return agent_name, agent.analyze(data)
                elif agent_name == 'feature':
                    return agent_name, agent.suggest_features(data, target)
                elif agent_name == 'model':
                    return agent_name, agent.recommend_model(data, target)
                elif agent_name == 'tuning':
                    return agent_name, agent.suggest_search_space(kwargs.get('model_type', 'rf'))
                elif agent_name == 'deploy':
                    return agent_name, agent.prepare_deployment(kwargs.get('model'), kwargs.get('metadata'))
                elif agent_name == 'insight':
                    return agent_name, agent.suggest_improvements(kwargs.get('metrics', {}))
            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")
                return agent_name, {'error': str(e)}
        
        with ThreadPoolExecutor(max_workers=len(agent_sequence)) as executor:
            futures = {executor.submit(execute_agent, name): name for name in agent_sequence}
            for future in as_completed(futures):
                agent_name, result = future.result()
                if result:
                    results[agent_name] = result
        
        results['workflow_type'] = 'parallel'
        results['agents_used'] = agent_sequence
        results['status'] = 'success'
        
        return results
    
    def _execute_pipeline(self, agent_sequence: List[str], data: np.ndarray,
                         target: Optional[np.ndarray], **kwargs) -> Dict:
        """Execute agents as a pipeline (data flows through)"""
        results = {}
        current_data = data
        current_target = target
        
        for agent_name in agent_sequence:
            if agent_name not in self.agents:
                continue
            
            agent = self.agents[agent_name]
            logger.info(f"[AgentOrchestrator] Pipeline stage: {agent_name}")
            
            try:
                if agent_name == 'data':
                    result = agent.analyze(current_data)
                    results['data_analysis'] = result
                elif agent_name == 'feature':
                    result = agent.suggest_features(current_data, current_target)
                    results['feature_suggestions'] = result
                    # In real implementation, would apply feature engineering
                elif agent_name == 'model':
                    result = agent.recommend_model(current_data, current_target)
                    results['model_recommendation'] = result
                elif agent_name == 'tuning':
                    result = agent.suggest_search_space(kwargs.get('model_type', 'rf'))
                    results['tuning_suggestions'] = result
                elif agent_name == 'deploy':
                    result = agent.prepare_deployment(kwargs.get('model'), kwargs.get('metadata'))
                    results['deployment_info'] = result
                elif agent_name == 'insight':
                    result = agent.suggest_improvements(kwargs.get('metrics', {}))
                    results['insights'] = result
            except Exception as e:
                logger.error(f"Pipeline stage {agent_name} failed: {e}")
                results[f'{agent_name}_error'] = str(e)
        
        results['workflow_type'] = 'pipeline'
        results['agents_used'] = agent_sequence
        results['status'] = 'success'
        
        return results
    
    def _execute_adaptive(self, agent_sequence: List[str], data: np.ndarray,
                         target: Optional[np.ndarray], **kwargs) -> Dict:
        """Execute adaptive workflow (adapts based on results)"""
        results = {}
        current_sequence = agent_sequence.copy()
        
        for agent_name in current_sequence:
            if agent_name not in self.agents:
                continue
            
            agent = self.agents[agent_name]
            logger.info(f"[AgentOrchestrator] Adaptive stage: {agent_name}")
            
            try:
                if agent_name == 'data':
                    result = agent.analyze(data)
                    results['data_analysis'] = result
                    # Adapt: if data quality is poor, add more preprocessing
                    if result.get('suggestions'):
                        if 'missing values' in str(result['suggestions']).lower():
                            if 'feature' not in current_sequence:
                                current_sequence.insert(1, 'feature')
                elif agent_name == 'feature':
                    result = agent.suggest_features(data, target)
                    results['feature_suggestions'] = result
                elif agent_name == 'model':
                    result = agent.recommend_model(data, target)
                    results['model_recommendation'] = result
                elif agent_name == 'tuning':
                    result = agent.suggest_search_space(kwargs.get('model_type', 'rf'))
                    results['tuning_suggestions'] = result
                elif agent_name == 'deploy':
                    result = agent.prepare_deployment(kwargs.get('model'), kwargs.get('metadata'))
                    results['deployment_info'] = result
                elif agent_name == 'insight':
                    result = agent.suggest_improvements(kwargs.get('metrics', {}))
                    results['insights'] = result
            except Exception as e:
                logger.error(f"Adaptive stage {agent_name} failed: {e}")
                results[f'{agent_name}_error'] = str(e)
        
        results['workflow_type'] = 'adaptive'
        results['agents_used'] = current_sequence
        results['status'] = 'success'
        
        return results
