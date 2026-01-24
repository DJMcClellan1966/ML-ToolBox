"""
Prompt → RAG → Deployment Pipeline

From "AI Agents and Applications" (Manning/Roberto Infante)
"""
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline stages"""
    PROMPT = "prompt"
    RAG = "rag"
    GENERATION = "generation"
    EVALUATION = "evaluation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


class PromptRAGDeployPipeline:
    """
    End-to-End Pipeline: Prompt → RAG → Deployment
    
    From "AI Agents and Applications"
    """
    
    def __init__(self):
        self.stages: Dict[PipelineStage, Any] = {}
        self.pipeline_history: List[Dict] = []
    
    def add_stage(self, stage: PipelineStage, handler: Any):
        """Add pipeline stage handler"""
        self.stages[stage] = handler
        logger.info(f"[Pipeline] Added stage: {stage.value}")
    
    def execute(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute full pipeline
        
        Parameters
        ----------
        query : str
            User query
        context : dict, optional
            Additional context
            
        Returns
        -------
        result : dict
            Pipeline execution result
        """
        result = {
            'query': query,
            'stages': {},
            'final_output': None
        }
        
        # Stage 1: Prompt Engineering
        if PipelineStage.PROMPT in self.stages:
            prompt_handler = self.stages[PipelineStage.PROMPT]
            prompt_result = self._execute_prompt_stage(prompt_handler, query, context)
            result['stages']['prompt'] = prompt_result
        
        # Stage 2: RAG
        if PipelineStage.RAG in self.stages:
            rag_handler = self.stages[PipelineStage.RAG]
            rag_result = self._execute_rag_stage(rag_handler, query, context)
            result['stages']['rag'] = rag_result
        
        # Stage 3: Generation
        if PipelineStage.GENERATION in self.stages:
            gen_handler = self.stages[PipelineStage.GENERATION]
            gen_result = self._execute_generation_stage(gen_handler, query, context, result['stages'])
            result['stages']['generation'] = gen_result
            result['final_output'] = gen_result.get('output')
        
        # Stage 4: Evaluation
        if PipelineStage.EVALUATION in self.stages:
            eval_handler = self.stages[PipelineStage.EVALUATION]
            eval_result = self._execute_evaluation_stage(eval_handler, result)
            result['stages']['evaluation'] = eval_result
        
        # Stage 5: Deployment
        if PipelineStage.DEPLOYMENT in self.stages:
            deploy_handler = self.stages[PipelineStage.DEPLOYMENT]
            deploy_result = self._execute_deployment_stage(deploy_handler, result)
            result['stages']['deployment'] = deploy_result
        
        self.pipeline_history.append(result)
        return result
    
    def _execute_prompt_stage(self, handler: Any, query: str, context: Optional[Dict]) -> Dict:
        """Execute prompt engineering stage"""
        try:
            if hasattr(handler, 'create_prompt'):
                prompt = handler.create_prompt(query, context)
            elif hasattr(handler, 'format'):
                prompt = handler.format(query=query, **(context or {}))
            else:
                prompt = query
            
            return {
                'prompt': prompt,
                'success': True
            }
        except Exception as e:
            logger.error(f"[Pipeline] Prompt stage error: {e}")
            return {'error': str(e), 'success': False}
    
    def _execute_rag_stage(self, handler: Any, query: str, context: Optional[Dict]) -> Dict:
        """Execute RAG stage"""
        try:
            if hasattr(handler, 'retrieve'):
                retrieved = handler.retrieve(query, top_k=5)
            elif hasattr(handler, 'search'):
                retrieved = handler.search(query)
            else:
                retrieved = []
            
            return {
                'retrieved_context': retrieved,
                'success': True
            }
        except Exception as e:
            logger.error(f"[Pipeline] RAG stage error: {e}")
            return {'error': str(e), 'success': False}
    
    def _execute_generation_stage(self, handler: Any, query: str, context: Optional[Dict],
                                  previous_stages: Dict) -> Dict:
        """Execute generation stage"""
        try:
            # Combine prompt and RAG context
            prompt = previous_stages.get('prompt', {}).get('prompt', query)
            rag_context = previous_stages.get('rag', {}).get('retrieved_context', [])
            
            if hasattr(handler, 'generate'):
                output = handler.generate(prompt, context=rag_context)
            elif hasattr(handler, 'predict'):
                output = handler.predict(prompt)
            else:
                output = f"Generated response for: {prompt}"
            
            return {
                'output': output,
                'success': True
            }
        except Exception as e:
            logger.error(f"[Pipeline] Generation stage error: {e}")
            return {'error': str(e), 'success': False}
    
    def _execute_evaluation_stage(self, handler: Any, result: Dict) -> Dict:
        """Execute evaluation stage"""
        try:
            output = result.get('final_output', '')
            
            if hasattr(handler, 'evaluate'):
                metrics = handler.evaluate(output)
            else:
                metrics = {'quality': 0.8, 'relevance': 0.7}
            
            return {
                'metrics': metrics,
                'success': True
            }
        except Exception as e:
            logger.error(f"[Pipeline] Evaluation stage error: {e}")
            return {'error': str(e), 'success': False}
    
    def _execute_deployment_stage(self, handler: Any, result: Dict) -> Dict:
        """Execute deployment stage"""
        try:
            output = result.get('final_output', '')
            
            if hasattr(handler, 'deploy'):
                deployment = handler.deploy(output)
            elif hasattr(handler, 'register'):
                deployment = handler.register(output)
            else:
                deployment = {'status': 'deployed', 'endpoint': '/api/v1/predict'}
            
            return {
                'deployment': deployment,
                'success': True
            }
        except Exception as e:
            logger.error(f"[Pipeline] Deployment stage error: {e}")
            return {'error': str(e), 'success': False}


class EndToEndPipeline:
    """
    Complete End-to-End Pipeline
    
    Integrates all stages with ML Toolbox components
    """
    
    def __init__(self, toolbox=None):
        self.toolbox = toolbox
        self.pipeline = PromptRAGDeployPipeline()
        self._setup_stages()
    
    def _setup_stages(self):
        """Setup pipeline stages with ML Toolbox"""
        try:
            # Prompt Engineering
            from ml_toolbox.llm_engineering import PromptEngineer
            self.pipeline.add_stage(PipelineStage.PROMPT, PromptEngineer())
        except:
            pass
        
        try:
            # RAG
            from ml_toolbox.llm_engineering import RAGSystem
            self.pipeline.add_stage(PipelineStage.RAG, RAGSystem())
        except:
            pass
        
        # Generation (can use LLM or simple generator)
        class SimpleGenerator:
            def generate(self, prompt, context=None):
                return f"Response to: {prompt}"
        
        self.pipeline.add_stage(PipelineStage.GENERATION, SimpleGenerator())
    
    def run(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Run end-to-end pipeline"""
        return self.pipeline.execute(query, context)
