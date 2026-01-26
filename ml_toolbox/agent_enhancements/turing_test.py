"""
Turing Test Framework (Alan Turing)

Implements:
- Turing Test evaluator
- Conversational intelligence metrics
- Imitation game framework
"""
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TuringTestResult:
    """Turing Test evaluation result"""
    agent_name: str
    pass_rate: float = 0.0
    total_tests: int = 0
    passed_tests: int = 0
    human_likeness_score: float = 0.0
    conversational_intelligence: float = 0.0
    judge_confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationalIntelligenceMetrics:
    """Conversational intelligence metrics"""
    coherence_score: float = 0.0
    relevance_score: float = 0.0
    naturalness_score: float = 0.0
    context_awareness: float = 0.0
    response_length_appropriateness: float = 0.0
    overall_intelligence: float = 0.0


class ConversationalIntelligenceEvaluator:
    """
    Evaluate conversational intelligence
    
    Measures how human-like and intelligent agent conversations are
    """
    
    def __init__(self):
        self.evaluations: List[Dict] = []
    
    def evaluate(self, agent_responses: List[str], 
                human_responses: List[str],
                contexts: Optional[List[str]] = None) -> ConversationalIntelligenceMetrics:
        """
        Evaluate conversational intelligence
        
        Parameters
        ----------
        agent_responses : list of str
            Agent's responses
        human_responses : list of str
            Human reference responses
        contexts : list of str, optional
            Conversation contexts
            
        Returns
        -------
        metrics : ConversationalIntelligenceMetrics
            Intelligence metrics
        """
        if len(agent_responses) != len(human_responses):
            raise ValueError("Agent and human responses must have same length")
        
        # Coherence: How well responses flow together
        coherence = self._calculate_coherence(agent_responses)
        
        # Relevance: How relevant responses are to context
        relevance = self._calculate_relevance(agent_responses, contexts) if contexts else 0.5
        
        # Naturalness: How natural responses sound (compared to human)
        naturalness = self._calculate_naturalness(agent_responses, human_responses)
        
        # Context awareness: How well agent maintains context
        context_awareness = self._calculate_context_awareness(agent_responses, contexts) if contexts else 0.5
        
        # Response length appropriateness
        length_appropriateness = self._calculate_length_appropriateness(agent_responses, human_responses)
        
        # Overall intelligence (weighted average)
        overall = (
            coherence * 0.2 +
            relevance * 0.2 +
            naturalness * 0.25 +
            context_awareness * 0.15 +
            length_appropriateness * 0.2
        )
        
        metrics = ConversationalIntelligenceMetrics(
            coherence_score=coherence,
            relevance_score=relevance,
            naturalness_score=naturalness,
            context_awareness=context_awareness,
            response_length_appropriateness=length_appropriateness,
            overall_intelligence=overall
        )
        
        self.evaluations.append({
            'timestamp': time.time(),
            'metrics': metrics,
            'agent_responses': agent_responses,
            'human_responses': human_responses
        })
        
        return metrics
    
    def _calculate_coherence(self, responses: List[str]) -> float:
        """Calculate coherence score"""
        if len(responses) < 2:
            return 1.0
        
        # Simple coherence: word overlap between consecutive responses
        coherence_scores = []
        for i in range(len(responses) - 1):
            words1 = set(responses[i].lower().split())
            words2 = set(responses[i + 1].lower().split())
            
            if words1 and words2:
                overlap = len(words1 & words2) / max(len(words1), len(words2))
                coherence_scores.append(overlap)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _calculate_relevance(self, responses: List[str], contexts: List[str]) -> float:
        """Calculate relevance to context"""
        if not contexts or len(responses) != len(contexts):
            return 0.5
        
        relevance_scores = []
        for response, context in zip(responses, contexts):
            response_words = set(response.lower().split())
            context_words = set(context.lower().split())
            
            if context_words:
                overlap = len(response_words & context_words) / len(context_words)
                relevance_scores.append(overlap)
        
        return np.mean(relevance_scores) if relevance_scores else 0.5
    
    def _calculate_naturalness(self, agent_responses: List[str], 
                             human_responses: List[str]) -> float:
        """Calculate naturalness (similarity to human responses)"""
        naturalness_scores = []
        
        for agent_resp, human_resp in zip(agent_responses, human_responses):
            # Word overlap
            agent_words = set(agent_resp.lower().split())
            human_words = set(human_resp.lower().split())
            
            if human_words:
                word_overlap = len(agent_words & human_words) / len(human_words)
            else:
                word_overlap = 0.0
            
            # Length similarity
            length_ratio = min(len(agent_resp), len(human_resp)) / max(len(agent_resp), len(human_resp), 1)
            
            # Combined naturalness
            naturalness = (word_overlap * 0.7 + length_ratio * 0.3)
            naturalness_scores.append(naturalness)
        
        return np.mean(naturalness_scores) if naturalness_scores else 0.5
    
    def _calculate_context_awareness(self, responses: List[str], 
                                    contexts: List[str]) -> float:
        """Calculate context awareness"""
        if len(responses) < 2:
            return 1.0
        
        # Check if responses reference previous context
        context_scores = []
        for i in range(1, len(responses)):
            prev_context = contexts[i - 1] if i > 0 else ""
            current_response = responses[i]
            
            if prev_context:
                context_words = set(prev_context.lower().split())
                response_words = set(current_response.lower().split())
                
                if context_words:
                    overlap = len(context_words & response_words) / len(context_words)
                    context_scores.append(overlap)
        
        return np.mean(context_scores) if context_scores else 0.5
    
    def _calculate_length_appropriateness(self, agent_responses: List[str],
                                         human_responses: List[str]) -> float:
        """Calculate if response lengths are appropriate"""
        length_ratios = []
        
        for agent_resp, human_resp in zip(agent_responses, human_responses):
            if len(human_resp) > 0:
                ratio = min(len(agent_resp), len(human_resp)) / max(len(agent_resp), len(human_resp), 1)
                length_ratios.append(ratio)
        
        return np.mean(length_ratios) if length_ratios else 0.5


class TuringTestEvaluator:
    """
    Turing Test Evaluator
    
    Implements the classic Turing Test: Can an agent pass as human?
    """
    
    def __init__(self, pass_threshold: float = 0.7):
        """
        Initialize Turing Test evaluator
        
        Parameters
        ----------
        pass_threshold : float
            Threshold for passing the test (0-1)
        """
        self.pass_threshold = pass_threshold
        self.test_results: List[TuringTestResult] = []
        self.intelligence_evaluator = ConversationalIntelligenceEvaluator()
    
    def evaluate(self, agent_name: str,
                agent_responses: List[str],
                human_responses: List[str],
                judge_responses: Optional[List[str]] = None,
                contexts: Optional[List[str]] = None) -> TuringTestResult:
        """
        Run Turing Test
        
        Parameters
        ----------
        agent_name : str
            Name of agent being tested
        agent_responses : list of str
            Agent's responses to test questions
        human_responses : list of str
            Human reference responses to same questions
        judge_responses : list of str, optional
            Judge's guesses ("agent" or "human")
            If not provided, uses automatic evaluation
        contexts : list of str, optional
            Conversation contexts/questions
            
        Returns
        -------
        result : TuringTestResult
            Turing Test result
        """
        if len(agent_responses) != len(human_responses):
            raise ValueError("Agent and human responses must have same length")
        
        total_tests = len(agent_responses)
        
        # If judge responses provided, use them
        if judge_responses:
            passed_tests = sum(1 for guess in judge_responses 
                            if guess.lower() in ['human', 'h'])
            pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
            judge_confidence = 0.7  # Default if judge provided
        else:
            # Automatic evaluation: compare to human responses
            passed_tests = 0
            judge_guesses = []
            
            for agent_resp, human_resp in zip(agent_responses, human_responses):
                # Simulate judge decision
                guess = self._simulate_judge(agent_resp, human_resp)
                judge_guesses.append(guess)
                if guess == 'human':
                    passed_tests += 1
            
            pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
            judge_confidence = self._calculate_judge_confidence(agent_responses, human_responses)
        
        # Calculate conversational intelligence
        intelligence_metrics = self.intelligence_evaluator.evaluate(
            agent_responses, human_responses, contexts
        )
        
        # Human likeness score (combination of pass rate and intelligence)
        human_likeness = (pass_rate * 0.6 + intelligence_metrics.overall_intelligence * 0.4)
        
        result = TuringTestResult(
            agent_name=agent_name,
            pass_rate=pass_rate,
            total_tests=total_tests,
            passed_tests=passed_tests,
            human_likeness_score=human_likeness,
            conversational_intelligence=intelligence_metrics.overall_intelligence,
            judge_confidence=judge_confidence,
            details={
                'intelligence_metrics': {
                    'coherence': intelligence_metrics.coherence_score,
                    'relevance': intelligence_metrics.relevance_score,
                    'naturalness': intelligence_metrics.naturalness_score,
                    'context_awareness': intelligence_metrics.context_awareness,
                    'length_appropriateness': intelligence_metrics.response_length_appropriateness
                },
                'passed': human_likeness >= self.pass_threshold
            }
        )
        
        self.test_results.append(result)
        
        return result
    
    def _simulate_judge(self, agent_response: str, human_response: str) -> str:
        """
        Simulate judge decision (agent or human)
        
        In real Turing Test, a human judge makes this decision.
        Here we simulate based on similarity to human response.
        """
        # Calculate similarity
        agent_words = set(agent_response.lower().split())
        human_words = set(human_response.lower().split())
        
        if human_words:
            word_overlap = len(agent_words & human_words) / len(human_words)
        else:
            word_overlap = 0.0
        
        # Length similarity
        length_ratio = min(len(agent_response), len(human_response)) / max(
            len(agent_response), len(human_response), 1
        )
        
        # Naturalness indicators
        has_common_words = len(agent_words & {'the', 'a', 'an', 'is', 'are', 'was', 'were'}) > 0
        has_punctuation = any(c in agent_response for c in ['.', '!', '?', ','])
        
        # Combined score
        similarity_score = (
            word_overlap * 0.4 +
            length_ratio * 0.3 +
            (0.2 if has_common_words else 0.0) +
            (0.1 if has_punctuation else 0.0)
        )
        
        # Judge guesses "human" if similarity is high enough
        return 'human' if similarity_score >= 0.6 else 'agent'
    
    def _calculate_judge_confidence(self, agent_responses: List[str],
                                   human_responses: List[str]) -> float:
        """Calculate judge confidence (how clear the distinction is)"""
        similarities = []
        
        for agent_resp, human_resp in zip(agent_responses, human_responses):
            agent_words = set(agent_resp.lower().split())
            human_words = set(human_resp.lower().split())
            
            if human_words:
                similarity = len(agent_words & human_words) / len(human_words)
                similarities.append(similarity)
        
        if not similarities:
            return 0.5
        
        # High confidence if similarities are consistently high or low
        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        
        # Confidence is high when std is low (consistent) or avg is extreme
        confidence = 1.0 - min(std_similarity, 0.5) + abs(avg_similarity - 0.5)
        return min(max(confidence, 0.0), 1.0)
    
    def get_report(self) -> Dict[str, Any]:
        """Get Turing Test report"""
        if not self.test_results:
            return {'message': 'No tests conducted yet'}
        
        return {
            'total_agents_tested': len(self.test_results),
            'results': [
                {
                    'agent': r.agent_name,
                    'pass_rate': r.pass_rate,
                    'human_likeness': r.human_likeness_score,
                    'intelligence': r.conversational_intelligence,
                    'passed': r.details.get('passed', False)
                }
                for r in self.test_results
            ],
            'summary': {
                'avg_pass_rate': np.mean([r.pass_rate for r in self.test_results]),
                'avg_intelligence': np.mean([r.conversational_intelligence for r in self.test_results]),
                'agents_passed': sum(1 for r in self.test_results if r.details.get('passed', False))
            }
        }


class ImitationGameFramework:
    """
    Imitation Game Framework
    
    A/B testing framework for comparing agents vs humans
    """
    
    def __init__(self):
        self.games: List[Dict] = []
        self.turing_evaluator = TuringTestEvaluator()
    
    def run_imitation_game(self, agent_name: str,
                          agent_responses: List[str],
                          human_responses: List[str],
                          test_questions: List[str],
                          judges: Optional[List[Callable]] = None) -> Dict[str, Any]:
        """
        Run imitation game (A/B test: agent vs human)
        
        Parameters
        ----------
        agent_name : str
            Name of agent
        agent_responses : list of str
            Agent's responses
        human_responses : list of str
            Human's responses
        test_questions : list of str
            Questions that were asked
        judges : list of callables, optional
            Judge functions that take (response, question) -> "agent" or "human"
            If not provided, uses automatic evaluation
            
        Returns
        -------
        result : dict
            Imitation game result
        """
        if len(agent_responses) != len(human_responses) != len(test_questions):
            raise ValueError("All lists must have same length")
        
        # Randomly mix agent and human responses
        import random
        mixed_responses = []
        response_labels = []
        
        for i in range(len(agent_responses)):
            # Randomly choose agent or human response
            if random.random() < 0.5:
                mixed_responses.append(agent_responses[i])
                response_labels.append('agent')
            else:
                mixed_responses.append(human_responses[i])
                response_labels.append('human')
        
        # Judges guess which is which
        judge_guesses = []
        if judges:
            for judge in judges:
                guesses = []
                for response, question in zip(mixed_responses, test_questions):
                    guess = judge(response, question)
                    guesses.append(guess)
                judge_guesses.append(guesses)
        else:
            # Automatic evaluation
            guesses = []
            for response, question, label in zip(mixed_responses, test_questions, response_labels):
                # Simulate judge
                guess = self._simulate_judge_guess(response, question, label)
                guesses.append(guess)
            judge_guesses.append(guesses)
        
        # Calculate accuracy for each judge
        judge_accuracies = []
        for guesses in judge_guesses:
            correct = sum(1 for guess, label in zip(guesses, response_labels) 
                         if guess.lower() == label.lower())
            accuracy = correct / len(response_labels) if response_labels else 0.0
            judge_accuracies.append(accuracy)
        
        # Agent "wins" if judges can't tell the difference (low accuracy)
        # Human "wins" if judges can tell the difference (high accuracy)
        avg_judge_accuracy = np.mean(judge_accuracies) if judge_accuracies else 0.5
        
        # Agent pass rate: inverse of judge accuracy (lower accuracy = agent more human-like)
        agent_pass_rate = 1.0 - avg_judge_accuracy
        
        # Run Turing Test for comparison
        turing_result = self.turing_evaluator.evaluate(
            agent_name, agent_responses, human_responses, contexts=test_questions
        )
        
        game_result = {
            'agent_name': agent_name,
            'total_questions': len(test_questions),
            'judge_accuracy': avg_judge_accuracy,
            'agent_pass_rate': agent_pass_rate,
            'agent_wins': agent_pass_rate > 0.5,
            'turing_test_result': {
                'pass_rate': turing_result.pass_rate,
                'human_likeness': turing_result.human_likeness_score,
                'intelligence': turing_result.conversational_intelligence
            },
            'details': {
                'response_labels': response_labels,
                'judge_guesses': judge_guesses,
                'judge_accuracies': judge_accuracies
            }
        }
        
        self.games.append(game_result)
        
        return game_result
    
    def _simulate_judge_guess(self, response: str, question: str, actual_label: str) -> str:
        """Simulate judge guess"""
        # Simple heuristic: check response characteristics
        has_common_words = len(set(response.lower().split()) & 
                              {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'I', 'you'}) > 0
        has_punctuation = any(c in response for c in ['.', '!', '?', ','])
        reasonable_length = 10 <= len(response.split()) <= 100
        
        # If response looks natural, guess human; otherwise agent
        if has_common_words and has_punctuation and reasonable_length:
            return 'human'
        else:
            return 'agent'
    
    def compare_agents(self, agent1_name: str, agent1_responses: List[str],
                      agent2_name: str, agent2_responses: List[str],
                      human_responses: List[str],
                      test_questions: List[str]) -> Dict[str, Any]:
        """
        Compare two agents in imitation game
        
        Parameters
        ----------
        agent1_name : str
            First agent name
        agent1_responses : list of str
            First agent's responses
        agent2_name : str
            Second agent name
        agent2_responses : list of str
            Second agent's responses
        human_responses : list of str
            Human reference responses
        test_questions : list of str
            Test questions
            
        Returns
        -------
        comparison : dict
            Comparison result
        """
        # Test agent 1
        game1 = self.run_imitation_game(
            agent1_name, agent1_responses, human_responses, test_questions
        )
        
        # Test agent 2
        game2 = self.run_imitation_game(
            agent2_name, agent2_responses, human_responses, test_questions
        )
        
        return {
            'agent1': {
                'name': agent1_name,
                'pass_rate': game1['agent_pass_rate'],
                'turing_pass_rate': game1['turing_test_result']['pass_rate'],
                'intelligence': game1['turing_test_result']['intelligence']
            },
            'agent2': {
                'name': agent2_name,
                'pass_rate': game2['agent_pass_rate'],
                'turing_pass_rate': game2['turing_test_result']['pass_rate'],
                'intelligence': game2['turing_test_result']['intelligence']
            },
            'winner': agent1_name if game1['agent_pass_rate'] > game2['agent_pass_rate'] else agent2_name,
            'difference': abs(game1['agent_pass_rate'] - game2['agent_pass_rate'])
        }
    
    def get_report(self) -> Dict[str, Any]:
        """Get imitation game report"""
        if not self.games:
            return {'message': 'No games conducted yet'}
        
        return {
            'total_games': len(self.games),
            'games': self.games,
            'summary': {
                'avg_agent_pass_rate': np.mean([g['agent_pass_rate'] for g in self.games]),
                'avg_judge_accuracy': np.mean([g['judge_accuracy'] for g in self.games]),
                'agents_passed': sum(1 for g in self.games if g['agent_wins'])
            }
        }
