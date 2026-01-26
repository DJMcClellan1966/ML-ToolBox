"""
Turing Test Framework Examples (Alan Turing)

Demonstrates:
1. Turing Test Evaluation
2. Conversational Intelligence Metrics
3. Imitation Game Framework
4. Agent Comparison
"""
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_toolbox.agent_enhancements.turing_test import (
    TuringTestEvaluator,
    ConversationalIntelligenceEvaluator,
    ImitationGameFramework
)

print("=" * 80)
print("Turing Test Framework Examples (Alan Turing)")
print("=" * 80)

# ============================================================================
# Example 1: Turing Test Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: Turing Test Evaluation")
print("=" * 80)

# Simulate agent and human responses
test_questions = [
    "What is the weather like today?",
    "How do you feel about artificial intelligence?",
    "Can you tell me a joke?",
    "What is your favorite color?",
    "How do you solve problems?"
]

# Agent responses (simulated - could be from actual agent)
agent_responses = [
    "The weather is sunny and warm today.",
    "I think artificial intelligence is fascinating and has great potential.",
    "Why did the chicken cross the road? To get to the other side!",
    "My favorite color is blue because it reminds me of the sky.",
    "I solve problems by breaking them down into smaller steps."
]

# Human responses (reference)
human_responses = [
    "It's pretty nice out, sunny and warm.",
    "AI is interesting but also a bit scary. It could change everything.",
    "Hmm, let me think... Why don't scientists trust atoms? Because they make up everything!",
    "I like blue, it's calming and reminds me of the ocean.",
    "I usually start by understanding the problem, then brainstorm solutions."
]

print("\nTest Questions:")
for i, q in enumerate(test_questions, 1):
    print(f"  {i}. {q}")

print("\nRunning Turing Test...")
turing_evaluator = TuringTestEvaluator(pass_threshold=0.7)
turing_result = turing_evaluator.evaluate(
    agent_name="TestAgent",
    agent_responses=agent_responses,
    human_responses=human_responses,
    contexts=test_questions
)

print("\nTuring Test Results:")
print(f"  Agent: {turing_result.agent_name}")
print(f"  Pass Rate: {turing_result.pass_rate:.2%}")
print(f"  Passed Tests: {turing_result.passed_tests}/{turing_result.total_tests}")
print(f"  Human Likeness Score: {turing_result.human_likeness_score:.4f}")
print(f"  Conversational Intelligence: {turing_result.conversational_intelligence:.4f}")
print(f"  Judge Confidence: {turing_result.judge_confidence:.4f}")
print(f"  Passed Turing Test: {'YES' if turing_result.details.get('passed') else 'NO'}")

print("\nIntelligence Metrics:")
intel_metrics = turing_result.details['intelligence_metrics']
print(f"  Coherence: {intel_metrics['coherence']:.4f}")
print(f"  Relevance: {intel_metrics['relevance']:.4f}")
print(f"  Naturalness: {intel_metrics['naturalness']:.4f}")
print(f"  Context Awareness: {intel_metrics['context_awareness']:.4f}")
print(f"  Length Appropriateness: {intel_metrics['length_appropriateness']:.4f}")

# ============================================================================
# Example 2: Conversational Intelligence Metrics
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Conversational Intelligence Metrics")
print("=" * 80)

# More detailed conversation
conversation_contexts = [
    "Hello, how are you?",
    "What did you do today?",
    "Tell me about your hobbies.",
    "What's your opinion on technology?",
    "Do you have any questions for me?"
]

agent_conversation = [
    "Hello! I'm doing well, thank you for asking. How about you?",
    "I spent the day learning and processing information. It's always interesting.",
    "I enjoy reading, problem-solving, and having conversations like this one.",
    "Technology is a powerful tool that can help solve many problems if used responsibly.",
    "Yes, what topics are you most interested in discussing?"
]

human_conversation = [
    "Hi! I'm good, thanks. How are you doing?",
    "I worked on some projects and went for a walk. Pretty productive day.",
    "I like reading, playing music, and hanging out with friends.",
    "Tech is cool but sometimes overwhelming. It changes so fast!",
    "Hmm, maybe what you think about the future of AI?"
]

print("\nEvaluating Conversational Intelligence...")
intelligence_evaluator = ConversationalIntelligenceEvaluator()
intelligence_metrics = intelligence_evaluator.evaluate(
    agent_responses=agent_conversation,
    human_responses=human_conversation,
    contexts=conversation_contexts
)

print("\nConversational Intelligence Metrics:")
print(f"  Coherence Score: {intelligence_metrics.coherence_score:.4f}")
print(f"  Relevance Score: {intelligence_metrics.relevance_score:.4f}")
print(f"  Naturalness Score: {intelligence_metrics.naturalness_score:.4f}")
print(f"  Context Awareness: {intelligence_metrics.context_awareness:.4f}")
print(f"  Length Appropriateness: {intelligence_metrics.response_length_appropriateness:.4f}")
print(f"  Overall Intelligence: {intelligence_metrics.overall_intelligence:.4f}")

# ============================================================================
# Example 3: Imitation Game Framework
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Imitation Game Framework")
print("=" * 80)

# Create imitation game
imitation_game = ImitationGameFramework()

print("\nRunning Imitation Game (Agent vs Human)...")
game_result = imitation_game.run_imitation_game(
    agent_name="TestAgent",
    agent_responses=agent_responses,
    human_responses=human_responses,
    test_questions=test_questions
)

print("\nImitation Game Results:")
print(f"  Agent: {game_result['agent_name']}")
print(f"  Total Questions: {game_result['total_questions']}")
print(f"  Judge Accuracy: {game_result['judge_accuracy']:.2%}")
print(f"  Agent Pass Rate: {game_result['agent_pass_rate']:.2%}")
print(f"  Agent Wins: {'YES' if game_result['agent_wins'] else 'NO'}")

print("\nTuring Test Comparison:")
tt_result = game_result['turing_test_result']
print(f"  Turing Pass Rate: {tt_result['pass_rate']:.2%}")
print(f"  Human Likeness: {tt_result['human_likeness']:.4f}")
print(f"  Intelligence: {tt_result['intelligence']:.4f}")

# ============================================================================
# Example 4: Agent Comparison
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Comparing Two Agents")
print("=" * 80)

# Simulate two different agents
agent1_responses = [
    "The weather is sunny and warm today.",
    "I think artificial intelligence is fascinating and has great potential.",
    "Why did the chicken cross the road? To get to the other side!",
    "My favorite color is blue because it reminds me of the sky.",
    "I solve problems by breaking them down into smaller steps."
]

agent2_responses = [
    "Weather data indicates sunny conditions with temperature 72F.",
    "Artificial intelligence represents computational systems capable of learning.",
    "A chicken crossed a road to reach the opposite side.",
    "Color preference: blue. Reason: association with sky imagery.",
    "Problem-solving methodology: decomposition into sub-problems."
]

print("\nAgent 1: More conversational style")
print("Agent 2: More formal/robotic style")

comparison = imitation_game.compare_agents(
    agent1_name="ConversationalAgent",
    agent1_responses=agent1_responses,
    agent2_name="FormalAgent",
    agent2_responses=agent2_responses,
    human_responses=human_responses,
    test_questions=test_questions
)

print("\nAgent Comparison Results:")
print(f"\n  {comparison['agent1']['name']}:")
print(f"    Pass Rate: {comparison['agent1']['pass_rate']:.2%}")
print(f"    Turing Pass Rate: {comparison['agent1']['turing_pass_rate']:.2%}")
print(f"    Intelligence: {comparison['agent1']['intelligence']:.4f}")

print(f"\n  {comparison['agent2']['name']}:")
print(f"    Pass Rate: {comparison['agent2']['pass_rate']:.2%}")
print(f"    Turing Pass Rate: {comparison['agent2']['turing_pass_rate']:.2%}")
print(f"    Intelligence: {comparison['agent2']['intelligence']:.4f}")

print(f"\n  Winner: {comparison['winner']}")
print(f"  Difference: {comparison['difference']:.2%}")

# ============================================================================
# Example 5: Comprehensive Report
# ============================================================================
print("\n" + "=" * 80)
print("Example 5: Comprehensive Evaluation Report")
print("=" * 80)

# Get reports
turing_report = turing_evaluator.get_report()
game_report = imitation_game.get_report()

print("\nTuring Test Report:")
print(f"  Total Agents Tested: {turing_report['total_agents_tested']}")
print(f"  Average Pass Rate: {turing_report['summary']['avg_pass_rate']:.2%}")
print(f"  Average Intelligence: {turing_report['summary']['avg_intelligence']:.4f}")
print(f"  Agents Passed: {turing_report['summary']['agents_passed']}")

print("\nImitation Game Report:")
print(f"  Total Games: {game_report['total_games']}")
print(f"  Average Agent Pass Rate: {game_report['summary']['avg_agent_pass_rate']:.2%}")
print(f"  Average Judge Accuracy: {game_report['summary']['avg_judge_accuracy']:.2%}")
print(f"  Agents Passed: {game_report['summary']['agents_passed']}")

print("\n" + "=" * 80)
print("[OK] All Turing Test Examples Completed!")
print("=" * 80)
