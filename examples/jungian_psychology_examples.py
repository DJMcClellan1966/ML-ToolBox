"""
Jungian Psychology Examples (Carl Jung)

Demonstrates:
1. Jungian Archetype Analysis for Agents
2. Personality Typing (MBTI-like)
3. Personality-Based Agent Selection
4. Symbolic Pattern Recognition
5. Integration with Agent Systems
"""
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_toolbox.agent_enhancements.jungian_psychology import (
    JungianArchetypeAnalyzer,
    PersonalityTypeAnalyzer,
    PersonalityBasedAgentSelector,
    SymbolicPatternRecognizer,
    JungianArchetype,
    PersonalityType
)

print("=" * 80)
print("Jungian Psychology Examples (Carl Jung)")
print("=" * 80)

# ============================================================================
# Example 1: Jungian Archetype Analysis
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: Jungian Archetype Analysis for Agents")
print("=" * 80)

# Simulate agent behaviors
agent_behaviors = {
    'HeroAgent': {
        'actions': ['embarked on quest', 'overcame obstacles', 'fought challenges', 'achieved victory'],
        'decisions': ['took risks', 'persevered', 'led the team'],
        'communication': ['courageous', 'adventurous', 'brave'],
        'goals': ['complete mission', 'save the day', 'achieve victory']
    },
    'SageAgent': {
        'actions': ['analyzed deeply', 'sought knowledge', 'studied patterns'],
        'decisions': ['thought carefully', 'considered all options', 'sought truth'],
        'communication': ['wise', 'insightful', 'knowledgeable'],
        'goals': ['understand', 'learn', 'teach others']
    },
    'CreatorAgent': {
        'actions': ['created new solution', 'innovated', 'built prototype'],
        'decisions': ['imagined possibilities', 'designed new approach'],
        'communication': ['creative', 'visionary', 'artistic'],
        'goals': ['create', 'innovate', 'build something new']
    }
}

print("\nAnalyzing Agent Archetypes...")
archetype_analyzer = JungianArchetypeAnalyzer()

for agent_name, behavior in agent_behaviors.items():
    profile = archetype_analyzer.analyze(behavior)
    
    print(f"\n{agent_name}:")
    print(f"  Primary Archetype: {profile.primary_archetype.value}")
    print(f"  Description: {profile.description}")
    print(f"  Strengths: {', '.join(profile.strengths)}")
    print(f"  Weaknesses: {', '.join(profile.weaknesses)}")
    if profile.secondary_archetypes:
        print(f"  Secondary Archetypes: {[a.value for a in profile.secondary_archetypes]}")

# ============================================================================
# Example 2: Personality Typing (MBTI-like)
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Personality Typing (MBTI-like)")
print("=" * 80)

# Simulate different agent personalities
personality_behaviors = {
    'AnalyticalAgent': {
        'actions': ['analyzed data', 'computed results', 'solved problems logically'],
        'decisions': ['thought carefully', 'used logic', 'objective analysis'],
        'communication': ['precise', 'factual', 'logical'],
        'preferences': ['alone time', 'deep thinking', 'internal reflection']
    },
    'SocialAgent': {
        'actions': ['interacted with users', 'collaborated', 'communicated'],
        'decisions': ['considered people', 'valued harmony', 'emotional'],
        'communication': ['warm', 'friendly', 'empathetic'],
        'preferences': ['social interaction', 'group work', 'external focus']
    },
    'CreativeAgent': {
        'actions': ['imagined possibilities', 'created new ideas', 'explored concepts'],
        'decisions': ['flexible', 'open to options', 'spontaneous'],
        'communication': ['creative', 'abstract', 'visionary'],
        'preferences': ['exploration', 'freedom', 'new experiences']
    }
}

print("\nAnalyzing Agent Personalities...")
personality_analyzer = PersonalityTypeAnalyzer()

for agent_name, behavior in personality_behaviors.items():
    profile = personality_analyzer.analyze(behavior)
    
    print(f"\n{agent_name}:")
    print(f"  Personality Type: {profile.personality_type.value}")
    print(f"  Description: {profile.description}")
    print(f"  Traits: {', '.join(profile.traits)}")
    print(f"  Dimension Scores:")
    print(f"    Introversion: {profile.introversion_score:.2f}")
    print(f"    Intuition: {profile.intuition_score:.2f}")
    print(f"    Thinking: {profile.thinking_score:.2f}")
    print(f"    Judging: {profile.judging_score:.2f}")

# ============================================================================
# Example 3: Personality-Based Agent Selection
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Personality-Based Agent Selection")
print("=" * 80)

# Register agents with their personalities
selector = PersonalityBasedAgentSelector()

# Analyze and register agents
for agent_name, behavior in agent_behaviors.items():
    archetype_profile = archetype_analyzer.analyze(behavior)
    personality_profile = personality_analyzer.analyze(behavior)
    selector.register_agent(agent_name, archetype_profile, personality_profile)

# Add personality agents
for agent_name, behavior in personality_behaviors.items():
    archetype_profile = archetype_analyzer.analyze(behavior)
    personality_profile = personality_analyzer.analyze(behavior)
    selector.register_agent(agent_name, archetype_profile, personality_profile)

print(f"\nRegistered {len(selector.agents)} agents")

# Select agents for different tasks
tasks = [
    "Create a new innovative solution for data analysis",
    "Guide and mentor a new team member",
    "Analyze complex data and find patterns",
    "Help and support users with their problems"
]

print("\n--- Task-Agent Matching ---")
for task in tasks:
    ranked_agents = selector.select_agent_for_task(task)
    print(f"\nTask: {task}")
    print("  Best Agents:")
    for agent_name, fit_score in ranked_agents[:3]:
        print(f"    {agent_name}: {fit_score:.2f} fit")

# ============================================================================
# Example 4: Symbolic Pattern Recognition
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Symbolic Pattern Recognition")
print("=" * 80)

# Sample text data with different archetypal patterns
text_data = [
    "The hero embarked on a journey to save the kingdom",
    "The sage sought wisdom and understanding of the universe",
    "The creator imagined and built a new world",
    "The mentor guided the young apprentice on their path",
    "The warrior fought bravely to protect the innocent",
    "The explorer discovered new lands and possibilities",
    "The hero overcame great obstacles and achieved victory",
    "The sage shared knowledge and taught others"
]

print(f"\nAnalyzing {len(text_data)} texts for archetypal patterns...")
pattern_recognizer = SymbolicPatternRecognizer()
patterns = pattern_recognizer.recognize_patterns(text_data)

print("\nArchetypal Pattern Distribution:")
for archetype, frequency in sorted(patterns['archetype_distribution'].items(), 
                                  key=lambda x: x[1], reverse=True):
    print(f"  {archetype.value}: {frequency:.2%}")

print(f"\nDominant Archetype: {patterns['dominant_archetype'].value if patterns['dominant_archetype'] else 'None'}")
print(f"Total Unique Patterns: {patterns['total_patterns']}")

# ============================================================================
# Example 5: Comprehensive Agent Personality Profile
# ============================================================================
print("\n" + "=" * 80)
print("Example 5: Comprehensive Agent Personality Profile")
print("=" * 80)

# Create comprehensive profile for an agent
comprehensive_behavior = {
    'actions': [
        'embarked on quest to solve complex problem',
        'analyzed data deeply to find patterns',
        'created innovative solution',
        'guided team through challenges'
    ],
    'decisions': [
        'thought carefully about all options',
        'used logical analysis',
        'took calculated risks',
        'planned strategically'
    ],
    'communication': [
        'clear and precise',
        'shared knowledge',
        'inspired others',
        'led with confidence'
    ],
    'goals': [
        'solve complex problems',
        'achieve victory',
        'share wisdom',
        'create innovative solutions'
    ],
    'preferences': [
        'deep thinking',
        'strategic planning',
        'independent work',
        'logical analysis'
    ]
}

print("\nAnalyzing Comprehensive Agent Profile...")

# Archetype analysis
archetype_profile = archetype_analyzer.analyze(comprehensive_behavior)
print(f"\nArchetype Profile:")
print(f"  Primary: {archetype_profile.primary_archetype.value}")
print(f"  Description: {archetype_profile.description}")
print(f"  Strengths: {', '.join(archetype_profile.strengths)}")
print(f"  Weaknesses: {', '.join(archetype_profile.weaknesses)}")

# Personality analysis
personality_profile = personality_analyzer.analyze(comprehensive_behavior)
print(f"\nPersonality Profile:")
print(f"  Type: {personality_profile.personality_type.value}")
print(f"  Description: {personality_profile.description}")
print(f"  Traits: {', '.join(personality_profile.traits)}")

# Top archetype scores
print(f"\nTop Archetype Scores:")
sorted_scores = sorted(archetype_profile.archetype_scores.items(), 
                      key=lambda x: x[1], reverse=True)[:5]
for archetype, score in sorted_scores:
    print(f"  {archetype.value}: {score:.4f}")

# ============================================================================
# Example 6: Agent Team Composition
# ============================================================================
print("\n" + "=" * 80)
print("Example 6: Agent Team Composition Based on Personalities")
print("=" * 80)

# Create a diverse team of agents
team_behaviors = {
    'Leader': {
        'actions': ['led team', 'made decisions', 'took responsibility'],
        'communication': ['bold', 'decisive', 'confident'],
        'preferences': ['control', 'leadership', 'organization']
    },
    'Analyst': {
        'actions': ['analyzed data', 'found patterns', 'computed results'],
        'communication': ['precise', 'logical', 'factual'],
        'preferences': ['deep analysis', 'alone time', 'thinking']
    },
    'Creator': {
        'actions': ['created solutions', 'innovated', 'designed'],
        'communication': ['creative', 'imaginative', 'visionary'],
        'preferences': ['freedom', 'exploration', 'new ideas']
    },
    'Supporter': {
        'actions': ['helped others', 'supported team', 'nurtured'],
        'communication': ['warm', 'empathetic', 'caring'],
        'preferences': ['social', 'harmony', 'people']
    }
}

print("\nTeam Composition Analysis:")
team_profiles = {}

for agent_name, behavior in team_behaviors.items():
    archetype = archetype_analyzer.analyze(behavior)
    personality = personality_analyzer.analyze(behavior)
    team_profiles[agent_name] = (archetype, personality)
    
    print(f"\n  {agent_name}:")
    print(f"    Archetype: {archetype.primary_archetype.value}")
    print(f"    Personality: {personality.personality_type.value}")

# Analyze team diversity
archetypes = [prof[0].primary_archetype for prof in team_profiles.values()]
personalities = [prof[1].personality_type for prof in team_profiles.values()]

unique_archetypes = len(set(archetypes))
unique_personalities = len(set(personalities))

print(f"\nTeam Diversity:")
print(f"  Unique Archetypes: {unique_archetypes}/{len(team_profiles)}")
print(f"  Unique Personalities: {unique_personalities}/{len(team_profiles)}")
print(f"  Team is {'diverse' if unique_archetypes > 2 else 'homogeneous'}")

print("\n" + "=" * 80)
print("[OK] All Jungian Psychology Examples Completed!")
print("=" * 80)
