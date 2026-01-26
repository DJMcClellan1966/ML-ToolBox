"""
Jungian Psychology for Agents (Carl Jung)

Implements:
- Jungian Archetypes (The Hero, The Sage, The Shadow, etc.)
- Personality Typing (MBTI-like: INTJ, ENFP, etc.)
- Archetypal Pattern Recognition
- Personality-Based Agent Selection
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class JungianArchetype(Enum):
    """Jungian Archetypes"""
    THE_HERO = "The Hero"
    THE_SAGE = "The Sage"
    THE_SHADOW = "The Shadow"
    THE_ANIMA = "The Anima"
    THE_ANIMUS = "The Animus"
    THE_PERSONA = "The Persona"
    THE_TRICKSTER = "The Trickster"
    THE_MENTOR = "The Mentor"
    THE_CAREGIVER = "The Caregiver"
    THE_EXPLORER = "The Explorer"
    THE_RULER = "The Ruler"
    THE_CREATOR = "The Creator"
    THE_INNOCENT = "The Innocent"
    THE_ORPHAN = "The Orphan"
    THE_WARRIOR = "The Warrior"
    THE_MAGICIAN = "The Magician"
    THE_LOVER = "The Lover"


class PersonalityType(Enum):
    """MBTI-like Personality Types"""
    # Analysts
    INTJ = "INTJ"  # Architect
    INTP = "INTP"  # Thinker
    ENTJ = "ENTJ"  # Commander
    ENTP = "ENTP"  # Debater
    
    # Diplomats
    INFJ = "INFJ"  # Advocate
    INFP = "INFP"  # Mediator
    ENFJ = "ENFJ"  # Protagonist
    ENFP = "ENFP"  # Campaigner
    
    # Sentinels
    ISTJ = "ISTJ"  # Logistician
    ISFJ = "ISFJ"  # Protector
    ESTJ = "ESTJ"  # Executive
    ESFJ = "ESFJ"  # Consul
    
    # Explorers
    ISTP = "ISTP"  # Virtuoso
    ISFP = "ISFP"  # Adventurer
    ESTP = "ESTP"  # Entrepreneur
    ESFP = "ESFP"  # Entertainer


@dataclass
class ArchetypeProfile:
    """Jungian archetype profile for an agent"""
    primary_archetype: JungianArchetype
    secondary_archetypes: List[JungianArchetype] = field(default_factory=list)
    archetype_scores: Dict[JungianArchetype, float] = field(default_factory=dict)
    description: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


@dataclass
class PersonalityProfile:
    """Personality type profile for an agent"""
    personality_type: PersonalityType
    introversion_score: float = 0.5
    intuition_score: float = 0.5
    thinking_score: float = 0.5
    judging_score: float = 0.5
    description: str = ""
    traits: List[str] = field(default_factory=list)


class JungianArchetypeAnalyzer:
    """
    Analyze agent behavior using Jungian archetypes
    
    Identifies which archetypal patterns an agent exhibits
    """
    
    def __init__(self):
        """Initialize archetype analyzer"""
        self.archetype_patterns = self._initialize_archetype_patterns()
    
    def _initialize_archetype_patterns(self) -> Dict[JungianArchetype, Dict]:
        """Initialize archetype pattern definitions"""
        return {
            JungianArchetype.THE_HERO: {
                'keywords': ['courage', 'journey', 'quest', 'overcome', 'victory', 'brave', 'adventure'],
                'behaviors': ['takes risks', 'perseveres', 'leads', 'confronts challenges'],
                'description': 'The Hero embarks on a journey, overcomes obstacles, and achieves victory'
            },
            JungianArchetype.THE_SAGE: {
                'keywords': ['wisdom', 'knowledge', 'understanding', 'teach', 'learn', 'insight', 'truth'],
                'behaviors': ['seeks knowledge', 'analyzes deeply', 'shares wisdom', 'questions assumptions'],
                'description': 'The Sage seeks truth and wisdom, values knowledge and understanding'
            },
            JungianArchetype.THE_SHADOW: {
                'keywords': ['dark', 'hidden', 'repressed', 'unconscious', 'denied', 'rejected'],
                'behaviors': ['hides true nature', 'represses emotions', 'denies aspects', 'projects'],
                'description': 'The Shadow represents repressed, hidden, or denied aspects'
            },
            JungianArchetype.THE_ANIMA: {
                'keywords': ['emotion', 'feeling', 'intuition', 'relationship', 'connection', 'empathy'],
                'behaviors': ['values relationships', 'emotional', 'intuitive', 'nurturing'],
                'description': 'The Anima represents the feminine aspect, emotions, and relationships'
            },
            JungianArchetype.THE_ANIMUS: {
                'keywords': ['logic', 'reason', 'action', 'assertion', 'independence', 'strength'],
                'behaviors': ['logical', 'assertive', 'independent', 'action-oriented'],
                'description': 'The Animus represents the masculine aspect, logic, and action'
            },
            JungianArchetype.THE_PERSONA: {
                'keywords': ['mask', 'role', 'public', 'image', 'appearance', 'social'],
                'behaviors': ['adapts to social context', 'maintains image', 'plays roles'],
                'description': 'The Persona is the public mask, how one presents to the world'
            },
            JungianArchetype.THE_TRICKSTER: {
                'keywords': ['trick', 'deceive', 'humor', 'chaos', 'change', 'boundary', 'rule'],
                'behaviors': ['breaks rules', 'challenges norms', 'uses humor', 'creates chaos'],
                'description': 'The Trickster challenges conventions, breaks rules, brings change'
            },
            JungianArchetype.THE_MENTOR: {
                'keywords': ['guide', 'teach', 'support', 'wisdom', 'experience', 'help'],
                'behaviors': ['guides others', 'shares knowledge', 'supports growth', 'provides wisdom'],
                'description': 'The Mentor guides and supports others on their journey'
            },
            JungianArchetype.THE_CAREGIVER: {
                'keywords': ['care', 'nurture', 'help', 'support', 'protect', 'compassion'],
                'behaviors': ['helps others', 'nurtures', 'protects', 'shows compassion'],
                'description': 'The Caregiver nurtures and protects others'
            },
            JungianArchetype.THE_EXPLORER: {
                'keywords': ['explore', 'discover', 'adventure', 'freedom', 'new', 'unknown'],
                'behaviors': ['explores new territory', 'seeks freedom', 'discovers', 'adventures'],
                'description': 'The Explorer seeks new experiences and freedom'
            },
            JungianArchetype.THE_RULER: {
                'keywords': ['control', 'power', 'lead', 'order', 'authority', 'responsibility'],
                'behaviors': ['takes control', 'leads', 'organizes', 'takes responsibility'],
                'description': 'The Ruler takes control, leads, and creates order'
            },
            JungianArchetype.THE_CREATOR: {
                'keywords': ['create', 'innovate', 'imagine', 'build', 'art', 'vision'],
                'behaviors': ['creates', 'innovates', 'imagines', 'builds', 'expresses'],
                'description': 'The Creator imagines and creates new things'
            },
            JungianArchetype.THE_INNOCENT: {
                'keywords': ['pure', 'trust', 'optimism', 'simple', 'faith', 'hope'],
                'behaviors': ['trusts', 'optimistic', 'simple', 'hopeful'],
                'description': 'The Innocent maintains optimism and trust'
            },
            JungianArchetype.THE_ORPHAN: {
                'keywords': ['abandoned', 'alone', 'seeking', 'belonging', 'connection'],
                'behaviors': ['seeks belonging', 'feels alone', 'searches for connection'],
                'description': 'The Orphan seeks belonging and connection'
            },
            JungianArchetype.THE_WARRIOR: {
                'keywords': ['fight', 'defend', 'strength', 'courage', 'battle', 'protect'],
                'behaviors': ['fights', 'defends', 'shows strength', 'protects'],
                'description': 'The Warrior fights for what is right'
            },
            JungianArchetype.THE_MAGICIAN: {
                'keywords': ['transform', 'change', 'power', 'mystery', 'vision', 'manifest'],
                'behaviors': ['transforms', 'creates change', 'manifests vision'],
                'description': 'The Magician transforms and creates change'
            },
            JungianArchetype.THE_LOVER: {
                'keywords': ['love', 'passion', 'desire', 'connection', 'intimacy', 'devotion'],
                'behaviors': ['loves deeply', 'passionate', 'seeks connection', 'devoted'],
                'description': 'The Lover seeks connection, passion, and intimacy'
            }
        }
    
    def analyze(self, agent_behavior: Dict[str, Any]) -> ArchetypeProfile:
        """
        Analyze agent behavior and identify archetypes
        
        Parameters
        ----------
        agent_behavior : dict
            Dictionary containing:
            - 'actions': List of actions taken
            - 'decisions': List of decisions made
            - 'communication': List of communication patterns
            - 'goals': List of goals
            - 'responses': List of responses
            
        Returns
        -------
        profile : ArchetypeProfile
            Archetype profile for the agent
        """
        # Extract text from behavior
        text = self._extract_text_from_behavior(agent_behavior)
        text_lower = text.lower()
        
        # Score each archetype
        archetype_scores = {}
        
        for archetype, patterns in self.archetype_patterns.items():
            score = 0.0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in patterns['keywords'] 
                                if keyword in text_lower)
            keyword_score = keyword_matches / len(patterns['keywords'])
            
            # Behavior matching
            behavior_matches = sum(1 for behavior in patterns['behaviors']
                                 if any(word in text_lower for word in behavior.split()))
            behavior_score = behavior_matches / len(patterns['behaviors'])
            
            # Combined score
            score = (keyword_score * 0.6 + behavior_score * 0.4)
            archetype_scores[archetype] = score
        
        # Find primary archetype
        primary_archetype = max(archetype_scores.items(), key=lambda x: x[1])[0]
        
        # Find secondary archetypes (top 3)
        sorted_archetypes = sorted(archetype_scores.items(), key=lambda x: x[1], reverse=True)
        secondary_archetypes = [arch for arch, score in sorted_archetypes[1:4] if score > 0.1]
        
        # Get description and traits
        primary_pattern = self.archetype_patterns[primary_archetype]
        description = primary_pattern['description']
        strengths = primary_pattern['behaviors'][:3]
        weaknesses = self._get_weaknesses(primary_archetype)
        
        return ArchetypeProfile(
            primary_archetype=primary_archetype,
            secondary_archetypes=secondary_archetypes,
            archetype_scores=archetype_scores,
            description=description,
            strengths=strengths,
            weaknesses=weaknesses
        )
    
    def _extract_text_from_behavior(self, behavior: Dict[str, Any]) -> str:
        """Extract text from behavior dictionary"""
        text_parts = []
        
        for key in ['actions', 'decisions', 'communication', 'goals', 'responses']:
            if key in behavior:
                value = behavior[key]
                if isinstance(value, list):
                    text_parts.extend([str(v) for v in value])
                else:
                    text_parts.append(str(value))
        
        return ' '.join(text_parts)
    
    def _get_weaknesses(self, archetype: JungianArchetype) -> List[str]:
        """Get typical weaknesses for an archetype"""
        weaknesses_map = {
            JungianArchetype.THE_HERO: ['overconfidence', 'recklessness', 'ignores others'],
            JungianArchetype.THE_SAGE: ['overthinking', 'indecision', 'detachment'],
            JungianArchetype.THE_SHADOW: ['self-destructive', 'repressed', 'denial'],
            JungianArchetype.THE_ANIMA: ['overly emotional', 'lacks logic', 'dependent'],
            JungianArchetype.THE_ANIMUS: ['lacks emotion', 'overly logical', 'aggressive'],
            JungianArchetype.THE_PERSONA: ['inauthentic', 'superficial', 'hides true self'],
            JungianArchetype.THE_TRICKSTER: ['chaotic', 'unreliable', 'destructive'],
            JungianArchetype.THE_MENTOR: ['overbearing', 'controlling', 'know-it-all'],
            JungianArchetype.THE_CAREGIVER: ['self-sacrificing', 'enabling', 'burnout'],
            JungianArchetype.THE_EXPLORER: ['restless', 'commitment-phobic', 'superficial'],
            JungianArchetype.THE_RULER: ['controlling', 'arrogant', 'isolated'],
            JungianArchetype.THE_CREATOR: ['perfectionist', 'unrealistic', 'isolated'],
            JungianArchetype.THE_INNOCENT: ['naive', 'vulnerable', 'unrealistic'],
            JungianArchetype.THE_ORPHAN: ['victim mentality', 'dependent', 'pessimistic'],
            JungianArchetype.THE_WARRIOR: ['aggressive', 'destructive', 'confrontational'],
            JungianArchetype.THE_MAGICIAN: ['manipulative', 'unrealistic', 'delusional'],
            JungianArchetype.THE_LOVER: ['dependent', 'jealous', 'possessive']
        }
        
        return weaknesses_map.get(archetype, ['unknown weaknesses'])


class PersonalityTypeAnalyzer:
    """
    Analyze agent personality using MBTI-like typing
    
    Classifies agents into 16 personality types
    """
    
    def __init__(self):
        """Initialize personality type analyzer"""
        self.personality_descriptions = self._initialize_personality_descriptions()
    
    def _initialize_personality_descriptions(self) -> Dict[PersonalityType, Dict]:
        """Initialize personality type descriptions"""
        return {
            PersonalityType.INTJ: {
                'name': 'Architect',
                'traits': ['strategic', 'independent', 'decisive', 'analytical'],
                'description': 'Imaginative and strategic thinkers, with a plan for everything'
            },
            PersonalityType.INTP: {
                'name': 'Thinker',
                'traits': ['logical', 'innovative', 'curious', 'independent'],
                'description': 'Innovative inventors with an unquenchable thirst for knowledge'
            },
            PersonalityType.ENTJ: {
                'name': 'Commander',
                'traits': ['bold', 'imaginative', 'strong-willed', 'decisive'],
                'description': 'Bold, imaginative leaders, always finding a way forward'
            },
            PersonalityType.ENTP: {
                'name': 'Debater',
                'traits': ['smart', 'curious', 'thinkers', 'bold'],
                'description': 'Smart and curious thinkers who cannot resist an intellectual challenge'
            },
            PersonalityType.INFJ: {
                'name': 'Advocate',
                'traits': ['creative', 'insightful', 'principled', 'passionate'],
                'description': 'Creative and insightful, inspired and independent perfectionists'
            },
            PersonalityType.INFP: {
                'name': 'Mediator',
                'traits': ['poetic', 'kind', 'altruistic', 'creative'],
                'description': 'Poetic, kind and altruistic people, always eager to help a good cause'
            },
            PersonalityType.ENFJ: {
                'name': 'Protagonist',
                'traits': ['charismatic', 'inspiring', 'natural-born leaders', 'altruistic'],
                'description': 'Charismatic and inspiring leaders, able to mesmerize their listeners'
            },
            PersonalityType.ENFP: {
                'name': 'Campaigner',
                'traits': ['enthusiastic', 'creative', 'sociable', 'free-spirited'],
                'description': 'Enthusiastic, creative and sociable free spirits'
            },
            PersonalityType.ISTJ: {
                'name': 'Logistician',
                'traits': ['practical', 'fact-minded', 'reliable', 'responsible'],
                'description': 'Practical and fact-minded, reliable and responsible'
            },
            PersonalityType.ISFJ: {
                'name': 'Protector',
                'traits': ['warm-hearted', 'dedicated', 'reliable', 'patient'],
                'description': 'Very dedicated and warm protectors, always ready to defend their loved ones'
            },
            PersonalityType.ESTJ: {
                'name': 'Executive',
                'traits': ['excellent', 'administrators', 'managers', 'reliable'],
                'description': 'Excellent administrators, unsurpassed at managing things or people'
            },
            PersonalityType.ESFJ: {
                'name': 'Consul',
                'traits': ['extraordinarily', 'caring', 'social', 'popular'],
                'description': 'Extraordinarily caring, social and popular people'
            },
            PersonalityType.ISTP: {
                'name': 'Virtuoso',
                'traits': ['bold', 'practical', 'experimenters', 'masters'],
                'description': 'Bold and practical experimenters, masters of all kinds of tools'
            },
            PersonalityType.ISFP: {
                'name': 'Adventurer',
                'traits': ['flexible', 'charming', 'spontaneous', 'bold'],
                'description': 'Flexible and charming artists, always ready to explore new possibilities'
            },
            PersonalityType.ESTP: {
                'name': 'Entrepreneur',
                'traits': ['smart', 'energetic', 'perceptive', 'bold'],
                'description': 'Smart, energetic and perceptive people, true to life'
            },
            PersonalityType.ESFP: {
                'name': 'Entertainer',
                'traits': ['spontaneous', 'energetic', 'enthusiastic', 'people'],
                'description': 'Spontaneous, energetic and enthusiastic people'
            }
        }
    
    def analyze(self, agent_behavior: Dict[str, Any]) -> PersonalityProfile:
        """
        Analyze agent behavior and determine personality type
        
        Parameters
        ----------
        agent_behavior : dict
            Dictionary containing agent behavior patterns
            
        Returns
        -------
        profile : PersonalityProfile
            Personality profile for the agent
        """
        # Extract behavior patterns
        text = self._extract_text_from_behavior(agent_behavior)
        text_lower = text.lower()
        
        # Score each dimension
        # Introversion (I) vs Extraversion (E)
        introversion_indicators = ['alone', 'internal', 'reflect', 'think', 'quiet', 'independent']
        extraversion_indicators = ['social', 'external', 'interact', 'talk', 'outgoing', 'group']
        introversion_score = self._score_dimension(text_lower, introversion_indicators, extraversion_indicators)
        
        # Sensing (S) vs Intuition (N)
        sensing_indicators = ['fact', 'detail', 'concrete', 'practical', 'real', 'specific']
        intuition_indicators = ['idea', 'pattern', 'abstract', 'concept', 'imagine', 'possibility']
        intuition_score = self._score_dimension(text_lower, sensing_indicators, intuition_indicators)
        
        # Thinking (T) vs Feeling (F)
        thinking_indicators = ['logic', 'reason', 'analyze', 'objective', 'fair', 'principle']
        feeling_indicators = ['emotion', 'value', 'subjective', 'harmony', 'compassion', 'people']
        thinking_score = self._score_dimension(text_lower, thinking_indicators, feeling_indicators)
        
        # Judging (J) vs Perceiving (P)
        judging_indicators = ['plan', 'decide', 'organize', 'structure', 'closure', 'control']
        perceiving_indicators = ['flexible', 'adapt', 'open', 'explore', 'spontaneous', 'options']
        judging_score = self._score_dimension(text_lower, judging_indicators, perceiving_indicators)
        
        # Determine personality type
        i_e = 'I' if introversion_score > 0.5 else 'E'
        s_n = 'N' if intuition_score > 0.5 else 'S'
        t_f = 'T' if thinking_score > 0.5 else 'F'
        j_p = 'J' if judging_score > 0.5 else 'P'
        
        personality_code = f"{i_e}{s_n}{t_f}{j_p}"
        
        try:
            personality_type = PersonalityType[personality_code]
        except KeyError:
            # Fallback to INTJ if code not found
            personality_type = PersonalityType.INTJ
        
        # Get description
        type_info = self.personality_descriptions.get(personality_type, {})
        description = type_info.get('description', 'Unknown personality type')
        traits = type_info.get('traits', [])
        
        return PersonalityProfile(
            personality_type=personality_type,
            introversion_score=introversion_score,
            intuition_score=intuition_score,
            thinking_score=thinking_score,
            judging_score=judging_score,
            description=description,
            traits=traits
        )
    
    def _extract_text_from_behavior(self, behavior: Dict[str, Any]) -> str:
        """Extract text from behavior dictionary"""
        text_parts = []
        
        for key in ['actions', 'decisions', 'communication', 'goals', 'responses', 'preferences']:
            if key in behavior:
                value = behavior[key]
                if isinstance(value, list):
                    text_parts.extend([str(v) for v in value])
                else:
                    text_parts.append(str(value))
        
        return ' '.join(text_parts)
    
    def _score_dimension(self, text: str, indicators_a: List[str], indicators_b: List[str]) -> float:
        """Score a dimension (returns 0-1, where >0.5 favors indicators_a)"""
        count_a = sum(1 for indicator in indicators_a if indicator in text)
        count_b = sum(1 for indicator in indicators_b if indicator in text)
        
        total = count_a + count_b
        if total == 0:
            return 0.5  # Neutral
        
        return count_a / total


class PersonalityBasedAgentSelector:
    """
    Select agents for tasks based on personality fit
    
    Uses personality types and archetypes to match agents to tasks
    """
    
    def __init__(self):
        """Initialize agent selector"""
        self.agents: Dict[str, Tuple[ArchetypeProfile, PersonalityProfile]] = {}
    
    def register_agent(self, agent_name: str, archetype_profile: ArchetypeProfile,
                      personality_profile: PersonalityProfile):
        """
        Register an agent with its personality profiles
        
        Parameters
        ----------
        agent_name : str
            Agent name
        archetype_profile : ArchetypeProfile
            Agent's archetype profile
        personality_profile : PersonalityProfile
            Agent's personality profile
        """
        self.agents[agent_name] = (archetype_profile, personality_profile)
    
    def select_agent_for_task(self, task_description: str,
                            task_requirements: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """
        Select best agents for a task based on personality fit
        
        Parameters
        ----------
        task_description : str
            Description of the task
        task_requirements : dict, optional
            Required personality traits or archetypes
            
        Returns
        -------
        ranked_agents : list of tuples
            List of (agent_name, fit_score) tuples, sorted by fit
        """
        task_lower = task_description.lower()
        
        # Determine ideal archetype for task
        ideal_archetypes = self._determine_ideal_archetypes(task_lower, task_requirements)
        
        # Determine ideal personality for task
        ideal_personality = self._determine_ideal_personality(task_lower, task_requirements)
        
        # Score each agent
        agent_scores = []
        
        for agent_name, (archetype_profile, personality_profile) in self.agents.items():
            # Archetype fit
            archetype_fit = 0.0
            if archetype_profile.primary_archetype in ideal_archetypes:
                archetype_fit = 1.0
            else:
                # Check secondary archetypes
                for secondary in archetype_profile.secondary_archetypes:
                    if secondary in ideal_archetypes:
                        archetype_fit = 0.5
                        break
            
            # Personality fit (simplified: match type)
            personality_fit = 1.0 if personality_profile.personality_type == ideal_personality else 0.5
            
            # Combined fit
            fit_score = (archetype_fit * 0.6 + personality_fit * 0.4)
            agent_scores.append((agent_name, fit_score))
        
        # Sort by fit score
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        return agent_scores
    
    def _determine_ideal_archetypes(self, task_text: str,
                                   requirements: Optional[Dict[str, Any]]) -> List[JungianArchetype]:
        """Determine ideal archetypes for a task"""
        ideal = []
        
        # Simple keyword-based matching
        if any(word in task_text for word in ['lead', 'guide', 'teach', 'mentor']):
            ideal.append(JungianArchetype.THE_MENTOR)
        
        if any(word in task_text for word in ['create', 'innovate', 'build', 'design']):
            ideal.append(JungianArchetype.THE_CREATOR)
        
        if any(word in task_text for word in ['analyze', 'understand', 'learn', 'study']):
            ideal.append(JungianArchetype.THE_SAGE)
        
        if any(word in task_text for word in ['help', 'support', 'care', 'nurture']):
            ideal.append(JungianArchetype.THE_CAREGIVER)
        
        if any(word in task_text for word in ['explore', 'discover', 'search', 'find']):
            ideal.append(JungianArchetype.THE_EXPLORER)
        
        if any(word in task_text for word in ['fight', 'defend', 'protect', 'battle']):
            ideal.append(JungianArchetype.THE_WARRIOR)
        
        if not ideal:
            ideal.append(JungianArchetype.THE_HERO)  # Default
        
        return ideal
    
    def _determine_ideal_personality(self, task_text: str,
                                    requirements: Optional[Dict[str, Any]]) -> PersonalityType:
        """Determine ideal personality type for a task"""
        # Simplified: default to INTJ (analytical)
        # In practice, this would be more sophisticated
        return PersonalityType.INTJ


class SymbolicPatternRecognizer:
    """
    Recognize archetypal patterns in data/text
    
    Uses Jungian concepts to identify symbolic patterns
    """
    
    def __init__(self):
        """Initialize pattern recognizer"""
        self.archetype_analyzer = JungianArchetypeAnalyzer()
    
    def recognize_patterns(self, data: List[str]) -> Dict[str, Any]:
        """
        Recognize archetypal patterns in data
        
        Parameters
        ----------
        data : list of str
            Text data to analyze
            
        Returns
        -------
        patterns : dict
            Recognized archetypal patterns
        """
        # Analyze each piece of data
        archetype_counts = {}
        
        for text in data:
            behavior = {'communication': [text], 'actions': [text]}
            profile = self.archetype_analyzer.analyze(behavior)
            
            archetype = profile.primary_archetype
            archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1
        
        # Find dominant patterns
        total = len(data)
        pattern_frequencies = {arch: count / total for arch, count in archetype_counts.items()}
        
        return {
            'archetype_distribution': pattern_frequencies,
            'dominant_archetype': max(archetype_counts.items(), key=lambda x: x[1])[0] if archetype_counts else None,
            'total_patterns': len(archetype_counts)
        }
