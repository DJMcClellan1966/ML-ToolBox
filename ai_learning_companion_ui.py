"""
AI Learning Companion - Enhanced UI/UX

Beautiful, user-friendly interface for learning ML/AI concepts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Try to use advanced companion, fallback to basic
try:
    from advanced_learning_companion import AdvancedLearningCompanion
    USE_ADVANCED = True
except ImportError:
    try:
        from ai_learning_companion import LearningCompanion
        USE_ADVANCED = False
    except ImportError:
        raise ImportError("No learning companion available")

from typing import Dict, List, Optional
import time


class EnhancedUI:
    """
    Enhanced UI/UX for AI Learning Companion
    
    Features:
    - Clean, organized display
    - Color-coded sections
    - Progress visualization
    - Interactive menus
    - Better formatting
    """
    
    def __init__(self, companion=None):
        if companion is None:
            if USE_ADVANCED:
                self.companion = AdvancedLearningCompanion()
                print("\n[Using ADVANCED Learning Companion with Brain Topology!]")
            else:
                self.companion = LearningCompanion()
                print("\n[Using Basic Learning Companion]")
        else:
            self.companion = companion
        self.session_history = []
    
    def print_header(self, title: str, width: int = 80):
        """Print beautiful header"""
        print("\n" + "="*width)
        print(title.center(width))
        print("="*width)
    
    def print_section(self, title: str, content: any, indent: int = 2):
        """Print section with title"""
        spaces = " " * indent
        print(f"\n{spaces}[{title}]")
        if isinstance(content, list):
            for item in content:
                print(f"{spaces}  • {item}")
        elif isinstance(content, dict):
            for key, value in content.items():
                print(f"{spaces}  {key}: {value}")
        else:
            print(f"{spaces}  {content}")
    
    def print_concept_card(self, result: Dict):
        """Print concept in a beautiful card format"""
        self.print_header(f"LEARNING: {result['concept'].upper().replace('_', ' ')}")
        
        # Explanation
        self.print_section("Explanation", result['explanation'])
        
        # Examples
        if result.get('examples'):
            self.print_section("Real-World Examples", result['examples'])
        
        # Key Terms
        if result.get('key_terms'):
            self.print_section("Key Terms to Master", result['key_terms'])
        
        # Learning Tips
        if result.get('learning_tips'):
            self.print_section("Learning Tips", result['learning_tips'])
        
        # Practice
        if result.get('practice_suggestions'):
            self.print_section("Practice Suggestions", result['practice_suggestions'])
        
        # Related
        if result.get('related_concepts'):
            self.print_section("Explore Next", result['related_concepts'])
        
        print("\n" + "="*80)
    
    def print_answer_card(self, result: Dict):
        """Print answer in a beautiful format"""
        self.print_header("ANSWER")
        
        print(f"\n  Question: {result['question']}")
        print(f"\n  Answer:")
        # Format answer with proper line breaks
        answer_lines = result['answer'].split('\n')
        for line in answer_lines:
            print(f"    {line}")
        
        if result.get('related_concepts'):
            self.print_section("Related Topics", result['related_concepts'])
        
        if result.get('confidence'):
            print(f"\n  Confidence: {result['confidence']:.1%}")
        
        print("\n" + "="*80)
    
    def print_path_card(self, result: Dict):
        """Print learning path in a beautiful format"""
        self.print_header(f"LEARNING PATH: {result['goal'].upper().replace('_', ' ')}")
        
        print(f"\n  Goal: {result['goal']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Estimated Time: {result['estimated_time']}")
        
        print(f"\n  Path:")
        for i, step in enumerate(result['path'], 1):
            marker = ">>>" if i == 1 else "   "
            print(f"    {marker} Step {i}: {step}")
        
        if result.get('next_step'):
            print(f"\n  [NEXT] Start with: '{result['next_step']}'")
        
        print("\n" + "="*80)
    
    def print_progress_card(self, result: Dict):
        """Print progress in a beautiful format"""
        self.print_header("YOUR LEARNING PROGRESS")
        
        print(f"\n  Topics Learned: {result['topics_learned']}")
        if result.get('topics_list'):
            print(f"    {', '.join(result['topics_list'][:5])}")
            if len(result['topics_list']) > 5:
                print(f"    ... and {len(result['topics_list']) - 5} more")
        
        print(f"\n  Learning Sessions: {result['learning_sessions']}")
        print(f"  Questions Asked: {result['questions_asked']}")
        print(f"  Current Level: {result['current_level'].upper()}")
        
        # Progress bar (simple)
        progress = min(result['topics_learned'] / 10, 1.0)  # 10 topics = 100%
        bar_length = 40
        filled = int(bar_length * progress)
        bar = "=" * filled + "-" * (bar_length - filled)
        print(f"\n  Overall Progress: [{bar}] {progress:.0%}")
        
        if result.get('recommendations'):
            self.print_section("Recommendations", result['recommendations'])
        
        print("\n" + "="*80)
    
    def print_menu(self):
        """Print main menu"""
        self.print_header("AI LEARNING COMPANION - MAIN MENU")
        
        print("\n  What would you like to do?")
        print("\n  1. Learn a Concept")
        print("  2. Ask a Question")
        print("  3. Get Learning Path")
        print("  4. Check Progress")
        print("  5. Browse Concepts")
        print("  6. Quick Start Guide")
        print("  0. Exit")
        
        print("\n" + "-"*80)
    
    def print_concept_browser(self):
        """Browse available concepts"""
        self.print_header("BROWSE CONCEPTS")
        
        print("\n  [BEGINNER]")
        beginner = ['machine_learning', 'supervised_learning', 'classification', 'regression']
        for i, concept in enumerate(beginner, 1):
            print(f"    {i}. {concept.replace('_', ' ').title()}")
        
        print("\n  [INTERMEDIATE]")
        intermediate = ['neural_networks', 'deep_learning', 'feature_engineering']
        for i, concept in enumerate(intermediate, 1):
            print(f"    {i}. {concept.replace('_', ' ').title()}")
        
        print("\n  [ADVANCED]")
        advanced = ['transformer', 'reinforcement_learning']
        for i, concept in enumerate(advanced, 1):
            print(f"    {i}. {concept.replace('_', ' ').title()}")
        
        print("\n" + "-"*80)
        print("\n  Type 'learn <concept_name>' to learn about any concept")
        print("  Example: learn classification")
    
    def print_quick_start(self):
        """Print quick start guide"""
        self.print_header("QUICK START GUIDE")
        
        print("\n  Welcome to AI Learning Companion!")
        print("\n  [Getting Started]")
        print("    1. Start with: learn machine_learning")
        print("    2. Then explore: learn classification")
        print("    3. Ask questions: ask what is the difference between classification and regression?")
        print("    4. Follow a path: path ml_fundamentals")
        
        print("\n  [Learning Paths]")
        print("    • ml_fundamentals - Start here!")
        print("    • deep_learning - For neural networks")
        print("    • practical_ml - For hands-on skills")
        
        print("\n  [Tips]")
        print("    • Start with beginner concepts")
        print("    • Ask questions when confused")
        print("    • Follow learning paths for structure")
        print("    • Check progress regularly")
        
        print("\n" + "-"*80)
    
    def interactive_menu(self):
        """Interactive menu-driven interface"""
        print("\n" + "="*80)
        print("AI LEARNING COMPANION - Enhanced UI".center(80))
        print("="*80)
        print("\nWelcome! I'm your AI learning companion.")
        print("I'll help you learn ML and AI concepts step by step.")
        
        while True:
            try:
                self.print_menu()
                choice = input("\n  Your choice: ").strip()
                
                if choice == '0' or choice.lower() == 'quit' or choice.lower() == 'exit':
                    print("\n  Thank you for learning! Keep growing!")
                    break
                
                elif choice == '1':
                    self._handle_learn()
                
                elif choice == '2':
                    self._handle_ask()
                
                elif choice == '3':
                    self._handle_path()
                
                elif choice == '4':
                    self._handle_progress()
                
                elif choice == '5':
                    self.print_concept_browser()
                    input("\n  Press Enter to continue...")
                
                elif choice == '6':
                    self.print_quick_start()
                    input("\n  Press Enter to continue...")
                
                else:
                    print("\n  Invalid choice. Please try again.")
                    time.sleep(1)
                
            except KeyboardInterrupt:
                print("\n\n  Goodbye! Keep learning!")
                break
            except Exception as e:
                print(f"\n  Error: {e}")
                input("  Press Enter to continue...")
    
    def _handle_learn(self):
        """Handle learn concept"""
        print("\n  Available concepts:")
        print("    machine_learning, classification, regression, neural_networks,")
        print("    deep_learning, feature_engineering, transformer, reinforcement_learning")
        
        concept = input("\n  Enter concept name: ").strip().lower()
        if not concept:
            print("  No concept entered.")
            return
        
        print("\n  Learning...")
        if USE_ADVANCED and hasattr(self.companion, 'learn_concept_advanced'):
            result = self.companion.learn_concept_advanced(concept, use_socratic=True)
        else:
            result = self.companion.learn_concept(concept)
        
        if not result.get('found', True):
            print(f"\n  {result.get('message', 'Concept not found')}")
            print("  Try: machine_learning, classification, regression, etc.")
        else:
            self.print_concept_card(result)
            self.session_history.append(('learn', concept))
        
        input("\n  Press Enter to continue...")
    
    def _handle_ask(self):
        """Handle ask question"""
        question = input("\n  Enter your question: ").strip()
        if not question:
            print("  No question entered.")
            return
        
        print("\n  Thinking...")
        if USE_ADVANCED and hasattr(self.companion, 'answer_question_advanced'):
            result = self.companion.answer_question_advanced(question)
        else:
            result = self.companion.answer_question(question)
        self.print_answer_card(result)
        self.session_history.append(('ask', question))
        
        input("\n  Press Enter to continue...")
    
    def _handle_path(self):
        """Handle learning path"""
        print("\n  Available paths:")
        print("    ml_fundamentals, deep_learning, practical_ml")
        
        goal = input("\n  Enter path name: ").strip().lower()
        if not goal:
            print("  No path entered.")
            return
        
        print("\n  Finding path...")
        if USE_ADVANCED and hasattr(self.companion, 'suggest_personalized_path'):
            result = self.companion.suggest_personalized_path(goal)
        elif hasattr(self.companion, 'suggest_learning_path'):
            result = self.companion.suggest_learning_path(goal)
        else:
            result = {'goal': goal, 'path': [], 'steps': 0, 'estimated_time': 'Unknown'}
        self.print_path_card(result)
        self.session_history.append(('path', goal))
        
        input("\n  Press Enter to continue...")
    
    def _handle_progress(self):
        """Handle progress check"""
        print("\n  Checking progress...")
        if USE_ADVANCED and hasattr(self.companion, 'assess_progress_advanced'):
            result = self.companion.assess_progress_advanced()
        else:
            result = self.companion.assess_progress()
        self.print_progress_card(result)
        self.session_history.append(('progress', None))
        
        input("\n  Press Enter to continue...")


def main():
    """Main function with enhanced UI"""
    ui = EnhancedUI()  # Will auto-create advanced companion if available
    ui.interactive_menu()


if __name__ == "__main__":
    main()
