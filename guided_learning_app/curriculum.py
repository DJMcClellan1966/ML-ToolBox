from __future__ import annotations

from typing import Dict, List, Optional


LESSONS: List[Dict] = [
    {
        "id": "basics_programs",
        "title": "What is a program?",
        "level": "beginner",
        "goal": "Understand inputs, processing, and outputs.",
        "explain": (
            "A program is a set of instructions that transforms inputs into outputs. "
            "You give data in, the program processes it, and you get a result. "
            "Variables are names that store those inputs or intermediate results."
        ),
        "example": "input = 3; output = input * 2  # output is 6",
        "practice": "Write down a real-life example of input → process → output.",
        "check": {
            "question": "In your own words, what does a program do?",
            "keywords": ["input", "output", "variable", "process"],
        },
    },
    {
        "id": "data_types",
        "title": "Data types and values",
        "level": "beginner",
        "goal": "Know the difference between numbers, text, and booleans.",
        "explain": (
            "Programs use data types to represent different kinds of information. "
            "Integers and floats are numbers, strings are text, and booleans are True/False."
        ),
        "example": "age = 42  # int; price = 19.99  # float; name = 'Ada'  # string; is_ready = True",
        "practice": "List one example of each type: int, float, string, bool.",
        "check": {
            "question": "Name at least two data types and explain what they represent.",
            "keywords": ["integer", "float", "string", "boolean"],
        },
    },
    {
        "id": "control_flow",
        "title": "Control flow with if/else",
        "level": "beginner",
        "goal": "Use conditions to make decisions.",
        "explain": (
            "Control flow lets a program choose different paths. "
            "An if/else checks a condition and runs the appropriate block of code."
        ),
        "example": "if temperature > 75: print('warm') else: print('cool')",
        "practice": "Describe a decision you make daily using an if/else structure.",
        "check": {
            "question": "What does an if/else do?",
            "keywords": ["condition", "if", "else", "branch"],
        },
    },
    {
        "id": "loops",
        "title": "Loops: repeating work",
        "level": "beginner",
        "goal": "Use loops to repeat steps safely.",
        "explain": (
            "Loops let you repeat a block of code. A for-loop runs a fixed number of times, "
            "and a while-loop runs while a condition is true."
        ),
        "example": "for i in range(3): print(i)  # prints 0,1,2",
        "practice": "Explain when you would use a for-loop vs a while-loop.",
        "check": {
            "question": "Why do we use loops?",
            "keywords": ["repeat", "loop", "iterate"],
        },
    },
    {
        "id": "functions",
        "title": "Functions and reuse",
        "level": "beginner",
        "goal": "Group steps into reusable pieces.",
        "explain": (
            "A function is a named block of code that can take inputs (parameters) and return outputs. "
            "Functions help you reuse logic and keep code organized."
        ),
        "example": "def add(a, b): return a + b",
        "practice": "Write a simple function in pseudocode that doubles a number.",
        "check": {
            "question": "What is a function and why use one?",
            "keywords": ["function", "parameter", "return", "reuse"],
        },
    },
    {
        "id": "collections",
        "title": "Collections: lists and dictionaries",
        "level": "beginner",
        "goal": "Store multiple values in one place.",
        "explain": (
            "Lists store ordered items, while dictionaries store key → value pairs. "
            "These let you organize and retrieve data efficiently."
        ),
        "example": "items = ['apple', 'banana']; prices = {'apple': 1.25, 'banana': 0.75}",
        "practice": "Describe a list and a dictionary you could use in a recipe app.",
        "check": {
            "question": "What is the difference between a list and a dictionary?",
            "keywords": ["list", "dictionary", "key", "value"],
        },
    },
    {
        "id": "errors_debugging",
        "title": "Errors and debugging",
        "level": "beginner",
        "goal": "Understand and fix mistakes.",
        "explain": (
            "Errors happen when code is written incorrectly or used the wrong way. "
            "Reading error messages and stepping through code helps you debug."
        ),
        "example": "NameError: name 'x' is not defined  # means you used x before creating it",
        "practice": "Describe how you would fix an error message you don't understand.",
        "check": {
            "question": "What is debugging and why is it important?",
            "keywords": ["error", "exception", "trace", "debug"],
        },
    },
    {
        "id": "algorithmic_thinking",
        "title": "Algorithmic thinking",
        "level": "beginner",
        "goal": "Break problems into clear steps.",
        "explain": (
            "An algorithm is a precise sequence of steps to solve a problem. "
            "Good algorithms are clear, correct, and efficient."
        ),
        "example": "To sort numbers: repeatedly select the smallest and place it next.",
        "practice": "Write the steps to make a sandwich as an algorithm.",
        "check": {
            "question": "What is an algorithm?",
            "keywords": ["algorithm", "steps", "procedure", "efficient"],
        },
    },
    {
        "id": "probability_basics",
        "title": "Probability basics",
        "level": "beginner",
        "goal": "Grasp uncertainty and chance.",
        "explain": (
            "Probability measures how likely something is to happen. "
            "It ranges from 0 (impossible) to 1 (certain)."
        ),
        "example": "P(heads) = 0.5 for a fair coin.",
        "practice": "Give one example of a 0.25 probability event.",
        "check": {
            "question": "What does probability represent?",
            "keywords": ["probability", "chance", "expected"],
        },
    },
    {
        "id": "learning_systems",
        "title": "What is a learning system?",
        "level": "beginner",
        "goal": "See how data and models fit together.",
        "explain": (
            "A learning system uses data to build a model that makes predictions. "
            "Training fits the model, and evaluation checks how well it works."
        ),
        "example": "Train a model to predict house prices from size and location.",
        "practice": "Describe a prediction problem you care about.",
        "check": {
            "question": "What are the main parts of a learning system?",
            "keywords": ["data", "model", "training", "evaluation"],
        },
    },
]


def get_lessons() -> List[Dict]:
    return LESSONS


def get_lesson(lesson_id: str) -> Optional[Dict]:
    for lesson in LESSONS:
        if lesson["id"] == lesson_id:
            return lesson
    return None


def get_next_lesson_id(lesson_id: str) -> Optional[str]:
    ids = [lesson["id"] for lesson in LESSONS]
    if lesson_id not in ids:
        return None
    idx = ids.index(lesson_id)
    if idx + 1 < len(ids):
        return ids[idx + 1]
    return None
