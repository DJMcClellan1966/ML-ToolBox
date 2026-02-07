# External Resources

Explore further with these standard ML/CS resources:
- [arXiv.org](https://arxiv.org/) — Research papers
- [Papers with Code](https://paperswithcode.com/) — Code and benchmarks
- [DeepAI](https://deepai.org/) — Tutorials and datasets
- [Stack Overflow](https://stackoverflow.com/) — Q&A
- [Reddit ML/AI](https://www.reddit.com/r/MachineLearning/) — Community discussions
- [YouTube ML/AI Channels](https://www.youtube.com/results?search_query=machine+learning) — Video lectures
- [Google Scholar](https://scholar.google.com/) — Academic search
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook) — Practical guides
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html) — ML library docs

Links are available in each lab and the hub for quick access.
# Mobile App Support

The platform is fully responsive and optimized for mobile devices:
- **Mobile Web App**: All labs and the hub work seamlessly on phones and tablets.
- **Touch-friendly UI**: Large buttons, swipe navigation, and mobile-friendly layouts.
- **Add to Home Screen**: Use browser's "Add to Home Screen" for app-like experience.
- **Future Native App**: Ready for React Native/Flutter integration if needed.

To use on mobile:
1. Open the Hub or any lab in your mobile browser.
2. Tap "Add to Home Screen" for quick access.
3. Enjoy all features, including AI tutor, progress tracking, and demos.
# Accessibility Features

This platform includes standard accessibility improvements:
- **Screen Reader Support**: All UI elements use semantic HTML and ARIA labels.
- **Keyboard Navigation**: Tab, arrow keys, and shortcuts (`Ctrl+K` for search, `Ctrl+T` for tutor chat) are supported throughout.
- **High Contrast & Colorblind Modes**: Toggle light/dark themes and color palettes for visual accessibility.
- **Multi-language UI (future)**: Ready for translation/localization.
- **Responsive Design**: Works on desktop, tablet, and mobile.

If you have accessibility needs or suggestions, please open an issue or contact the maintainer.
# Learning Apps — World-Class Learning Platform

A comprehensive learning platform featuring **13 specialized labs** with **AI tutors** inspired by the greatest minds in computer science and machine learning. Each lab provides personalized, Socratic-style tutoring from virtual mentors like Richard Sutton, Ian Goodfellow, Stuart Russell, and more.

---

## 🎓 AI Tutors — Learn from the Masters

Each lab features a unique AI tutor character based on famous CS/ML figures:

| Lab | AI Tutor | Specialization |
|-----|----------|----------------|
| CLRS Algorithms | **Prof. Thomas Cormen** | Algorithm analysis, complexity |
| Deep Learning | **Dr. Ian Goodfellow** | Neural networks, GANs |
| RL Lab | **Prof. Richard Sutton** | Reinforcement learning |
| AI Concepts | **Prof. Stuart Russell** | Search, planning, rationality |
| ML Theory | **Prof. Shai Shalev-Shwartz** | PAC learning, VC dimension |
| Probabilistic ML | **Prof. Kevin Murphy** | Bayesian methods, graphical models |
| LLM Engineers | **Andrej Karpathy** | LLMs, transformers, prompting |
| Math for ML | **Prof. Gilbert Strang** | Linear algebra, optimization |
| Practical ML | **Aurélien Géron** | Hands-on ML, production |
| Python Practice | **Prof. John Zelle** | Programming fundamentals |
| SICP | **Prof. Hal Abelson** | Abstraction, recursion |
| Cross-Domain | **Alan Turing** | Cross-disciplinary thinking |

### Tutor Features:
- **Socratic Method**: Guides understanding through thoughtful questions
- **Adaptive Difficulty**: Adjusts to beginner/intermediate/advanced/expert
- **Progressive Hints**: Scaffolded hints before giving answers
- **LLM Integration**: Works with Ollama (local) or OpenAI

**Keyboard Shortcut**: Press `Ctrl+T` to open the tutor chat in any lab.

---

## 🚀 Quick Start

```bash
# Start the Hub
python learning_apps/hub.py
# Open http://127.0.0.1:5000

# Or start a specific lab
python learning_apps/deep_learning_lab/app.py  # Port 5003
```

---

## Overview

| App | Port | Books / sources | Run command |
|-----|------|-----------------|-------------|
| **ml_learning_lab** | 5001 | TAOCP (Knuth), Skiena & Bentley, Sedgewick, Bishop/Goodfellow/R&N, Info theory, Algorithm design | (at repo root) `python ml_learning_lab/app.py` |
| **clrs_algorithms_lab** | 5002 | CLRS (Cormen et al.) | `python learning_apps/clrs_algorithms_lab/app.py` |
| **deep_learning_lab** | 5003 | Goodfellow, Bishop, ESL, Burkov | `python learning_apps/deep_learning_lab/app.py` |
| **ai_concepts_lab** | 5004 | Russell & Norvig style (ml_toolbox.ai_concepts) | `python learning_apps/ai_concepts_lab/app.py` |
| **cross_domain_lab** | 5005 | textbook_concepts: quantum, stat mech, linguistics, precognition, self-organization | `python learning_apps/cross_domain_lab/app.py` |
| **python_practice_lab** | 5006 | Reed & Zelle (reed_zelle_patterns.py) | `python learning_apps/python_practice_lab/app.py` |
| **sicp_lab** | 5007 | SICP (Abelson, Sussman — sicp_methods.py) | `python learning_apps/sicp_lab/app.py` |
| **practical_ml_lab** | 5008 | Hands-On ML (Géron — practical_ml.py) | `python learning_apps/practical_ml_lab/app.py` |
| **rl_lab** | 5009 | Sutton & Barto (Reinforcement Learning) | `python learning_apps/rl_lab/app.py` |
| **probabilistic_ml_lab** | 5010 | Murphy (A Probabilistic Perspective) | `python learning_apps/probabilistic_ml_lab/app.py` |
| **ml_theory_lab** | 5011 | Shalev-Shwartz & Ben-David (Understanding ML) | `python learning_apps/ml_theory_lab/app.py` |
| **llm_engineers_lab** | 5012 | ML/LLM Engineers Handbook + Build Your Own LLM | `python learning_apps/llm_engineers_lab/app.py` |
| **math_for_ml_lab** | 5013 | Basic math for ML (linear algebra, calculus, probability, optimization) | `python learning_apps/math_for_ml_lab/app.py` |

All run commands assume you are in the **repo root** (`ML-ToolBox`).

---

## What each app contains

- **ml_learning_lab** (existing): Compass (Learn/Decide/Practice/Think), By Book, By Level, Build Knuth Machine; covers Knuth, Skiena, Sedgewick, textbook concepts, information/communication theory, algorithm design.
- **clrs_algorithms_lab**: CLRS — DP (Optimal BST, LIS, Coin Change, Rod Cutting), Greedy (Prim’s MST, Activity Selection), Graph (Bellman-Ford). Curriculum + runnable demos using `clrs_complete_algorithms`.
- **deep_learning_lab**: Goodfellow (regularization, optimization), Bishop (Gaussian processes, EM), ESL (SVM, boosting), Burkov (workflow, ensembles). Curriculum + ESL SVM demo via `three_books_methods`.
- **ai_concepts_lab**: Game theory (Nash, cooperative games), search & planning, reinforcement learning, probabilistic reasoning, clustering; uses `ml_toolbox.ai_concepts`. Curriculum only (demos stub).
- **cross_domain_lab**: “Unusual” cross-domain: quantum mechanics, statistical mechanics (e.g. simulated annealing), linguistics (parsing, grammar features), precognition (forecaster), self-organization (SOM, dissipative structures); uses `ml_toolbox.textbook_concepts`. Curriculum only (demos stub).
- **python_practice_lab**: Reed & Zelle — problem decomposition, algorithm patterns (divide-and-conquer, greedy, recursive, iterative), data structure optimizer, code organizer; uses `reed_zelle_patterns.py`. Curriculum only (demos stub).
- **sicp_lab**: SICP (Abelson, Sussman) — functional ML pipeline (map, filter, reduce, compose, pipe), streams, data abstraction (pairs, trees), symbolic computation; uses `sicp_methods.py`. Curriculum + runnable demos (map, compose, stream, pair, symbolic).
- **practical_ml_lab**: Hands-On ML (Géron) — feature engineering, model selection, hyperparameter tuning, ensembles, cross-validation, production ML; uses `ml_toolbox.textbook_concepts.practical_ml`. Curriculum only (demos stub).
- **rl_lab**: Sutton & Barto — MDPs, value functions, Bellman equations, TD learning, Q-learning, SARSA, policy gradient; uses `ml_toolbox.ai_concepts.reinforcement_learning`. Curriculum only (demos stub).
- **probabilistic_ml_lab**: Murphy — graphical models, EM, variational inference, Bayesian learning; uses `ml_toolbox.textbook_concepts.probabilistic_ml`. Curriculum only (demos stub).
- **ml_theory_lab**: Shalev-Shwartz & Ben-David — PAC learning, VC dimension, generalization bounds, stability, Rademacher complexity. Theory-focused curriculum (demos stub).
- **llm_engineers_lab**: ML/LLM Engineers Handbook (RAG, prompt engineering, evaluation, safety, optimization) + Build Your Own LLM (transformer architecture, tokenization, training/finetuning, scaling, LLM apps). Uses `ml_toolbox.llm_engineering` and `ml_toolbox.agent_pipelines`. Curriculum + demos (RAG retrieve, prompt template).
- **math_for_ml_lab**: Basic math for ML — linear algebra (vectors, matrices, SVD, eigen), calculus (derivative, gradient, Jacobian, Hessian), probability & statistics (Gaussian, Bayes, MLE), optimization (gradient descent, SGD, Adam). Uses `ml_toolbox.math_foundations`. Curriculum + demos (dot, SVD, derivative, gradient, Gaussian, GD).

---

## Conventions

- **Paths**: From each lab’s `app.py`, repo root is `Path(__file__).resolve().parents[2]`; lab root is `Path(__file__).parent`. Same in each lab’s `demos.py` for REPO.
- **Structure**: Each lab has `curriculum.py` (LEVELS, BOOKS, CURRICULUM + getters), `demos.py` (`run_demo`, DEMO_HANDLERS or stub), `app.py` (Flask, `/api/health`, `/api/curriculum`, `/api/curriculum/book/<id>`, `/api/curriculum/level/<level>`, `/api/try/<demo_id>`, `/`), and its own `README.md`.
- **Ports**: 5001 (ml_learning_lab), 5002–5013 for the labs under `learning_apps/` (clrs, deep_learning, ai_concepts, cross_domain, python_practice, sicp, practical_ml, rl, probabilistic_ml, ml_theory, llm_engineers, math_for_ml).

---

## Adding demos

Labs that currently have a demos stub (ai_concepts_lab, cross_domain_lab, python_practice_lab, practical_ml_lab, rl_lab, probabilistic_ml_lab, ml_theory_lab) can add runnable demos in their `demos.py`: implement `run_demo(demo_id)` and, if desired, wire `try_demo` in curriculum items to those demo IDs.

# UI/UX Enhancements for Learning Paths

The platform now includes world-class UI/UX features for guided learning:
- **Visual Learning Paths**: Beginner-to-expert and specialist journeys are shown in the Hub, with progress tracking and "Start/Continue" buttons.
- **Smart Recommendations**: The AI tutor and Hub suggest your next step based on your progress and goals.
- **Onboarding & Tooltips**: New users see helpful guides and tooltips explaining features and navigation.
- **Seamless Cross-Lab Navigation**: "Next Lab" and "Continue Path" buttons let you move smoothly between labs as you progress.
- **Accessibility**: High contrast, keyboard navigation, ARIA labels, and responsive design ensure everyone can use the platform.
- **Milestone Feedback**: Celebrate achievements with badges, confetti, and progress stats.

If you have suggestions for further improvements, please open an issue or contact the maintainer.
