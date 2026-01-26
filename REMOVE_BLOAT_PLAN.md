# Remove Bloat - Action Plan

## User Feedback: "All the extra math, philosophy, etc is just waste"

**You're absolutely right.** Let's strip it down to what actually works.

---

## What to Remove/Disable

### ❌ **Remove These Modules** (Pure Bloat)

1. **Philosophy/Religion**
   - `ml_toolbox/agent_enhancements/socratic_method.py`
   - `ml_toolbox/agent_enhancements/moral_laws.py`
   - `ml_toolbox/multi_agent_design/divine_omniscience.py`

2. **Science Fiction**
   - `ml_toolbox/infrastructure/neural_lace.py`
   - `ml_toolbox/textbook_concepts/precognition.py`
   - `ml_toolbox/optimization/multiverse.py`
   - `ml_toolbox/automl/singularity.py`

3. **Experimental Psychology**
   - `ml_toolbox/agent_enhancements/jungian_psychology.py`

4. **Quantum Mechanics** (Experimental, unproven)
   - `ml_toolbox/textbook_concepts/quantum_mechanics.py`

5. **Experimental Math** (Most of it)
   - `ml_toolbox/ai_concepts/game_theory.py` (keep if useful)
   - `ml_toolbox/optimization/evolutionary_algorithms.py` (keep - actually works for optimization)
   - `ml_toolbox/textbook_concepts/statistical_mechanics.py` (keep SimulatedAnnealing if useful)
   - `ml_toolbox/textbook_concepts/linguistics.py` (remove)
   - `ml_toolbox/textbook_concepts/communication_theory.py` (remove)
   - `ml_toolbox/textbook_concepts/self_organization.py` (remove)
   - `ml_toolbox/optimization/control_theory.py` (maybe keep PID controller)
   - `ml_toolbox/optimization/bounded_rationality.py` (remove)
   - `ml_toolbox/optimization/systems_theory.py` (remove)
   - `ml_toolbox/ai_concepts/cooperative_games.py` (remove)
   - `ml_toolbox/ai_concepts/network_theory.py` (remove)

### ⚠️ **Keep But Mark as Experimental**

1. **Information Theory** (Actually useful for feature selection)
   - Keep but mark as experimental
   - Test thoroughly

2. **Evolutionary Algorithms** (Works for optimization)
   - Keep - proven useful for TSP, etc.

3. **Concept Drift** (Works - major win)
   - Keep - this is actually valuable

---

## What to Keep (Core ML)

### ✅ **Essential Core**

1. **Core Models**
   - `ml_toolbox/core_models/` - All of it
   - Regression, Classification, Neural Networks, Evaluation

2. **Data Preprocessing**
   - `ml_toolbox/compartment1_data.py` - Basic preprocessing
   - Remove advanced/experimental parts

3. **Pipelines**
   - `ml_toolbox/pipelines/` - Keep if working
   - Feature, Training, Inference pipelines

4. **Basic Agents** (If actually used)
   - `ml_toolbox/agent_fundamentals/` - Basic agent functionality
   - Remove experimental enhancements

5. **Math Foundations** (Basic)
   - `ml_toolbox/math_foundations/` - Linear algebra, calculus, etc.
   - Keep basic math, remove experimental

---

## Implementation Plan

### Step 1: Create "Core Only" Mode

Add flag to disable experimental features:

```python
toolbox = MLToolbox(experimental_features=False)
```

### Step 2: Remove Experimental Imports

Don't import experimental modules by default.

### Step 3: Move Experimental to Separate Package

Move all experimental features to `ml_toolbox_experimental/` or similar.

### Step 4: Update Documentation

Be honest: "Core ML features only. Experimental features disabled."

---

## Quick Win: Disable Experimental Features

Easiest fix: Add flag to disable loading experimental features.
