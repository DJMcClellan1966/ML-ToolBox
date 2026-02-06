# ML Learning Lab

**Learn · Decide · Practice · Think · By Book · By Level · Build Knuth Machine**

A single-page learning app that integrates **ML_Compass**, **ml_toolbox**, and content from **Knuth (TAOCP), Skiena, Bentley, Sedgewick, textbook concepts, and information/communication theory**. Basics to expert, with a **Learn** and **Try it** section per topic, plus a **Build Knuth Machine** pipeline.

---

## Run (from repo root)

```powershell
cd c:\Users\DJMcC\OneDrive\Desktop\toolbox\ML-ToolBox
python ml_learning_lab/app.py
```

Then open: **http://127.0.0.1:5001**

(Port 5001 avoids conflict with other apps on 5000. Set `PORT=8080` to use 8080.)

---

## What you get

| Section | What it does |
|--------|----------------|
| **Compass** | Learn (entropy, bias_variance, capacity), Decide (oracle + guidance), Practice (toolbox demo), Think (Socratic debate). |
| **By Book** | Pick a book (Knuth, Skiena & Bentley, Sedgewick, Textbook concepts, Information & communication, Algorithm design). Each topic has **Learn** text and **Try it** (run demo). |
| **By Level** | Basics → Intermediate → Advanced → Expert. Same topics, filtered by difficulty. |
| **Build Knuth Machine** | Chain TAOCP-style steps: **LCG** (random numbers) → **Shuffle** (Fisher-Yates) → **Sample** (k from n). Set seed and params, run pipeline. |

---

## Books and levels

- **Knuth (TAOCP)**: LCG, Fisher-Yates, sample, heapsort, quicksort, binary search, combinatorics.
- **Skiena & Bentley**: Backtracking/greedy, Kadane max subarray.
- **Sedgewick**: Red-Black tree.
- **Textbook concepts**: Bishop (bias-variance), practical ML (cross-validation), statistical mechanics (simulated annealing), communication theory (error-correcting predictions).
- **Information & communication**: Shannon entropy, channel capacity.
- **Algorithm design patterns**: Greedy, DP templates.

Levels: **basics** → **intermediate** → **advanced** → **expert**.

---

## Requirements

- **Python 3.8+**
- **Flask** (`pip install flask`)
- **ML_Compass** (in this repo)
- **ml_toolbox** (in this repo) — optional; Practice uses sklearn if missing
- **knuth_algorithms**, **foundational_algorithms**, **algorithm_design_patterns** (repo root) — for By Book / By Level demos and Knuth Machine

Socratic debate needs `ml_toolbox.agent_enhancements.socratic_method`; if that’s not available, Think returns a short message.
