# Top 4: Kaggle-Style and Other Tests

Ways to run and validate the Combined Top 4 sandbox against real data and benchmarks. **No Kaggle API is required** for the tests below; they use sklearn built-in datasets (Iris, Digits, Wine, Breast Cancer), which are standard benchmark/UCI-style data.

---

## 1. Combined test runner (Iris + Digits)

Runs the full Top 4 and Top 8 sandboxes and an extra **Digits** ensemble test. Writes `combined_test_results.json`.

```powershell
cd c:\Users\DJMcC\OneDrive\Desktop\toolbox\ML-ToolBox
python sandboxes/unusual_book_ideas/run_combined_tests.py
```

**Validates:** Oracle, explainer, Socratic, real-world Iris pipeline, ensemble on Digits (ensemble should be >= best single).

---

## 2. Top 4 benchmarks (Iris, Digits, Wine, Breast Cancer)

Runs the Top 4 pipeline (oracle + explainer + 3-model ensemble) on **four** sklearn datasets. Writes `top4_benchmark_results.json`.

```powershell
python sandboxes/unusual_book_ideas/run_top4_benchmarks.py
```

**Datasets:**
- **iris_binary** – 3-class Iris, binarized (class 0 vs rest)
- **digits** – 10-class digit images (64 features)
- **wine** – 3-class Wine (UCI-style)
- **breast_cancer** – 2-class Breast Cancer (30 features)

**Pass criteria:** Oracle and explainer return OK on all datasets. Ensemble accuracy can be >= or occasionally &lt; best single model (majority vote can be marginally worse on some splits; that’s expected).

---

## 3. Pytest (CI-style)

A pytest file runs Top 4 and ensemble checks on Iris, Digits, Wine, and Breast Cancer. Run only if your environment has pytest and no plugin conflicts:

```powershell
python -m pytest tests/test_combined_top4.py -v --tb=short
```

**Tests:**
- `test_top4_sandbox_full_run` – full Top 4 run, all components + real_world pass
- `test_top4_oracle_returns_suggestion` – oracle returns pattern/suggestion/why
- `test_top4_explainer_three_views` – explainer returns three entropy views
- `test_top4_ensemble_on_digits` – Digits: corrected >= best single
- `test_top4_ensemble_on_wine` – Wine: corrected >= best single
- `test_top4_ensemble_on_breast_cancer` – Breast Cancer: corrected >= best single

**Note:** On some seeds/splits, Breast Cancer ensemble can be slightly below best single; if that test fails, it’s a known majority-vote effect. You can relax that assertion to `corrected_acc >= best_single - 0.02` if needed.

---

## 4. Using real Kaggle datasets (optional)

The repo does **not** include the Kaggle API or downloaded competition data. To run Top 4 on a Kaggle dataset:

1. **Install Kaggle CLI:** `pip install kaggle`
2. **Configure credentials:** Place `kaggle.json` (from Kaggle account) in `~/.kaggle/` (or `%USERPROFILE%\.kaggle\` on Windows).
3. **Download a dataset**, e.g. Titanic or House Prices:
   ```powershell
   kaggle competitions download -c titanic
   # or: kaggle datasets download -d heptapod/titanic
   ```
4. **Load the CSV** in a small script, then call the same Top 4 flow (oracle, explainer, train_test_split, 3 models, ErrorCorrectingPredictions). You can add a `run_top4_on_csv(path)` helper that reads the CSV, infers or accepts target column, and runs the ensemble.

Suggested Kaggle-style datasets for Top 4 (tabular classification):
- **Titanic** – binary survival
- **Spaceship Titanic** – multiclass
- **House Prices** – regression (would need a regression path: median/mean of 3 models instead of majority vote)

---

## 5. Summary: what to run

| Goal | Command | Output |
|------|--------|--------|
| Full Top 4 + Top 8 + Digits | `python sandboxes/unusual_book_ideas/run_combined_tests.py` | `combined_test_results.json` |
| Top 4 on 4 benchmarks | `python sandboxes/unusual_book_ideas/run_top4_benchmarks.py` | `top4_benchmark_results.json` |
| Pytest (if env OK) | `python -m pytest tests/test_combined_top4.py -v` | Console pass/fail |
| Real Kaggle data | Manual: download via Kaggle CLI, then run Top 4 on CSV | Your own script |

**Bottom line:** You have **Kaggle-style / UCI-style** coverage via sklearn’s Iris, Digits, Wine, and Breast Cancer. For real Kaggle competitions, add a small loader and point it at a downloaded CSV.
