# ML Compass

**Decide. Understand. Build. Think.**

Combined Top 4 app: **Oracle** (what to do) + **Explainers** (why, in three disciplines) + **Theory-as-Channel** (reliable ensembles) + **Socratic** (debate + question).

---

## Run from repo root

All commands assume you are in the ML-ToolBox repo root:

```powershell
cd c:\Users\DJMcC\OneDrive\Desktop\toolbox\ML-ToolBox
```

**Two ways to run:**

- `python -m ML_Compass <command> ...` (recommended)
- `python ML_Compass/run.py <command> ...`

---

## CLI

### Oracle – get algorithm suggestion

```powershell
python -m ML_Compass oracle --tabular --classification --n_samples medium
```

### Explain – cross-domain concept (entropy, bias_variance, capacity)

```powershell
python -m ML_Compass explain entropy
python -m ML_Compass explain bias_variance
python -m ML_Compass explain capacity
```

### Debate – Socratic question on a statement

```powershell
python -m ML_Compass debate "I used an ensemble of three models."
```

### Capacity – channel capacity and redundancy recommendation

```powershell
python -m ML_Compass capacity --signal 10 --noise 1
```

### Run all – oracle + explain + debate (defaults)

```powershell
python -m ML_Compass run_all
```

---

## Web API (optional)

Install FastAPI and uvicorn, then:

```powershell
python -m ML_Compass serve --port 8000
```

Then open:

- http://127.0.0.1:8000/  
- http://127.0.0.1:8000/docs (Swagger UI)

**Endpoints:**

- `POST /oracle` – body: `{"tabular": true, "classification": true, "n_samples": "medium"}`  
- `GET /explain/{concept}` – e.g. `/explain/entropy`  
- `POST /debate` – body: `"I used an ensemble."` (as string or JSON)  
- `GET /capacity?signal_power=10&noise_power=1`

---

## As a library

From repo root (or with repo root on `PYTHONPATH`):

```python
from ML_Compass import oracle_suggest, explain_concept, debate_and_question
from ML_Compass import channel_capacity_bits, recommend_redundancy, correct_predictions

# Oracle
out = oracle_suggest({"tabular": True, "classification": True, "n_samples": "medium"})
print(out["suggestion"], out["why"])

# Explainer
views = explain_concept("entropy")
print(views["views"]["information_theory"])

# Socratic
d = debate_and_question("I used an ensemble.")
print(d["question"])

# Theory-as-Channel (needs list of fitted sklearn models and X_test)
# corrected = correct_predictions(prediction_matrix, method="majority_vote")
```

---

## Dependencies

- **Python 3.8+**
- **ml_toolbox** (this repo) – for `textbook_concepts.communication_theory` and `agent_enhancements.socratic_method`
- **numpy**
- **Optional:** `sklearn` (only if you train models and use `correct_predictions` / ensemble)
- **Optional (serve):** `fastapi`, `uvicorn`

No separate `requirements.txt` in ML_Compass; use the repo’s `requirements.txt` for core deps.
