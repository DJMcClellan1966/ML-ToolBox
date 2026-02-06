"""
Combined Top 4 sandbox: Oracle + Explainers + Theory-as-Channel + Socratic.
Tested against real sklearn data and models.
"""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Optional sklearn
try:
    from sklearn.datasets import load_iris, load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def oracle_suggest(profile: dict) -> dict:
    """08: Problem profile -> pattern, suggestion, why."""
    rules = [
        {"profile": {"tabular": True, "classification": True, "n_samples": "small"}, "pattern": "Classification (small n)", "suggestion": "LogisticRegression or RandomForest with default regularization.", "why": "Small n favors regularized models; avoid overfitting."},
        {"profile": {"tabular": True, "classification": True, "n_samples": "medium"}, "pattern": "Classification (medium n)", "suggestion": "RandomForest or SVC; consider ensemble.", "why": "Ensemble reduces variance; theory-as-channel suggests 3-5 models with majority vote."},
        {"profile": {"high_dim": True, "classification": True}, "pattern": "High-dim classification", "suggestion": "PCA or feature selection + LogisticRegression/RandomForest.", "why": "Bishop: bias-variance; reduce dimensions or regularize."},
    ]
    for r in rules:
        if r["profile"] == profile:
            return {"ok": True, "pattern": r["pattern"], "suggestion": r["suggestion"], "why": r["why"]}
    return {"ok": False, "pattern": None, "suggestion": "No match; use default: RandomForest.", "why": "Fallback."}


def explainer_entropy() -> dict:
    """06: Three views of entropy."""
    return {
        "ok": True,
        "information_theory": "Shannon entropy H = -sum p log p measures uncertainty in bits.",
        "thermodynamics": "Boltzmann entropy: disorder; same logarithmic form.",
        "ml": "In ML: decision tree splitting (information gain), softmax loss, regularization.",
    }


def theory_as_channel_ensemble(models, X_test, y_test):
    """01: Ensemble predictions + majority vote; return accuracies."""
    from ml_toolbox.textbook_concepts.communication_theory import ErrorCorrectingPredictions
    preds = np.column_stack([m.predict(X_test) for m in models])
    ec = ErrorCorrectingPredictions(redundancy_factor=len(models))
    corrected = ec.correct_predictions(preds, method="majority_vote")
    single_accs = [np.mean(m.predict(X_test) == y_test) for m in models]
    corrected_acc = np.mean(corrected == y_test)
    return {
        "single_accs": single_accs,
        "corrected_acc": float(corrected_acc),
        "best_single": float(max(single_accs)),
        "improvement": float(corrected_acc - max(single_accs)),
    }


def socratic_debate(statement: str) -> dict:
    """03: Debate snippet + Socratic question."""
    from ml_toolbox.agent_enhancements.socratic_method import SocraticQuestioner
    q = SocraticQuestioner()
    question = q.generate_question(statement, question_type="evidence")
    debate = f"Theory: Ensembles reduce variance (Bishop). Practice: Start with 3 models, add if needed. Question: {question}"
    return {"ok": True, "debate_preview": debate[:120], "has_question": "?" in question}


def run_top4():
    results = {"oracle": {}, "explainer": {}, "theory_channel": {}, "socratic": {}, "real_world": {}}

    # --- Oracle ---
    profile = {"tabular": True, "classification": True, "n_samples": "medium"}
    results["oracle"] = oracle_suggest(profile)
    results["oracle"]["profile_used"] = profile

    # --- Explainer ---
    results["explainer"] = explainer_entropy()

    # --- Socratic ---
    results["socratic"] = socratic_debate("I used an ensemble of three models.")

    # --- Real-world: sklearn data + models ---
    if not SKLEARN_AVAILABLE:
        results["real_world"] = {"ok": False, "reason": "sklearn not available"}
        return results

    rng = np.random.RandomState(42)
    # Iris: 150 samples, 4 features, 3 classes -> binarize for majority vote
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    y_bin = (y == 0).astype(int)  # binary for clearer ensemble test
    X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, random_state=rng)

    models = [
        LogisticRegression(random_state=rng, max_iter=500),
        RandomForestClassifier(n_estimators=50, random_state=rng),
        SVC(kernel="rbf", random_state=rng),
    ]
    for m in models:
        m.fit(X_train, y_train)

    tc = theory_as_channel_ensemble(models, X_test, y_test)
    results["real_world"] = {
        "ok": True,
        "dataset": "iris_binary",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "single_accs": tc["single_accs"],
        "corrected_acc": tc["corrected_acc"],
        "best_single": tc["best_single"],
        "improvement_over_best": tc["improvement"],
        "ensemble_helps": tc["corrected_acc"] >= tc["best_single"],
    }

    return results


if __name__ == "__main__":
    try:
        r = run_top4()
        print("TOP4 RESULTS", r)
        ok = r["oracle"]["ok"] and r["explainer"]["ok"] and r["socratic"]["ok"]
        if r.get("real_world", {}).get("ok"):
            ok = ok and r["real_world"].get("ensemble_helps", False) or r["real_world"].get("corrected_acc", 0) >= r["real_world"].get("best_single", 0)
        print("PASS" if ok else "FAIL")
    except Exception as e:
        print("FAIL", str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)
