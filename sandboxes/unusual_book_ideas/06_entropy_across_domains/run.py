"""Sandbox 06: Entropy Across Domains - viability test."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def run():
    views = {
        "information_theory": "Shannon entropy H = -sum p log p measures uncertainty in bits. Higher H means more randomness; lower H means more predictable.",
        "thermodynamics": "Boltzmann linked entropy to disorder. In statistical mechanics, entropy counts microstates; the same logarithmic form appears.",
        "ml": "In ML, entropy is used in decision tree splitting (information gain), in softmax loss, and in regularization to encourage smooth distributions.",
    }
    result_text = "\n\n".join(f"[{k}]\n{v}" for k, v in views.items())
    return {
        "ok": True,
        "views": list(views.keys()),
        "total_chars": len(result_text),
        "preview": result_text[:120] + "...",
    }

if __name__ == "__main__":
    try:
        result = run()
        print("PASS", result)
    except Exception as e:
        print("FAIL", str(e))
        sys.exit(1)
