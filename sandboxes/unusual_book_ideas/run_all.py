"""Run all 8 sandbox viability tests and collect results."""
import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SANDBOXES = [
    ("01_theory_as_channel", "Theory-as-Channel"),
    ("02_self_organizing_curriculum", "Self-Organizing Curriculum"),
    ("03_socratic_multi_viewpoint", "Socratic Multi-Viewpoint"),
    ("04_precognition_failure_modes", "Precognition + Failure Modes"),
    ("05_linguistics_augmentation", "Linguistics Augmentation"),
    ("06_entropy_across_domains", "Entropy Across Domains"),
    ("07_dissipative_training", "Dissipative Training"),
    ("08_algorithm_oracle", "Algorithm Oracle"),
]

def main():
    results = []
    for dirname, label in SANDBOXES:
        run_py = Path(__file__).parent / dirname / "run.py"
        if not run_py.exists():
            results.append((label, "SKIP", "run.py not found", {}))
            continue
        try:
            proc = subprocess.run(
                [sys.executable, str(run_py)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=60,
            )
            out = (proc.stdout or "").strip()
            err = (proc.stderr or "").strip()
            if proc.returncode == 0:
                if out.startswith("PASS"):
                    results.append((label, "PASS", out, err))
                else:
                    results.append((label, "FAIL", out or "no output", err))
            else:
                results.append((label, "FAIL", out or err or f"exit {proc.returncode}", err))
        except subprocess.TimeoutExpired:
            results.append((label, "FAIL", "timeout", ""))
        except Exception as e:
            results.append((label, "FAIL", str(e), ""))
    return results

if __name__ == "__main__":
    results = main()
    for label, status, out, _ in results:
        print(f"{status}\t{label}\t{out[:80]}")
    # Exit 1 if any failed
    sys.exit(0 if all(r[1] == "PASS" for r in results) else 1)
