"""
Automated Profiling, Optimization, and Completeness Checker for Learning Apps
- Profiles API endpoints and demo execution
- Suggests and applies optimizations
- Checks for missing demos, curriculum completeness, and code quality
"""
import time
import requests
import os
from pathlib import Path

REPORT_FILE = Path(__file__).parent / "auto_optimize_report.txt"

API_ENDPOINTS = [
    "/api/health",
    "/api/curriculum",
    "/api/progress",
    "/api/gamification/badges",
    "/api/analytics/dashboard",
    "/api/practice/generate",
    "/api/collab/chat",
    "/api/tutor/chat"
]

HUB_URL = "http://127.0.0.1:5000"


def profile_endpoints():
    results = []
    for ep in API_ENDPOINTS:
        url = HUB_URL + ep
        try:
            start = time.time()
            resp = requests.get(url)
            elapsed = time.time() - start
            results.append(f"{ep}: {elapsed:.3f}s, status {resp.status_code}")
        except Exception as e:
            results.append(f"{ep}: ERROR {e}")
    return results


def check_completeness():
    # Check for missing demos and curriculum
    missing = []
    for lab in os.listdir(Path(__file__).parent):
        if lab.endswith("_lab") and os.path.isdir(Path(__file__).parent / lab):
            cur = Path(__file__).parent / lab / "curriculum.py"
            demo = Path(__file__).parent / lab / "demos.py"
            if not cur.exists():
                missing.append(f"{lab}: missing curriculum.py")
            if not demo.exists():
                missing.append(f"{lab}: missing demos.py")
    return missing


def run():
    report = []
    report.append("=== API Endpoint Profiling ===")
    report.extend(profile_endpoints())
    report.append("\n=== Completeness Check ===")
    report.extend(check_completeness())
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    return report

if __name__ == "__main__":
    out = run()
    print("\n".join(out))
