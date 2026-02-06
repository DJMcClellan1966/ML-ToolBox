"""
ML Compass - App entry point.
CLI and optional FastAPI server. Run from repo root so ml_toolbox is on path.
"""
import sys
import json
import argparse
from pathlib import Path

# Ensure repo root is on path when running as script or module
_REPO_ROOT = Path(__file__).resolve().parents[1]  # ML_Compass dir
_REPO_ROOT = _REPO_ROOT.parent  # repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ML_Compass.oracle import suggest as oracle_suggest
from ML_Compass.explainers import explain_concept
from ML_Compass.theory_channel import channel_capacity_bits, recommend_redundancy
from ML_Compass.socratic import debate_and_question


def cmd_oracle(profile_dict: dict) -> None:
    """Print oracle suggestion for a problem profile."""
    out = oracle_suggest(profile_dict)
    print("Pattern:", out.get("pattern"))
    print("Suggestion:", out.get("suggestion"))
    print("Why:", out.get("why"))


def cmd_explain(concept: str) -> None:
    """Print cross-domain explanation for a concept."""
    out = explain_concept(concept)
    if not out["ok"]:
        print("Unknown concept. Available:", out.get("available", []))
        return
    for view_name, text in out["views"].items():
        print(f"[{view_name}]", text)


def cmd_debate(statement: str) -> None:
    """Print debate + Socratic question."""
    out = debate_and_question(statement)
    print(out.get("debate", out.get("question", "")))


def cmd_capacity(signal: float, noise: float) -> None:
    """Print channel capacity and redundancy recommendation."""
    C = channel_capacity_bits(signal_power=signal, noise_power=noise)
    rec = recommend_redundancy(signal_power=signal, noise_power=noise)
    print("Channel capacity (bits):", C)
    print("Recommended ensemble size:", rec["recommended_models"])


def cmd_run_all(profile: dict, concept: str, statement: str) -> None:
    """Run full flow: oracle + explain + debate."""
    print("=== Oracle ===")
    o = oracle_suggest(profile)
    print("Pattern:", o.get("pattern"))
    print("Suggestion:", o.get("suggestion"))
    print("Why:", o.get("why"))
    print()
    print("=== Explain:", concept, "===")
    e = explain_concept(concept)
    if e["ok"]:
        for k, v in e["views"].items():
            print(f"[{k}]", v[:80] + "..." if len(v) > 80 else v)
    print()
    print("=== Debate ===")
    d = debate_and_question(statement)
    print(d.get("debate", "")[:300])


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="ML Compass - Decide. Understand. Build. Think.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ML_Compass oracle --tabular --classification --n_samples medium
  python -m ML_Compass explain entropy
  python -m ML_Compass debate "I used an ensemble."
  python -m ML_Compass capacity --signal 10 --noise 1
  python -m ML_Compass run_all
        """,
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # oracle
    p_oracle = sub.add_parser("oracle", help="Get algorithm suggestion for problem profile")
    p_oracle.add_argument("--tabular", action="store_true", help="Tabular data")
    p_oracle.add_argument("--text", action="store_true", help="Text data")
    p_oracle.add_argument("--classification", action="store_true", help="Classification task")
    p_oracle.add_argument("--n_samples", choices=["small", "medium", "large"], default="medium")
    p_oracle.add_argument("--high_dim", action="store_true")
    p_oracle.add_argument("--need_safety", action="store_true")
    p_oracle.add_argument("--high_volume", action="store_true")
    p_oracle.add_argument("--unsupervised", action="store_true")
    p_oracle.add_argument("--cluster_shape", default=None)
    p_oracle.set_defaults(func=lambda a: cmd_oracle(_profile_from_args(a)))

    # explain
    p_explain = sub.add_parser("explain", help="Explain a concept (entropy, bias_variance, capacity)")
    p_explain.add_argument("concept", nargs="?", default="entropy")
    p_explain.set_defaults(func=lambda a: cmd_explain(a.concept))

    # debate
    p_debate = sub.add_parser("debate", help="Debate + Socratic question")
    p_debate.add_argument("statement", nargs="?", default="I used an ensemble of three models.")
    p_debate.set_defaults(func=lambda a: cmd_debate(a.statement))

    # capacity
    p_cap = sub.add_parser("capacity", help="Channel capacity and redundancy recommendation")
    p_cap.add_argument("--signal", type=float, default=10.0)
    p_cap.add_argument("--noise", type=float, default=1.0)
    p_cap.set_defaults(func=lambda a: cmd_capacity(a.signal, a.noise))

    # run_all
    p_all = sub.add_parser("run_all", help="Run oracle + explain + debate with defaults")
    p_all.set_defaults(
        func=lambda a: cmd_run_all(
            {"tabular": True, "classification": True, "n_samples": "medium"},
            "entropy",
            "I used an ensemble.",
        )
    )

    # serve (FastAPI)
    p_serve = sub.add_parser("serve", help="Start FastAPI server (requires: pip install fastapi uvicorn)")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.set_defaults(func=lambda a: _run_serve(a.host, a.port))

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    args.func(args)


def _run_serve(host: str, port: int) -> None:
    app = create_app()
    if app is None:
        print("Install fastapi and uvicorn: pip install fastapi uvicorn")
        sys.exit(1)
    import uvicorn
    uvicorn.run(app, host=host, port=port)


def _profile_from_args(a) -> dict:
    profile = {}
    if getattr(a, "tabular", False):
        profile["tabular"] = True
    if getattr(a, "text", False):
        profile["text"] = True
    if getattr(a, "classification", False):
        profile["classification"] = True
    if getattr(a, "n_samples", None):
        profile["n_samples"] = a.n_samples
    if getattr(a, "high_dim", False):
        profile["high_dim"] = True
    if getattr(a, "need_safety", False):
        profile["need_safety"] = True
    if getattr(a, "high_volume", False):
        profile["high_volume"] = True
    if getattr(a, "unsupervised", False):
        profile["unsupervised"] = True
    if getattr(a, "cluster_shape", None):
        profile["cluster_shape"] = a.cluster_shape
    if not profile:
        profile = {"tabular": True, "classification": True, "n_samples": "medium"}
    return profile


def create_app():
    """Create FastAPI app for ML Compass (optional)."""
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError:
        return None
    app = FastAPI(title="ML Compass", description="Decide. Understand. Build. Think.")

    @app.get("/")
    def root():
        return {"name": "ML Compass", "tagline": "Decide. Understand. Build. Think."}

    @app.post("/oracle")
    def api_oracle(profile: dict):
        return oracle_suggest(profile)

    @app.get("/explain/{concept}")
    def api_explain(concept: str):
        return explain_concept(concept)

    @app.post("/debate")
    def api_debate(statement: str):
        return debate_and_question(statement)

    @app.get("/capacity")
    def api_capacity(signal_power: float = 10.0, noise_power: float = 1.0):
        try:
            C = channel_capacity_bits(signal_power, noise_power)
            rec = recommend_redundancy(signal_power, noise_power)
            return {"channel_capacity_bits": C, "recommend_redundancy": rec}
        except ImportError as e:
            raise HTTPException(500, str(e))

    return app


if __name__ == "__main__":
    main_cli()
