"""
Default LLM Practice Problem Generator for Learning Apps
Auto-generates practice questions and coding challenges for any topic using OpenAI or Ollama.
"""
from flask import request, jsonify
import os

# Try OpenAI or Ollama
try:
    import openai
except ImportError:
    openai = None
try:
    import ollama
except ImportError:
    ollama = None


def register_practice_routes(app):
    @app.route("/api/practice/generate", methods=["POST"])
    def api_practice_generate():
        data = request.get_json(silent=True) or {}
        topic = data.get("topic", "machine learning")
        prompt = f"Generate a practice problem and coding challenge for: {topic}. Include a solution and explanation."
        response = None
        if ollama:
            try:
                resp = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
                response = resp["message"]["content"]
            except Exception:
                response = None
        elif openai and os.getenv("OPENAI_API_KEY"):
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500
                )
                response = resp["choices"][0]["message"]["content"]
            except Exception:
                response = None
        if not response:
            response = f"Practice problem for {topic}:\n[Sample question]\nSolution: [Sample solution]\nExplanation: [Sample explanation]"
        return jsonify({"ok": True, "problem": response})
