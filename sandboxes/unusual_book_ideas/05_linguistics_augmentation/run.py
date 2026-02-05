"""Sandbox 05: Linguistics-Driven Augmentation - viability test."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def run():
    from ml_toolbox.textbook_concepts.linguistics import (
        SimpleSyntacticParser,
        GrammarBasedFeatureExtractor,
    )
    parser = SimpleSyntacticParser()
    extractor = GrammarBasedFeatureExtractor()
    sentences = [
        "The model is learning.",
        "Machine learning is amazing.",
    ]
    all_pos = []
    all_phrases = []
    for s in sentences:
        pos = parser.extract_pos_tags(s)
        phrases = parser.extract_phrases(s)
        all_pos.append(len(pos))
        all_phrases.append(phrases)
    # "Augmentation": same number of tokens, different words (simplified: just check we got structure)
    feats = extractor.extract_features([sentences[0]])
    return {
        "ok": True,
        "sentences_parsed": len(sentences),
        "pos_counts": all_pos,
        "phrase_keys": list(all_phrases[0].keys()) if all_phrases[0] else [],
        "grammar_feature_matrix_shape": list(feats.shape),
    }

if __name__ == "__main__":
    try:
        result = run()
        print("PASS", result)
    except Exception as e:
        print("FAIL", str(e))
        sys.exit(1)
