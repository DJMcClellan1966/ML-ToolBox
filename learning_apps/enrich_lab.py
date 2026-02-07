"""
Universal Lab Enrichment Tool
==============================

Uses curriculum_extractor.py to enrich any learning lab.

Usage:
    python enrich_lab.py rl_lab ml_toolbox/ai_concepts/reinforcement_learning.py
    python enrich_lab.py practical_ml_lab ml_toolbox/textbook_concepts/practical_ml.py
"""

import sys
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from learning_apps.curriculum_extractor import CurriculumExtractor


# Lab mapping: lab_id → (book_id, target_count, topics)
LAB_CONFIG = {
    "rl_lab": {
        "book_id": "sutton",
        "target_count": 20,
        "topics": ["MDP", "policy", "value function", "Q-learning", "actor-critic", "policy gradient"],
        "sources": [
            "ml_toolbox/ai_concepts/reinforcement_learning.py",
            "corpus/reinforcement_learning.md"
        ]
    },
    "practical_ml_lab": {
        "book_id": "burkov",
        "target_count": 25,
        "topics": ["feature engineering", "cross-validation", "model selection", "hyperparameter tuning"],
        "sources": [
            "ml_toolbox/textbook_concepts/practical_ml.py",
            "ml_toolbox/textbook_concepts/data_quality.py"
        ]
    },
    "probabilistic_ml_lab": {
        "book_id": "bishop",
        "target_count": 20,
        "topics": ["Bayesian inference", "graphical models", "variational inference", "MCMC"],
        "sources": [
            "ml_toolbox/textbook_concepts/probabilistic_ml.py",
            "ml_toolbox/textbook_concepts/bayesian_networks.py"
        ]
    },
    "causal_inference_lab": {
        "book_id": "pearl",
        "target_count": 20,
        "topics": ["causal graphs", "intervention", "counterfactuals", "do-calculus"],
        "sources": [
            "ml_toolbox/ai_concepts/causal_reasoning.py",
            "corpus/causality.md"
        ]
    },
    "sicp_lab": {
        "book_id": "sicp",
        "target_count": 30,
        "topics": ["recursion", "higher-order functions", "closures", "lazy evaluation", "streams"],
        "sources": [
            "ml_toolbox/textbook_concepts/linguistics.py",
            "corpus/functional_programming.md"
        ]
    }
}


def extract_from_sources(sources: list[str], topics: list[str]) -> list:
    """Extract concepts from multiple source files."""
    
    extractor = CurriculumExtractor()
    all_concepts = []
    
    for source_path in sources:
        full_path = REPO_ROOT / source_path
        
        if not full_path.exists():
            print(f"⚠️  Source not found: {source_path}")
            continue
        
        print(f"📖 Reading: {source_path}")
        content = full_path.read_text(encoding='utf-8')
        
        # Extract concepts
        concepts = extractor.extract_from_text(content, source=source_path)
        
        # Filter by topics (keyword matching)
        filtered = []
        for concept in concepts:
            text = (concept.title + " " + concept.content).lower()
            if any(topic.lower() in text for topic in topics):
                filtered.append(concept)
        
        print(f"   ✅ Extracted {len(filtered)} relevant concepts")
        all_concepts.extend(filtered)
    
    return all_concepts


def generate_curriculum(lab_id: str) -> dict:
    """Generate curriculum for a lab."""
    
    if lab_id not in LAB_CONFIG:
        print(f"❌ Unknown lab: {lab_id}")
        print(f"Available labs: {', '.join(LAB_CONFIG.keys())}")
        return None
    
    config = LAB_CONFIG[lab_id]
    
    print("=" * 70)
    print(f"ENRICHING: {lab_id}")
    print("=" * 70)
    print(f"Target: {config['target_count']} items")
    print(f"Topics: {', '.join(config['topics'])}")
    print()
    
    # Extract concepts
    concepts = extract_from_sources(config["sources"], config["topics"])
    
    if not concepts:
        print("❌ No concepts extracted!")
        return None
    
    # Convert to curriculum format
    extractor = CurriculumExtractor()
    curriculum = extractor.to_curriculum_format(
        concepts,
        book_id=config["book_id"],
        default_level="intermediate"
    )
    
    # Save to cache
    output_dir = REPO_ROOT / "learning_apps" / ".cache"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"{lab_id}_enriched.json"
    output_file.write_text(json.dumps(curriculum, indent=2))
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Extracted: {len(curriculum)} curriculum items")
    print(f"Saved to: {output_file}")
    
    # Show distribution
    from collections import Counter
    level_counts = Counter(item.get('level', 'unknown') for item in curriculum)
    print(f"\nBy level:")
    for level, count in level_counts.items():
        print(f"  {level:15s}: {count}")
    
    # Show sample items
    print(f"\nSample items:")
    for item in curriculum[:3]:
        print(f"  • {item['title']}")
    
    return curriculum


def main():
    if len(sys.argv) < 2:
        print("Usage: python enrich_lab.py <lab_id>")
        print(f"\nAvailable labs:")
        for lab_id, config in LAB_CONFIG.items():
            print(f"  {lab_id:25s} - {config['target_count']} items, topics: {', '.join(config['topics'][:3])}")
        return
    
    lab_id = sys.argv[1]
    curriculum = generate_curriculum(lab_id)
    
    if curriculum:
        print(f"\n✅ SUCCESS: Generated {len(curriculum)} items for {lab_id}")
        print(f"\nNext steps:")
        print(f"1. Review .cache/{lab_id}_enriched.json")
        print(f"2. Manually curate and add to {lab_id}/curriculum.py")
        print(f"3. Test: python hub.py → {lab_id}")


if __name__ == "__main__":
    main()
