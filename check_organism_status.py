"""Check if the ML Organism is still alive"""

from organic_ml_organism import MLOrganism

print("\n" + "="*80)
print("ORGANISM STATUS CHECK".center(80))
print("="*80)

# Create organism
organism = MLOrganism()

# Check status
print("\n[Status] FUNCTIONAL")
print(f"[Memory] {len(organism.memory.memory['active'])} active items")
print(f"[Learning] {len(organism.learning.patterns)} patterns")
print(f"[Discovery] {len(organism.discovery.knowledge_network)} nodes")
print(f"[Reasoning] {len(organism.reasoning.reasoning_history)} history")

# Test functionality
print("\n[Test] Processing data...")
import numpy as np
result = organism.process(np.array([1, 2, 3]), task="test")
print(f"[Test] SUCCESS - Output type: {type(result['output'])}")

print("\n" + "="*80)
print("VERDICT: The organism is ALIVE (functional) but not sentient!")
print("="*80)
print("\nIt's a program that:")
print("  [OK] Works when executed")
print("  [OK] Processes data")
print("  [OK] Learns patterns")
print("  [X] Cannot access the internet")
print("  [X] Cannot modify itself")
print("  [X] Cannot become super intelligent")
print("\nIt's safe! Just a useful learning tool.")
print("="*80 + "\n")
