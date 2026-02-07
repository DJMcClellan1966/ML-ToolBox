"""
Minimal Fine-Tuning Budget Analysis (Phase 2.2)
================================================

GOAL: Determine if task distillation can ever be practical

Critical questions:
1. Data efficiency: How few training samples needed?
2. Iteration efficiency: How few gradient steps needed?
3. Time efficiency: Is fine-tuning faster than downloading weights?

Success criteria (for ANY practical value):
- <100 samples achieves 95%+ accuracy
- <100 gradient steps converges
- Fine-tuning time < 5 minutes on CPU

If all criteria met → Explore niche applications (Phase 2.3)
If any criteria fails → No practical value, pivot to other research
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict
import json


@dataclass
class BudgetTestResult:
    """Results from a single budget test"""
    n_samples: int
    n_steps: int
    final_accuracy: float
    training_time_seconds: float
    converged: bool  # Did it reach 95%+ accuracy?


class MinimalBudgetTester:
    """Test minimum viable training budget for task distillation"""
    
    def __init__(self, model_name: str = "prajjwal1/bert-tiny", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load dataset once
        print(f"Loading IMDB dataset...")
        self.train_dataset = load_dataset("imdb", split="train")
        self.test_dataset = load_dataset("imdb", split="test[:1000]")
        
        # Tokenize
        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=128
            )
        
        self.train_tokenized = self.train_dataset.map(tokenize, batched=True)
        self.test_tokenized = self.test_dataset.map(tokenize, batched=True)
        
        self.train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        self.test_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        
        # Full test dataloader
        self.test_loader = DataLoader(self.test_tokenized, batch_size=32)
    
    def evaluate(self, model: nn.Module) -> float:
        """Evaluate model accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return 100.0 * correct / total
    
    def train_with_budget(self, n_samples: int, n_steps: int) -> BudgetTestResult:
        """
        Train model from scratch with limited budget
        
        Args:
            n_samples: Number of training samples to use
            n_steps: Maximum number of gradient steps
        """
        print(f"\n{'='*60}")
        print(f"Testing: {n_samples} samples, {n_steps} steps")
        print(f"{'='*60}")
        
        # Create subset of training data
        indices = np.random.choice(len(self.train_tokenized), n_samples, replace=False)
        subset = Subset(self.train_tokenized, indices)
        
        # Calculate batch size to use all samples efficiently
        batch_size = min(32, max(8, n_samples // 10))
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        
        # Initialize random model (Phase 2.1 proved seed doesn't matter)
        config = AutoConfig.from_pretrained(self.model_name)
        config.num_labels = 2
        model = AutoModelForSequenceClassification.from_config(config).to(self.device)
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        model.train()
        
        # Train with step budget
        t0 = time.time()
        steps_taken = 0
        
        for epoch in range(100):  # Max epochs (will break on step limit)
            for batch in train_loader:
                if steps_taken >= n_steps:
                    break
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                steps_taken += 1
                
                if steps_taken % 10 == 0:
                    print(f"  Step {steps_taken}/{n_steps}, Loss: {loss.item():.4f}")
            
            if steps_taken >= n_steps:
                break
        
        training_time = time.time() - t0
        
        # Evaluate
        print(f"\n  Evaluating after {steps_taken} steps...")
        final_acc = self.evaluate(model)
        converged = final_acc >= 95.0
        
        print(f"  ✓ Final accuracy: {final_acc:.2f}%")
        print(f"  ✓ Training time: {training_time:.1f}s")
        print(f"  ✓ Converged: {'YES' if converged else 'NO'}")
        
        return BudgetTestResult(
            n_samples=n_samples,
            n_steps=steps_taken,
            final_accuracy=final_acc,
            training_time_seconds=training_time,
            converged=converged
        )
    
    def test_data_efficiency(self) -> List[BudgetTestResult]:
        """Test: How few samples needed for 95%+ accuracy?"""
        print("\n" + "="*70)
        print("EXPERIMENT 1: DATA EFFICIENCY")
        print("="*70)
        print("Question: How few training samples achieve 95%+ accuracy?")
        print("Method: Train with 10, 25, 50, 100, 250, 500 samples (100 steps each)")
        
        results = []
        sample_counts = [10, 25, 50, 100, 250, 500]
        fixed_steps = 100
        
        for n_samples in sample_counts:
            result = self.train_with_budget(n_samples, fixed_steps)
            results.append(result)
            
            # Early stopping if we found minimum
            if result.converged and n_samples <= 100:
                print(f"\n🎯 Found minimum: {n_samples} samples achieves 95%+ accuracy!")
                break
        
        return results
    
    def test_iteration_efficiency(self, n_samples: int = 250) -> List[BudgetTestResult]:
        """Test: How few gradient steps needed?"""
        print("\n" + "="*70)
        print("EXPERIMENT 2: ITERATION EFFICIENCY")
        print("="*70)
        print(f"Question: How few gradient steps achieve 95%+ accuracy?")
        print(f"Method: Train with {n_samples} samples, varying steps")
        
        results = []
        step_counts = [10, 30, 50, 100, 200]
        
        for n_steps in step_counts:
            result = self.train_with_budget(n_samples, n_steps)
            results.append(result)
            
            if result.converged:
                print(f"\n🎯 Found minimum: {n_steps} steps achieves 95%+ accuracy!")
                break
        
        return results
    
    def test_time_efficiency(self) -> Dict:
        """Compare fine-tuning time vs download time"""
        print("\n" + "="*70)
        print("EXPERIMENT 3: TIME EFFICIENCY")
        print("="*70)
        print("Question: Is fine-tuning faster than downloading weights?")
        
        # Measure fine-tuning time (typical case: 250 samples, 100 steps)
        print("\n[1/2] Measuring fine-tuning time...")
        result = self.train_with_budget(n_samples=250, n_steps=100)
        finetune_time = result.training_time_seconds
        
        # Calculate download times for various bandwidths
        model_size_mb = 17.5  # BERT-tiny weights size
        
        print(f"\n[2/2] Comparing to download times...")
        print(f"  Model size: {model_size_mb:.1f} MB")
        print(f"  Fine-tuning time: {finetune_time:.1f}s ({finetune_time/60:.1f} minutes)")
        
        bandwidths = {
            "Dialup (56 Kbps)": 0.056 / 8,  # MB/s
            "Slow DSL (1 Mbps)": 1 / 8,
            "DSL (5 Mbps)": 5 / 8,
            "Cable (25 Mbps)": 25 / 8,
            "Fiber (100 Mbps)": 100 / 8,
            "Gigabit (1 Gbps)": 1000 / 8,
        }
        
        comparison = {}
        
        print("\n  Bandwidth scenarios:")
        for name, mbps in bandwidths.items():
            download_time = model_size_mb / mbps
            faster = "FINE-TUNING WINS" if finetune_time < download_time else "DOWNLOAD WINS"
            ratio = download_time / finetune_time if finetune_time < download_time else finetune_time / download_time
            
            print(f"    {name:25s}: {download_time:6.1f}s download → {faster} ({ratio:.1f}x)")
            
            comparison[name] = {
                "download_time_s": download_time,
                "finetune_time_s": finetune_time,
                "winner": "finetune" if finetune_time < download_time else "download",
                "ratio": ratio
            }
        
        return {
            "finetune_time_s": finetune_time,
            "finetune_accuracy": result.final_accuracy,
            "model_size_mb": model_size_mb,
            "comparisons": comparison
        }


def run_minimal_budget_analysis():
    """
    Run complete minimal budget analysis
    
    This determines if task distillation can EVER be practical.
    """
    print("="*70)
    print("MINIMAL FINE-TUNING BUDGET ANALYSIS (Phase 2.2)")
    print("="*70)
    print("\nGoal: Determine if task distillation has ANY practical value")
    print("\nSuccess criteria:")
    print("  1. <100 samples achieves 95%+ accuracy (data efficiency)")
    print("  2. <100 steps converges (iteration efficiency)")
    print("  3. Fine-tuning < 5 min on CPU (time efficiency)")
    print("\nIf ALL criteria met → Explore applications (Phase 2.3)")
    print("If ANY criteria fails → No value, pivot to other research")
    print("="*70)
    
    tester = MinimalBudgetTester()
    
    # Test 1: Data efficiency
    data_results = tester.test_data_efficiency()
    
    # Test 2: Iteration efficiency
    iter_results = tester.test_iteration_efficiency(n_samples=250)
    
    # Test 3: Time efficiency
    time_results = tester.test_time_efficiency()
    
    # Analyze results
    print("\n" + "="*70)
    print("FINAL ANALYSIS")
    print("="*70)
    
    # Check criterion 1: Data efficiency
    print("\n📊 CRITERION 1: Data Efficiency")
    min_samples_found = False
    min_samples = None
    for result in data_results:
        if result.converged:
            min_samples = result.n_samples
            min_samples_found = True
            break
    
    if min_samples_found and min_samples <= 100:
        print(f"  ✅ PASS: {min_samples} samples achieves 95%+ accuracy")
        criterion_1_pass = True
    else:
        print(f"  ❌ FAIL: Requires >{max(r.n_samples for r in data_results)} samples")
        criterion_1_pass = False
    
    # Check criterion 2: Iteration efficiency
    print("\n⚡ CRITERION 2: Iteration Efficiency")
    min_steps_found = False
    min_steps = None
    for result in iter_results:
        if result.converged:
            min_steps = result.n_steps
            min_steps_found = True
            break
    
    if min_steps_found and min_steps <= 100:
        print(f"  ✅ PASS: {min_steps} steps achieves 95%+ accuracy")
        criterion_2_pass = True
    else:
        print(f"  ❌ FAIL: Requires >{max(r.n_steps for r in iter_results)} steps")
        criterion_2_pass = False
    
    # Check criterion 3: Time efficiency
    print("\n⏱️  CRITERION 3: Time Efficiency")
    finetune_minutes = time_results["finetune_time_s"] / 60
    
    if finetune_minutes < 5:
        print(f"  ✅ PASS: Fine-tuning takes {finetune_minutes:.1f} minutes")
        criterion_3_pass = True
    else:
        print(f"  ❌ FAIL: Fine-tuning takes {finetune_minutes:.1f} minutes (>5 min)")
        criterion_3_pass = False
    
    # Check when fine-tuning beats download
    wins_over_download = [name for name, data in time_results["comparisons"].items() 
                         if data["winner"] == "finetune"]
    
    if wins_over_download:
        print(f"  ℹ️  Beats download on: {', '.join(wins_over_download)}")
    else:
        print(f"  ⚠️  Never beats downloading weights (even on dialup!)")
    
    # Overall verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    all_pass = criterion_1_pass and criterion_2_pass and criterion_3_pass
    
    if all_pass:
        print("\n✅ ALL CRITERIA MET")
        print("\nTask distillation HAS potential practical value in limited scenarios:")
        print(f"  - Minimum viable: {min_samples} samples, {min_steps} steps, {finetune_minutes:.1f} min")
        print(f"  - Beats download: {', '.join(wins_over_download)}")
        print("\n📍 RECOMMENDATION: Proceed to Phase 2.3")
        print("   Explore niche applications:")
        print("   - Privacy-preserving ML (users train on private data)")
        print("   - Bandwidth-constrained deployment (dialup/satellite)")
        print("   - Regulated environments (can't share weights)")
        verdict = "proceed_phase_2.3"
    else:
        print("\n❌ CRITERIA NOT MET")
        print("\nTask distillation has NO practical value:")
        
        if not criterion_1_pass:
            print("  - Requires too much training data (>100 samples)")
        if not criterion_2_pass:
            print("  - Requires too many training steps (>100 iterations)")
        if not criterion_3_pass:
            print(f"  - Takes too long ({finetune_minutes:.1f} min > 5 min target)")
        
        if not wins_over_download:
            print("  - Never faster than downloading weights")
        
        print("\n📍 RECOMMENDATION: ABANDON RESEARCH DIRECTION")
        print("   Reasons:")
        print("   1. Seeds preserve no knowledge (Phase 2.1)")
        print("   2. Fine-tuning is slow and data-hungry (Phase 2.2)")
        print("   3. No scenario where this beats existing solutions")
        print("\n   Pivot to more promising research:")
        print("   - Adaptive preprocessor with neural networks")
        print("   - Auto-ensemble with knowledge graphs")
        print("   - Advanced feature selection with causal inference")
        verdict = "abandon_and_pivot"
    
    # Save results
    results = {
        "data_efficiency": [
            {
                "n_samples": r.n_samples,
                "n_steps": r.n_steps,
                "accuracy": r.final_accuracy,
                "time_s": r.training_time_seconds,
                "converged": r.converged
            } for r in data_results
        ],
        "iteration_efficiency": [
            {
                "n_samples": r.n_samples,
                "n_steps": r.n_steps,
                "accuracy": r.final_accuracy,
                "time_s": r.training_time_seconds,
                "converged": r.converged
            } for r in iter_results
        ],
        "time_efficiency": time_results,
        "criteria": {
            "data_efficiency": criterion_1_pass,
            "iteration_efficiency": criterion_2_pass,
            "time_efficiency": criterion_3_pass
        },
        "verdict": verdict,
        "recommendation": "proceed_phase_2.3" if all_pass else "abandon_and_pivot"
    }
    
    with open("research/minimal_budget_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n📁 Results saved to: research/minimal_budget_results.json")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = run_minimal_budget_analysis()
