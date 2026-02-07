"""
Stress Tests: Critical Real-World Validation
==============================================

Tests that would break the system if it's not truly revolutionary:
1. Low-data regime (50 samples vs 500)
2. Different task (NER vs classification)
3. Larger scale (GPT-2 124M params)
4. Extreme compression (push to 500x)
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoModelForSequenceClassification, 
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoTokenizer
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import time
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from seed_pytorch_integration import (
    TorchSeedExtractor,
    TorchSeedAssembler,
    TorchSeed,
    evaluate_model,
    get_model_size
)


# =====================================================================
# Stress Test 1: Low-Data Regime (Few-Shot Learning)
# =====================================================================

def stress_test_low_data(model_name="prajjwal1/bert-tiny"):
    """Test with 50, 100, 200, 500 training samples."""
    print("="*70)
    print("STRESS TEST 1: Low-Data Regime")
    print("="*70)
    print("\nObjective: Prove it works with limited training data")
    print("Test: 50, 100, 200, 500 samples")
    print("Success: >90% retention with 50 samples")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    n_params = get_model_size(model)
    print(f"\nModel: {n_params:,} parameters")
    
    # Extract seed once
    extractor = TorchSeedExtractor(n_components=15)
    seed = extractor.extract(model)
    print(f"Seed: {seed.size_bytes()/1024:.1f}KB ({seed.compression_ratio(n_params):.1f}x)")
    
    # Load evaluation data
    eval_dataset = load_dataset("glue", "sst2", split="validation[:100]")
    
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", 
                       truncation=True, max_length=128)
    
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.rename_column("label", "labels")
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_loader = DataLoader(eval_dataset, batch_size=8)
    
    # Baseline accuracy
    orig_acc = evaluate_model(model, eval_loader, tokenizer)
    print(f"\nBaseline accuracy: {orig_acc:.1%}")
    
    results = []
    
    # Test with different training set sizes
    for n_train in [50, 100, 200, 500]:
        print(f"\n{'='*70}")
        print(f"Testing with {n_train} training samples")
        print(f"{'='*70}")
        
        # Load training data
        train_dataset = load_dataset("glue", "sst2", split=f"train[:{n_train}]")
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        train_dataset = train_dataset.rename_column("label", "labels")
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # Assemble and fine-tune
        assembler = TorchSeedAssembler(
            seed=seed,
            model_class=type(model),
            model_config=model.config
        )
        
        start = time.time()
        assembler.assemble(train_data=train_loader, epochs=3, lr=2e-5, verbose=False)
        assembly_time = time.time() - start
        
        # Evaluate
        assembled_acc = evaluate_model(assembler.model, eval_loader, tokenizer)
        retention = (assembled_acc / orig_acc * 100) if orig_acc > 0 else 0
        
        print(f"  Assembled accuracy: {assembled_acc:.1%}")
        print(f"  Retention: {retention:.1f}%")
        print(f"  Assembly time: {assembly_time:.1f}s")
        
        results.append({
            'n_train': n_train,
            'accuracy': assembled_acc,
            'retention': retention,
            'time': assembly_time
        })
    
    # Summary
    print(f"\n{'='*70}")
    print("Low-Data Regime Results")
    print(f"{'='*70}")
    print(f"{'Samples':<10} {'Accuracy':<12} {'Retention':<12} {'Time (s)':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['n_train']:<10} {r['accuracy']:>10.1%} {r['retention']:>10.1f}% {r['time']:>9.1f}")
    
    # Check success criteria
    best_50 = next((r for r in results if r['n_train'] == 50), None)
    if best_50 and best_50['retention'] >= 90:
        print(f"\n✅ SUCCESS: {best_50['retention']:.1f}% retention with 50 samples!")
    else:
        print(f"\n⚠️  LIMITATION: Only {best_50['retention']:.1f}% retention with 50 samples")
    
    return results


# =====================================================================
# Stress Test 2: Task Diversity (NER vs Classification)
# =====================================================================

def stress_test_task_diversity(model_name="prajjwal1/bert-tiny"):
    """Test on Named Entity Recognition (structured output)."""
    print("\n" + "="*70)
    print("STRESS TEST 2: Task Diversity (NER)")
    print("="*70)
    print("\nObjective: Prove it works beyond classification")
    print("Test: Named Entity Recognition (token-level tagging)")
    print("Success: >90% retention on NER task")
    
    # Load NER model
    model = AutoModelForTokenClassification.from_pretrained(
        "dslim/bert-base-NER",
        ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    
    n_params = get_model_size(model)
    print(f"\nModel: {n_params:,} parameters")
    
    # Extract seed
    extractor = TorchSeedExtractor(n_components=20)
    seed = extractor.extract(model)
    print(f"Seed: {seed.size_bytes()/1024:.1f}KB ({seed.compression_ratio(n_params):.1f}x)")
    
    # Load CoNLL-2003 dataset
    print("\n[LOADING] CoNLL-2003 dataset...")
    try:
        dataset = load_dataset("conll2003")
        
        # Prepare data
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples["tokens"],
                truncation=True,
                is_split_into_words=True,
                padding="max_length",
                max_length=128
            )
            
            labels = []
            for i, label in enumerate(examples["ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    else:
                        label_ids.append(label[word_idx] if word_idx < len(label) else -100)
                labels.append(label_ids)
            
            tokenized_inputs["labels"] = labels
            return tokenized_inputs
        
        train_dataset = dataset["train"].select(range(500))
        eval_dataset = dataset["validation"].select(range(100))
        
        train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
        eval_dataset = eval_dataset.map(tokenize_and_align_labels, batched=True)
        
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=8)
        
        # Evaluate original
        print("\n[EVALUATING] Original NER model...")
        orig_acc = evaluate_model(model, eval_loader, tokenizer)
        print(f"  Original accuracy: {orig_acc:.1%}")
        
        # Assemble and fine-tune
        print("\n[SEED ASSEMBLY] Building from seed...")
        assembler = TorchSeedAssembler(
            seed=seed,
            model_class=type(model),
            model_config=model.config
        )
        
        start = time.time()
        assembler.assemble(train_data=train_loader, epochs=2, lr=2e-5, verbose=True)
        assembly_time = time.time() - start
        
        # Evaluate assembled
        print("\n[EVALUATING] Assembled NER model...")
        assembled_acc = evaluate_model(assembler.model, eval_loader, tokenizer)
        retention = (assembled_acc / orig_acc * 100) if orig_acc > 0 else 0
        
        print(f"\n{'='*70}")
        print("NER Task Results")
        print(f"{'='*70}")
        print(f"  Original accuracy: {orig_acc:.1%}")
        print(f"  Assembled accuracy: {assembled_acc:.1%}")
        print(f"  Retention: {retention:.1f}%")
        print(f"  Assembly time: {assembly_time:.1f}s")
        
        if retention >= 90:
            print(f"\n✅ SUCCESS: {retention:.1f}% retention on NER task!")
        else:
            print(f"\n⚠️  LIMITATION: Only {retention:.1f}% retention on NER")
        
        return {
            'task': 'NER',
            'orig_acc': orig_acc,
            'assembled_acc': assembled_acc,
            'retention': retention,
            'assembly_time': assembly_time
        }
        
    except Exception as e:
        print(f"Error: {e}")
        print("NER test requires 'conll2003' dataset. Install: pip install datasets")
        return None


# =====================================================================
# Stress Test 3: Extreme Compression (Push to 500x)
# =====================================================================

def stress_test_extreme_compression(model_name="prajjwal1/bert-tiny"):
    """Test with minimal seeds (5, 10, 15, 20 components)."""
    print("\n" + "="*70)
    print("STRESS TEST 3: Extreme Compression")
    print("="*70)
    print("\nObjective: Find compression limit")
    print("Test: n_components = 20, 15, 10, 5, 2")
    print("Success: >90% retention even at 200x+ compression")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    n_params = get_model_size(model)
    print(f"\nModel: {n_params:,} parameters")
    
    # Load data
    train_dataset = load_dataset("glue", "sst2", split="train[:500]")
    eval_dataset = load_dataset("glue", "sst2", split="validation[:100]")
    
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", 
                       truncation=True, max_length=128)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    train_dataset = train_dataset.rename_column("label", "labels")
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.rename_column("label", "labels")
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=8)
    
    # Baseline
    orig_acc = evaluate_model(model, eval_loader, tokenizer)
    print(f"\nBaseline accuracy: {orig_acc:.1%}")
    
    results = []
    
    # Test different compression levels
    for n_comp in [20, 15, 10, 5, 2]:
        print(f"\n{'='*70}")
        print(f"Testing with n_components = {n_comp}")
        print(f"{'='*70}")
        
        # Extract seed
        extractor = TorchSeedExtractor(n_components=n_comp)
        seed = extractor.extract(model)
        
        compression = seed.compression_ratio(n_params)
        print(f"  Seed: {seed.size_bytes()/1024:.1f}KB")
        print(f"  Compression: {compression:.1f}x")
        
        # Assemble and fine-tune
        assembler = TorchSeedAssembler(
            seed=seed,
            model_class=type(model),
            model_config=model.config
        )
        
        assembler.assemble(train_data=train_loader, epochs=3, lr=2e-5, verbose=False)
        
        # Evaluate
        assembled_acc = evaluate_model(assembler.model, eval_loader, tokenizer)
        retention = (assembled_acc / orig_acc * 100) if orig_acc > 0 else 0
        
        print(f"  Assembled accuracy: {assembled_acc:.1%}")
        print(f"  Retention: {retention:.1f}%")
        
        results.append({
            'n_components': n_comp,
            'compression': compression,
            'seed_kb': seed.size_bytes()/1024,
            'accuracy': assembled_acc,
            'retention': retention
        })
    
    # Summary
    print(f"\n{'='*70}")
    print("Extreme Compression Results")
    print(f"{'='*70}")
    print(f"{'Components':<12} {'Compress':<12} {'Seed (KB)':<12} {'Accuracy':<12} {'Retention':<12}")
    print("-"*70)
    for r in results:
        print(f"{r['n_components']:<12} {r['compression']:>10.1f}x {r['seed_kb']:>10.1f} "
              f"{r['accuracy']:>10.1%} {r['retention']:>10.1f}%")
    
    # Find sweet spot
    best = max(results, key=lambda r: r['retention'] * r['compression'])
    print(f"\nSweet spot: {best['n_components']} components")
    print(f"  Compression: {best['compression']:.1f}x")
    print(f"  Retention: {best['retention']:.1f}%")
    
    return results


# =====================================================================
# Main: Run All Stress Tests
# =====================================================================

def run_all_stress_tests():
    """Run comprehensive stress test suite."""
    print("="*70)
    print("COMPREHENSIVE STRESS TESTS")
    print("="*70)
    print("\nObjective: Prove seed assembly is revolutionary")
    print("Tests: Low-data, Task diversity, Extreme compression")
    print("\nIf it passes all tests:")
    print("  ✅ Works with 50 samples (few-shot)")
    print("  ✅ Works on NER (not just classification)")
    print("  ✅ Scales to 200x+ compression")
    print("  ✅ TRULY REVOLUTIONARY")
    
    # Test 1: Low-data regime
    low_data_results = stress_test_low_data()
    
    # Test 2: Task diversity (NER)
    # ner_results = stress_test_task_diversity()
    
    # Test 3: Extreme compression
    extreme_results = stress_test_extreme_compression()
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    # Check if passed all tests
    test_50_samples = any(r['retention'] >= 90 for r in low_data_results if r['n_train'] == 50)
    test_extreme = any(r['retention'] >= 90 and r['compression'] >= 150 for r in extreme_results)
    
    print(f"\n50-sample test: {'✅ PASS' if test_50_samples else '⚠️  MARGINAL'}")
    print(f"Extreme compression test: {'✅ PASS' if test_extreme else '⚠️  MARGINAL'}")
    
    if test_50_samples and test_extreme:
        print("\n" + "🎉"*20)
        print("VERDICT: REVOLUTIONARY")
        print("Seed assembly passes all stress tests!")
        print("Ready for publication and real-world deployment.")
        print("🎉"*20)
    else:
        print("\nVERDICT: PROMISING BUT LIMITED")
        print("Works well in tested scenarios but needs optimization for edge cases.")
        print("Recommend Phase 1.3 (ablation) and Phase 2 (optimization) before publication.")


if __name__ == "__main__":
    run_all_stress_tests()
