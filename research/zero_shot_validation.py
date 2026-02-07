"""
Zero-Shot Validation (Phase 2.1 - CRITICAL TEST)
================================================

THE CRITICAL QUESTION:
Do seeds preserve ANY pre-trained knowledge without fine-tuning?

Previous tests (Phase 1.1-1.3) all included fine-tuning, which masked whether
seeds actually preserve knowledge or just provide initialization for re-training.

This test answers the question definitively by evaluating assembled models
with ZERO fine-tuning steps.

Expected outcomes:
1. Accuracy ~0-5%: Seeds preserve NOTHING → Pure task distillation
2. Accuracy 50-80%: Seeds preserve PARTIAL knowledge → Lossy compression
3. Accuracy >80%: Seeds preserve MOST knowledge → Compression works!

This single test determines the entire research direction.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
import time


class ZeroShotValidator:
    """Test assembled models WITHOUT any fine-tuning"""
    
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def evaluate_model(self, model, dataloader, task_type="classification"):
        """Evaluate model accuracy with no training"""
        model.eval()
        
        if task_type == "classification":
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
            
            accuracy = 100.0 * correct / total
            return accuracy
        
        elif task_type == "generation":
            # For language models, use perplexity as quality metric
            total_loss = 0
            total_tokens = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    
                    if outputs.loss is not None:
                        total_loss += outputs.loss.item() * input_ids.size(0)
                        total_tokens += input_ids.size(0)
            
            avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
            perplexity = np.exp(avg_loss)
            return perplexity
    
    def test_bert_zero_shot(self):
        """
        Test BERT-tiny on IMDB sentiment classification
        WITHOUT any fine-tuning
        """
        print("\n" + "="*70)
        print("ZERO-SHOT TEST 1: BERT-tiny (4.4M params)")
        print("="*70)
        
        model_name = "prajjwal1/bert-tiny"
        
        # Load original pre-trained model
        print("\n[1/4] Loading original pre-trained model...")
        original_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        ).to(self.device)
        
        # Prepare data
        print("[2/4] Loading evaluation dataset (IMDB)...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dataset = load_dataset("imdb", split="test[:1000]")
        
        def tokenize(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=128
            )
        
        tokenized = dataset.map(tokenize, batched=True)
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        dataloader = DataLoader(tokenized, batch_size=32)
        
        # Evaluate original
        print("[3/4] Evaluating original pre-trained model (baseline)...")
        original_acc = self.evaluate_model(original_model, dataloader)
        print(f"✓ Original model accuracy: {original_acc:.2f}%")
        
        # Extract seed and assemble WITHOUT fine-tuning
        print("[4/4] Extracting seed and assembling WITHOUT fine-tuning...")
        from seed_pytorch_integration import TorchSeedExtractor
        from transformers import BertForSequenceClassification
        
        extractor = TorchSeedExtractor(n_components=10)
        seed = extractor.extract(original_model)
        
        print(f"Seed size: {seed.size_bytes():,} bytes ({seed.size_bytes()/1024:.1f} KB)")
        
        # Assemble from seed - NO FINE-TUNING
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 2
        assembled_model = BertForSequenceClassification(config).to(self.device)
        
        # Initialize from seed statistics only
        seed_idx = 0
        for name, param in assembled_model.named_parameters():
            if param.requires_grad and seed_idx < len(seed.layer_names):
                mean = seed.layer_means[seed_idx][0]
                std = seed.layer_stds[seed_idx]
                with torch.no_grad():
                    param.data.normal_(mean=mean, std=std)
                seed_idx += 1
        
        # Evaluate assembled model - ZERO FINE-TUNING
        print("\n[CRITICAL] Evaluating assembled model with ZERO fine-tuning...")
        assembled_acc = self.evaluate_model(assembled_model, dataloader)
        
        print("\n" + "="*70)
        print("BERT-TINY ZERO-SHOT RESULTS")
        print("="*70)
        print(f"Original model:   {original_acc:.2f}%")
        print(f"Assembled model:  {assembled_acc:.2f}% (NO fine-tuning)")
        print(f"Knowledge retained: {100 * assembled_acc / original_acc:.1f}%")
        
        # Interpret results
        print("\n" + "-"*70)
        print("INTERPRETATION:")
        if assembled_acc < 5:
            print("❌ Seeds preserve ZERO knowledge (pure random initialization)")
            print("   → Seed assembly is TASK DISTILLATION, not compression")
            print("   → Fine-tuning is essential, seed is irrelevant")
        elif assembled_acc < 50:
            print("⚠️  Seeds preserve MINIMAL knowledge (~noise level)")
            print("   → Still requires heavy fine-tuning to be useful")
        elif assembled_acc >= 50 and assembled_acc < 80:
            print("⚡ Seeds preserve PARTIAL knowledge (lossy compression)")
            print("   → Has value as rough initialization")
            print("   → Light fine-tuning can recover full performance")
        else:
            print("✅ Seeds preserve STRONG knowledge (effective compression)")
            print("   → Original value proposition is valid")
            print("   → Fine-tuning just polishes, seed does heavy lifting")
        print("-"*70)
        
        return {
            "model": "BERT-tiny",
            "params": "4.4M",
            "task": "IMDB sentiment",
            "original_acc": original_acc,
            "assembled_acc": assembled_acc,
            "retention": 100 * assembled_acc / original_acc,
            "seed_size_kb": seed.size_bytes() / 1024,
            "compression": (original_model.num_parameters() * 4) / seed.size_bytes()
        }
    
    def test_gpt2_zero_shot(self):
        """
        Test GPT-2 (124M params) on text generation
        WITHOUT any fine-tuning
        """
        print("\n" + "="*70)
        print("ZERO-SHOT TEST 2: GPT-2 (124M params)")
        print("="*70)
        
        model_name = "gpt2"
        
        # Load original model
        print("\n[1/4] Loading original GPT-2 model...")
        original_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # Prepare data
        print("[2/4] Loading evaluation dataset (WikiText-2)...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        def tokenize(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=128
            )
        
        tokenized = dataset.map(tokenize, batched=True)
        tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 10)
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
        dataloader = DataLoader(tokenized, batch_size=8)
        
        # Take subset for speed
        subset = []
        for i, batch in enumerate(dataloader):
            if i >= 50:  # 400 samples
                break
            subset.append(batch)
        
        # Evaluate original
        print("[3/4] Evaluating original GPT-2 perplexity (baseline)...")
        original_ppl = self.evaluate_model(original_model, subset, task_type="generation")
        print(f"✓ Original GPT-2 perplexity: {original_ppl:.2f}")
        
        # Extract seed and assemble
        print("[4/4] Extracting seed and assembling WITHOUT fine-tuning...")
        from seed_pytorch_integration import TorchSeedExtractor
        from transformers import GPT2LMHeadModel
        
        extractor = TorchSeedExtractor(n_components=10)
        seed = extractor.extract(original_model)
        
        print(f"Seed size: {seed.size_bytes():,} bytes ({seed.size_bytes()/1024:.1f} KB)")
        
        # Assemble - NO FINE-TUNING
        config = AutoConfig.from_pretrained(model_name)
        assembled_model = GPT2LMHeadModel(config).to(self.device)
        
        # Initialize from seed
        seed_idx = 0
        for name, param in assembled_model.named_parameters():
            if param.requires_grad and seed_idx < len(seed.layer_names):
                mean = seed.layer_means[seed_idx][0]
                std = seed.layer_stds[seed_idx]
                with torch.no_grad():
                    param.data.normal_(mean=mean, std=std)
                seed_idx += 1
        
        # Evaluate assembled - ZERO FINE-TUNING
        print("\n[CRITICAL] Evaluating assembled model with ZERO fine-tuning...")
        assembled_ppl = self.evaluate_model(assembled_model, subset, task_type="generation")
        
        print("\n" + "="*70)
        print("GPT-2 ZERO-SHOT RESULTS")
        print("="*70)
        print(f"Original model:   {original_ppl:.2f} perplexity")
        print(f"Assembled model:  {assembled_ppl:.2f} perplexity (NO fine-tuning)")
        print(f"Quality ratio:    {assembled_ppl / original_ppl:.1f}x worse")
        
        # Interpret (lower perplexity is better)
        print("\n" + "-"*70)
        print("INTERPRETATION:")
        if assembled_ppl > original_ppl * 100:
            print("❌ Assembled model is RANDOM (100x+ worse perplexity)")
            print("   → Seed preserves NO language modeling knowledge")
        elif assembled_ppl > original_ppl * 10:
            print("⚠️  Assembled model is POOR (10-100x worse)")
            print("   → Seed preserves minimal structure")
        elif assembled_ppl > original_ppl * 2:
            print("⚡ Assembled model is DEGRADED (2-10x worse)")
            print("   → Seed preserves partial knowledge")
        else:
            print("✅ Assembled model is COMPARABLE (<2x worse)")
            print("   → Seed effectively preserves language knowledge")
        print("-"*70)
        
        return {
            "model": "GPT-2",
            "params": "124M",
            "task": "WikiText-2 generation",
            "original_ppl": original_ppl,
            "assembled_ppl": assembled_ppl,
            "quality_ratio": assembled_ppl / original_ppl,
            "seed_size_kb": seed.size_bytes() / 1024,
            "compression": (original_model.num_parameters() * 4) / seed.size_bytes()
        }


def run_zero_shot_validation():
    """
    Run the critical zero-shot validation tests
    
    This determines the entire future direction of seed assembly research.
    """
    print("="*70)
    print("CRITICAL ZERO-SHOT VALIDATION (Phase 2.1)")
    print("="*70)
    print("\nQuestion: Do seeds preserve ANY knowledge without fine-tuning?")
    print("\nTests:")
    print("  1. BERT-tiny (4.4M) on IMDB classification")
    print("  2. GPT-2 (124M) on WikiText-2 generation")
    print("\nMethod: Extract seed → Assemble → Evaluate (NO fine-tuning)")
    print("="*70)
    
    validator = ZeroShotValidator()
    
    # Test 1: BERT classification
    t0 = time.time()
    bert_results = validator.test_bert_zero_shot()
    bert_time = time.time() - t0
    
    # Test 2: GPT-2 generation
    t0 = time.time()
    gpt2_results = validator.test_gpt2_zero_shot()
    gpt2_time = time.time() - t0
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    print("\n📊 SUMMARY:")
    print(f"  BERT-tiny: {bert_results['retention']:.1f}% knowledge retained")
    print(f"  GPT-2:     {gpt2_results['quality_ratio']:.1f}x perplexity degradation")
    
    # Decision logic
    bert_preserves = bert_results['assembled_acc'] > 50
    gpt2_preserves = gpt2_results['quality_ratio'] < 5
    
    print("\n🎯 RESEARCH DIRECTION:")
    if not bert_preserves and not gpt2_preserves:
        print("❌ SEEDS PRESERVE NO KNOWLEDGE")
        print("   → Seed assembly is TASK DISTILLATION (fine-tuning from scratch)")
        print("   → Value: Replace model storage with (architecture + task recipe)")
        print("   → Applications: Federated learning, privacy-preserving ML")
        print("   → NOT suitable for: Zero-shot deployment, transfer learning")
        print("\n   Next steps: Phase 2.2 (minimal fine-tuning budget)")
    elif bert_preserves or gpt2_preserves:
        print("⚡ SEEDS PRESERVE PARTIAL KNOWLEDGE")
        print("   → Seed assembly is LOSSY COMPRESSION")
        print("   → Value: Reduced storage + light fine-tuning")
        print("   → Applications: Edge deployment with training capability")
        print("   → Requires: Small labeled dataset at deployment time")
        print("\n   Next steps: Optimize fine-tuning budget + transfer learning")
    
    if bert_preserves and gpt2_preserves:
        print("✅ SEEDS PRESERVE STRONG KNOWLEDGE")
        print("   → Seed assembly is EFFECTIVE COMPRESSION")
        print("   → Value: Original value proposition validated")
        print("   → Applications: All originally planned (edge, distribution, etc.)")
        print("   → Can use: Zero-shot or with minimal fine-tuning")
        print("\n   Next steps: Phase 3 applications (edge deployment, etc.)")
    
    print("\n⏱️  TEST DURATION:")
    print(f"  BERT: {bert_time:.1f}s")
    print(f"  GPT-2: {gpt2_time:.1f}s")
    print(f"  Total: {bert_time + gpt2_time:.1f}s")
    
    print("\n📁 RESULTS SAVED TO: research/zero_shot_results.json")
    
    # Save results
    import json
    results = {
        "bert_tiny": {
            "model": bert_results["model"],
            "params": bert_results["params"],
            "task": bert_results["task"],
            "original_acc": float(bert_results["original_acc"]),
            "assembled_acc": float(bert_results["assembled_acc"]),
            "retention": float(bert_results["retention"]),
            "seed_size_kb": float(bert_results["seed_size_kb"]),
            "compression": float(bert_results["compression"])
        },
        "gpt2": {
            "model": gpt2_results["model"],
            "params": gpt2_results["params"],
            "task": gpt2_results["task"],
            "original_ppl": float(gpt2_results["original_ppl"]),
            "assembled_ppl": float(gpt2_results["assembled_ppl"]),
            "quality_ratio": float(gpt2_results["quality_ratio"]),
            "seed_size_kb": float(gpt2_results["seed_size_kb"]),
            "compression": float(gpt2_results["compression"])
        },
        "verdict": {
            "bert_preserves_knowledge": bool(bert_preserves),
            "gpt2_preserves_knowledge": bool(gpt2_preserves),
            "research_direction": "task_distillation" if not (bert_preserves or gpt2_preserves) else "compression"
        }
    }
    
    with open("research/zero_shot_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = run_zero_shot_validation()
