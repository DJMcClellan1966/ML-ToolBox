"""
Seed Assembly - Ablation Studies (Phase 1.3)
============================================

Critical questions:
1. Which seed components matter most?
   - PCA structure vs statistical moments vs sparsity patterns
2. What's the minimal viable seed?
3. Does random init + fine-tuning work as well as seed init?
4. Can we optimize compression further?

Tests:
- Component removal experiments
- Minimal seed variants
- Assembly algorithm comparisons
- Fine-tuning hyperparameter sweeps
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import json


@dataclass
class MinimalSeed:
    """Minimal seed with only essential components"""
    model_name: str
    # Global statistics only - no per-layer structure
    global_mean: float
    global_std: float
    # Tiny PCA of concatenated weights
    pca_components: np.ndarray  # (n_components, flattened_dim)
    
    def size_bytes(self) -> int:
        """Calculate seed size in bytes"""
        return (
            8 +  # global_mean (float64)
            8 +  # global_std (float64)
            self.pca_components.nbytes
        )


@dataclass
class NoStructureSeed:
    """Seed without PCA structure - only statistics"""
    model_name: str
    # Per-layer statistics but no structure
    layer_means: Dict[str, float]
    layer_stds: Dict[str, float]
    
    def size_bytes(self) -> int:
        return len(json.dumps({
            'means': self.layer_means,
            'stds': self.layer_stds
        }).encode())


@dataclass
class NoPCASeed:
    """Seed without PCA - only moments and patterns"""
    model_name: str
    layer_means: Dict[str, float]
    layer_stds: Dict[str, float]
    sparsity_patterns: Dict[str, float]  # % of near-zero values
    
    def size_bytes(self) -> int:
        return len(json.dumps({
            'means': self.layer_means,
            'stds': self.layer_stds,
            'sparsity': self.sparsity_patterns
        }).encode())


class AblationExperiments:
    """Run systematic ablation studies"""
    
    def __init__(self, model_name: str = "prajjwal1/bert-tiny", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load evaluation data
        print(f"Loading dataset for {model_name}...")
        self.eval_data = self._prepare_eval_data()
        
    def _prepare_eval_data(self):
        """Load and prepare evaluation dataset"""
        dataset = load_dataset("imdb", split="test[:1000]")  # 1000 samples
        
        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=128
            )
        
        tokenized = dataset.map(tokenize, batched=True)
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        return DataLoader(tokenized, batch_size=32)
    
    def evaluate_model(self, model: nn.Module) -> float:
        """Evaluate model accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.eval_data:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return 100.0 * correct / total
    
    def fine_tune(self, model: nn.Module, epochs: int = 3, lr: float = 2e-5) -> nn.Module:
        """Fine-tune model"""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(self.eval_data):
                if batch_idx >= 10:  # Quick fine-tune on 10 batches
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
        
        return model
    
    # ============================================
    # EXPERIMENT 1: Minimal Seed (Global Stats Only)
    # ============================================
    
    def extract_minimal_seed(self, n_components: int = 5) -> MinimalSeed:
        """Extract minimal seed with only global statistics"""
        print("\n[1/3] Loading original model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        ).to(self.device)
        
        # Get all weights
        all_weights = []
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                all_weights.append(param.data.cpu().numpy().flatten())
        
        weights_concat = np.concatenate(all_weights)
        
        print(f"[2/3] Computing global statistics...")
        global_mean = float(np.mean(weights_concat))
        global_std = float(np.std(weights_concat))
        
        print(f"[3/3] Computing tiny PCA ({n_components} components)...")
        # Create a few samples by adding noise (for PCA)
        n_samples = max(10, n_components * 2)
        noise_samples = []
        for _ in range(n_samples):
            noise = np.random.normal(0, global_std * 0.01, weights_concat.shape)
            noise_samples.append(weights_concat + noise)
        
        samples_matrix = np.vstack(noise_samples)  # (n_samples, n_params)
        
        pca = PCA(n_components=n_components)
        pca.fit(samples_matrix)
        
        seed = MinimalSeed(
            model_name=self.model_name,
            global_mean=global_mean,
            global_std=global_std,
            pca_components=pca.components_
        )
        
        print(f"\n✓ Minimal seed size: {seed.size_bytes():,} bytes")
        return seed
    
    def assemble_from_minimal_seed(self, seed: MinimalSeed) -> nn.Module:
        """Assemble model from minimal seed"""
        print("\n[1/2] Initializing model...")
        config = AutoConfig.from_pretrained(seed.model_name)
        model = AutoModelForSequenceClassification.from_config(config).to(self.device)
        
        # Initialize with global statistics
        print(f"[2/2] Applying minimal seed (mean={seed.global_mean:.6f}, std={seed.global_std:.6f})...")
        for param in model.parameters():
            if param.requires_grad:
                # Initialize with global distribution
                param.data.normal_(mean=seed.global_mean, std=seed.global_std)
        
        return model
    
    # ============================================
    # EXPERIMENT 2: Random Init Baseline
    # ============================================
    
    def random_init_baseline(self) -> nn.Module:
        """Pure random initialization (PyTorch default)"""
        print("\n[Random Init] Creating randomly initialized model...")
        config = AutoConfig.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_config(config).to(self.device)
        return model
    
    # ============================================
    # EXPERIMENT 3: No Structure Seed
    # ============================================
    
    def extract_no_structure_seed(self) -> NoStructureSeed:
        """Extract seed without PCA structure"""
        print("\n[No Structure] Loading model and extracting layer statistics...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        ).to(self.device)
        
        layer_means = {}
        layer_stds = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                weights = param.data.cpu().numpy()
                layer_means[name] = float(np.mean(weights))
                layer_stds[name] = float(np.std(weights))
        
        seed = NoStructureSeed(
            model_name=self.model_name,
            layer_means=layer_means,
            layer_stds=layer_stds
        )
        
        print(f"✓ No-structure seed size: {seed.size_bytes():,} bytes")
        return seed
    
    def assemble_from_no_structure_seed(self, seed: NoStructureSeed) -> nn.Module:
        """Assemble from statistics-only seed"""
        print("\n[Assembling] Per-layer statistics without structure...")
        config = AutoConfig.from_pretrained(seed.model_name)
        model = AutoModelForSequenceClassification.from_config(config).to(self.device)
        
        for name, param in model.named_parameters():
            if name in seed.layer_means and param.requires_grad:
                mean = seed.layer_means[name]
                std = seed.layer_stds[name]
                param.data.normal_(mean=mean, std=std)
        
        return model
    
    # ============================================
    # EXPERIMENT 4: Full Seed (from Phase 1.1)
    # ============================================
    
    def load_full_seed_baseline(self):
        """Load results from Phase 1.1 for comparison"""
        # Import from seed_pytorch_integration
        from seed_pytorch_integration import TorchSeedExtractor
        from transformers import AutoModelForSequenceClassification, AutoConfig, BertForSequenceClassification
        
        print("\n[Full Seed] Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        ).to(self.device)
        
        print(f"[Full Seed] Extracting with PCA + moments + sparsity...")
        extractor = TorchSeedExtractor(n_components=10)
        seed = extractor.extract(model)
        
        print(f"[Full Seed] Assembling from seed...")
        # Create model from config using concrete class
        config = AutoConfig.from_pretrained(self.model_name)
        config.num_labels = 2
        assembled_model = BertForSequenceClassification(config).to(self.device)
        
        # Initialize from seed
        print("[Full Seed] Initializing from seed statistics...")
        seed_idx = 0
        for name, param in assembled_model.named_parameters():
            if param.requires_grad and seed_idx < len(seed.layer_names):
                mean = seed.layer_means[seed_idx][0]
                std = seed.layer_stds[seed_idx]
                with torch.no_grad():
                    param.data.normal_(mean=mean, std=std)
                seed_idx += 1
        
        return seed, assembled_model


def run_ablation_study():
    """
    Run complete ablation study comparing:
    1. Full seed (Phase 1.1 baseline)
    2. Minimal seed (global stats only)
    3. No structure seed (per-layer stats, no PCA)
    4. Random init (PyTorch default)
    
    All variants get same fine-tuning budget.
    """
    print("=" * 60)
    print("SEED ASSEMBLY ABLATION STUDY (Phase 1.3)")
    print("=" * 60)
    
    experiments = AblationExperiments(model_name="prajjwal1/bert-tiny")
    
    # Get original accuracy as ground truth
    print("\n" + "=" * 60)
    print("BASELINE: Original Pre-trained Model")
    print("=" * 60)
    original_model = AutoModelForSequenceClassification.from_pretrained(
        "prajjwal1/bert-tiny",
        num_labels=2
    ).to(experiments.device)
    original_acc = experiments.evaluate_model(original_model)
    print(f"\n✓ Original accuracy: {original_acc:.2f}%")
    
    results = {
        "original": {
            "accuracy": original_acc,
            "seed_size_bytes": original_model.num_parameters() * 4,  # float32
            "compression": 1.0
        }
    }
    
    # ============================================
    # EXPERIMENT 1: Random Init + Fine-Tuning
    # ============================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Random Init + Fine-Tuning")
    print("=" * 60)
    
    t0 = time.time()
    random_model = experiments.random_init_baseline()
    random_acc_before = experiments.evaluate_model(random_model)
    print(f"Before fine-tuning: {random_acc_before:.2f}%")
    
    print("\nFine-tuning (3 epochs, lr=2e-5)...")
    random_model = experiments.fine_tune(random_model)
    random_acc_after = experiments.evaluate_model(random_model)
    random_time = time.time() - t0
    
    print(f"\n✓ After fine-tuning: {random_acc_after:.2f}%")
    print(f"  Retention: {100 * random_acc_after / original_acc:.1f}%")
    print(f"  Time: {random_time:.1f}s")
    
    results["random_init"] = {
        "accuracy_before": random_acc_before,
        "accuracy_after": random_acc_after,
        "retention": 100 * random_acc_after / original_acc,
        "seed_size_bytes": 0,  # No seed
        "compression": float('inf'),
        "time": random_time
    }
    
    # ============================================
    # EXPERIMENT 2: Minimal Seed (Global Stats)
    # ============================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Minimal Seed (Global Mean + Std)")
    print("=" * 60)
    
    t0 = time.time()
    minimal_seed = experiments.extract_minimal_seed(n_components=5)
    minimal_model = experiments.assemble_from_minimal_seed(minimal_seed)
    minimal_acc_before = experiments.evaluate_model(minimal_model)
    print(f"Before fine-tuning: {minimal_acc_before:.2f}%")
    
    print("\nFine-tuning (3 epochs, lr=2e-5)...")
    minimal_model = experiments.fine_tune(minimal_model)
    minimal_acc_after = experiments.evaluate_model(minimal_model)
    minimal_time = time.time() - t0
    
    original_size = original_model.num_parameters() * 4
    minimal_compression = original_size / minimal_seed.size_bytes()
    
    print(f"\n✓ After fine-tuning: {minimal_acc_after:.2f}%")
    print(f"  Retention: {100 * minimal_acc_after / original_acc:.1f}%")
    print(f"  Seed size: {minimal_seed.size_bytes():,} bytes")
    print(f"  Compression: {minimal_compression:.1f}x")
    print(f"  Time: {minimal_time:.1f}s")
    
    results["minimal_seed"] = {
        "accuracy_before": minimal_acc_before,
        "accuracy_after": minimal_acc_after,
        "retention": 100 * minimal_acc_after / original_acc,
        "seed_size_bytes": minimal_seed.size_bytes(),
        "compression": minimal_compression,
        "time": minimal_time
    }
    
    # ============================================
    # EXPERIMENT 3: No Structure Seed (Layer Stats)
    # ============================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: No Structure Seed (Per-Layer Stats, No PCA)")
    print("=" * 60)
    
    t0 = time.time()
    no_struct_seed = experiments.extract_no_structure_seed()
    no_struct_model = experiments.assemble_from_no_structure_seed(no_struct_seed)
    no_struct_acc_before = experiments.evaluate_model(no_struct_model)
    print(f"Before fine-tuning: {no_struct_acc_before:.2f}%")
    
    print("\nFine-tuning (3 epochs, lr=2e-5)...")
    no_struct_model = experiments.fine_tune(no_struct_model)
    no_struct_acc_after = experiments.evaluate_model(no_struct_model)
    no_struct_time = time.time() - t0
    
    no_struct_compression = original_size / no_struct_seed.size_bytes()
    
    print(f"\n✓ After fine-tuning: {no_struct_acc_after:.2f}%")
    print(f"  Retention: {100 * no_struct_acc_after / original_acc:.1f}%")
    print(f"  Seed size: {no_struct_seed.size_bytes():,} bytes")
    print(f"  Compression: {no_struct_compression:.1f}x")
    print(f"  Time: {no_struct_time:.1f}s")
    
    results["no_structure_seed"] = {
        "accuracy_before": no_struct_acc_before,
        "accuracy_after": no_struct_acc_after,
        "retention": 100 * no_struct_acc_after / original_acc,
        "seed_size_bytes": no_struct_seed.size_bytes(),
        "compression": no_struct_compression,
        "time": no_struct_time
    }
    
    # ============================================
    # EXPERIMENT 4: Full Seed (Phase 1.1 Baseline)
    # ============================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Full Seed (PCA + Moments + Sparsity)")
    print("=" * 60)
    
    t0 = time.time()
    full_seed, full_model = experiments.load_full_seed_baseline()
    full_acc_before = experiments.evaluate_model(full_model)
    print(f"Before fine-tuning: {full_acc_before:.2f}%")
    
    print("\nFine-tuning (3 epochs, lr=2e-5)...")
    full_model = experiments.fine_tune(full_model)
    full_acc_after = experiments.evaluate_model(full_model)
    full_time = time.time() - t0
    
    full_compression = original_size / full_seed.size_bytes()
    
    print(f"\n✓ After fine-tuning: {full_acc_after:.2f}%")
    print(f"  Retention: {100 * full_acc_after / original_acc:.1f}%")
    print(f"  Seed size: {full_seed.size_bytes():,} bytes")
    print(f"  Compression: {full_compression:.1f}x")
    print(f"  Time: {full_time:.1f}s")
    
    results["full_seed"] = {
        "accuracy_before": full_acc_before,
        "accuracy_after": full_acc_after,
        "retention": 100 * full_acc_after / original_acc,
        "seed_size_bytes": full_seed.size_bytes(),
        "compression": full_compression,
        "time": full_time
    }
    
    # ============================================
    # COMPARATIVE ANALYSIS
    # ============================================
    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)
    
    print("\n{:<25} {:>12} {:>12} {:>12} {:>10}".format(
        "Method", "Seed Size", "Compression", "Retention", "Time"
    ))
    print("-" * 75)
    
    for name, data in results.items():
        if name == "original":
            continue
        
        size_str = f"{data['seed_size_bytes']:,}B" if data['seed_size_bytes'] > 0 else "0B"
        comp_str = f"{data['compression']:.0f}x" if data['compression'] != float('inf') else "∞"
        
        print("{:<25} {:>12} {:>12} {:>11.1f}% {:>9.1f}s".format(
            name.replace('_', ' ').title(),
            size_str,
            comp_str,
            data['retention'],
            data['time']
        ))
    
    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    # Compare random vs minimal vs full
    random_ret = results["random_init"]["retention"]
    minimal_ret = results["minimal_seed"]["retention"]
    full_ret = results["full_seed"]["retention"]
    
    print(f"\n1. Does seed initialization help?")
    if minimal_ret > random_ret + 5:
        print(f"   ✓ YES - Minimal seed beats random init by {minimal_ret - random_ret:.1f}%")
    else:
        print(f"   ✗ NO - Minimal seed similar to random ({minimal_ret:.1f}% vs {random_ret:.1f}%)")
    
    print(f"\n2. Does per-layer structure matter?")
    no_struct_ret = results["no_structure_seed"]["retention"]
    if no_struct_ret > minimal_ret + 5:
        print(f"   ✓ YES - Per-layer stats beat global by {no_struct_ret - minimal_ret:.1f}%")
    else:
        print(f"   ✗ NO - Per-layer similar to global ({no_struct_ret:.1f}% vs {minimal_ret:.1f}%)")
    
    print(f"\n3. Does PCA structure matter?")
    if full_ret > no_struct_ret + 5:
        print(f"   ✓ YES - PCA structure beats stats-only by {full_ret - no_struct_ret:.1f}%")
    else:
        print(f"   ✗ NO - PCA similar to stats-only ({full_ret:.1f}% vs {no_struct_ret:.1f}%)")
    
    print(f"\n4. Compression-retention tradeoff:")
    print(f"   - Full seed: {results['full_seed']['compression']:.0f}x @ {full_ret:.1f}%")
    print(f"   - Minimal seed: {results['minimal_seed']['compression']:.0f}x @ {minimal_ret:.1f}%")
    
    # Save results
    with open('research/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to research/ablation_results.json")
    
    return results


if __name__ == "__main__":
    results = run_ablation_study()
