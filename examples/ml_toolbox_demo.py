"""
Demonstration: Machine Learning Toolbox
Shows how to use the three-compartment ML toolbox
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_toolbox import MLToolbox
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def demonstrate_ml_toolbox():
    """Demonstrate the three-compartment ML toolbox"""
    
    print("="*80)
    print("MACHINE LEARNING TOOLBOX DEMONSTRATION")
    print("="*80)
    
    # Initialize toolbox
    print("\n[Initializing ML Toolbox]")
    toolbox = MLToolbox()
    print(f"  Toolbox initialized: {toolbox}")
    
    # Show compartments
    print("\n[Toolbox Structure]")
    print(f"  Compartment 1 (Data): {toolbox.data.get_info()['component_count']} components")
    print(f"  Compartment 2 (Infrastructure): {toolbox.infrastructure.get_info()['component_count']} components")
    print(f"  Compartment 3 (Algorithms): {toolbox.algorithms.get_info()['component_count']} components")
    
    # Sample data
    print("\n[Sample Data]")
    texts = [
        "Python programming is great for data science",
        "Machine learning uses neural networks",
        "Revenue increased by twenty percent",
        "Customer satisfaction drives business growth",
        "I need help with technical issues",
        "Support team provides assistance",
        "Learn Python through online courses",
        "Educational content helps students learn"
    ]
    labels = [0, 0, 1, 1, 2, 2, 3, 3]  # 0=technical, 1=business, 2=support, 3=education
    
    print(f"  Texts: {len(texts)}")
    print(f"  Labels: {len(labels)} classes")
    
    # Compartment 1: Data Preprocessing
    print("\n" + "="*80)
    print("COMPARTMENT 1: DATA PREPROCESSING")
    print("="*80)
    
    print("\n[Using AdvancedDataPreprocessor]")
    print("  Location: Compartment 1 (Data)")
    print("  Purpose: Preprocess raw text data")
    
    results = toolbox.data.preprocess(
        texts,
        advanced=True,
        dedup_threshold=0.70,
        enable_compression=True,
        compression_ratio=0.5,
        verbose=False
    )
    
    print(f"\n  Preprocessing Results:")
    print(f"    Original samples: {len(texts)}")
    print(f"    After deduplication: {len(results['deduplicated'])}")
    print(f"    Duplicates removed: {len(results['duplicates'])}")
    
    if 'compressed_embeddings' in results and results['compressed_embeddings'] is not None:
        X = results['compressed_embeddings']
        print(f"    Compressed embeddings shape: {X.shape}")
        print(f"    Features created: {X.shape[1]} dimensions")
    else:
        # Fallback to original embeddings
        kernel = toolbox.infrastructure.get_kernel()
        X = np.array([kernel.embed(text) for text in results['deduplicated']])
        print(f"    Embeddings shape: {X.shape}")
    
    # Filter labels
    y = np.array(labels[:len(X)])
    
    print(f"\n  Why AdvancedDataPreprocessor is in Compartment 1:")
    print(f"    - Preprocesses raw data")
    print(f"    - Transforms text to features")
    print(f"    - Cleans and validates data")
    print(f"    - Prepares data for ML models")
    
    # Compartment 2: Infrastructure (if needed)
    print("\n" + "="*80)
    print("COMPARTMENT 2: INFRASTRUCTURE")
    print("="*80)
    
    print("\n[Available Infrastructure Components]")
    print("  - Quantum Kernel: Semantic understanding")
    print("  - AI Components: Understanding, knowledge graph, search")
    print("  - LLM: Text generation")
    print("  - Adaptive Neuron: Neural-like learning")
    
    # Show kernel usage
    kernel = toolbox.infrastructure.get_kernel()
    print(f"\n[Using Quantum Kernel]")
    print(f"  Location: Compartment 2 (Infrastructure)")
    print(f"  Purpose: Semantic operations")
    
    similarity = kernel.similarity("Python programming", "machine learning")
    print(f"  Similarity('Python programming', 'machine learning'): {similarity:.4f}")
    
    # Compartment 3: Algorithms
    print("\n" + "="*80)
    print("COMPARTMENT 3: ALGORITHMS")
    print("="*80)
    
    # Split data
    if len(X) >= 4:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        print(f"\n[Training Model]")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        print(f"\n[Evaluating Model]")
        print(f"  Location: Compartment 3 (Algorithms)")
        print(f"  Purpose: Model evaluation and tuning")
        
        try:
            evaluator = toolbox.algorithms.get_evaluator()
            eval_results = evaluator.evaluate_model(
                model=model,
                X=X_train,
                y=y_train,
                cv=min(3, len(X_train) // 2)  # Adjust CV folds
            )
            
            print(f"\n  Evaluation Results:")
            print(f"    Accuracy: {eval_results.get('accuracy', 0):.4f}")
            print(f"    Precision: {eval_results.get('precision', 0):.4f}")
            print(f"    Recall: {eval_results.get('recall', 0):.4f}")
            print(f"    F1 Score: {eval_results.get('f1', 0):.4f}")
        except Exception as e:
            print(f"  Evaluation error: {e}")
        
        # Test accuracy
        test_accuracy = model.score(X_test, y_test)
        print(f"\n  Test Accuracy: {test_accuracy:.4f}")
    else:
        print(f"\n  Not enough samples for train/test split (need at least 4, have {len(X)})")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\n[Compartment Organization]")
    print("\n  Compartment 1: DATA")
    print("    - AdvancedDataPreprocessor (preprocessing)")
    print("    - ConventionalPreprocessor (basic preprocessing)")
    print("    - Purpose: Prepare data for ML")
    
    print("\n  Compartment 2: INFRASTRUCTURE")
    print("    - Quantum Kernel (semantic operations)")
    print("    - AI Components (understanding, knowledge graph)")
    print("    - LLM (text generation)")
    print("    - Purpose: Provide AI infrastructure")
    
    print("\n  Compartment 3: ALGORITHMS")
    print("    - ML Evaluation (model evaluation)")
    print("    - Hyperparameter Tuning (optimization)")
    print("    - Ensemble Learning (combining models)")
    print("    - Purpose: Train and evaluate models")
    
    print("\n[Key Point]")
    print("  AdvancedDataPreprocessor is correctly placed in Compartment 1 (Data)")
    print("  because it preprocesses and transforms data before ML models.")
    
    print("\n[Workflow]")
    print("  1. Compartment 1: Preprocess data (AdvancedDataPreprocessor)")
    print("  2. Compartment 2: Use infrastructure as needed (Quantum Kernel)")
    print("  3. Compartment 3: Train and evaluate models (ML Evaluation)")
    
    print("="*80 + "\n")
    
    return toolbox


if __name__ == "__main__":
    try:
        toolbox = demonstrate_ml_toolbox()
        print("[+] Demonstration complete!")
        print("\nYou can now use the ML Toolbox with:")
        print("  - toolbox.data.preprocess() for data preprocessing")
        print("  - toolbox.infrastructure.get_kernel() for semantic operations")
        print("  - toolbox.algorithms.get_evaluator() for model evaluation")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
