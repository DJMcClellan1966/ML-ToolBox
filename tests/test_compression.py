"""
Test Dimensionality Reduction / Compression in Advanced Data Preprocessor
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_preprocessor import AdvancedDataPreprocessor


def test_compression():
    """Test compression capabilities"""
    print("="*80)
    print("DIMENSIONALITY REDUCTION / COMPRESSION TEST")
    print("="*80)
    
    # Create test data
    test_data = [
        "Python is great for data science and machine learning",
        "Machine learning uses algorithms to find patterns in data",
        "I need help with programming errors in my code",
        "Business revenue increased by 20% this quarter",
        "Customer support is available 24/7 for assistance",
        "Learn Python programming through online tutorials",
        "JavaScript is used for web development and frontend",
        "Sales team achieved record profits this year",
        "Fix errors in your code with debugging tools",
        "Education courses teach programming fundamentals"
    ]
    
    # Test with compression enabled
    print("\n[TEST 1] Preprocessing with Compression (PCA)")
    print("-" * 80)
    preprocessor = AdvancedDataPreprocessor(
        enable_compression=True,
        compression_ratio=0.5,  # 50% compression
        compression_method='pca'
    )
    
    results = preprocessor.preprocess(test_data, verbose=True)
    
    if results['compressed_embeddings'] is not None:
        compressed = results['compressed_embeddings']
        info = results['compression_info']
        
        print("\n[Compression Results]")
        print(f"  Original dimensions: {info['original_dim']}")
        print(f"  Compressed dimensions: {info['compressed_dim']}")
        print(f"  Compression ratio: {info['compression_ratio_achieved']:.1%}")
        print(f"  Memory reduction: {info['memory_reduction']:.1%}")
        if 'variance_retained' in info:
            print(f"  Variance retained: {info['variance_retained']:.2%}")
        
        # Test similarity preservation
        print("\n[Similarity Preservation Test]")
        original_embeddings = np.array([preprocessor.quantum_kernel.embed(item) for item in results['deduplicated']])
        
        # Calculate original similarities
        original_sim = np.dot(original_embeddings[0], original_embeddings[1])
        
        # Calculate compressed similarities
        compressed_sim = np.dot(compressed[0], compressed[1])
        
        print(f"  Original similarity (item 0 vs 1): {original_sim:.4f}")
        print(f"  Compressed similarity (item 0 vs 1): {compressed_sim:.4f}")
        print(f"  Similarity preservation: {abs(compressed_sim - original_sim) / abs(original_sim) * 100:.1f}% difference")
        
        # Test decompression
        print("\n[Decompression Test]")
        decompressed = preprocessor.decompress_embeddings(compressed)
        if decompressed is not None:
            decompressed_sim = np.dot(decompressed[0], decompressed[1])
            print(f"  Decompressed similarity: {decompressed_sim:.4f}")
            print(f"  Reconstruction error: {np.mean(np.abs(decompressed - original_embeddings)):.4f}")
        else:
            print("  Decompression not available for this method")
    
    # Test with different compression ratios
    print("\n" + "="*80)
    print("[TEST 2] Different Compression Ratios")
    print("="*80)
    
    compression_ratios = [0.3, 0.5, 0.7, 0.9]
    
    for ratio in compression_ratios:
        print(f"\n[Compression Ratio: {ratio:.0%}]")
        print("-" * 80)
        preprocessor = AdvancedDataPreprocessor(
            enable_compression=True,
            compression_ratio=ratio,
            compression_method='pca'
        )
        
        results = preprocessor.preprocess(test_data, verbose=False)
        
        if results['compressed_embeddings'] is not None:
            info = results['compression_info']
            print(f"  Original: {info['original_dim']} dims")
            print(f"  Compressed: {info['compressed_dim']} dims")
            print(f"  Memory saved: {info['memory_reduction']:.1%}")
            if 'variance_retained' in info:
                print(f"  Variance retained: {info['variance_retained']:.2%}")
    
    # Test without compression
    print("\n" + "="*80)
    print("[TEST 3] Without Compression (Baseline)")
    print("="*80)
    
    preprocessor_no_comp = AdvancedDataPreprocessor(
        enable_compression=False
    )
    
    results_no_comp = preprocessor_no_comp.preprocess(test_data, verbose=True)
    
    print(f"\n  Compression applied: {results_no_comp['stats']['compression_applied']}")
    
    # Compare memory usage
    print("\n" + "="*80)
    print("[MEMORY COMPARISON]")
    print("="*80)
    
    if results['compressed_embeddings'] is not None:
        original_size = results['compression_info']['original_dim'] * len(results['deduplicated']) * 4  # 4 bytes per float32
        compressed_size = results['compression_info']['compressed_dim'] * len(results['deduplicated']) * 4
        
        print(f"  Original size: {original_size:,} bytes ({original_size/1024:.2f} KB)")
        print(f"  Compressed size: {compressed_size:,} bytes ({compressed_size/1024:.2f} KB)")
        print(f"  Space saved: {original_size - compressed_size:,} bytes ({(original_size - compressed_size)/1024:.2f} KB)")
        print(f"  Compression ratio: {compressed_size/original_size:.1%}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        test_compression()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: sklearn is required for PCA/SVD compression.")
        print("      Install with: pip install scikit-learn")
