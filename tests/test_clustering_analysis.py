"""
Test AdvancedDataPreprocessor for Clustering Analysis
Unsupervised learning with unlabeled data
"""
import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_preprocessor import AdvancedDataPreprocessor, ConventionalPreprocessor

# Try to import sklearn for clustering
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class ClusteringAnalysisTest:
    """Test clustering analysis with AdvancedDataPreprocessor on unlabeled data"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = []
    
    def generate_unlabeled_data(self, n_samples: int = 200) -> list:
        """Generate unlabeled text data for clustering"""
        # Create diverse text data across different topics
        data = {
            'technical': [
                "Python programming language is great for data science",
                "Machine learning algorithms use neural networks",
                "Software development requires coding skills",
                "JavaScript is used for web development",
                "Database systems store structured data",
                "API endpoints provide data access",
                "Code optimization improves performance",
                "Version control systems track changes",
                "Debugging tools help find errors",
                "Cloud computing enables scalability",
                "Python is excellent for data analysis",
                "ML algorithms employ neural networks",
                "Programming requires coding abilities",
                "JS is utilized for website development",
                "Databases store structured information"
            ],
            'business': [
                "Revenue increased by twenty percent this quarter",
                "Customer satisfaction drives business growth",
                "Market analysis shows positive trends",
                "Sales team achieved record profits",
                "Business strategy focuses on expansion",
                "Profit margins improved significantly",
                "Customer retention is important",
                "Market share increased this year",
                "Business development creates opportunities",
                "Revenue growth exceeded expectations",
                "Quarterly earnings report shows strong performance",
                "Marketing campaign increased brand awareness",
                "Strategic partnerships expand market reach",
                "Customer acquisition cost decreased",
                "Return on investment improved",
                "Supply chain optimization reduced costs",
                "Product launch exceeded sales targets",
                "Customer lifetime value increased",
                "Market penetration strategy succeeded",
                "Competitive analysis revealed opportunities"
            ],
            'support': [
                "I need help with technical issues",
                "Customer support is available twenty four seven",
                "How do I fix errors in my code",
                "Troubleshooting guide helps resolve problems",
                "Support team provides assistance",
                "Error messages indicate problems",
                "Help documentation explains solutions",
                "Technical support resolves issues",
                "Customer service helps users",
                "Problem solving requires patience",
                "Ticket system tracks support requests",
                "Knowledge base contains solutions",
                "Live chat provides instant help",
                "Email support handles complex issues",
                "Phone support offers personal assistance",
                "FAQ section answers common questions",
                "Video tutorials demonstrate solutions",
                "Community forum shares experiences",
                "Bug reports help improve software",
                "Feature requests guide development"
            ],
            'education': [
                "Learn Python programming through online courses",
                "Educational tutorials teach coding skills",
                "Training programs cover programming fundamentals",
                "Study materials help students learn",
                "Teaching programming requires patience",
                "Learning resources are available online",
                "Educational content improves understanding",
                "Course materials explain concepts",
                "Training sessions teach new skills",
                "Study guides help with learning",
                "Online learning platforms offer flexibility",
                "Certification programs validate skills",
                "Workshops provide hands-on experience",
                "Textbooks explain theoretical concepts",
                "Practice exercises reinforce learning",
                "Peer learning enhances understanding",
                "Mentorship programs guide students",
                "Assessment tests measure progress",
                "Curriculum design ensures coverage",
                "Student feedback improves courses"
            ]
        }
        
        texts = []
        
        # Add all texts (no labels - unlabeled data)
        for category_texts in data.values():
            texts.extend(category_texts)
        
        # Add more diverse variations to reach desired size
        import random
        random.seed(self.random_state)
        
        # Create more diverse variations
        variation_templates = [
            lambda t: t + ".",
            lambda t: t.lower(),
            lambda t: t.replace("Python", "Python programming"),
            lambda t: t.replace("code", "source code"),
            lambda t: t.replace("business", "company"),
            lambda t: t.replace("help", "assistance"),
            lambda t: t.replace("learn", "study"),
            lambda t: t.replace("customer", "client"),
            lambda t: t.replace("support", "help"),
            lambda t: t.replace("programming", "coding")
        ]
        
        while len(texts) < n_samples:
            # Pick random base text
            base_text = random.choice([text for category_texts in data.values() for text in category_texts])
            
            # Apply random variation
            variation = random.choice(variation_templates)
            try:
                varied_text = variation(base_text)
                if varied_text not in texts:
                    texts.append(varied_text)
            except:
                pass
        
        # Shuffle for randomness
        random.shuffle(texts)
        return texts[:n_samples]
    
    def test_clustering_with_preprocessor(
        self,
        texts: list,
        preprocessor_name: str,
        preprocessor_results: dict,
        n_clusters: int = 4,
        use_embeddings: bool = True,
        verbose: bool = True
    ) -> dict:
        """Test clustering models with preprocessed data"""
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        start_time = time.time()
        
        # Use preprocessed data
        processed_texts = preprocessor_results['deduplicated']
        
        # Create features
        if use_embeddings and 'compressed_embeddings' in preprocessor_results and preprocessor_results['compressed_embeddings'] is not None:
            # Use compressed embeddings
            X = preprocessor_results['compressed_embeddings']
            feature_type = 'compressed_embeddings'
        elif use_embeddings:
            # Use original embeddings
            preprocessor = AdvancedDataPreprocessor() if preprocessor_name == 'AdvancedDataPreprocessor' else None
            if preprocessor and preprocessor.quantum_kernel:
                X = np.array([preprocessor.quantum_kernel.embed(text) for text in processed_texts])
                feature_type = 'quantum_embeddings'
            else:
                # Fallback to TF-IDF
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
                X = vectorizer.fit_transform(processed_texts).toarray()
                feature_type = 'tfidf'
        else:
            # Use TF-IDF
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            X = vectorizer.fit_transform(processed_texts).toarray()
            feature_type = 'tfidf'
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Adjust n_clusters if needed
        actual_n_clusters = min(n_clusters, len(processed_texts))
        
        # Test multiple clustering algorithms
        clustering_algorithms = {
            'KMeans': KMeans(n_clusters=actual_n_clusters, random_state=self.random_state, n_init=10),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=3),
            'Agglomerative': AgglomerativeClustering(n_clusters=actual_n_clusters)
        }
        
        results = {
            'preprocessor': preprocessor_name,
            'feature_type': feature_type,
            'n_samples': len(processed_texts),
            'n_features': X.shape[1],
            'n_clusters': actual_n_clusters,
            'algorithms': {},
            'processing_time': time.time() - start_time
        }
        
        # Evaluate each clustering algorithm
        for algo_name, algo in clustering_algorithms.items():
            try:
                # Fit clustering
                labels = algo.fit_predict(X_scaled)
                
                # Handle DBSCAN (may return -1 for noise)
                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                
                # Calculate metrics (only if we have valid clusters)
                if n_clusters_found > 1 and len(set(labels)) > 1:
                    try:
                        silhouette = silhouette_score(X_scaled, labels)
                    except:
                        silhouette = -1.0
                    
                    try:
                        davies_bouldin = davies_bouldin_score(X_scaled, labels)
                    except:
                        davies_bouldin = float('inf')
                    
                    try:
                        calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
                    except:
                        calinski_harabasz = 0.0
                else:
                    silhouette = -1.0
                    davies_bouldin = float('inf')
                    calinski_harabasz = 0.0
                
                # Count points per cluster
                unique_labels, counts = np.unique(labels, return_counts=True)
                cluster_sizes = dict(zip(unique_labels.tolist(), counts.tolist()))
                
                results['algorithms'][algo_name] = {
                    'n_clusters_found': int(n_clusters_found),
                    'silhouette_score': float(silhouette),
                    'davies_bouldin_score': float(davies_bouldin),
                    'calinski_harabasz_score': float(calinski_harabasz),
                    'cluster_sizes': cluster_sizes,
                    'noise_points': int(cluster_sizes.get(-1, 0)) if -1 in cluster_sizes else 0
                }
                
            except Exception as e:
                results['algorithms'][algo_name] = {
                    'error': str(e)
                }
        
        if verbose:
            self._print_results(results, preprocessor_name)
        
        return results
    
    def _print_results(self, results: dict, preprocessor_name: str):
        """Print clustering test results"""
        print("\n" + "="*80)
        print(f"CLUSTERING RESULTS: {preprocessor_name.upper()}")
        print("="*80)
        
        print(f"\nFeature Type: {results['feature_type']}")
        print(f"Preprocessed Samples: {results['n_samples']}")
        print(f"Features: {results['n_features']}")
        print(f"Target Clusters: {results['n_clusters']}")
        print(f"Processing Time: {results['processing_time']:.3f}s")
        
        print("\n[Clustering Performance]")
        print("-" * 80)
        for algo_name, metrics in results['algorithms'].items():
            if 'error' in metrics:
                print(f"\n{algo_name}:")
                print(f"  Error: {metrics['error']}")
            else:
                print(f"\n{algo_name}:")
                print(f"  Clusters Found: {metrics['n_clusters_found']}")
                print(f"  Silhouette Score: {metrics['silhouette_score']:.4f} (higher is better, max=1.0)")
                print(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f} (lower is better)")
                print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f} (higher is better)")
                if metrics['noise_points'] > 0:
                    print(f"  Noise Points: {metrics['noise_points']}")
                print(f"  Cluster Sizes: {metrics['cluster_sizes']}")
        
        print("="*80)
    
    def compare_preprocessors(self, n_samples: int = 200, n_clusters: int = 4, verbose: bool = True) -> dict:
        """Compare AdvancedDataPreprocessor vs ConventionalPreprocessor for clustering"""
        print("\n" + "="*80)
        print("CLUSTERING ANALYSIS COMPARISON TEST")
        print("="*80)
        
        # Generate unlabeled data
        print(f"\n[Generating Unlabeled Data]")
        texts = self.generate_unlabeled_data(n_samples)
        print(f"  Generated: {len(texts)} unlabeled samples")
        print(f"  Target Clusters: {n_clusters}")
        
        # Preprocess with AdvancedDataPreprocessor
        print(f"\n[AdvancedDataPreprocessor]")
        print("-" * 80)
        advanced_preprocessor = AdvancedDataPreprocessor(
            dedup_threshold=0.80,  # Lower threshold to keep more samples for clustering
            enable_compression=True,
            compression_ratio=0.5
        )
        advanced_results = advanced_preprocessor.preprocess(texts.copy(), verbose=verbose)
        
        # Test clustering with advanced preprocessor
        advanced_clustering_results = self.test_clustering_with_preprocessor(
            texts, "AdvancedDataPreprocessor",
            advanced_results, n_clusters=n_clusters,
            use_embeddings=True, verbose=verbose
        )
        
        # Preprocess with ConventionalPreprocessor
        print(f"\n[ConventionalPreprocessor]")
        print("-" * 80)
        conventional_preprocessor = ConventionalPreprocessor()
        conventional_results = conventional_preprocessor.preprocess(texts.copy(), verbose=verbose)
        
        # Test clustering with conventional preprocessor
        conventional_clustering_results = self.test_clustering_with_preprocessor(
            texts, "ConventionalPreprocessor",
            conventional_results, n_clusters=n_clusters,
            use_embeddings=False, verbose=verbose  # Use TF-IDF for conventional
        )
        
        # Compare results
        comparison = self._compare_results(advanced_clustering_results, conventional_clustering_results, verbose)
        
        return {
            'advanced': advanced_clustering_results,
            'conventional': conventional_clustering_results,
            'comparison': comparison
        }
    
    def _compare_results(self, advanced: dict, conventional: dict, verbose: bool = True) -> dict:
        """Compare clustering results between preprocessors"""
        comparison = {
            'samples': {
                'advanced': advanced['n_samples'],
                'conventional': conventional['n_samples'],
                'difference': advanced['n_samples'] - conventional['n_samples']
            },
            'features': {
                'advanced': advanced['n_features'],
                'conventional': conventional['n_features'],
                'difference': advanced['n_features'] - conventional['n_features']
            },
            'algorithms': {}
        }
        
        # Compare each algorithm
        for algo_name in advanced['algorithms'].keys():
            if algo_name not in conventional['algorithms']:
                continue
            
            adv_metrics = advanced['algorithms'][algo_name]
            conv_metrics = conventional['algorithms'][algo_name]
            
            if 'error' in adv_metrics or 'error' in conv_metrics:
                continue
            
            comparison['algorithms'][algo_name] = {
                'silhouette_score': {
                    'advanced': adv_metrics['silhouette_score'],
                    'conventional': conv_metrics['silhouette_score'],
                    'improvement': adv_metrics['silhouette_score'] - conv_metrics['silhouette_score']
                },
                'davies_bouldin_score': {
                    'advanced': adv_metrics['davies_bouldin_score'],
                    'conventional': conv_metrics['davies_bouldin_score'],
                    'improvement': conv_metrics['davies_bouldin_score'] - adv_metrics['davies_bouldin_score']  # Lower is better
                },
                'calinski_harabasz_score': {
                    'advanced': adv_metrics['calinski_harabasz_score'],
                    'conventional': conv_metrics['calinski_harabasz_score'],
                    'improvement': adv_metrics['calinski_harabasz_score'] - conv_metrics['calinski_harabasz_score']
                },
                'n_clusters_found': {
                    'advanced': adv_metrics['n_clusters_found'],
                    'conventional': conv_metrics['n_clusters_found'],
                    'difference': adv_metrics['n_clusters_found'] - conv_metrics['n_clusters_found']
                }
            }
        
        if verbose:
            self._print_comparison(comparison)
        
        return comparison
    
    def _print_comparison(self, comparison: dict):
        """Print comparison results"""
        print("\n" + "="*80)
        print("CLUSTERING COMPARISON SUMMARY")
        print("="*80)
        
        print(f"\n[Samples]")
        print(f"  Advanced: {comparison['samples']['advanced']}")
        print(f"  Conventional: {comparison['samples']['conventional']}")
        print(f"  Difference: {comparison['samples']['difference']:+d}")
        
        print(f"\n[Features]")
        print(f"  Advanced: {comparison['features']['advanced']}")
        print(f"  Conventional: {comparison['features']['conventional']}")
        print(f"  Difference: {comparison['features']['difference']:+d}")
        
        print(f"\n[Clustering Performance]")
        print("-" * 80)
        for algo_name, metrics in comparison['algorithms'].items():
            print(f"\n{algo_name}:")
            print(f"  Silhouette Score (higher is better):")
            print(f"    Advanced: {metrics['silhouette_score']['advanced']:.4f}")
            print(f"    Conventional: {metrics['silhouette_score']['conventional']:.4f}")
            print(f"    Improvement: {metrics['silhouette_score']['improvement']:+.4f}")
            
            print(f"  Davies-Bouldin Score (lower is better):")
            print(f"    Advanced: {metrics['davies_bouldin_score']['advanced']:.4f}")
            print(f"    Conventional: {metrics['davies_bouldin_score']['conventional']:.4f}")
            print(f"    Improvement: {metrics['davies_bouldin_score']['improvement']:+.4f}")
            
            print(f"  Calinski-Harabasz Score (higher is better):")
            print(f"    Advanced: {metrics['calinski_harabasz_score']['advanced']:.4f}")
            print(f"    Conventional: {metrics['calinski_harabasz_score']['conventional']:.4f}")
            print(f"    Improvement: {metrics['calinski_harabasz_score']['improvement']:+.4f}")
            
            print(f"  Clusters Found:")
            print(f"    Advanced: {metrics['n_clusters_found']['advanced']}")
            print(f"    Conventional: {metrics['n_clusters_found']['conventional']}")
            print(f"    Difference: {metrics['n_clusters_found']['difference']:+d}")
        
        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)
        
        findings = []
        
        # Check for improvements
        for algo_name, metrics in comparison['algorithms'].items():
            if metrics['silhouette_score']['improvement'] > 0.1:
                findings.append(f"{algo_name}: Advanced preprocessor improves silhouette score by {metrics['silhouette_score']['improvement']:.4f}")
            elif metrics['silhouette_score']['improvement'] < -0.1:
                findings.append(f"{algo_name}: Conventional preprocessor improves silhouette score by {abs(metrics['silhouette_score']['improvement']):.4f}")
            
            if metrics['davies_bouldin_score']['improvement'] > 0.1:
                findings.append(f"{algo_name}: Advanced preprocessor improves Davies-Bouldin score by {metrics['davies_bouldin_score']['improvement']:.4f}")
            elif metrics['davies_bouldin_score']['improvement'] < -0.1:
                findings.append(f"{algo_name}: Conventional preprocessor improves Davies-Bouldin score by {abs(metrics['davies_bouldin_score']['improvement']):.4f}")
        
        if comparison['samples']['difference'] < 0:
            findings.append(f"Advanced preprocessor removes {abs(comparison['samples']['difference'])} more duplicates")
        
        for finding in findings:
            print(f"  • {finding}")
        
        if not findings:
            print("  • Results are similar between both methods")
        
        print("="*80 + "\n")


def main():
    """Run clustering comparison test"""
    if not SKLEARN_AVAILABLE:
        print("Error: sklearn not available. Install with: pip install scikit-learn")
        return
    
    try:
        test = ClusteringAnalysisTest(random_state=42)
        results = test.compare_preprocessors(n_samples=200, n_clusters=4, verbose=True)
        
        print("\n" + "="*80)
        print("CLUSTERING TEST COMPLETE")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
