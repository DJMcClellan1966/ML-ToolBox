# Information Theory for Machine Learning

Concepts from information theory that are used in ML: entropy, mutual information, KL divergence, and information gain.

## Entropy

**Entropy** measures the uncertainty or randomness of a random variable. For a discrete distribution with probabilities p_i, Shannon entropy is defined as H = -Σ p_i log(p_i). Higher entropy means more uncertainty; lower entropy means the variable is more predictable. In ML, entropy is used in decision trees (splitting criteria), in clustering (e.g. soft assignments), and in probabilistic models to measure how peaked or spread a distribution is.

The toolbox implements entropy for probability arrays and uses it in data quality assessment and in feature pipelines (e.g. for discretized variables).

## Mutual Information

**Mutual information** I(X; Y) measures how much knowing one variable reduces uncertainty about the other. It is symmetric: I(X; Y) = I(Y; X). It is zero only if X and Y are independent. In ML, mutual information is used for feature selection (select features that share high mutual information with the target and low redundancy with each other), and for evaluating clustering (e.g. adjusted mutual information score). The toolbox uses it in data quality metrics, in feature pipelines for selection, and in the learning companion for structuring learning content.

## KL Divergence

**Kullback–Leibler (KL) divergence** measures how one probability distribution differs from another. It is asymmetric: KL(P || Q) is not the same as KL(Q || P). It is non-negative and zero only when the two distributions are equal. In ML, KL divergence is used in variational inference (to regularize the approximate posterior), in generative models (e.g. as part of the loss), and in reinforcement learning (policy updates). It is not a true distance (does not satisfy symmetry or triangle inequality) but is widely used as a divergence.

## Information Gain

**Information gain** is the reduction in entropy (or impurity) achieved by splitting data on a feature. In decision trees, each split is chosen to maximize information gain (or equivalently minimize the impurity of the child nodes). Information gain tends to favor features with many levels; **gain ratio** or other normalized variants can correct for that. The toolbox uses information gain in tree-based logic and in feature importance and selection utilities.

## Using These in the Toolbox

- **data_quality**: Entropy and mutual information help assess feature informativeness and redundancy.
- **feature_pipeline**: Mutual information and information gain can drive feature selection and automatic feature creation.
- **regression_classification**: Entropy and information gain appear in tree-based and probabilistic components.
- **learning companion**: Information-theoretic notions can guide ordering and presentation of concepts (e.g. by informativeness or surprise).

These concepts do not replace standard ML algorithms; they augment preprocessing, feature selection, and evaluation with principled, interpretable measures.
