"""
Quantum Mechanics Foundations Examples (Heisenberg, Schrödinger, Bohr, etc.)

Demonstrates:
1. Heisenberg Uncertainty Regularization
2. Schrödinger Wave Function Evolution
3. Bell Inequality for Entanglement Detection
4. Born Rule (Probability Interpretation)
5. Wave-Particle Duality
"""
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_toolbox.textbook_concepts.quantum_mechanics import (
    HeisenbergUncertaintyRegularizer,
    SchrodingerWaveFunction,
    BellInequality,
    BornRule,
    WaveParticleDuality
)

print("=" * 80)
print("Quantum Mechanics Foundations Examples")
print("=" * 80)

# ============================================================================
# Example 1: Heisenberg Uncertainty Regularization
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: Heisenberg Uncertainty Regularization")
print("=" * 80)

# Simulate model weights
np.random.seed(42)
weights = np.random.randn(100)
gradients = np.random.randn(100) * 0.1

print(f"\nModel weights shape: {weights.shape}")
print(f"Weights std (position uncertainty): {np.std(weights):.4f}")
print(f"Gradients std (momentum uncertainty): {np.std(gradients):.4f}")

# Create uncertainty regularizer
regularizer = HeisenbergUncertaintyRegularizer(uncertainty_weight=0.01)

# Compute regularization penalty
penalty = regularizer.regularize(weights, gradients)
print(f"\nUncertainty Regularization Penalty: {penalty:.6f}")

# Apply to loss
original_loss = 0.5
regularized_loss = regularizer.apply_regularization(original_loss, weights, gradients)
print(f"Original Loss: {original_loss:.4f}")
print(f"Regularized Loss: {regularized_loss:.4f}")
print(f"Penalty Added: {regularized_loss - original_loss:.6f}")

# Test with different weight distributions
print("\n--- Testing Different Weight Distributions ---")
for name, test_weights in [
    ("Uniform (low uncertainty)", np.ones(100) * 0.5),
    ("Normal (medium uncertainty)", np.random.randn(100)),
    ("Spread (high uncertainty)", np.random.randn(100) * 5)
]:
    penalty = regularizer.compute_uncertainty_penalty(test_weights)
    print(f"  {name}: Penalty = {penalty:.6f}")

# ============================================================================
# Example 2: Schrödinger Wave Function Evolution
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Schrödinger Wave Function Evolution")
print("=" * 80)

# Create initial wave function (superposition state)
n_states = 8
initial_state = np.ones(n_states, dtype=complex) / np.sqrt(n_states)

# Create Hamiltonian (energy operator)
# Simple example: diagonal Hamiltonian with different energy levels
hamiltonian = np.diag(np.linspace(0, 1, n_states)) + 0.1 * np.eye(n_states, dtype=complex)

print(f"\nInitial Wave Function (superposition):")
print(f"  States: {n_states}")
print(f"  Initial probabilities: {np.abs(initial_state)**2}")

# Create wave function
wave_function = SchrodingerWaveFunction(initial_state, hamiltonian)

# Get initial energy
initial_energy = wave_function.get_energy()
print(f"  Initial Energy: {initial_energy:.4f}")

# Evolve wave function
print("\nEvolving wave function (time-dependent Schrödinger equation)...")
evolved_state = wave_function.evolve(time_steps=100, dt=0.01)

# Get evolved probabilities
evolved_probabilities = wave_function.get_probability_density()
final_energy = wave_function.get_energy()

print(f"\nEvolved Wave Function:")
print(f"  Final probabilities: {evolved_probabilities}")
print(f"  Final Energy: {final_energy:.4f}")
print(f"  Energy Change: {final_energy - initial_energy:.4f}")

# Quantum measurement
print("\n--- Quantum Measurement (Wave Function Collapse) ---")
outcome, prob = wave_function.measure()
print(f"  Measurement Outcome: State {outcome}")
print(f"  Probability: {prob:.4f}")
print(f"  Wave function collapsed to state {outcome}")

# ============================================================================
# Example 3: Bell Inequality for Entanglement Detection
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: Bell Inequality for Entanglement Detection")
print("=" * 80)

# Create quantum states (simulated)
np.random.seed(42)

# State 1: Superposition
state1 = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)

# State 2: Entangled state (Bell state)
state2 = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)

# State 3: Product state (not entangled)
state3 = np.array([1.0, 0.0], dtype=complex)

print(f"\nTesting Entanglement:")
print(f"  State 1 shape: {state1.shape}")
print(f"  State 2 shape: {state2.shape}")
print(f"  State 3 shape: {state3.shape}")

# Test Bell inequality
bell_tester = BellInequality()

# Test state1 vs state2 (if compatible dimensions)
if len(state1) == len(state2):
    bell_result = bell_tester.measure_bell_inequality(state1, state2)
    print(f"\nBell Inequality Test (State 1 vs State 2):")
    print(f"  Bell Value: {bell_result['bell_value']:.4f}")
    print(f"  Classical Limit: {bell_result['classical_limit']:.4f}")
    print(f"  Quantum Limit: {bell_result['quantum_limit']:.4f}")
    print(f"  Is Entangled: {bell_result['is_entangled']}")
    print(f"  Entanglement Strength: {bell_result['entanglement_strength']:.4f}")

# Test multiple states
print("\n--- Testing Multiple States ---")
states = [state1, state3]
if len(state1) == len(state3):
    multi_result = bell_tester.test_entanglement(states)
    print(f"  Total Pairs: {multi_result['total_pairs']}")
    print(f"  Entangled Pairs: {multi_result['entangled_pairs']}")
    print(f"  Max Bell Value: {multi_result['max_bell_value']:.4f}")

# ============================================================================
# Example 4: Born Rule (Probability Interpretation)
# ============================================================================
print("\n" + "=" * 80)
print("Example 4: Born Rule (Probability Interpretation)")
print("=" * 80)

# Create wave function
wave_function = np.array([1.0, 1j, 0.5, 0.5j], dtype=complex)
wave_function = wave_function / np.linalg.norm(wave_function)

print(f"\nWave Function: {wave_function}")

# Compute probability density |psi|^2
probability = BornRule.probability_density(wave_function)
print(f"\nProbability Density (|psi|^2): {probability}")
print(f"  Sum of probabilities: {np.sum(probability):.4f} (should be 1.0)")

# Normalize wave function
normalized = BornRule.normalize(wave_function)
print(f"\nNormalized Wave Function: {normalized}")
print(f"  Norm: {np.linalg.norm(normalized):.4f}")

# Expectation value
operator = np.diag([1, 2, 3, 4])
expectation = BornRule.expectation_value(operator, wave_function)
print(f"\nExpectation Value <psi|O|psi>: {expectation:.4f}")

# Measurement probability
eigenstate = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
measurement_prob = BornRule.measurement_probability(wave_function, eigenstate)
print(f"\nMeasurement Probability for eigenstate [1,0,0,0]: {measurement_prob:.4f}")

# ============================================================================
# Example 5: Wave-Particle Duality
# ============================================================================
print("\n" + "=" * 80)
print("Example 5: Wave-Particle Duality")
print("=" * 80)

# Create sample data (particle-like: localized)
np.random.seed(42)
data = np.zeros(100)
data[45:55] = 1.0  # Localized "particle"

print(f"\nOriginal Data (Particle-like):")
print(f"  Shape: {data.shape}")
print(f"  Localized region: indices 45-55")

# Create wave-particle duality analyzer
duality = WaveParticleDuality()

# Get dual representations
representations = duality.dual_representation(data)

print(f"\nWave Representation (Fourier Transform):")
print(f"  Shape: {representations['wave'].shape}")
print(f"  Magnitude range: [{np.min(np.abs(representations['wave'])):.4f}, {np.max(np.abs(representations['wave'])):.4f}]")

print(f"\nParticle Representation (Inverse Fourier Transform):")
print(f"  Shape: {representations['particle'].shape}")
print(f"  Recovered data matches original: {np.allclose(representations['particle'], data, atol=1e-10)}")

# Complementarity score
complementarity = duality.complementarity_score(data)
print(f"\nComplementarity Score: {complementarity:.4f}")
print(f"  (Higher = more complementary, exhibits both wave and particle properties)")

# Test with different data types
print("\n--- Testing Different Data Types ---")
particle_data = np.zeros(100)
particle_data[45:55] = 1.0

test_cases = [
    ("Localized (Particle-like)", particle_data),
    ("Oscillating (Wave-like)", np.sin(np.linspace(0, 4*np.pi, 100))),
    ("Mixed", np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.1)
]

for name, test_data in test_cases:
    comp = duality.complementarity_score(test_data)
    print(f"  {name}: Complementarity = {comp:.4f}")

# de Broglie wavelength
print("\n--- de Broglie Wavelength ---")
momentum = 1.0
wavelength = duality.de_broglie_wavelength_from_momentum(momentum)
print(f"  Momentum: {momentum} kg*m/s")
print(f"  de Broglie Wavelength: {wavelength:.2e} m")

# ============================================================================
# Example 6: Integration with ML Models
# ============================================================================
print("\n" + "=" * 80)
print("Example 6: Integration with ML Models")
print("=" * 80)

# Simulate training a simple model with uncertainty regularization
print("\n--- Training with Heisenberg Uncertainty Regularization ---")

# Simulate training loop
weights = np.random.randn(50)
learning_rate = 0.01
regularizer = HeisenbergUncertaintyRegularizer(uncertainty_weight=0.001)

print(f"Initial weights std: {np.std(weights):.4f}")

for epoch in range(5):
    # Simulate gradients
    gradients = np.random.randn(50) * 0.1
    
    # Compute regularization penalty
    penalty = regularizer.regularize(weights, gradients)
    
    # Update weights (with regularization)
    weights = weights - learning_rate * (gradients + penalty * weights)
    
    print(f"  Epoch {epoch+1}: Weight std = {np.std(weights):.4f}, Penalty = {penalty:.6f}")

print("\n[OK] Uncertainty regularization prevents overfitting by maintaining uncertainty")

# Sequence modeling with wave function evolution
print("\n--- Sequence Modeling with Schrödinger Wave Function ---")

# Model sequence as wave function evolution
sequence_length = 10
initial_state = np.ones(sequence_length, dtype=complex) / np.sqrt(sequence_length)

# Hamiltonian for sequence evolution
hamiltonian = np.diag(np.arange(sequence_length)) + 0.1 * np.eye(sequence_length, dtype=complex)

wave_func = SchrodingerWaveFunction(initial_state, hamiltonian)

# Evolve sequence
evolved = wave_func.evolve(time_steps=50, dt=0.01)
probabilities = wave_func.get_probability_density()

print(f"Sequence Evolution:")
print(f"  Initial probabilities: {np.abs(initial_state)**2}")
print(f"  Evolved probabilities: {probabilities}")
print(f"  Most likely state: {np.argmax(probabilities)}")

print("\n" + "=" * 80)
print("[OK] All Quantum Mechanics Examples Completed!")
print("=" * 80)
