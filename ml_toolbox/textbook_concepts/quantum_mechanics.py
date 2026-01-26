"""
Quantum Mechanics Foundations (Heisenberg, Schrödinger, Bohr, etc.)

Implements:
- Heisenberg Uncertainty Principle (regularization)
- Schrödinger Wave Equation (evolution)
- Bell Inequality (entanglement detection)
- Born Rule (probability interpretation)
- Wave-Particle Duality (dual representations)
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)

# Reduced Planck constant (approximate)
HBAR = 1.054571817e-34  # J⋅s (for theoretical calculations, we'll use normalized units)


class HeisenbergUncertaintyRegularizer:
    """
    Heisenberg Uncertainty Principle-based Regularization
    
    Uses uncertainty principle: Δx · Δp ≥ ℏ/2
    Applied to ML: Regularization based on position-momentum uncertainty
    """
    
    def __init__(self, hbar: float = 1.0, uncertainty_weight: float = 0.01):
        """
        Initialize uncertainty regularizer
        
        Parameters
        ----------
        hbar : float
            Reduced Planck constant (normalized, default 1.0)
        uncertainty_weight : float
            Weight for uncertainty regularization term
        """
        self.hbar = hbar
        self.uncertainty_weight = uncertainty_weight
    
    def compute_uncertainty_penalty(self, weights: np.ndarray) -> float:
        """
        Compute uncertainty-based regularization penalty
        
        Treats weights as "position" and gradients as "momentum"
        Uncertainty: Δw · Δg ≥ ℏ/2
        
        Parameters
        ----------
        weights : array
            Model weights (position)
            
        Returns
        -------
        penalty : float
            Uncertainty regularization penalty
        """
        weights = np.asarray(weights)
        
        # Position uncertainty (variance of weights)
        position_uncertainty = np.var(weights)
        
        # Minimum momentum uncertainty (from uncertainty principle)
        # Δp ≥ ℏ/(2·Δx)
        if position_uncertainty > 1e-10:
            min_momentum_uncertainty = self.hbar / (2.0 * position_uncertainty)
        else:
            min_momentum_uncertainty = 1e10  # Large penalty for zero uncertainty
        
        # Penalty: encourage weights to satisfy uncertainty principle
        # Penalize if uncertainty is too low (overfitting)
        penalty = self.uncertainty_weight * (1.0 / (position_uncertainty + 1e-10))
        
        return float(penalty)
    
    def regularize(self, weights: np.ndarray, gradients: Optional[np.ndarray] = None) -> float:
        """
        Compute regularization term based on uncertainty principle
        
        Parameters
        ----------
        weights : array
            Model weights
        gradients : array, optional
            Model gradients (momentum)
            
        Returns
        -------
        regularization : float
            Regularization term to add to loss
        """
        weights = np.asarray(weights)
        
        # Position uncertainty
        delta_x = np.std(weights)
        
        if gradients is not None:
            # Momentum uncertainty (from gradients)
            gradients = np.asarray(gradients)
            delta_p = np.std(gradients)
            
            # Uncertainty principle: Δx · Δp ≥ ℏ/2
            uncertainty_product = delta_x * delta_p
            min_uncertainty = self.hbar / 2.0
            
            # Penalty if uncertainty product is too small (violates principle)
            if uncertainty_product < min_uncertainty:
                penalty = self.uncertainty_weight * (min_uncertainty - uncertainty_product) / min_uncertainty
            else:
                penalty = 0.0
        else:
            # Simplified: just use position uncertainty
            penalty = self.compute_uncertainty_penalty(weights)
        
        return float(penalty)
    
    def apply_regularization(self, loss: float, weights: np.ndarray,
                            gradients: Optional[np.ndarray] = None) -> float:
        """
        Apply uncertainty regularization to loss
        
        Parameters
        ----------
        loss : float
            Original loss
        weights : array
            Model weights
        gradients : array, optional
            Model gradients
            
        Returns
        -------
        regularized_loss : float
            Loss with uncertainty regularization
        """
        penalty = self.regularize(weights, gradients)
        return loss + penalty


class SchrodingerWaveFunction:
    """
    Schrödinger Wave Equation
    
    Implements time-dependent and time-independent Schrödinger equations
    for sequence modeling and state evolution
    """
    
    def __init__(self, initial_state: np.ndarray, hamiltonian: Optional[np.ndarray] = None):
        """
        Initialize wave function
        
        Parameters
        ----------
        initial_state : array
            Initial wave function (complex-valued)
        hamiltonian : array, optional
            Hamiltonian operator (energy operator)
            If None, uses identity (no evolution)
        """
        self.state = np.asarray(initial_state, dtype=complex)
        self.initial_state = self.state.copy()
        
        # Normalize wave function
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state = self.state / norm
        
        if hamiltonian is not None:
            self.hamiltonian = np.asarray(hamiltonian, dtype=complex)
        else:
            # Default: identity (no evolution)
            self.hamiltonian = np.eye(len(self.state), dtype=complex)
        
        self.time = 0.0
        self.hbar = 1.0  # Normalized
    
    def evolve(self, time_steps: int, dt: float = 0.01) -> np.ndarray:
        """
        Evolve wave function using time-dependent Schrödinger equation
        
        iℏ ∂ψ/∂t = Ĥψ
        
        Uses numerical integration
        
        Parameters
        ----------
        time_steps : int
            Number of time steps
        dt : float
            Time step size
            
        Returns
        -------
        evolved_state : array
            Evolved wave function
        """
        current_state = self.state.copy()
        
        for _ in range(time_steps):
            # Time-dependent Schrödinger equation: iℏ ∂ψ/∂t = Ĥψ
            # ∂ψ/∂t = -i/ℏ · Ĥψ
            # Euler method: ψ(t+dt) = ψ(t) + dt · ∂ψ/∂t
            
            # Compute H|ψ>
            h_psi = self.hamiltonian @ current_state
            
            # Time derivative
            dpsi_dt = -1j / self.hbar * h_psi
            
            # Update state
            current_state = current_state + dt * dpsi_dt
            
            # Normalize
            norm = np.linalg.norm(current_state)
            if norm > 0:
                current_state = current_state / norm
            
            self.time += dt
        
        self.state = current_state
        return current_state
    
    def get_probability_density(self) -> np.ndarray:
        """
        Get probability density |ψ|² (Born rule)
        
        Returns
        -------
        probability : array
            Probability density
        """
        return np.abs(self.state) ** 2
    
    def measure(self) -> Tuple[int, float]:
        """
        Quantum measurement (collapse wave function)
        
        Returns
        -------
        outcome : tuple
            (index, probability) of measurement outcome
        """
        probabilities = self.get_probability_density()
        
        # Sample from probability distribution
        outcome = np.random.choice(len(probabilities), p=probabilities)
        
        # Collapse to measured state
        self.state = np.zeros_like(self.state, dtype=complex)
        self.state[outcome] = 1.0
        
        return outcome, float(probabilities[outcome])
    
    def get_energy(self) -> float:
        """
        Compute expected energy <ψ|Ĥ|ψ>
        
        Returns
        -------
        energy : float
            Expected energy
        """
        h_psi = self.hamiltonian @ self.state
        energy = np.real(np.vdot(self.state, h_psi))
        return float(energy)


class BellInequality:
    """
    Bell Inequality for Entanglement Detection
    
    Tests for non-local correlations (entanglement)
    Classical limit: |E(a,b) - E(a,b') + E(a',b) + E(a',b')| ≤ 2
    Quantum limit: Can exceed 2 (up to 2√2 ≈ 2.828)
    """
    
    def __init__(self):
        """Initialize Bell inequality tester"""
        pass
    
    def measure_bell_inequality(self, state1: np.ndarray, state2: np.ndarray,
                               measurement_bases: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Measure Bell inequality
        
        Parameters
        ----------
        state1 : array
            First quantum state
        state2 : array
            Second quantum state
        measurement_bases : list of tuples, optional
            Measurement bases (angles) for a, a', b, b'
            Default: [0, π/4, π/8, 3π/8]
            
        Returns
        -------
        result : dict
            Dictionary containing:
            - 'bell_value': Bell inequality value
            - 'is_entangled': Whether states are entangled
            - 'classical_limit': 2.0
            - 'quantum_limit': 2.828
            - 'correlations': Individual correlation values
        """
        state1 = np.asarray(state1)
        state2 = np.asarray(state2)
        
        # Default measurement bases
        if measurement_bases is None:
            measurement_bases = [(0, 0), (0, np.pi/4), (np.pi/4, 0), (np.pi/4, np.pi/4)]
        
        # Compute correlations for different measurement bases
        correlations = []
        for a, b in measurement_bases:
            # Simplified correlation: E(a,b) = <state1|Pa|state1> · <state2|Pb|state2>
            # For simplicity, use cosine similarity with rotated states
            corr = self._compute_correlation(state1, state2, a, b)
            correlations.append(corr)
        
        # Bell inequality: |E(a,b) - E(a,b') + E(a',b) + E(a',b')|
        if len(correlations) >= 4:
            bell_value = abs(correlations[0] - correlations[1] + correlations[2] + correlations[3])
        else:
            # Simplified: use first correlation
            bell_value = abs(correlations[0]) if correlations else 0.0
        
        classical_limit = 2.0
        quantum_limit = 2.0 * np.sqrt(2)  # ≈ 2.828
        
        is_entangled = bell_value > classical_limit
        
        return {
            'bell_value': float(bell_value),
            'is_entangled': is_entangled,
            'classical_limit': classical_limit,
            'quantum_limit': quantum_limit,
            'correlations': [float(c) for c in correlations],
            'entanglement_strength': float((bell_value - classical_limit) / (quantum_limit - classical_limit))
        }
    
    def _compute_correlation(self, state1: np.ndarray, state2: np.ndarray,
                            angle1: float, angle2: float) -> float:
        """Compute correlation for given measurement angles"""
        # Rotate states by measurement angles
        rotated1 = state1 * np.exp(1j * angle1)
        rotated2 = state2 * np.exp(1j * angle2)
        
        # Correlation: real part of <state1|rotated1> · <state2|rotated2>
        corr1 = np.real(np.vdot(state1, rotated1))
        corr2 = np.real(np.vdot(state2, rotated2))
        
        return corr1 * corr2
    
    def test_entanglement(self, quantum_states: List[np.ndarray]) -> Dict[str, Any]:
        """
        Test entanglement between multiple quantum states
        
        Parameters
        ----------
        quantum_states : list of arrays
            List of quantum states to test
            
        Returns
        -------
        result : dict
            Entanglement test results
        """
        if len(quantum_states) < 2:
            return {'error': 'Need at least 2 states to test entanglement'}
        
        results = []
        max_bell_value = 0.0
        
        # Test all pairs
        for i in range(len(quantum_states)):
            for j in range(i + 1, len(quantum_states)):
                bell_result = self.measure_bell_inequality(quantum_states[i], quantum_states[j])
                results.append({
                    'pair': (i, j),
                    'bell_value': bell_result['bell_value'],
                    'is_entangled': bell_result['is_entangled']
                })
                max_bell_value = max(max_bell_value, bell_result['bell_value'])
        
        return {
            'total_pairs': len(results),
            'entangled_pairs': sum(1 for r in results if r['is_entangled']),
            'max_bell_value': float(max_bell_value),
            'pair_results': results
        }


class BornRule:
    """
    Born Rule - Probability Interpretation of Quantum Mechanics
    
    |ψ|² = probability density
    """
    
    @staticmethod
    def probability_density(wave_function: np.ndarray) -> np.ndarray:
        """
        Compute probability density from wave function
        
        P(x) = |ψ(x)|²
        
        Parameters
        ----------
        wave_function : array
            Wave function (complex-valued)
            
        Returns
        -------
        probability : array
            Probability density
        """
        wave_function = np.asarray(wave_function, dtype=complex)
        return np.abs(wave_function) ** 2
    
    @staticmethod
    def normalize(wave_function: np.ndarray) -> np.ndarray:
        """
        Normalize wave function so total probability = 1
        
        Parameters
        ----------
        wave_function : array
            Wave function
            
        Returns
        -------
        normalized : array
            Normalized wave function
        """
        wave_function = np.asarray(wave_function, dtype=complex)
        norm = np.linalg.norm(wave_function)
        
        if norm > 1e-10:
            return wave_function / norm
        else:
            return wave_function
    
    @staticmethod
    def expectation_value(operator: np.ndarray, wave_function: np.ndarray) -> complex:
        """
        Compute expectation value <ψ|O|ψ>
        
        Parameters
        ----------
        operator : array
            Quantum operator
        wave_function : array
            Wave function
            
        Returns
        -------
        expectation : complex
            Expectation value
        """
        operator = np.asarray(operator, dtype=complex)
        wave_function = np.asarray(wave_function, dtype=complex)
        
        # O|ψ>
        op_psi = operator @ wave_function
        
        # <ψ|O|ψ>
        expectation = np.vdot(wave_function, op_psi)
        
        return expectation
    
    @staticmethod
    def measurement_probability(wave_function: np.ndarray, eigenstate: np.ndarray) -> float:
        """
        Probability of measuring specific eigenstate
        
        P = |<eigenstate|ψ>|²
        
        Parameters
        ----------
        wave_function : array
            Wave function
        eigenstate : array
            Eigenstate to measure
            
        Returns
        -------
        probability : float
            Measurement probability
        """
        wave_function = np.asarray(wave_function, dtype=complex)
        eigenstate = np.asarray(eigenstate, dtype=complex)
        
        # Normalize eigenstate
        eigenstate = eigenstate / np.linalg.norm(eigenstate)
        
        # Overlap: <eigenstate|ψ>
        overlap = np.vdot(eigenstate, wave_function)
        
        # Probability: |overlap|²
        return float(np.abs(overlap) ** 2)


class WaveParticleDuality:
    """
    Wave-Particle Duality
    
    Objects can exhibit both wave-like and particle-like properties
    """
    
    def __init__(self, de_broglie_wavelength: Optional[float] = None):
        """
        Initialize wave-particle duality
        
        Parameters
        ----------
        de_broglie_wavelength : float, optional
            de Broglie wavelength λ = h/p
        """
        self.de_broglie_wavelength = de_broglie_wavelength
        self.h = 6.626e-34  # Planck constant (J⋅s)
    
    def wave_representation(self, data: np.ndarray) -> np.ndarray:
        """
        Convert to wave-like representation (Fourier transform)
        
        Parameters
        ----------
        data : array
            Input data (particle-like)
            
        Returns
        -------
        wave : array
            Wave representation (frequency domain)
        """
        data = np.asarray(data)
        # Fourier transform gives wave representation
        wave = np.fft.fft(data)
        return wave
    
    def particle_representation(self, wave: np.ndarray) -> np.ndarray:
        """
        Convert to particle-like representation (inverse Fourier transform)
        
        Parameters
        ----------
        wave : array
            Wave representation (frequency domain)
            
        Returns
        -------
        particle : array
            Particle representation (spatial domain)
        """
        wave = np.asarray(wave, dtype=complex)
        # Inverse Fourier transform gives particle representation
        particle = np.fft.ifft(wave)
        return np.real(particle)
    
    def dual_representation(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get both wave and particle representations
        
        Parameters
        ----------
        data : array
            Input data
            
        Returns
        -------
        representations : dict
            Dictionary with 'wave' and 'particle' representations
        """
        wave = self.wave_representation(data)
        particle = self.particle_representation(wave)
        
        return {
            'wave': wave,
            'particle': particle,
            'original': data
        }
    
    def complementarity_score(self, data: np.ndarray) -> float:
        """
        Measure complementarity (how much data exhibits both properties)
        
        Parameters
        ----------
        data : array
            Input data
            
        Returns
        -------
        score : float
            Complementarity score (0-1, higher = more complementary)
        """
        representations = self.dual_representation(data)
        
        wave = representations['wave']
        particle = representations['particle']
        
        # Complementarity: how different are wave and particle views?
        # Higher difference = more complementary
        wave_magnitude = np.abs(wave)
        particle_magnitude = np.abs(particle)
        
        # Normalize
        if np.max(wave_magnitude) > 0:
            wave_magnitude = wave_magnitude / np.max(wave_magnitude)
        if np.max(particle_magnitude) > 0:
            particle_magnitude = particle_magnitude / np.max(particle_magnitude)
        
        # Complementarity: correlation between wave and particle
        # Lower correlation = higher complementarity
        correlation = np.corrcoef(wave_magnitude, particle_magnitude)[0, 1]
        complementarity = 1.0 - abs(correlation)
        
        return float(complementarity)
    
    def de_broglie_wavelength_from_momentum(self, momentum: float) -> float:
        """
        Calculate de Broglie wavelength from momentum
        
        λ = h/p
        
        Parameters
        ----------
        momentum : float
            Momentum
            
        Returns
        -------
        wavelength : float
            de Broglie wavelength
        """
        if abs(momentum) < 1e-10:
            return float('inf')
        return self.h / abs(momentum)
