"""Core module for trajectory representation using prime factorizations.

This module provides functionality to represent N-dimensional discrete trajectories
as prime factorizations, where each axis corresponds to a prime number (x->2, y->3, z->5, etc.).
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from functools import reduce
import operator


class PrimeTrajectory:
    """Represents a discrete trajectory in N-dimensional space using prime factorizations.
    
    Each axis in N-dimensional space corresponds to a prime number:
    - x-axis: 2
    - y-axis: 3
    - z-axis: 5
    - and so on...
    
    Attributes:
        dimension (int): The dimensionality of the space.
        primes (List[int]): The first N prime numbers corresponding to each axis.
        trajectory_steps (List[np.ndarray]): List of unit vectors representing each step.
        factorizations (List[int]): Prime factorization representation of each step.
    """
    
    def __init__(self, dimension: int = 3):
        """Initialize a PrimeTrajectory object.
        
        Args:
            dimension (int): The dimensionality of the space (default: 3).
        """
        self.dimension = dimension
        self.primes = self._get_first_n_primes(dimension)
        self.trajectory_steps: List[np.ndarray] = []
        self.factorizations: List[int] = []
    
    def _get_first_n_primes(self, n: int) -> List[int]:
        """Get the first n prime numbers.
        
        Args:
            n (int): Number of primes to generate.
            
        Returns:
            List[int]: List of the first n prime numbers.
        """
        if n <= 0:
            return []
        
        primes = []
        candidate = 2
        
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        
        return primes
    
    def add_step(self, direction_vector: np.ndarray) -> int:
        """Add a step to the trajectory from a direction vector.
        
        Args:
            direction_vector (np.ndarray): The direction vector for this step.
                Should be of length equal to the dimension.
        
        Returns:
            int: The prime factorization representation of this step.
            
        Raises:
            ValueError: If direction_vector has incorrect dimension.
        """
        if len(direction_vector) != self.dimension:
            raise ValueError(f"Direction vector must have dimension {self.dimension}")
        
        # Normalize to unit vector
        norm = np.linalg.norm(direction_vector)
        if norm == 0:
            raise ValueError("Direction vector cannot be zero")
        
        unit_vector = direction_vector / norm
        self.trajectory_steps.append(unit_vector)
        
        # Convert to prime factorization
        factorization = self.vector_to_factorization(unit_vector)
        self.factorizations.append(factorization)
        
        return factorization
    
    def vector_to_factorization(self, unit_vector: np.ndarray) -> int:
        """Convert a unit vector to its prime factorization representation.
        
        The factorization is computed as: product(prime_i ^ round(component_i * scale))
        where scale is chosen to preserve information while keeping numbers manageable.
        
        Args:
            unit_vector (np.ndarray): A unit vector in N-dimensional space.
            
        Returns:
            int: The prime factorization representation.
        """
        # Scale factor to convert continuous components to discrete exponents
        scale = 10
        
        factorization = 1
        for i, component in enumerate(unit_vector):
            # Map component to non-negative exponent
            # Shift and scale: component in [-1, 1] -> exponent in [0, 2*scale]
            exponent = int(round((component + 1) * scale))
            factorization *= (self.primes[i] ** exponent)
        
        return factorization
    
    def factorization_to_vector(self, factorization: int) -> np.ndarray:
        """Convert a prime factorization back to an approximate unit vector.
        
        Args:
            factorization (int): The prime factorization representation.
            
        Returns:
            np.ndarray: The reconstructed unit vector.
        """
        scale = 10
        exponents = self._extract_exponents(factorization)
        
        # Reverse the mapping: exponent -> component
        components = np.array([(exp / scale) - 1 for exp in exponents])
        
        # Normalize to unit vector
        norm = np.linalg.norm(components)
        if norm > 0:
            components = components / norm
        
        return components
    
    def _extract_exponents(self, factorization: int) -> List[int]:
        """Extract exponents of each prime from a factorization.
        
        Args:
            factorization (int): The number to factorize.
            
        Returns:
            List[int]: List of exponents for each prime.
        """
        exponents = []
        for prime in self.primes:
            exp = 0
            temp = factorization
            while temp % prime == 0:
                exp += 1
                temp //= prime
            exponents.append(exp)
        
        return exponents
    
    def get_trajectory_encoding(self) -> int:
        """Get a unique encoding for the entire trajectory.
        
        This combines all step factorizations into a single representation.
        Note: This can produce very large numbers for long trajectories.
        
        Returns:
            int: The encoded trajectory.
        """
        if not self.factorizations:
            return 1
        
        # Use product of all factorizations (can be very large)
        return reduce(operator.mul, self.factorizations, 1)
    
    def get_trajectory_hash(self) -> str:
        """Get a hash representation of the trajectory.
        
        Returns:
            str: Hexadecimal hash of the trajectory.
        """
        import hashlib
        trajectory_str = ','.join(map(str, self.factorizations))
        return hashlib.sha256(trajectory_str.encode()).hexdigest()
    
    def get_step_count(self) -> int:
        """Get the number of steps in the trajectory.
        
        Returns:
            int: Number of steps.
        """
        return len(self.trajectory_steps)
    
    def clear(self) -> None:
        """Clear all trajectory data."""
        self.trajectory_steps.clear()
        self.factorizations.clear()


def create_trajectory_from_points(points: np.ndarray, dimension: Optional[int] = None) -> PrimeTrajectory:
    """Create a PrimeTrajectory from a sequence of points.
    
    Args:
        points (np.ndarray): Array of shape (T+1, N) representing T+1 points in N-dimensional space.
        dimension (Optional[int]): Override dimension if needed.
        
    Returns:
        PrimeTrajectory: The constructed trajectory object.
        
    Raises:
        ValueError: If points array is invalid.
    """
    if points.ndim != 2:
        raise ValueError("Points must be a 2D array")
    
    if points.shape[0] < 2:
        raise ValueError("Need at least 2 points to define a trajectory")
    
    dim = dimension if dimension is not None else points.shape[1]
    trajectory = PrimeTrajectory(dimension=dim)
    
    # Compute direction vectors between consecutive points
    for i in range(len(points) - 1):
        direction = points[i + 1] - points[i]
        if np.linalg.norm(direction) > 0:
            trajectory.add_step(direction)
    
    return trajectory


def compare_trajectories(traj1: PrimeTrajectory, traj2: PrimeTrajectory) -> Dict[str, float]:
    """Compare two trajectories and return similarity metrics.
    
    Args:
        traj1 (PrimeTrajectory): First trajectory.
        traj2 (PrimeTrajectory): Second trajectory.
        
    Returns:
        Dict[str, float]: Dictionary containing similarity metrics.
    """
    metrics = {}
    
    # Length similarity
    len1, len2 = traj1.get_step_count(), traj2.get_step_count()
    metrics['length_ratio'] = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    
    # Hash equality
    metrics['hash_match'] = 1.0 if traj1.get_trajectory_hash() == traj2.get_trajectory_hash() else 0.0
    
    # Step-by-step comparison (for trajectories of same length)
    if len1 == len2 and len1 > 0:
        factorization_matches = sum(1 for f1, f2 in zip(traj1.factorizations, traj2.factorizations) if f1 == f2)
        metrics['factorization_match_rate'] = factorization_matches / len1
        
        # Vector similarity (average cosine similarity)
        cosine_sims = []
        for v1, v2 in zip(traj1.trajectory_steps, traj2.trajectory_steps):
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cosine_sims.append(cos_sim)
        metrics['avg_cosine_similarity'] = np.mean(cosine_sims)
    else:
        metrics['factorization_match_rate'] = 0.0
        metrics['avg_cosine_similarity'] = 0.0
    
    return metrics
