"""Utility functions for trajectory analysis and prime number operations.

This module provides helper functions for working with prime factorizations,
trajectory visualization, and mathematical operations.
"""

from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import math


def is_prime(n: int) -> bool:
    """Check if a number is prime.
    
    Args:
        n (int): The number to check.
        
    Returns:
        bool: True if n is prime, False otherwise.
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def generate_primes(limit: int) -> List[int]:
    """Generate all prime numbers up to a given limit using Sieve of Eratosthenes.
    
    Args:
        limit (int): The upper limit (inclusive).
        
    Returns:
        List[int]: List of prime numbers up to limit.
    """
    if limit < 2:
        return []
    
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    
    return [i for i in range(limit + 1) if sieve[i]]


def prime_factorization(n: int) -> Dict[int, int]:
    """Compute the prime factorization of a number.
    
    Args:
        n (int): The number to factorize.
        
    Returns:
        Dict[int, int]: Dictionary mapping prime factors to their exponents.
    """
    if n <= 1:
        return {}
    
    factors = {}
    d = 2
    
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    
    return factors


def factorization_to_number(factors: Dict[int, int]) -> int:
    """Convert a prime factorization back to the original number.
    
    Args:
        factors (Dict[int, int]): Dictionary mapping prime factors to exponents.
        
    Returns:
        int: The reconstructed number.
    """
    result = 1
    for prime, exponent in factors.items():
        result *= prime ** exponent
    return result


def discretize_vector(vector: np.ndarray, resolution: int = 10) -> np.ndarray:
    """Discretize a continuous vector to a grid.
    
    Args:
        vector (np.ndarray): The continuous vector.
        resolution (int): The discretization resolution.
        
    Returns:
        np.ndarray: The discretized vector.
    """
    return np.round(vector * resolution).astype(int)


def continuous_to_discrete_direction(vector: np.ndarray, num_bins: int = 8) -> int:
    """Convert a continuous direction vector to a discrete direction index.
    
    For 2D: 8 cardinal/ordinal directions (N, NE, E, SE, S, SW, W, NW)
    For 3D and higher: discretized based on dominant component.
    
    Args:
        vector (np.ndarray): The direction vector.
        num_bins (int): Number of discrete directions per dimension.
        
    Returns:
        int: Discrete direction index.
    """
    if len(vector) == 0:
        return 0
    
    # Normalize
    norm = np.linalg.norm(vector)
    if norm == 0:
        return 0
    
    unit_vector = vector / norm
    
    # For 2D, use angle-based discretization
    if len(vector) == 2:
        angle = np.arctan2(unit_vector[1], unit_vector[0])
        # Map angle to [0, 2π) and discretize
        angle = (angle + 2 * np.pi) % (2 * np.pi)
        direction_idx = int(angle / (2 * np.pi) * num_bins) % num_bins
        return direction_idx
    
    # For higher dimensions, use a hash-based approach
    discretized = discretize_vector(unit_vector, resolution=num_bins)
    # Create a unique index from discretized components
    direction_idx = hash(tuple(discretized)) % (num_bins ** len(vector))
    return direction_idx


def compute_trajectory_length(points: np.ndarray) -> float:
    """Compute the total length of a trajectory defined by points.
    
    Args:
        points (np.ndarray): Array of shape (N, D) representing N points in D dimensions.
        
    Returns:
        float: Total trajectory length.
    """
    if len(points) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(len(points) - 1):
        segment_length = np.linalg.norm(points[i + 1] - points[i])
        total_length += segment_length
    
    return total_length


def compute_curvature(points: np.ndarray) -> np.ndarray:
    """Compute discrete curvature at each point along a trajectory.
    
    Curvature is approximated using the angle between consecutive segments.
    
    Args:
        points (np.ndarray): Array of shape (N, D) representing N points in D dimensions.
        
    Returns:
        np.ndarray: Array of curvature values (length N-2).
    """
    if len(points) < 3:
        return np.array([])
    
    curvatures = []
    
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 0 and norm2 > 0:
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            # Clamp to [-1, 1] to avoid numerical errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            curvatures.append(angle)
        else:
            curvatures.append(0.0)
    
    return np.array(curvatures)


def generate_random_trajectory(num_steps: int, dimension: int = 3, 
                              step_size: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
    """Generate a random trajectory in N-dimensional space.
    
    Args:
        num_steps (int): Number of steps in the trajectory.
        dimension (int): Dimensionality of the space.
        step_size (float): Average step size.
        seed (Optional[int]): Random seed for reproducibility.
        
    Returns:
        np.ndarray: Array of shape (num_steps+1, dimension) representing the trajectory points.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Start at origin
    points = [np.zeros(dimension)]
    current_pos = np.zeros(dimension)
    
    for _ in range(num_steps):
        # Random direction
        direction = np.random.randn(dimension)
        direction = direction / np.linalg.norm(direction) * step_size
        
        current_pos = current_pos + direction
        points.append(current_pos.copy())
    
    return np.array(points)


def trajectory_to_string(factorizations: List[int], separator: str = ',') -> str:
    """Convert a list of factorizations to a string representation.
    
    Args:
        factorizations (List[int]): List of prime factorization values.
        separator (str): Separator between values.
        
    Returns:
        str: String representation of the trajectory.
    """
    return separator.join(map(str, factorizations))


def string_to_trajectory(trajectory_str: str, separator: str = ',') -> List[int]:
    """Parse a string representation back to a list of factorizations.
    
    Args:
        trajectory_str (str): String representation of trajectory.
        separator (str): Separator between values.
        
    Returns:
        List[int]: List of prime factorization values.
    """
    if not trajectory_str:
        return []
    return [int(x) for x in trajectory_str.split(separator)]


def compute_bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the bounding box of a set of points.
    
    Args:
        points (np.ndarray): Array of shape (N, D) representing N points in D dimensions.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (min_coords, max_coords) defining the bounding box.
    """
    if len(points) == 0:
        return np.array([]), np.array([])
    
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    return min_coords, max_coords


def normalize_trajectory(points: np.ndarray) -> np.ndarray:
    """Normalize a trajectory to fit within a unit hypercube.
    
    Args:
        points (np.ndarray): Array of shape (N, D) representing N points in D dimensions.
        
    Returns:
        np.ndarray: Normalized trajectory.
    """
    if len(points) == 0:
        return points
    
    min_coords, max_coords = compute_bounding_box(points)
    ranges = max_coords - min_coords
    
    # Avoid division by zero
    ranges = np.where(ranges == 0, 1, ranges)
    
    normalized = (points - min_coords) / ranges
    return normalized


def get_prime_mapping(dimension: int) -> Dict[int, int]:
    """Get the mapping from axis index to prime number.
    
    Args:
        dimension (int): Number of dimensions.
        
    Returns:
        Dict[int, int]: Dictionary mapping axis index (0, 1, 2, ...) to prime (2, 3, 5, ...).
    """
    primes = []
    candidate = 2
    
    while len(primes) < dimension:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    
    return {i: prime for i, prime in enumerate(primes)}


def axis_name_to_prime(axis_name: str) -> Optional[int]:
    """Convert axis name (x, y, z, ...) to corresponding prime number.
    
    Args:
        axis_name (str): Axis name ('x', 'y', 'z', etc.).
        
    Returns:
        Optional[int]: Corresponding prime number, or None if invalid.
    """
    axis_map = {
        'x': 2,
        'y': 3,
        'z': 5,
        'w': 7,
        'u': 11,
        'v': 13
    }
    return axis_map.get(axis_name.lower())


def prime_to_axis_name(prime: int) -> Optional[str]:
    """Convert prime number to axis name.
    
    Args:
        prime (int): Prime number.
        
    Returns:
        Optional[str]: Axis name, or None if not in standard mapping.
    """
    prime_map = {
        2: 'x',
        3: 'y',
        5: 'z',
        7: 'w',
        11: 'u',
        13: 'v'
    }
    return prime_map.get(prime)
