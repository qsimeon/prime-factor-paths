#!/usr/bin/env python3
"""
Prime Trajectory Demo Script

This script demonstrates the concept of representing discrete trajectories in N-dimensional
space using prime factorizations. Each axis corresponds to a prime number (x=2, y=3, z=5, etc.),
and trajectory steps are encoded as products of prime powers.

The demo includes:
1. Creating trajectories from continuous paths
2. Encoding/decoding trajectory steps using prime factorizations
3. Comparing different trajectories
4. Visualizing trajectories in 2D and 3D
5. Serialization and deserialization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple
import sys

# Import from the provided modules
sys.path.insert(0, 'lib')
from core import PrimeTrajectory, create_trajectory_from_points, compare_trajectories
from utils import (
    generate_primes, prime_factorization, factorization_to_number,
    compute_trajectory_length, compute_curvature, generate_random_trajectory,
    trajectory_to_string, string_to_trajectory, normalize_trajectory,
    get_prime_mapping, axis_name_to_prime, prime_to_axis_name
)


def create_spiral_trajectory(num_points: int = 50, dimension: int = 3) -> np.ndarray:
    """
    Create a spiral trajectory in N-dimensional space.
    
    Args:
        num_points: Number of points in the trajectory
        dimension: Dimensionality of the space
    
    Returns:
        Array of shape (num_points, dimension) representing the trajectory
    """
    t = np.linspace(0, 4 * np.pi, num_points)
    
    if dimension == 2:
        x = t * np.cos(t)
        y = t * np.sin(t)
        return np.column_stack([x, y])
    elif dimension == 3:
        x = t * np.cos(t)
        y = t * np.sin(t)
        z = t
        return np.column_stack([x, y, z])
    else:
        # For higher dimensions, create a generalized spiral
        points = np.zeros((num_points, dimension))
        for i in range(dimension):
            phase = (2 * np.pi * i) / dimension
            if i % 2 == 0:
                points[:, i] = t * np.cos(t + phase)
            else:
                points[:, i] = t * np.sin(t + phase)
        return points


def create_circle_trajectory(num_points: int = 30, radius: float = 5.0) -> np.ndarray:
    """
    Create a circular trajectory in 2D space.
    
    Args:
        num_points: Number of points in the trajectory
        radius: Radius of the circle
    
    Returns:
        Array of shape (num_points, 2) representing the trajectory
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y])


def create_lissajous_trajectory(num_points: int = 100, a: int = 3, b: int = 2) -> np.ndarray:
    """
    Create a Lissajous curve trajectory in 3D space.
    
    Args:
        num_points: Number of points in the trajectory
        a, b: Frequency parameters for the Lissajous curve
    
    Returns:
        Array of shape (num_points, 3) representing the trajectory
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    x = np.sin(a * t)
    y = np.sin(b * t)
    z = np.sin((a + b) * t / 2)
    return np.column_stack([x, y, z])


def visualize_trajectory_2d(points: np.ndarray, title: str = "2D Trajectory"):
    """
    Visualize a 2D trajectory.
    
    Args:
        points: Array of shape (n, 2) representing the trajectory
        title: Title for the plot
    """
    plt.figure(figsize=(8, 8))
    plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
    plt.plot(points[0, 0], points[0, 1], 'go', markersize=10, label='Start')
    plt.plot(points[-1, 0], points[-1, 1], 'ro', markersize=10, label='End')
    
    # Add arrows to show direction
    for i in range(0, len(points) - 1, max(1, len(points) // 10)):
        dx = points[i + 1, 0] - points[i, 0]
        dy = points[i + 1, 1] - points[i, 1]
        plt.arrow(points[i, 0], points[i, 1], dx, dy, 
                 head_width=0.3, head_length=0.2, fc='blue', ec='blue', alpha=0.5)
    
    plt.xlabel('X (prime 2)')
    plt.ylabel('Y (prime 3)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')


def visualize_trajectory_3d(points: np.ndarray, title: str = "3D Trajectory"):
    """
    Visualize a 3D trajectory.
    
    Args:
        points: Array of shape (n, 3) representing the trajectory
        title: Title for the plot
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
    ax.scatter(points[0, 0], points[0, 1], points[0, 2], c='green', s=100, label='Start')
    ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], c='red', s=100, label='End')
    
    ax.set_xlabel('X (prime 2)')
    ax.set_ylabel('Y (prime 3)')
    ax.set_zlabel('Z (prime 5)')
    ax.set_title(title)
    ax.legend()


def print_trajectory_info(traj: PrimeTrajectory, name: str = "Trajectory"):
    """
    Print detailed information about a trajectory.
    
    Args:
        traj: PrimeTrajectory object
        name: Name of the trajectory for display
    """
    print(f"\n{'=' * 60}")
    print(f"{name} Information")
    print(f"{'=' * 60}")
    print(f"Dimension: {traj.dimension}")
    print(f"Number of steps: {traj.get_step_count()}")
    print(f"Prime mapping: {get_prime_mapping(traj.dimension)}")
    
    # Display axis names
    axis_names = [prime_to_axis_name(p) for p in generate_primes(100)[:traj.dimension]]
    print(f"Axes: {', '.join([f'{name}={prime}' for name, prime in zip(axis_names, generate_primes(100)[:traj.dimension])])}")
    
    # Get encoding
    encoding = traj.get_trajectory_encoding()
    print(f"\nTrajectory encoding (first 10 steps):")
    for i, enc in enumerate(encoding[:10]):
        factors = prime_factorization(enc)
        print(f"  Step {i}: {enc} = {factors}")
    
    if len(encoding) > 10:
        print(f"  ... ({len(encoding) - 10} more steps)")
    
    print(f"\nTrajectory hash: {traj.get_trajectory_hash()}")
    
    # Serialization
    traj_string = trajectory_to_string(encoding)
    print(f"\nSerialized (first 100 chars): {traj_string[:100]}...")


def demonstrate_basic_usage():
    """Demonstrate basic trajectory creation and manipulation."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Trajectory Creation and Encoding")
    print("=" * 80)
    
    # Create a simple 2D trajectory
    print("\n--- Creating a simple 2D square trajectory ---")
    square_points = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0]
    ], dtype=float)
    
    square_traj = create_trajectory_from_points(square_points)
    print_trajectory_info(square_traj, "Square Trajectory")
    
    # Create a 3D trajectory
    print("\n--- Creating a 3D cubic trajectory ---")
    cube_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 0, 1]
    ], dtype=float)
    
    cube_traj = create_trajectory_from_points(cube_points)
    print_trajectory_info(cube_traj, "Cube Trajectory")


def demonstrate_complex_trajectories():
    """Demonstrate complex trajectory patterns."""
    print("\n" + "=" * 80)
    print("DEMO 2: Complex Trajectory Patterns")
    print("=" * 80)
    
    # Spiral trajectory
    print("\n--- Creating a 3D spiral trajectory ---")
    spiral_points = create_spiral_trajectory(num_points=30, dimension=3)
    spiral_points = normalize_trajectory(spiral_points) * 10  # Scale for better visualization
    spiral_traj = create_trajectory_from_points(spiral_points)
    print_trajectory_info(spiral_traj, "Spiral Trajectory")
    
    # Compute trajectory properties
    length = compute_trajectory_length(spiral_points)
    curvature = compute_curvature(spiral_points)
    print(f"\nTrajectory length: {length:.2f}")
    print(f"Average curvature: {np.mean(curvature):.4f}")
    print(f"Max curvature: {np.max(curvature):.4f}")
    
    # Lissajous curve
    print("\n--- Creating a Lissajous curve trajectory ---")
    lissajous_points = create_lissajous_trajectory(num_points=50, a=3, b=2)
    lissajous_points = normalize_trajectory(lissajous_points) * 10
    lissajous_traj = create_trajectory_from_points(lissajous_points)
    print_trajectory_info(lissajous_traj, "Lissajous Trajectory")
    
    return spiral_points, lissajous_points


def demonstrate_trajectory_comparison():
    """Demonstrate trajectory comparison functionality."""
    print("\n" + "=" * 80)
    print("DEMO 3: Trajectory Comparison")
    print("=" * 80)
    
    # Create two similar trajectories
    print("\n--- Comparing similar trajectories ---")
    traj1_points = create_spiral_trajectory(num_points=25, dimension=3)
    traj1_points = normalize_trajectory(traj1_points) * 10
    traj1 = create_trajectory_from_points(traj1_points)
    
    # Slightly perturbed version
    np.random.seed(42)
    traj2_points = traj1_points + np.random.normal(0, 0.1, traj1_points.shape)
    traj2 = create_trajectory_from_points(traj2_points)
    
    comparison = compare_trajectories(traj1, traj2)
    print("\nComparison of similar trajectories:")
    for key, value in comparison.items():
        print(f"  {key}: {value:.4f}")
    
    # Create two different trajectories
    print("\n--- Comparing different trajectories ---")
    circle_points = create_circle_trajectory(num_points=25)
    circle_points = np.column_stack([circle_points, np.zeros(len(circle_points))])
    circle_traj = create_trajectory_from_points(circle_points)
    
    comparison2 = compare_trajectories(traj1, circle_traj)
    print("\nComparison of different trajectories:")
    for key, value in comparison2.items():
        print(f"  {key}: {value:.4f}")


def demonstrate_serialization():
    """Demonstrate trajectory serialization and deserialization."""
    print("\n" + "=" * 80)
    print("DEMO 4: Trajectory Serialization")
    print("=" * 80)
    
    # Create a trajectory
    points = create_spiral_trajectory(num_points=20, dimension=3)
    points = normalize_trajectory(points) * 10
    original_traj = create_trajectory_from_points(points)
    
    # Serialize
    encoding = original_traj.get_trajectory_encoding()
    serialized = trajectory_to_string(encoding)
    print(f"\nOriginal trajectory steps: {original_traj.get_step_count()}")
    print(f"Serialized length: {len(serialized)} characters")
    print(f"Serialized (first 200 chars): {serialized[:200]}...")
    
    # Deserialize
    deserialized_encoding = string_to_trajectory(serialized)
    print(f"\nDeserialized steps: {len(deserialized_encoding)}")
    print(f"Encoding match: {encoding == deserialized_encoding}")
    
    # Verify hash
    print(f"\nOriginal hash: {original_traj.get_trajectory_hash()}")


def demonstrate_high_dimensional():
    """Demonstrate trajectories in higher dimensions."""
    print("\n" + "=" * 80)
    print("DEMO 5: High-Dimensional Trajectories")
    print("=" * 80)
    
    for dim in [4, 5, 6]:
        print(f"\n--- {dim}D Trajectory ---")
        points = generate_random_trajectory(num_steps=15, dimension=dim, step_size=1.0, seed=42)
        traj = create_trajectory_from_points(points)
        
        print(f"Dimension: {dim}")
        print(f"Steps: {traj.get_step_count()}")
        print(f"Prime mapping: {get_prime_mapping(dim)}")
        
        # Show first few encodings
        encoding = traj.get_trajectory_encoding()
        print(f"First 3 step encodings:")
        for i, enc in enumerate(encoding[:3]):
            factors = prime_factorization(enc)
            print(f"  Step {i}: {enc} = {factors}")


def demonstrate_prime_factorization_details():
    """Demonstrate the prime factorization encoding in detail."""
    print("\n" + "=" * 80)
    print("DEMO 6: Prime Factorization Encoding Details")
    print("=" * 80)
    
    # Create a simple trajectory with known steps
    print("\n--- Manual trajectory construction ---")
    traj = PrimeTrajectory(dimension=3)
    
    # Add specific steps and show encoding
    steps = [
        np.array([1, 0, 0]),   # Move in X direction (prime 2)
        np.array([0, 1, 0]),   # Move in Y direction (prime 3)
        np.array([0, 0, 1]),   # Move in Z direction (prime 5)
        np.array([1, 1, 0]),   # Move in X and Y (primes 2 and 3)
        np.array([1, 1, 1]),   # Move in all directions (primes 2, 3, and 5)
    ]
    
    print("\nAdding steps and showing prime factorization:")
    for i, step in enumerate(steps):
        encoding = traj.add_step(step)
        factors = prime_factorization(encoding)
        print(f"Step {i}: direction={step} -> encoding={encoding} -> factors={factors}")
        
        # Explain the encoding
        explanation = []
        primes = generate_primes(100)
        for axis_idx, value in enumerate(step):
            if value != 0:
                axis_name = prime_to_axis_name(primes[axis_idx])
                explanation.append(f"{axis_name}({primes[axis_idx]})^{abs(int(value))}")
        print(f"         Explanation: {' × '.join(explanation) if explanation else '1 (no movement)'}")


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("PRIME TRAJECTORY DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo shows how discrete trajectories in N-dimensional space")
    print("can be uniquely encoded using prime factorizations.")
    print("\nKey concept: Each axis corresponds to a prime number:")
    print("  X-axis = 2 (first prime)")
    print("  Y-axis = 3 (second prime)")
    print("  Z-axis = 5 (third prime)")
    print("  W-axis = 7 (fourth prime), etc.")
    print("\nEach step in the trajectory is encoded as a product of prime powers,")
    print("making each trajectory uniquely identifiable by its factorization sequence.")
    
    try:
        # Run all demonstrations
        demonstrate_basic_usage()
        spiral_points, lissajous_points = demonstrate_complex_trajectories()
        demonstrate_trajectory_comparison()
        demonstrate_serialization()
        demonstrate_high_dimensional()
        demonstrate_prime_factorization_details()
        
        # Visualization
        print("\n" + "=" * 80)
        print("VISUALIZATION")
        print("=" * 80)
        print("\nGenerating visualizations...")
        
        # 2D visualization
        circle_points = create_circle_trajectory(num_points=30)
        visualize_trajectory_2d(circle_points, "2D Circle Trajectory")
        
        # 3D visualizations
        visualize_trajectory_3d(spiral_points, "3D Spiral Trajectory")
        visualize_trajectory_3d(lissajous_points, "3D Lissajous Trajectory")
        
        plt.tight_layout()
        plt.show()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("\nKey takeaways:")
        print("1. Trajectories are uniquely encoded using prime factorizations")
        print("2. Each dimension maps to a unique prime number")
        print("3. Trajectories can be serialized, compared, and analyzed")
        print("4. The system works in arbitrary dimensions")
        print("5. Prime factorization provides a unique 'fingerprint' for each trajectory")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
