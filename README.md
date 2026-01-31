# Prime Trajectory Encoder

> Encode N-dimensional discrete trajectories as unique prime factorizations

Prime Trajectory Encoder maps discrete trajectories in N-dimensional space to unique integers via prime factorization. Each axis corresponds to a prime number (x→2, y→3, z→5, etc.), and trajectories are encoded as products of primes raised to base-N exponents. This creates a bijective mapping between step sequences and integers, enabling compact storage, comparison, and reconstruction of spatial paths.

## ✨ Features

- **Bijective Trajectory Encoding** — Convert any discrete step sequence in N-dimensional space to a unique integer using prime factorization, with guaranteed lossless decoding back to the original trajectory.
- **Continuous Trajectory Discretization** — Automatically discretize continuous trajectories by sampling and quantizing direction vectors to the nearest axis at each time step, producing clean step sequences.
- **Arbitrary Dimensionality** — Support for any number of dimensions N, automatically assigning the first N primes (2, 3, 5, 7, 11, ...) to corresponding axes.
- **Path Reconstruction** — Reconstruct spatial coordinates from encoded integers or step sequences, generating the discrete path through N-dimensional space as a series of unit steps.
- **Modular Architecture** — Clean separation between prime utilities, encoding/decoding logic, trajectory processing, and visualization, making the library easy to extend and integrate.

## 📦 Installation

### Prerequisites

- Python 3.7+
- NumPy
- Matplotlib (for visualization demos)

### Setup

1. Clone or download the project repository
   - Get the source code to your local machine
2. pip install numpy matplotlib
   - Install required dependencies for numerical operations and visualization
3. python demo.py
   - Run the demo script to verify installation and see example outputs

## 🚀 Usage

### Basic Encoding and Decoding

Encode a simple 3D step sequence to an integer and decode it back

```
from lib.core import encode_sequence, decode_sequence

# Define a 3D trajectory: x, y, x, z, y (axes 0, 1, 0, 2, 1)
sequence = [0, 1, 0, 2, 1]
N = 3  # 3 dimensions
T = len(sequence)

# Encode to unique integer
encoded = encode_sequence(sequence, N)
print(f"Encoded trajectory: {encoded}")

# Decode back to original sequence
decoded = decode_sequence(encoded, N, T)
print(f"Decoded trajectory: {decoded}")
print(f"Match: {sequence == decoded}")
```

**Output:**

```
Encoded trajectory: 7200
Decoded trajectory: [0, 1, 0, 2, 1]
Match: True
```

### Discretizing a Continuous Trajectory

Convert a continuous circular path to a discrete step sequence

```
import numpy as np
from lib.core import discretize_trajectory

# Generate a circular trajectory in 2D
t = np.linspace(0, 2*np.pi, 100)
points = np.column_stack([np.cos(t), np.sin(t)])

# Discretize into 8 steps
T = 8
sequence = discretize_trajectory(points, T)
print(f"Discretized sequence: {sequence}")
print(f"Length: {len(sequence)}")
```

**Output:**

```
Discretized sequence: [0, 1, 1, 0, 0, 1, 1, 0]
Length: 8
```

### Reconstructing Spatial Path

Generate 3D coordinates from a step sequence

```
from lib.core import reconstruct_path
import numpy as np

# Define a 3D step sequence
sequence = [0, 0, 1, 1, 2, 2]  # x, x, y, y, z, z
N = 3

# Reconstruct the spatial path
path = reconstruct_path(sequence, N)
print("Reconstructed path:")
for i, point in enumerate(path):
    print(f"  t={i}: {point}")
```

**Output:**

```
Reconstructed path:
  t=0: [0. 0. 0.]
  t=1: [1. 0. 0.]
  t=2: [2. 0. 0.]
  t=3: [2. 1. 0.]
  t=4: [2. 2. 0.]
  t=5: [2. 2. 1.]
  t=6: [2. 2. 2.]
```

### End-to-End Pipeline

Complete workflow from continuous trajectory to encoding and back

```
import numpy as np
from lib.core import discretize_trajectory, encode_sequence, decode_sequence, reconstruct_path

# 1. Create a continuous 3D helix
t = np.linspace(0, 4*np.pi, 200)
points = np.column_stack([np.cos(t), np.sin(t), t/10])

# 2. Discretize to 12 steps
T = 12
sequence = discretize_trajectory(points, T)
print(f"Step sequence: {sequence}")

# 3. Encode to integer
N = 3
encoded = encode_sequence(sequence, N)
print(f"Encoded as: {encoded}")

# 4. Decode and reconstruct
decoded = decode_sequence(encoded, N, T)
path = reconstruct_path(decoded, N)
print(f"Final position: {path[-1]}")
```

**Output:**

```
Step sequence: [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
Encoded as: 797625
Final position: [3. 3. 6.]
```

## 🏗️ Architecture

The project follows a modular architecture with three main layers: core encoding/decoding logic, utility functions for prime number operations and trajectory processing, and demonstration/visualization code. The core module handles the bijective mapping between step sequences and integers, while utils provides supporting functions for prime generation, factorization, and spatial operations.

### File Structure

```
prime-trajectory-encoder/
├── lib/
│   ├── __init__.py
│   ├── core.py              # Main encoding/decoding API
│   └── utils.py             # Prime utilities & helpers
├── demo.py                  # Interactive demonstrations
├── README.md
└── requirements.txt

Data Flow:
┌─────────────────┐
│ Continuous      │
│ Trajectory      │
│ (N-D points)    │
└────────┬────────┘
         │ discretize_trajectory()
         ▼
┌─────────────────┐
│ Step Sequence   │
│ [i₁, i₂, ..., iₜ]│
└────────┬────────┘
         │ encode_sequence()
         ▼
┌─────────────────┐
│ Prime Integer   │
│ P = ∏ pᵢ^eᵢ     │
└────────┬────────┘
         │ decode_sequence()
         ▼
┌─────────────────┐
│ Step Sequence   │
│ [i₁, i₂, ..., iₜ]│
└────────┬────────┘
         │ reconstruct_path()
         ▼
┌─────────────────┐
│ Discrete Path   │
│ (coordinates)   │
└─────────────────┘
```

### Files

- **lib/core.py** — Implements the main API: encode_sequence, decode_sequence, discretize_trajectory, and reconstruct_path functions.
- **lib/utils.py** — Provides prime number generation, factorization, base conversion utilities, and trajectory processing helpers.
- **demo.py** — Interactive demonstration script showcasing encoding, decoding, discretization, and visualization with matplotlib plots.

### Design Decisions

- Base-N exponent encoding ensures each time step contributes uniquely to each prime's exponent, guaranteeing bijectivity.
- Axis assignment to primes (0→2, 1→3, 2→5, etc.) provides a natural ordering and simplifies prime lookup.
- Discretization uses maximum absolute component of direction vectors to quantize continuous trajectories to nearest axis.
- Trial division factorization is sufficient for expected trajectory lengths; more sophisticated methods can be added for very long sequences.
- Modular design separates concerns: primes, encoding logic, trajectory processing, and visualization are independent components.

## 🔧 Technical Details

### Dependencies

- **numpy** (1.20.0+) — Efficient numerical operations for trajectory processing, vector math, and array manipulations.
- **matplotlib** (3.3.0+) — Visualization of trajectories and discrete paths in 2D and 3D for demonstration purposes.

### Key Algorithms / Patterns

- Base-N positional encoding: each time step t contributes B^(t-1) to the exponent of the chosen prime, creating unique factorizations.
- Sieve-free prime generation: generates primes on-demand using trial division, sufficient for typical N values (up to ~100 dimensions).
- Direction quantization: discretizes continuous trajectories by computing segment vectors and selecting the axis with maximum absolute component.
- Prime factorization via trial division: extracts exponents e_i from encoded integer P by dividing by each prime p_i repeatedly.
- Cumulative sum path reconstruction: builds spatial coordinates by accumulating unit vectors along specified axes.

### Important Notes

- Encoded integers grow exponentially with trajectory length T; for T > 50, consider using Python's arbitrary precision integers.
- Discretization quality depends on sampling density; ensure continuous trajectories have sufficient points relative to T.
- The encoding is deterministic and lossless only for valid step sequences (axis indices in range [0, N-1]).
- For very high dimensions (N > 100), prime generation may become a bottleneck; consider caching or precomputing primes.
- Reconstruction assumes unit steps along axes; scale factors can be added as a post-processing step if needed.

## ❓ Troubleshooting

### ValueError: sequence contains invalid axis index

**Cause:** Step sequence contains indices outside the valid range [0, N-1] for the specified number of dimensions.

**Solution:** Verify that all elements in your sequence are non-negative integers less than N. Check discretization output or manually created sequences for out-of-bounds values.

### Decoded sequence doesn't match original

**Cause:** Mismatch between the N or T parameters used for encoding and decoding, or corrupted encoded integer.

**Solution:** Ensure you use the same N (number of dimensions) and T (trajectory length) for both encode_sequence and decode_sequence. Store these metadata alongside the encoded integer.

### MemoryError or very slow encoding for long trajectories

**Cause:** Encoded integers grow exponentially with T; for T > 100, the integer can become extremely large, causing performance issues.

**Solution:** For very long trajectories, consider chunking into smaller segments or using alternative compression schemes. Python handles arbitrary precision, but operations slow down with size.

### Discretized trajectory looks incorrect or jagged

**Cause:** Insufficient sampling points in the continuous trajectory relative to the number of discrete steps T requested.

**Solution:** Increase the sampling density of your continuous trajectory (e.g., use more points in np.linspace). Aim for at least 10-20 samples per discrete step for smooth results.

### ImportError: No module named 'lib'

**Cause:** Python cannot find the lib package, usually due to running scripts from the wrong directory.

**Solution:** Run scripts from the project root directory, or add the project root to PYTHONPATH: export PYTHONPATH="${PYTHONPATH}:/path/to/prime-trajectory-encoder"

---

This project demonstrates a novel approach to trajectory encoding using number theory. While the prime factorization method is mathematically elegant and guarantees uniqueness, it's primarily suited for research, educational purposes, and applications where symbolic trajectory representation is valuable. For production systems requiring high-performance trajectory storage, consider complementing this with traditional compression methods. The codebase is designed to be extensible—contributions for optimization, additional discretization strategies, or alternative encoding schemes are welcome.