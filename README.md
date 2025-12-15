# Numerical Experiments for Gadi Solver

This repository contains the source code and scripts for the numerical experiments presented in the paper. The experiments evaluate the performance of the proposed Gadi solver against baselines like cuDSS and Mixed-Precision GMRES.

## Prerequisites

### Hardware
- **NVIDIA GPU**: Compute Capability 8.0 or higher (e.g., NVIDIA A100) is recommended.
  - *Note: The Makefiles are currently configured with `-arch=sm_80`. Please adjust this flag in the Makefiles if you are using a different GPU architecture.*

### Software
- **CUDA Toolkit**: Version 12.4 or higher.
- **cuDSS Library**: NVIDIA cuDSS sparse linear solver library.
- **C++ Compiler**: Must support C++17 (e.g., `nvcc`, `g++`).
- **Python 3**: Required for running data generation and testing scripts.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Build the project:**
   A root `Makefile` is provided to compile all experiments and solvers at once.
   ```bash
   make
   ```
   This command will recursively build the executables in each subdirectory. All generated executables will have the `.exe` suffix.

   To clean the build artifacts:
   ```bash
   make clean
   ```

## Matrix Generation

Each experiment folder (`exp1`, `exp2`, `exp3`) contains a `generate_A.py` script to generate the test matrices.

### Experiment 1: Sylvester Equations (`exp1`)
Generates the coefficient matrix for a Sylvester equation.
```bash
python exp1/generate_A.py -n <dimension> -r <parameter_r>
```
- `-n`: Dimension of the matrix (resulting matrix size will be $n^2 \times n^2$).
- `-r`: Parameter $r$ (default: 1.0).
- **Output**: `exp1/sylve{n}.mtx`

### Experiment 2: 3D Finite Difference (`exp2`)
Generates a 3D finite difference matrix.
```bash
python exp2/generate_A.py -n <size> [-o <output_file>]
```
- `-n`: Dimension size $n$ (resulting matrix size will be $n^3 \times n^3$).
- `-o`: Output filename (optional, default: `exp2/3d{n}.mtx`).

### Experiment 3: Complex Reaction-Diffusion (`exp3`)
Generates a matrix for a complex reaction-diffusion problem.
```bash
python exp3/generate_A.py <n> [v_scale] [nu]
```
- `<n>`: Grid dimension (resulting matrix size will be $2n^2 \times 2n^2$).
- `[v_scale]`: Scale of the random potential $V(x)$ (default: 10000.0).
- `[nu]`: Diffusion coefficient (optional, auto-scaled if not provided).
- **Output**: `exp3/complex_diff_n{n}.mtx` (Note: script might default to `exp4` folder in some versions, please check script output).

## Usage & Repository Structure

The repository is organized into folders corresponding to different experiments and solvers mentioned in the paper.

### 1. Baseline: cuDSS (`cudss_fp32`, `cudss_fp64`)
These folders contain the benchmark code using NVIDIA's cuDSS solver in single (`fp32`) and double (`fp64`) precision.

**Usage:**
```bash
./cudss_fp32/cudss_solver.exe <matrix_file.mtx>
./cudss_fp64/cudss_solver.exe <matrix_file.mtx>
```

### 2. Numerical Experiments (`exp1`, `exp2`, `exp3`)
These folders correspond to the three main numerical experiments detailed in the paper. They contain the implementation of the proposed Gadi solver.

- **exp1**: Numerical Experiment 1 (Likely Sylvester equation benchmarks, see `sylve*.mtx`)
- **exp2**: Numerical Experiment 2 (Likely 3D problems, see `3d*.mtx`)
- **exp3**: Numerical Experiment 3 (Likely Complex Diffusion problems, see `complex_diff*.mtx`)

**Usage:**
The Gadi solver executables (`gadi_bf16.exe`, `gadi_fp32.exe`, `gadi_fp64.exe`) require the matrix file and two parameters: $\alpha$ and $\omega$.

```bash
./exp{N}/gadi_{precision}.exe <matrix_file.mtx> <alpha> <omega>
```

*Example:*
```bash
# Run Experiment 1 with FP32 precision, alpha=1.0, omega=0.5
./exp1/gadi_fp32.exe ./exp1/sylve108.mtx 1.0 0.5
```

### 3. Comparison: Mixed-Precision GMRES (`mix_gmres`)
This folder contains the Mixed-Precision GMRES solver used for comparison.

**Usage:**
```bash
./mix_gmres/mix_gmres_test.exe <matrix_file.mtx> mgs <restart> <max_iters> <rtol>
```

**Parameters:**
- `matrix_file.mtx`: Path to the matrix file in Matrix Market format.
- `mgs`: Modified Gram-Schmidt parameter (string).
- `restart`: Restart parameter for GMRES (integer).
- `max_iters`: Maximum number of iterations (integer).
- `rtol`: Relative tolerance for convergence (float).

*Example:*
```bash
./mix_gmres/mix_gmres_test.exe ./exp1/sylve108.mtx mgs 30 100 1e-6
```

## Folder Overview

| Folder | Description |
|--------|-------------|
| `cudss_fp32/` | Baseline cuDSS solver (Single Precision) |
| `cudss_fp64/` | Baseline cuDSS solver (Double Precision) |
| `exp1/` | **Experiment 1**: Benchmarks on Sylvester equations (`sylve*.mtx`) |
| `exp2/` | **Experiment 2**: Benchmarks on 3D problems (`3d*.mtx`) |
| `exp3/` | **Experiment 3**: Benchmarks on Complex Diffusion problems (`complex_diff*.mtx`) |
| `mix_gmres/` | Mixed-Precision GMRES implementation for comparison |
| `gpr/` | Gaussian Process Regression scripts (used for parameter prediction) |
| `debug/` | Debugging scripts |

## Citation

If you use this code in your research, please cite our paper:

```bibtex
<!-- @article{YourPaper2025,
  title={Title of the Paper},
  author={Author Names},
  journal={Journal Name},
  year={2025}
} -->
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

