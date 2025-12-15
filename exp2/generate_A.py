import numpy as np
from scipy.sparse import spdiags, eye, kron
from scipy.io import mmwrite
import argparse
import os

def generate_3d_matrix(n):
    """Generate 3D finite difference matrix for given dimension n"""
    # Parameters
    k = n**3
    r = 1 / (2 * n + 2)

    # Define tridiagonal matrix components
    diagx = 6 * np.ones(n)
    diagy = np.zeros(n)
    xz = (-1 + r) * np.ones(n)
    xf = (-1 - r) * np.ones(n)

    # Create sparse tridiagonal matrices tx, ty, tz
    tx = spdiags([xf, diagx, xz], [-1, 0, 1], n, n)
    ty = spdiags([xf, diagy, xz], [-1, 0, 1], n, n)
    tz = spdiags([xf, diagy, xz], [-1, 0, 1], n, n)

    # Kronecker product to form the system matrix a
    I = eye(n)
    a = (kron(kron(tx, I), I) + kron(kron(I, ty), I) + kron(kron(I, I), tz))
    
    return a

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate 3D finite difference matrix')
    parser.add_argument('-n', '--size', type=int, required=True, 
                       help='Dimension size n (matrix will be n^3 x n^3)')
    parser.add_argument('-o', '--output', type=str, 
                       help='Output filename (default: 3d{n}.mtx)')
    
    args = parser.parse_args()
    n = args.size
    
    # Set output filename
    if args.output:
        output_file = args.output
    else:
        output_file = f"exp2/3d{n}.mtx"
    
    print(f"Generating 3D matrix with dimension n={n} (matrix size: {n**3}x{n**3})")
    
    # Generate the matrix
    matrix = generate_3d_matrix(n)
    
    # Save the matrix in Matrix Market format
    print(f"Saving matrix to {output_file}")
    mmwrite(output_file, matrix)
    
    # Print matrix information
    print(f"Matrix saved successfully!")
    print(f"Matrix dimensions: {matrix.shape}")
    print(f"Number of non-zeros: {matrix.nnz}")
    print(f"Sparsity ratio: {1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.6f}")

if __name__ == "__main__":
    main()
