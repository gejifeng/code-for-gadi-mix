import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import sys

def generate_complex_reaction_diffusion(n, v_scale=100.0, nu=1e-3):
    # Discretize -nu Delta u + i V(x) u = f on [0,1]^2
    h = 1.0 / (n - 1)
    N = n * n
    
    # nu is now passed as argument
    
    data = []
    rows = []
    cols = []
    
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            # -nu * Delta (Standard 5-point stencil)
            data.append(4.0 * nu / h**2)
            rows.append(idx)
            cols.append(idx)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n:
                    nidx = ni * n + nj
                    data.append(-1.0 * nu / h**2)
                    rows.append(idx)
                    cols.append(nidx)
                    
    L = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    
    # Large random potential V(x)
    np.random.seed(42)
    V_vals = v_scale * np.random.rand(N)
    V_mat = sp.diags(V_vals)
    
    # Construct Block Matrix for Complex System
    # (L + iV) u = f  =>  [ L  -V ] [ur] = [fr]
    #                     [ V   L ] [ui]   [fi]
    A = sp.bmat([[L, -V_mat], [V_mat, L]], format='csr')
    return A

if __name__ == "__main__":
    n = 64
    v_scale = 10000.0
    
    # Base configuration for "hard" problem
    # At n=64, nu=1e-3 provides a good challenge (Potential dominated)
    n_base = 64.0
    nu_base = 1e-3
    
    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    # Auto-scale nu to maintain difficulty
    # We want nu * n^2 to be roughly constant so the Laplacian term doesn't overwhelm V
    nu = nu_base * (n_base / n)**2

    if len(sys.argv) > 2:
        v_scale = float(sys.argv[2])
    if len(sys.argv) > 3:
        nu = float(sys.argv[3])
        
    print(f"Generating Complex RD matrix for n={n}, v_scale={v_scale}, nu={nu:.2e}")
    print(f"(Auto-scaled nu to maintain hardness relative to n=64 case)")
    A = generate_complex_reaction_diffusion(n, v_scale, nu)
    print(f"Matrix shape: {A.shape}, nnz: {A.nnz}")
    
    # Ensure output directory exists
    import os
    if not os.path.exists('exp3'):
        os.makedirs('exp3')

    sio.mmwrite(f"exp3/complex_diff_n{n}.mtx", A)
    print(f"Saved to exp3/complex_diff_n{n}.mtx")
