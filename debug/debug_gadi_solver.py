
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import scipy.sparse.linalg as spla
import sys
import time

def read_mtx(filename):
    return sio.mmread(filename).tocsr()

def solve_hss_inner_cg(A, b, alpha, omega, tol=1e-9, max_iter=1000):
    n = A.shape[0]
    x = np.zeros(n)
    
    print("Constructing H and S...")
    H = 0.5 * (A + A.T)
    S = 0.5 * (A - A.T)
    I = sp.eye(n, format='csr')
    
    print("Constructing operators...")
    AH = alpha * I + H
    AIMS = alpha * I - S
    AIPS = alpha * I + S
    
    # Operator for S step: M = (alpha*I - S)(alpha*I + S)
    def matvec_S_step(v):
        return AIMS @ (AIPS @ v)
    
    LO_S = spla.LinearOperator((n,n), matvec=matvec_S_step)
    
    norm_b = np.linalg.norm(b)
    
    history = []
    
    print(f"Starting HSS with alpha={alpha}, omega={omega}")
    start_time = time.time()
    
    for k in range(max_iter):
        r = b - A @ x
        res_norm = np.linalg.norm(r)
        rel_res = res_norm / norm_b
        history.append(rel_res)
        
        if k % 10 == 0:
            print(f"Iter {k}: Rel Res = {rel_res:.2e}")
            
        if rel_res < tol:
            print(f"Converged at iter {k} with Rel Res {rel_res:.2e}")
            break
            
        # Inner CG counters
        h_iters = [0]
        s_iters = [0]
        
        def cb_h(xk): h_iters[0] += 1
        def cb_s(xk): s_iters[0] += 1
        
        # Step 1: Solve (alpha I + H) r1 = r using CG
        # Inner CG 1
        try:
            r1, info1 = spla.cg(AH, r, rtol=1e-4, maxiter=100, callback=cb_h)
        except TypeError:
            r1, info1 = spla.cg(AH, r, tol=1e-4, maxiter=100, callback=cb_h)
        
        # Inner CG 2
        r2 = (2 - omega) * alpha * (AIMS @ r1)
        try:
            r3, info2 = spla.cg(LO_S, r2, rtol=1e-6, maxiter=100, callback=cb_s)
        except TypeError:
            r3, info2 = spla.cg(LO_S, r2, tol=1e-6, maxiter=100, callback=cb_s)
        
        x = x + r3
        
        if k % 10 == 0:
            print(f"  Inner H iters: {h_iters[0]}, Info: {info1}")
            print(f"  Inner S iters: {s_iters[0]}, Info: {info2}")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.4f}s")
    return x, history

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_gadi_solver.py <mtx_file> [alpha] [omega]")
        sys.exit(1)
        
    filename = sys.argv[1]
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    omega = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    
    print(f"Reading matrix {filename}...")
    A = read_mtx(filename)
    n = A.shape[0]
    b = A @ np.ones(n) # Use ones as exact solution
    
    solve_hss_inner_cg(A, b, alpha, omega)
