import subprocess
import re
import os
import math
import sys

# Configuration
N_START = 12
N_END = 64
N_STEP = 4
W = 0.1
# EPSILON = 0.0001
CUDA_DEVICE = "5"

PRECISIONS = ["bf16", "fp32", "fp64"]
EXECUTABLES = {
    "bf16": "./gadi_bf16",
    "fp32": "./gadi_fp32",
    "fp64": "./gadi_fp64"
}
OUTPUT_PATTERNS = {
    "bf16": r"Half-precision iterations: (\d+)",
    "fp32": r"Single-precision iterations: (\d+)",
    "fp64": r"Double-precision iterations: (\d+)"
}

# Initial alphas for N=50
current_alphas = {
    "bf16": 0.3,
    "fp32": 0.2,
    "fp64": 0.6
}

def compile_code():
    print("Compiling code...")
    subprocess.run(["make"], cwd="exp2", check=True)

def generate_matrix(n):
    print(f"Generating matrix for N={n}...")
    # python exp2/generate_A.py <N> <EPSILON>
    script_path = os.path.join("exp2", "generate_A.py")
    subprocess.run(["python", script_path, "-n", str(n)], check=True)
    return os.path.join("exp2", f"3d{n}.mtx")

def run_solver(precision, matrix_file, alpha):
    exe = os.path.join("exp2", EXECUTABLES[precision])
    # Command: CUDA_VISIBLE_DEVICES=4 ./exp2/gadi_bf16 <matrix> <alpha> <w>
    cmd = [exe, matrix_file, str(alpha), str(W)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        output = result.stdout
        
        match = re.search(OUTPUT_PATTERNS[precision], output)
        if match:
            return int(match.group(1))
        else:
            print(f"Error: Could not parse iterations from output for {precision} alpha={alpha}")
            # print(output) # Reduce noise
            return 999999 # Return high number on failure
    except subprocess.CalledProcessError as e:
        print(f"Error running {precision} with alpha={alpha}: {e}")
        # print(e.stdout)
        # print(e.stderr)
        return 999999

def get_step_sizes(alpha):
    # 2 significant digits logic
    if alpha <= 1e-10: return 0.05, 0.005 # Fallback
    exponent = int(math.floor(math.log10(alpha)))
    fine_step = 10 ** (exponent - 1)
    coarse_step = fine_step * 5 
    return coarse_step, fine_step

def search_alpha(precision, matrix_file, start_alpha):
    print(f"Searching {precision} starting at {start_alpha}...")
    
    current_alpha = start_alpha
    coarse_step, fine_step = get_step_sizes(current_alpha)
    
    print(f"  Steps: Coarse={coarse_step}, Fine={fine_step}")
    
    # 1. Determine Direction & Coarse Search
    base_iter = run_solver(precision, matrix_file, current_alpha)
    print(f"  Base: alpha={current_alpha}, iter={base_iter}")
    current_iter = base_iter
    
    # Try + direction
    next_alpha = current_alpha + coarse_step
    next_iter = run_solver(precision, matrix_file, next_alpha)
    print(f"  Try +: alpha={next_alpha}, iter={next_iter}")
    
    direction = 0
    if next_iter < base_iter:
        direction = 1
        current_alpha = next_alpha
        current_iter = next_iter
    else:
        # Try - direction
        prev_alpha = current_alpha - coarse_step
        if prev_alpha > 0:
            prev_iter = run_solver(precision, matrix_file, prev_alpha)
            print(f"  Try -: alpha={prev_alpha}, iter={prev_iter}")
            if prev_iter < base_iter:
                direction = -1
                current_alpha = prev_alpha
                current_iter = prev_iter
    
    if direction != 0:
        print(f"  Direction found: {direction}")
        while True:
            next_val = current_alpha + (direction * coarse_step)
            if next_val <= 0: break
            
            iter_val = run_solver(precision, matrix_file, next_val)
            print(f"  Coarse step: alpha={next_val}, iter={iter_val}")
            
            if iter_val < current_iter:
                current_iter = iter_val
                current_alpha = next_val
                # Update step sizes dynamically? Maybe not for stability.
            else:
                # Found the valley bottom (roughly)
                break
    else:
        print("  Already at local minimum (coarse).")

    # 2. Fine Search
    # Search around current_alpha in range [current_alpha - coarse_step, current_alpha + coarse_step]
    # with fine_step
    print(f"  Fine search around {current_alpha}...")
    best_alpha = current_alpha
    best_iter = current_iter
    
    # We scan from -coarse_step to +coarse_step
    start_fine = current_alpha - coarse_step
    end_fine = current_alpha + coarse_step
    
    # Ensure positive
    start_fine = max(fine_step, start_fine)
    
    # Align start_fine to fine_step grid?
    # start_fine = round(start_fine / fine_step) * fine_step
    
    val = start_fine
    while val <= end_fine:
        if abs(val - current_alpha) < 1e-9: # Already computed (approx)
            val += fine_step
            continue
            
        iter_val = run_solver(precision, matrix_file, val)
        # print(f"    Fine: alpha={val}, iter={iter_val}")
        
        if iter_val < best_iter:
            best_iter = iter_val
            best_alpha = val
        
        val += fine_step
        
    print(f"  Found optimal: alpha={best_alpha}, iter={best_iter}")
    return best_alpha, best_iter

def main():
    compile_code()
    
    output_file = os.path.join("exp2", "search_results.txt")
    header = "N, bf16_alpha, bf16_iter, fp32_alpha, fp32_iter, fp64_alpha, fp64_iter"
    
    # Initialize file with header
    with open(output_file, "w") as f:
        f.write(header + "\n")
    print(f"Results will be saved to {output_file}")

    results = []
    
    for n in range(N_START, N_END + 1, N_STEP):
        print(f"\n=== Processing N={n} ===")
        matrix_file = generate_matrix(n)
        
        row = {"N": n}
        
        for prec in PRECISIONS:
            start_alpha = current_alphas[prec]
            best_alpha, best_iter = search_alpha(prec, matrix_file, start_alpha)
            
            current_alphas[prec] = best_alpha # Update for next N
            row[f"{prec}_alpha"] = best_alpha
            row[f"{prec}_iter"] = best_iter
            
        results.append(row)
        print(f"Result for N={n}: {row}")
        
        # Write immediately
        line = f"{row['N']}, {row['bf16_alpha']}, {row['bf16_iter']}, {row['fp32_alpha']}, {row['fp32_iter']}, {row['fp64_alpha']}, {row['fp64_iter']}"
        with open(output_file, "a") as f:
            f.write(line + "\n")
            f.flush()

    print("\n=== Final Results ===")
    print(header)
    for r in results:
        line = f"{r['N']}, {r['bf16_alpha']}, {r['bf16_iter']}, {r['fp32_alpha']}, {r['fp32_iter']}, {r['fp64_alpha']}, {r['fp64_iter']}"
        print(line)

if __name__ == "__main__":
    main()