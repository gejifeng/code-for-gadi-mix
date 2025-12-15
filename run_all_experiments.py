import os
import sys
import subprocess
import re
import shutil
import time

# Configuration
EXP1_SIZES = [960, 2560, 4096, 8192, 10000]
EXP2_SIZES = [180, 256, 320, 360, 400, 450, 512]
EXP3_SIZES = [1024, 2048, 4096, 5120, 8192]

# Paths
ROOT_DIR = os.getcwd()
RESULTS_FILE = os.path.join(ROOT_DIR, "all_experiments_results.txt")

def log(message):
    print(message)
    with open(RESULTS_FILE, "a") as f:
        f.write(message + "\n")

def check_prerequisites():
    log("=== Checking Prerequisites ===")
    
    # Check Python
    log(f"Python version: {sys.version}")
    
    # Check NumPy/SciPy
    try:
        import numpy
        import scipy
        log(f"NumPy version: {numpy.__version__}")
        log(f"SciPy version: {scipy.__version__}")
    except ImportError as e:
        log(f"Error: Missing required Python packages. {e}")
        sys.exit(1)
        
    # Check GPU
    try:
        nvidia_smi = subprocess.check_output("nvidia-smi", shell=True).decode()
        log("GPU detected:")
        # Extract GPU name (simple heuristic)
        for line in nvidia_smi.split('\n'):
            if "NVIDIA" in line and "MiB" in line: # Typical header line or GPU line
                log(line.strip())
                break
    except subprocess.CalledProcessError:
        log("Error: nvidia-smi not found or failed. Is a GPU available?")
        sys.exit(1)

    # Compile Software
    log("Compiling software...")
    try:
        subprocess.check_call(["make", "clean"], cwd=ROOT_DIR)
        subprocess.check_call(["make", "-j"], cwd=ROOT_DIR)
        log("Compilation successful.")
    except subprocess.CalledProcessError:
        log("Error: Compilation failed.")
        sys.exit(1)

def parse_output(output):
    """
    Parses stdout to extract time, success status, and memory usage.
    Adapts to different output formats from cudss, mix_gmres, gadi.
    """
    time_sec = "N/A"
    memory_mb = "N/A"
    success = "Unknown"
    
    # --- Time Parsing ---
    # 1. Try to find Chinese "耗时: X ms" (common in cudss output)
    # We sum up all occurrences (e.g. factorization + solve)
    chinese_time_matches = re.findall(r"耗时:\s*(\d+)\s*ms", output)
    if chinese_time_matches:
        total_ms = sum(float(t) for t in chinese_time_matches)
        time_sec = f"{total_ms / 1000.0:.4f}"
    else:
        # 2. Fallback to English patterns
        time_match = re.search(r"(?:Time|Duration|Elapsed).*?(\d+\.?\d*)\s*(?:s|ms|sec)", output, re.IGNORECASE)
        if time_match:
            val = float(time_match.group(1))
            if "ms" in time_match.group(0).lower():
                val /= 1000.0
            time_sec = f"{val:.4f}"
    
    # --- Memory Parsing ---
    # Find all memory usage reports and take the maximum
    # Matches "Used=123" or "Memory Used: 123" or "Used: 123"
    mem_matches = re.findall(r"(?:Used=|Memory Used:|Used:)\s*(\d+\.?\d*)", output, re.IGNORECASE)
    if mem_matches:
        # Convert all to floats and find max
        max_mem = max(float(m) for m in mem_matches)
        memory_mb = f"{max_mem:.2f}"
        
    # --- Success Parsing ---
    output_lower = output.lower()
    if "cudss error" in output_lower:
        success = "No"
    elif "=== 求解完成 ===" in output:
        success = "Yes"
    elif "res =" in output_lower: # Gadi success pattern
        success = "Yes"
    elif "converged" in output_lower or "success" in output_lower:
        success = "Yes"
    elif "failed" in output_lower or "not converged" in output_lower:
        success = "No"
        
    return time_sec, success, memory_mb

def run_command(cmd_args, cwd=ROOT_DIR):
    try:
        result = subprocess.run(
            cmd_args, 
            cwd=cwd,
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            check=False
        )
        return result.stdout, result.returncode
    except Exception as e:
        return str(e), -1

def load_alpha_map(filename):
    mapping = {}
    if not os.path.exists(filename):
        log(f"Warning: Alpha map file {filename} not found.")
        return mapping
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # Assuming format: size alpha
                    try:
                        size = int(parts[0])
                        alpha = float(parts[1])
                        mapping[size] = alpha
                    except ValueError:
                        continue
    except Exception as e:
        log(f"Error reading {filename}: {e}")
    return mapping

def run_experiment_1():
    log("\n=== Running Experiment 1 (2D Convection-Diffusion) ===")
    exp_dir = os.path.join(ROOT_DIR, "exp1")
    
    for size in EXP1_SIZES:
        log(f"\n--- Size: {size} ---")
        
        # 1. Generate Matrix
        mtx_file = os.path.join(exp_dir, f"sylve{size}.mtx")
        if not os.path.exists(mtx_file):
            log(f"Generating matrix for size {size}...")
            # exp1/generate_A.py -n <size> -r 1.0 (default)
            cmd = ["python", "exp1/generate_A.py", "-n", str(size)]
            out, rc = run_command(cmd)
            if rc != 0:
                log(f"Matrix generation failed:\n{out}")
                continue
        
        # 2. cuDSS FP64
        log("Running cuDSS FP64...")
        cmd = ["./cudss_fp64/cudss_solver.exe", mtx_file]
        out, rc = run_command(cmd)
        t, s, m = parse_output(out)
        log(f"cuDSS FP64: Time={t}s, Success={s}, Mem={m}MB")
        
        # 3. Mix GMRES
        log("Running Mix GMRES...")
        # mix_gmres_test.exe <mtx> <mgs> <restart> <max_iters> <rtol>
        # mgs=1 (MGS), restart=50, max_iter=20000, rtol=1e-10
        cmd = ["./mix_gmres/mix_gmres_test.exe", mtx_file, "1", "50", "20000", "1e-10"]
        out, rc = run_command(cmd)
        t, s, m = parse_output(out)
        log(f"Mix GMRES: Time={t}s, Success={s}, Mem={m}MB")
        
        # 4. Gadi (Mixed Precision)
        # User said: "use mixed precision gadi... all omega 0.1, all alpha 0.95"
        # We run BF16 and FP32 variants as "mixed precision".
        alpha = 0.95
        omega = 0.1
        
        for prec in ["bf16", "fp32", "fp64"]:
            exe = os.path.join(exp_dir, f"gadi_{prec}.exe")
            if not os.path.exists(exe): continue
            
            log(f"Running Gadi {prec.upper()}...")
            # gadi_xx.exe <mtx> <alpha> <omega>
            cmd = [exe, mtx_file, str(alpha), str(omega)]
            out, rc = run_command(cmd)
            t, s, m = parse_output(out)
            log(f"Gadi {prec.upper()}: Time={t}s, Success={s}, Mem={m}MB")

def run_experiment_2():
    log("\n=== Running Experiment 2 (3D) ===")
    exp_dir = os.path.join(ROOT_DIR, "exp2")
    
    # Load Alpha Maps
    float_map = load_alpha_map(os.path.join(ROOT_DIR, "tmp", "float_info_exp2.txt"))
    double_map = load_alpha_map(os.path.join(ROOT_DIR, "tmp", "double_info_exp2.txt"))
    
    for size in EXP2_SIZES:
        log(f"\n--- Size: {size} ---")
        
        # 1. Generate Matrix
        mtx_file = os.path.join(exp_dir, f"3d{size}.mtx")
        if not os.path.exists(mtx_file):
            log(f"Generating matrix for size {size}...")
            # exp2/generate_A.py -n <size>
            cmd = ["python", "exp2/generate_A.py", "-n", str(size)]
            out, rc = run_command(cmd)
            if rc != 0:
                log(f"Matrix generation failed:\n{out}")
                continue

        # 2. cuDSS FP32 (Single Precision)
        log("Running cuDSS FP32...")
        cmd = ["./cudss_fp32/cudss_solver.exe", mtx_file]
        out, rc = run_command(cmd)
        t, s, m = parse_output(out)
        log(f"cuDSS FP32: Time={t}s, Success={s}, Mem={m}MB")
        
        # 3. Mix GMRES (rtol=1e-6)
        log("Running Mix GMRES...")
        # mgs=1, restart=50, max_iter=20000, rtol=1e-6
        cmd = ["./mix_gmres/mix_gmres_test.exe", mtx_file, "1", "50", "20000", "1e-6"]
        out, rc = run_command(cmd)
        t, s, m = parse_output(out)
        log(f"Mix GMRES: Time={t}s, Success={s}, Mem={m}MB")
        
        # 4. Gadi
        omega = 0.1
        
        # Gadi FP32
        alpha_fp32 = float_map.get(size, 0.95) # Default if missing?
        log(f"Running Gadi FP32 (alpha={alpha_fp32})...")
        cmd = [os.path.join(exp_dir, "gadi_fp32.exe"), mtx_file, str(alpha_fp32), str(omega)]
        out, rc = run_command(cmd)
        t, s, m = parse_output(out)
        log(f"Gadi FP32: Time={t}s, Success={s}, Mem={m}MB")
        
        # Gadi FP64
        alpha_fp64 = double_map.get(size, 0.95)
        log(f"Running Gadi FP64 (alpha={alpha_fp64})...")
        cmd = [os.path.join(exp_dir, "gadi_fp64.exe"), mtx_file, str(alpha_fp64), str(omega)]
        out, rc = run_command(cmd)
        t, s, m = parse_output(out)
        log(f"Gadi FP64: Time={t}s, Success={s}, Mem={m}MB")
        
        # Gadi BF16
        if size <= 450:
            alpha_bf16 = 0.003
        else:
            alpha_bf16 = 0.002
        log(f"Running Gadi BF16 (alpha={alpha_bf16})...")
        cmd = [os.path.join(exp_dir, "gadi_bf16.exe"), mtx_file, str(alpha_bf16), str(omega)]
        out, rc = run_command(cmd)
        t, s, m = parse_output(out)
        log(f"Gadi BF16: Time={t}s, Success={s}, Mem={m}MB")

def run_experiment_3():
    log("\n=== Running Experiment 3 (Complex Diffusion) ===")
    exp_dir = os.path.join(ROOT_DIR, "exp3")
    
    for size in EXP3_SIZES:
        log(f"\n--- Size: {size} ---")
        
        # 1. Generate Matrix
        # exp3/generate_A.py <n>
        # Output is complex_diff_n{n}.mtx
        mtx_file = os.path.join(exp_dir, f"complex_diff_n{size}.mtx")
        if not os.path.exists(mtx_file):
            log(f"Generating matrix for size {size}...")
            cmd = ["python", "exp3/generate_A.py", str(size)]
            out, rc = run_command(cmd)
            if rc != 0:
                log(f"Matrix generation failed:\n{out}")
                continue
        
        # 2. cuDSS FP32
        log("Running cuDSS FP32...")
        cmd = ["./cudss_fp32/cudss_solver.exe", mtx_file]
        out, rc = run_command(cmd)
        t, s, m = parse_output(out)
        log(f"cuDSS FP32: Time={t}s, Success={s}, Mem={m}MB")
        
        # 3. Mix GMRES (rtol=1e-6)
        log("Running Mix GMRES...")
        cmd = ["./mix_gmres/mix_gmres_test.exe", mtx_file, "1", "50", "20000", "1e-6"]
        out, rc = run_command(cmd)
        t, s, m = parse_output(out)
        log(f"Mix GMRES: Time={t}s, Success={s}, Mem={m}MB")
        
        # 4. Gadi
        omega = 0.1
        
        # Gadi FP32 (alpha=300)
        log(f"Running Gadi FP32 (alpha=300)...")
        cmd = [os.path.join(exp_dir, "gadi_fp32.exe"), mtx_file, "300", str(omega)]
        out, rc = run_command(cmd)
        t, s, m = parse_output(out)
        log(f"Gadi FP32: Time={t}s, Success={s}, Mem={m}MB")
        
        # Gadi FP64 (alpha=300)
        log(f"Running Gadi FP64 (alpha=300)...")
        cmd = [os.path.join(exp_dir, "gadi_fp64.exe"), mtx_file, "300", str(omega)]
        out, rc = run_command(cmd)
        t, s, m = parse_output(out)
        log(f"Gadi FP64: Time={t}s, Success={s}, Mem={m}MB")
        
        # Gadi BF16 (alpha=270)
        log(f"Running Gadi BF16 (alpha=270)...")
        cmd = [os.path.join(exp_dir, "gadi_bf16.exe"), mtx_file, "270", str(omega)]
        out, rc = run_command(cmd)
        t, s, m = parse_output(out)
        log(f"Gadi BF16: Time={t}s, Success={s}, Mem={m}MB")

def main():
    # Clear previous results
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
        
    log("Starting One-Click Test Script")
    log("==============================")
    
    check_prerequisites()
    
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    
    log("\nAll experiments completed. Results saved to " + RESULTS_FILE)

if __name__ == "__main__":
    main()
