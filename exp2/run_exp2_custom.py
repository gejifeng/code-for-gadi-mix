import os
import sys
import subprocess
import re

# Configuration
EXP2_SIZES = [180, 256, 320, 360, 400, 450, 512]

# Alpha values hardcoded from tmp/float_info_exp2.txt and tmp/double_info_exp2.txt
ALPHA_MAP = {
    180: 0.003348,
    256: 0.002597,
    320: 0.002243,
    360: 0.002086,
    400: 0.001961,
    450: 0.001836,
    512: 0.001714
}

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
RESULTS_FILE = os.path.join(CURRENT_DIR, "exp2_custom_results.txt")

def log(message):
    print(message)
    with open(RESULTS_FILE, "a") as f:
        f.write(message + "\n")

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

def run_command(cmd_args, cwd=CURRENT_DIR):
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

def main():
    # Clear previous results
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
        
    log("=== Running Experiment 2 (Custom Gadi Sequence) ===")
    log(f"Results will be saved to {RESULTS_FILE}")

    # Compile
    log("Compiling executables...")
    out, rc = run_command(["make"], cwd=CURRENT_DIR)
    if rc != 0:
        log(f"Compilation failed:\n{out}")
        sys.exit(1)
    
    # Use hardcoded Alpha Maps
    log("Using hardcoded alpha values.")
    float_map = ALPHA_MAP
    double_map = ALPHA_MAP
    
    omega = 0.1
    
    for size in EXP2_SIZES:
        log(f"\n--- Size: {size} ---")
        
        # 1. Generate Matrix
        mtx_file = os.path.join(CURRENT_DIR, f"3d{size}.mtx")
        if not os.path.exists(mtx_file):
            log(f"Generating matrix for size {size}...")
            # exp2/generate_A.py -n <size>
            cmd = ["python", "generate_A.py", "-n", str(size)]
            out, rc = run_command(cmd)
            if rc != 0:
                log(f"Matrix generation failed:\n{out}")
                continue

        # Order: FP64 -> FP32 -> BF16
        
        # 1. Gadi FP64
        alpha_fp64 = double_map.get(size, 0.95)
        log(f"Running Gadi FP64 (alpha={alpha_fp64}, omega={omega})...")
        cmd = ["./gadi_fp64.exe", mtx_file, str(alpha_fp64), str(omega)]
        out, rc = run_command(cmd)
        t, s, m = parse_output(out)
        log(f"Gadi FP64: Time={t}s, Success={s}, Mem={m}MB")
        
        # 2. Gadi FP32 (Mixed Precision)
        alpha_fp32 = float_map.get(size, 0.95)
        log(f"Running Gadi FP32 (alpha={alpha_fp32}, omega={omega})...")
        cmd = ["./gadi_fp32.exe", mtx_file, str(alpha_fp32), str(omega)]
        out, rc = run_command(cmd)
        t, s, m = parse_output(out)
        log(f"Gadi FP32: Time={t}s, Success={s}, Mem={m}MB")
        
        # 3. Gadi BF16 (Mixed Precision)
        # Logic from run_all_experiments.py
        if size <= 450:
            alpha_bf16 = 0.003
        else:
            alpha_bf16 = 0.002
        log(f"Running Gadi BF16 (alpha={alpha_bf16}, omega={omega})...")
        cmd = ["./gadi_bf16.exe", mtx_file, str(alpha_bf16), str(omega)]
        out, rc = run_command(cmd)
        t, s, m = parse_output(out)
        log(f"Gadi BF16: Time={t}s, Success={s}, Mem={m}MB")

    log("\nExperiment 2 completed.")

if __name__ == "__main__":
    main()
