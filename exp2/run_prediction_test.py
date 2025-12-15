
import os
import sys
import subprocess
import re
import argparse
import time

def parse_predictions(filepath):
    predictions = {}
    current_precision = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Match header like "--- bf16 (LogX=True) ---"
            header_match = re.match(r'---\s+(\w+)\s+\(LogX=True\)\s+---', line)
            if header_match:
                current_precision = header_match.group(1)
                predictions[current_precision] = []
                continue
            
            # Match data like "N=960: alpha=1.447043 (std=0.035324)"
            data_match = re.match(r'N=(\d+):\s+alpha=([\d\.]+)', line)
            if data_match and current_precision:
                n = int(data_match.group(1))
                alpha = float(data_match.group(2))
                predictions[current_precision].append((n, alpha))
                
    return predictions

def run_command(cmd, env=None):
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Stderr: {e.stderr}")
        return f"Error: {e.stderr}"

def main():
    parser = argparse.ArgumentParser(description='Run prediction tests for Exp2')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    args = parser.parse_args()

    # Set GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pred_file = os.path.join(base_dir, "predictions_log.txt")
    output_file = os.path.join(base_dir, "prediction_test_results.txt")
    
    # Parse predictions
    if not os.path.exists(pred_file):
        print(f"Prediction file not found: {pred_file}")
        return

    predictions = parse_predictions(pred_file)
    
    # Clear output file
    with open(output_file, 'w') as f:
        f.write("Precision, N, Alpha, Output\n")
    
    w = 0.1
    
    for precision, values in predictions.items():
        executable = os.path.join(base_dir, f"gadi_{precision}")
        if not os.path.exists(executable):
            print(f"Executable not found: {executable}")
            continue
            
        for n, alpha in values:
            matrix_file = os.path.join(base_dir, f"3d{n}.mtx")
            
            # Generate matrix if needed
            if not os.path.exists(matrix_file):
                print(f"Generating matrix for N={n}...")
                gen_script = os.path.join(base_dir, "generate_A.py")
                # Use -o to specify absolute path to avoid CWD issues
                run_command(f"python {gen_script} -n {n} -o {matrix_file}")
            
            if not os.path.exists(matrix_file):
                print(f"Failed to generate matrix: {matrix_file}")
                continue
                
            # Run experiment
            cmd = f"{executable} {matrix_file} {alpha} {w}"
            output = run_command(cmd, env=env)
            
            # Save result immediately
            with open(output_file, 'a') as f:
                f.write(f"{precision}, {n}, {alpha}, {output}\n")
                f.flush()
            
            print(f"Finished {precision} N={n}")

if __name__ == "__main__":
    main()
