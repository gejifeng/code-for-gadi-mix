
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic
from scipy.optimize import curve_fit
import os

# Configuration
experiments = [
    {
        "name": "exp1",
        "file": "exp1/search_results.txt",
        "targets": [960, 2560, 4096, 8192, 10000],
        "trend_type": "inverse", # decreasing
        "outlier_threshold": 2.0,
        "manual_outliers": lambda df: df[df['N'] >= 48] # Exp1 specific: remove small N
    },
    {
        "name": "exp2",
        "file": "exp2/search_results.txt",
        "targets": [180, 256, 320, 360, 400, 450, 512],
        "trend_type": "inverse", # decreasing
        "outlier_threshold": 2.0,
        "manual_outliers": lambda df: df # No manual removal
    },
    {
        "name": "exp3",
        "file": "exp3/search_results.txt",
        "targets": [960, 2560, 4096, 8192, 10000],
        "trend_type": "poly", # increasing
        "outlier_threshold": 2.0,
        "manual_outliers": lambda df: df
    }
]

precisions = ['bf16', 'fp32', 'fp64']

# Trend Models
def asymptotic_model(x, a, b, c):
    # c + a / (x + b)
    return c + a / (x + b)

def power_model(x, a, b, c):
    # a * x^b + c
    return a * np.power(x, b) + c

def linear_model(x, a, b):
    return a * x + b

def log_linear_model(x, a, b):
    # a * log(x) + b
    return a * np.log(x) + b

def fit_trend(x, y, model_type='inverse'):
    if model_type == 'inverse':
        # Try multiple initial guesses
        guesses = [
            [100.0, 0.0, 0.0],   # a large, b small
            [1000.0, 100.0, 0.0], # a very large, b large
            [1.0, 1.0, 0.0],     # default
            [-100.0, 0.0, 5.0]   # decreasing
        ]
        
        best_popt = None
        best_err = np.inf
        
        for p0 in guesses:
            try:
                popt, _ = curve_fit(asymptotic_model, x, y, p0=p0, maxfev=10000)
                # Calculate error
                y_est = asymptotic_model(x, *popt)
                err = np.sum((y - y_est)**2)
                if err < best_err:
                    best_err = err
                    best_popt = popt
            except:
                continue
                
        if best_popt is not None:
            return lambda x_new: asymptotic_model(x_new, *best_popt), best_popt, "asymptotic"
        else:
            # Fallback to linear if inverse fails
            z = np.polyfit(x, y, 1)
            return lambda x_new: np.polyval(z, x_new), z, "linear_fallback"
            
    elif model_type == 'poly':
        # Try power law first
        try:
            # p0 = [scale, power, offset]
            p0 = [0.1, 1.0, 0.0]
            popt, _ = curve_fit(power_model, x, y, p0=p0, maxfev=20000)
            return lambda x_new: power_model(x_new, *popt), popt, "power"
        except:
            # Fallback to linear
            z = np.polyfit(x, y, 1)
            return lambda x_new: np.polyval(z, x_new), z, "linear"
    
    return lambda x_new: np.zeros_like(x_new), [], "none"

def remove_outliers(x, y, trend_func, threshold=2.0):
    y_pred = trend_func(x)
    residuals = y - y_pred
    std_res = np.std(residuals)
    if std_res < 1e-9:
        return x, y, np.zeros_like(x, dtype=bool)
    
    z_scores = np.abs(residuals / std_res)
    mask = z_scores < threshold
    
    # Safety check: if we are removing too many points (>50%), assume the fit was bad and keep all
    if np.sum(mask) < 0.5 * len(x):
        print("    Warning: Outlier removal would drop >50% of data. Keeping all.")
        return x, y, np.zeros_like(x, dtype=bool)
        
    return x[mask], y[mask], ~mask

def run_gpr(x_train, y_train, x_target, trend_func, use_log_x=False):
    # Prepare data
    X = x_train.reshape(-1, 1)
    y = y_train
    
    X_target = x_target.reshape(-1, 1)
    
    if use_log_x:
        X = np.log(X)
        X_target = np.log(X_target)
    
    # Calculate residuals from trend
    trend_train = trend_func(x_train)
    residuals = y - trend_train
    
    # Kernel definition
    # Mix of RBF (smooth), Matern (rougher), and WhiteKernel (noise)
    k1 = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5)
    k2 = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=10.0)
    kernel = k1 + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, normalize_y=True, random_state=42)
    gpr.fit(X, residuals)
    
    # Predict
    trend_target = trend_func(x_target)
    res_pred, sigma = gpr.predict(X_target, return_std=True)
    
    y_pred = trend_target + res_pred
    
    # Also get prediction for training range for plotting
    x_plot = np.linspace(min(x_train.min(), x_target.min()), max(x_train.max(), x_target.max()), 500)
    X_plot = x_plot.reshape(-1, 1)
    if use_log_x:
        X_plot = np.log(X_plot)
        
    trend_plot = trend_func(x_plot)
    res_plot, sigma_plot = gpr.predict(X_plot, return_std=True)
    y_plot = trend_plot + res_plot
    
    return y_pred, sigma, x_plot, y_plot, sigma_plot

def process_experiment(exp_config, use_log_x=False):
    print(f"Processing {exp_config['name']} (Log X: {use_log_x})...")
    
    # Load data
    try:
        df = pd.read_csv(exp_config['file'])
        # Strip whitespace from columns
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"Error reading {exp_config['file']}: {e}")
        return

    # Apply manual outlier removal (e.g. N < 48 for exp1)
    df = exp_config['manual_outliers'](df)
    
    results_txt = []
    
    for precision in precisions:
        col_name = f"{precision}_alpha"
        if col_name not in df.columns:
            continue
            
        print(f"  Precision: {precision}")
        
        x_data = df['N'].values
        y_data = df[col_name].values
        
        # 1. Fit Trend
        trend_func, params, trend_name = fit_trend(x_data, y_data, exp_config['trend_type'])
        
        # 2. Remove Statistical Outliers
        x_clean, y_clean, mask_outliers = remove_outliers(x_data, y_data, trend_func, exp_config['outlier_threshold'])
        
        if np.sum(mask_outliers) > 0:
            print(f"    Removed {np.sum(mask_outliers)} outliers.")
            # Refit trend on clean data
            trend_func, params, trend_name = fit_trend(x_clean, y_clean, exp_config['trend_type'])
        
        # 3. Run GPR
        targets = np.array(exp_config['targets'])
        y_pred, sigma, x_plot, y_plot, sigma_plot = run_gpr(x_clean, y_clean, targets, trend_func, use_log_x)
        
        # 4. Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, y_data, c='red', label='Data', alpha=0.5)
        plt.scatter(x_clean, y_clean, c='blue', label='Clean Data', marker='x')
        
        plt.plot(x_plot, y_plot, 'g-', label='GPR Fit')
        plt.fill_between(x_plot, y_plot - 1.96*sigma_plot, y_plot + 1.96*sigma_plot, color='green', alpha=0.2)
        
        plt.scatter(targets, y_pred, c='purple', marker='*', s=100, label='Predictions')
        
        plt.title(f"{exp_config['name']} - {precision} (LogX={use_log_x})")
        plt.xlabel("N")
        plt.ylabel("Alpha")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        suffix = "_log" if use_log_x else ""
        plot_filename = f"{exp_config['name']}/{precision}_fit{suffix}.png"
        plt.savefig(plot_filename)
        plt.close()
        
        # 5. Collect Results
        results_txt.append(f"--- {precision} (LogX={use_log_x}) ---")
        for t, p, s in zip(targets, y_pred, sigma):
            results_txt.append(f"N={t}: alpha={p:.6f} (std={s:.6f})")
            
    # Save results to text file
    suffix = "_log" if use_log_x else ""
    with open(f"{exp_config['name']}/predictions{suffix}.txt", "w") as f:
        f.write("\n".join(results_txt))

def main():
    for exp in experiments:
        # Standard version
        process_experiment(exp, use_log_x=False)
        # Log regularization version
        process_experiment(exp, use_log_x=True)

if __name__ == "__main__":
    main()
