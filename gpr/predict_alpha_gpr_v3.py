
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
from scipy.optimize import curve_fit
import os

# Configuration
predictions = {
    'exp1': [960, 2560, 4096, 8192, 10000],
    'exp2': [180, 256, 320, 360, 400, 450, 512],
    'exp3': [960, 2560, 4096, 8192, 10000]
}

# Trend models
def asymptotic_model(x, a, b, c):
    # c + a / (x + b)
    return c + a / (x + b + 1e-5)

def power_model(x, a, b, c):
    # a * x^b + c
    return a * np.power(x, b) + c

def linear_model(x, a, b):
    return a * x + b

def remove_outliers(x, y, model_func, threshold=2.0):
    # Iterative outlier removal
    mask = np.ones(len(x), dtype=bool)
    
    # Initial guess & Bounds
    if model_func == asymptotic_model:
        p0 = [np.max(y) - np.min(y), 1.0, np.min(y)]
        bounds = ([0, -np.min(x)+1e-3, 0], [np.inf, np.inf, np.inf])
    elif model_func == power_model:
        p0 = [1e-4, 2.0, 0]
        bounds = ([0, 0, -np.inf], [np.inf, 5, np.inf])
    else:
        p0 = [0, np.mean(y)]
        bounds = (-np.inf, np.inf)

    for i in range(2): 
        x_curr = x[mask]
        y_curr = y[mask]
        
        if len(x_curr) < 4: break
        
        try:
            if bounds != (-np.inf, np.inf):
                popt, _ = curve_fit(model_func, x_curr, y_curr, p0=p0, bounds=bounds, maxfev=50000)
            else:
                popt, _ = curve_fit(model_func, x_curr, y_curr, p0=p0, maxfev=50000)
                
            y_pred = model_func(x, *popt)
            residuals = np.abs(y - y_pred)
            std_res = np.std(residuals[mask])
            
            if std_res == 0: break
            
            new_mask = residuals < threshold * std_res
            if np.sum(new_mask) < len(x) * 0.7:
                new_mask = residuals < (threshold * 2) * std_res
            
            mask = new_mask
            p0 = popt
        except Exception as e:
            break
            
    return mask

def process_experiment(exp_name, file_path, pred_points):
    print(f"Processing {exp_name}...")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    x_orig = df['N'].values
    
    precisions = ['bf16', 'fp32', 'fp64']
    
    for prec in precisions:
        col_name = f'{prec}_alpha'
        if col_name not in df.columns:
            continue
            
        y = df[col_name].values
        x = x_orig.copy()
        
        # Select Trend Model
        if exp_name == 'exp3':
            # Use Power Law for Exp3 to capture super-linear or linear growth safely
            trend_func = power_model
            trend_name = "PowerLaw"
            p0 = [1e-3, 1.5, 0]
            bounds = ([0, 0, -np.inf], [np.inf, 4, np.inf])
        else:
            trend_func = asymptotic_model
            trend_name = "Asymptotic"
            p0 = [100, 10, 0]
            bounds = ([0, -np.min(x)+1e-3, 0], [np.inf, np.inf, np.inf])

        # Special handling for Exp1 outliers
        if exp_name == 'exp1':
            if np.max(y) > 3.0:
                manual_mask = ~((x < 50) & (y < 2.0))
                x = x[manual_mask]
                y = y[manual_mask]

        # Automatic outlier removal
        mask = remove_outliers(x, y, trend_func)
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Fit Trend
        try:
            if bounds:
                popt, _ = curve_fit(trend_func, x_clean, y_clean, p0=p0, bounds=bounds, maxfev=50000)
            else:
                popt, _ = curve_fit(trend_func, x_clean, y_clean, p0=p0, maxfev=50000)
        except:
            print(f"Trend fit failed for {exp_name} {prec}, using linear fallback")
            trend_func = linear_model
            popt, _ = curve_fit(trend_func, x_clean, y_clean, maxfev=50000)
            trend_name = "Linear"

        trend_fit = trend_func(x_clean, *popt)
        residuals = y_clean - trend_fit
        
        # GPR
        l_scale = (max(x) - min(x)) / 5.0
        kernel = ConstantKernel(1.0) * Matern(length_scale=l_scale, nu=2.5) + WhiteKernel(noise_level=1e-2)
        
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True, random_state=42)
        gpr.fit(x_clean.reshape(-1, 1), residuals)
        
        # Prediction
        x_max_plot = max(max(pred_points), max(x))
        x_plot = np.linspace(min(x), x_max_plot, 500).reshape(-1, 1)
        
        trend_plot = trend_func(x_plot.ravel(), *popt)
        res_plot, sigma_plot = gpr.predict(x_plot, return_std=True)
        y_plot = trend_plot + res_plot
        
        # Target Prediction
        x_target = np.array(pred_points).reshape(-1, 1)
        trend_target = trend_func(x_target.ravel(), *popt)
        res_target, sigma_target = gpr.predict(x_target, return_std=True)
        y_target = trend_target + res_target
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='gray', label='Original Data', alpha=0.5)
        plt.scatter(x_clean, y_clean, color='red', label='Cleaned Data')
        plt.plot(x_plot, y_plot, color='blue', label=f'{trend_name} + GPR')
        plt.fill_between(x_plot.ravel(), 
                         y_plot - 1.96 * sigma_plot, 
                         y_plot + 1.96 * sigma_plot, 
                         color='blue', alpha=0.2, label='95% CI')
        
        plt.scatter(x_target, y_target, color='green', marker='x', s=100, label='Predictions')
        
        plt.title(f'{exp_name} {prec} - {trend_name} Trend + GPR')
        plt.xlabel('N')
        plt.ylabel('Alpha')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_filename = f'{exp_name}_{prec}_gpr.png'
        plt.savefig(plot_filename)
        plt.close()
        
        print(f"Saved plot to {plot_filename}")
        print(f"Predictions for {exp_name} {prec}:")
        for i, val in enumerate(y_target):
            print(f"  N={pred_points[i]}: {val:.4f}")

if __name__ == "__main__":
    process_experiment('exp1', 'exp1/search_results.txt', predictions['exp1'])
    process_experiment('exp2', 'exp2/search_results.txt', predictions['exp2'])
    process_experiment('exp3', 'exp3/search_results.txt', predictions['exp3'])
