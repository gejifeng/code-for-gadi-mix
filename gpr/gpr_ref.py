# data
# 3d8.mtx: alpha=0.0735, 
# 3d12.mtx: alpha=0.0452, 
# 3d16.mtx: alpha=0.0328, 
# 3d20.mtx: alpha=0.0245, 
# 3d24.mtx: alpha=0.0192, 
# 3d28.mtx: alpha=0.0189, 
# 3d32.mtx: alpha=0.0171, 
# 3d36.mtx: alpha=0.0152, 
# 3d40.mtx: alpha=0.0124, 
# 3d44.mtx: alpha=0.0106, 
# 3d48.mtx: alpha=0.0108, 
# 3d56.mtx: alpha=0.0086, 
# 3d60.mtx: alpha=0.0078, 
# 3d64.mtx: alpha=0.0085,

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.optimize import curve_fit

# ===============================
# 1. 原始数据
# ===============================
xx = np.array([8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 60, 64])
alpha = np.array([0.0735, 0.0452, 0.0328, 0.0245, 0.0192, 0.0189, 0.0171,
                  0.0152, 0.0124, 0.0106, 0.0108, 0.0086, 0.0078, 0.0085])

# ===============================
# 2. 参数化趋势：渐近模型 alpha(x) = c + a / (x + b)
# ===============================
def asymptotic_model(x, a, b, c):
    return c + a / (x + b)

# 初始参数估计
p0 = [0.5, 1.0, 0.002]
popt, _ = curve_fit(asymptotic_model, xx, alpha, p0=p0, maxfev=10000)
a_fit, b_fit, c_fit = popt
print(f"Fitted parameters: a={a_fit:.4f}, b={b_fit:.4f}, c={c_fit:.6f}")

# 拟合趋势与残差
trend = asymptotic_model(xx, *popt)
residuals = alpha - trend

# ===============================
# 3. 高斯过程拟合残差
# ===============================
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=30.0, nu=1.5) \
          + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-12, 1e0))

gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                               normalize_y=True, random_state=42)
gpr.fit(xx.reshape(-1, 1), residuals)

# ===============================
# 4. 预测与外推
# ===============================
xx_pred = np.linspace(8, 520, 500).reshape(-1, 1)
trend_pred = asymptotic_model(xx_pred.ravel(), *popt)
res_pred, sigma_res = gpr.predict(xx_pred, return_std=True)
alpha_pred = trend_pred + res_pred

# 要预测的目标点
xx_target = np.array([180, 256, 320, 360, 400, 450, 512]).reshape(-1, 1)
trend_target = asymptotic_model(xx_target.ravel(), *popt)
res_t, sigma_t = gpr.predict(xx_target, return_std=True)
alpha_target = trend_target + res_t

# ===============================
# 5. 输出预测结果
# ===============================
print("\nPredictions:")
for x, a, s in zip(xx_target.ravel(), alpha_target, sigma_t):
    print(f"3d{x}.mtx: alpha = {a:.6f} ± {s:.6f}")

# ===============================
# 6. 绘图
# ===============================
plt.figure(figsize=(9, 4))
plt.scatter(xx, alpha, color='red', label='Observed data')
plt.plot(xx_pred, alpha_pred, color='orange', lw=2, label='Trend + GPR residual')
plt.plot(xx_pred, trend_pred, '--', color='skyblue', lw=1.5, label='Asymptotic trend only')

plt.fill_between(xx_pred.ravel(),
                 alpha_pred - 1.96 * sigma_res,
                 alpha_pred + 1.96 * sigma_res,
                 color='orange', alpha=0.2, label='95% confidence interval')

plt.scatter(xx_target, alpha_target, color='black', s=60, marker='x', label='Predicted targets')
plt.xlabel('size', fontsize=12)
plt.ylabel('alpha', fontsize=12)
plt.title('Asymptotic Trend + GPR Residuals Prediction', fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
