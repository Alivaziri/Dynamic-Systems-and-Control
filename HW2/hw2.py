import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# -----------------------------
# Data Processing
# -----------------------------
# Load the dryer data (assumed format: first column = u(t), second column = y(t))
data = np.loadtxt('dryer.dat')
u = data[:, 0]
y = data[:, 1]

# Optional: Detrend or normalize data if necessary
# For example, remove the mean:
u = u - np.mean(u)
y = y - np.mean(y)

# Split the data into training and validation sets (e.g., 70%-30% split)
N = len(y)
train_size = int(0.7 * N)
u_train, y_train = u[:train_size], y[:train_size]
u_val, y_val = u[train_size:], y[train_size:]

# Define ARX model orders (example: 2 past outputs and 2 past inputs)
n = 2  # Number of past outputs
m = 2  # Number of past inputs

def build_regressor(u, y, n, m):
    """
    Build the regression matrix Phi and output vector Y
    for an ARX model: y(t) = -a1*y(t-1)-...-an*y(t-n) + b1*u(t-1)+...+bm*u(t-m) + e(t).
    """
    N = len(y)
    start = max(n, m)
    Phi = []
    Y = []
    for t in range(start, N):
        # Construct regressor: first n elements (negative past outputs), then m elements (past inputs)
        phi_t = [-y[t - i] for i in range(1, n+1)]
        phi_t += [u[t - i] for i in range(1, m+1)]
        Phi.append(phi_t)
        Y.append(y[t])
    return np.array(Phi), np.array(Y)

# Build regressors for training data
Phi_train, Y_train = build_regressor(u_train, y_train, n, m)

# -----------------------------
# Batch Least Squares (LS)
# -----------------------------
theta_ls, residuals, rank, s = np.linalg.lstsq(Phi_train, Y_train, rcond=None)
print("Batch LS Estimate of theta:")
print(theta_ls)

# Predict on validation data using the LS estimate
Phi_val, Y_val = build_regressor(u_val, y_val, n, m)
Y_pred_ls = Phi_val @ theta_ls

# Compute performance metrics for LS
rmse_ls = np.sqrt(mean_squared_error(Y_val, Y_pred_ls))
fit_ls = 100 * (1 - (np.linalg.norm(Y_val - Y_pred_ls) / np.linalg.norm(Y_val - np.mean(Y_val))))
print(f"Batch LS Validation RMSE: {rmse_ls:.4f}")
print(f"Batch LS Fit Percentage: {fit_ls:.2f}%")

# -----------------------------
# Recursive Least Squares (RLS)
# -----------------------------
# Initialize RLS parameters
theta_rls = np.zeros(n + m)          # initial estimate
P = 1000 * np.eye(n + m)               # large initial covariance
lam = 1                           # forgetting factor here is 1 but can be less than and close to 1 (like 0.98)
theta_rls_history = []               # to store evolution

# Run RLS on training data
start = max(n, m)
for t in range(start, len(y_train)):
    # Build the regressor vector for time t
    phi_t = np.array([-y_train[t - i] for i in range(1, n+1)] +
                     [u_train[t - i] for i in range(1, m+1)])
    phi_t = phi_t.reshape(-1, 1)  # column vector
    # Compute the Kalman gain
    K = P @ phi_t / (lam + phi_t.T @ P @ phi_t)
    # Prediction
    y_pred = float(np.dot(theta_rls, phi_t.flatten()))
    error = y_train[t] - y_pred
    # Update parameter estimate and covariance matrix
    theta_rls = theta_rls + (K.flatten() * error)
    P = (P - K @ phi_t.T @ P) / lam
    theta_rls_history.append(theta_rls.copy())

theta_rls_history = np.array(theta_rls)
print("Final RLS Estimate of theta:")
print(theta_rls)


# Predict on validation data using final RLS estimate
Y_pred_rls = Phi_val @ theta_rls
rmse_rls = np.sqrt(mean_squared_error(Y_val, Y_pred_rls))
fit_rls = 100 * (1 - (np.linalg.norm(Y_val - Y_pred_rls) /
                       np.linalg.norm(Y_val - np.mean(Y_val))))
print(f"RLS Validation RMSE: {rmse_rls:.4f}")
print(f"RLS Fit Percentage: {fit_rls:.2f}%")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 18,
    "font.size": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

# -----------------------------
# LS: Measured vs Predicted Output
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(Y_val, color='black', label=r'\textbf{Measured Output}')
plt.plot(Y_pred_ls, color='blue', linestyle='--', label=r'\textbf{LS Predicted Output}')
plt.xlabel(r'\textbf{Time Index}')
plt.ylabel(r'\textbf{Output}')
plt.title(r'\textbf{Batch LS: Measured vs Predicted Output}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('LS_measured_vs_predicted.pdf', format='pdf', bbox_inches='tight')


# -----------------------------
# RLS: Measured vs Predicted Output
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(Y_val, color='black', label=r'\textbf{Measured Output}')
plt.plot(Y_pred_rls, color='red', linestyle='--', label=r'\textbf{RLS Predicted Output}')
plt.xlabel(r'\textbf{Time Index}')
plt.ylabel(r'\textbf{Output}')
plt.title(r'\textbf{RLS: Measured vs Predicted Output}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('RLS_measured_vs_predicted.pdf', format='pdf', bbox_inches='tight')


# -----------------------------
# Residuals for LS model
# -----------------------------
residuals_rls = Y_val - Y_pred_rls
plt.figure(figsize=(10, 4))
plt.plot(residuals_rls, color='blue', label=r'\textbf{RLS Residuals}')

# -----------------------------
# Residuals for RLS model
# -----------------------------
residuals_ls = Y_val - Y_pred_ls
plt.plot(residuals_ls, linestyle='dashed', color='red', label=r'\textbf{LS Residuals}')
plt.xlabel(r'\textbf{Time Index}')
plt.ylabel(r'\textbf{Residual Error}')
plt.title(r'\textbf{Residuals of RLS and LS Models}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('residuals.pdf', format='pdf', bbox_inches='tight')
