import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from nlrs.linear_model import MultiTaskRegressor


seed = 42
rng = np.random.RandomState(seed)

n_samples, n_features, n_tasks = 100, 20, 15
n_relevant_features = 5

# Ground Truth Coefficients (Sine waves across tasks globally)
true_coef = np.zeros((n_tasks, n_features))
times = np.linspace(0, 2 * np.pi, n_tasks)

for k in range(n_relevant_features):
    # We apply random scaling to make coefficients robust but not uniform
    scale = (rng.rand() * 5) + 3.0
    true_coef[:, k] = scale * np.sin((1.0 + rng.randn(1)) * times + 3 * rng.randn(1))

X = rng.randn(n_samples, n_features)

# Generate Targets with standard Gaussian noise
Y_clean = np.dot(X, true_coef.T) + rng.randn(n_samples, n_tasks) * 1.0

# Introduce severe outliers to Y
Y_corrupted = Y_clean.copy()
n_outliers = int(0.15 * n_samples)  # 15% outliers

for t in range(n_tasks):
    outlier_indices = rng.choice(n_samples, n_outliers, replace=False)
    # Outliers have a magnitude 10x larger than normal scale
    Y_corrupted[outlier_indices, t] += rng.randn(n_outliers) * 50.0

# 1. Standard Multi-Task Lasso (Squared Error Loss)
# Highly susceptible to outliers
mt_lasso = MultiTaskRegressor(loss="squared_error", penalty="l1", alpha=0.1)
mt_lasso.fit(X, Y_corrupted)

# 2. Multi-Task SVR (Epsilon-Insensitive Loss) with L1 Penalty
# Epsilon-insensitive loss ignores errors smaller than epsilon and scales linearly for large errors (robust to outliers)
mt_svr = MultiTaskRegressor(
    loss="epsilon_insensitive", epsilon=1, penalty="l1", alpha=1.1
)
mt_svr.fit(X, Y_corrupted)

# 3. Adaptive Multi-Task SVR
# Yields the robustness of epsilon-insensitive loss with the Oracle feature selection of Adaptive Lasso
mt_adaptive_svr = MultiTaskRegressor(
    loss="epsilon_insensitive", epsilon=1, penalty="l1", alpha=1.1, adaptive_weights_power=0.8, adaptive=True,
)
mt_adaptive_svr.fit(X, Y_corrupted)

# Calculate Mean Absolute Error (MAE corresponds better to epsilon-insensitive loss)
# We evaluate on the clean target to see how well they recovered the true signal!
mae_lasso = mean_absolute_error(Y_clean, mt_lasso.predict(X))
mae_svr = mean_absolute_error(Y_clean, mt_svr.predict(X))
mae_adapt = mean_absolute_error(Y_clean, mt_adaptive_svr.predict(X))

print(f"Standard Multi-Task Lasso (Squared Loss) Score on clean data: {mae_lasso:.2f} MAE")
print(f"Multi-Task SVR (Epsilon-Insensitive) Score on clean data: {mae_svr:.2f} MAE")
print(f"Adaptive Multi-Task SVR Score on clean data: {mae_adapt:.2f} MAE")

# --- Build non-zero masks (group sparsity view) ---
nz_true = (true_coef != 0).T.astype(float)
nz_lasso = (mt_lasso.coef_ != 0).T.astype(float)
nz_svr = (mt_svr.coef_ != 0).T.astype(float)
nz_adapt_svr = (mt_adaptive_svr.coef_ != 0).T.astype(float)

# --- Integrated plot: coefficients + selection ---
fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

plots = [
    (true_coef.T, "Ground Truth"),
    (mt_lasso.coef_.T, "MT Lasso\n(Corrupted)"),
    (mt_svr.coef_.T, "MT SVR\n(Robust)"),
    (mt_adaptive_svr.coef_.T, "Adaptive MT SVR\n(Robust + Sparse)")
]

masks = [
    (nz_true, "Ground Truth"),
    (nz_lasso, "MT Lasso"),
    (nz_svr, "MT SVR"),
    (nz_adapt_svr, "Adaptive MT SVR"),
]

# --- Shared color scale (coefficients only!) ---
vmax = np.percentile(np.abs(true_coef), 99)
vmin = -vmax

# =========================
# Top row: coefficients
# =========================
im_coef = None  # keep reference for colorbar

for col, (mat, title) in enumerate(plots):
    ax = axes[0, col]
    im_coef = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Task", size=11)
    ax.set_ylabel("Feature", size=11)

# shared colorbar ONLY for top row
cbar = fig.colorbar(im_coef, ax=axes[0, :], shrink=0.8)
cbar.set_label("Coefficient value", fontsize=12)

# =========================
# Bottom row: selection masks (FIXED)
# =========================
for col, (mask, title) in enumerate(masks):
    ax = axes[1, col]
    ax.imshow(
        mask,                     # already float (0/1)
        aspect="auto",
        origin="lower",
        cmap="gray_r",            # black = selected
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_title(f"{title} (Selection)", fontsize=12)
    ax.set_xlabel("Task", size=11)
    ax.set_ylabel("Feature", size=11)

# --- Title ---
fig.suptitle(
    "Robust Multi-Task Learning: Coefficients vs Group Sparsity",
    fontsize=16,
)
plt.show()
