from nlrs.linear_model import MultiTaskRegressor, Lasso
import matplotlib.pyplot as plt
import numpy as np


rng = np.random.RandomState(42)

# Generate some 2D coefficients with sine waves with random frequency and phase
n_samples, n_features, n_tasks = 100, 30, 40
n_relevant_features = 5
coef = np.zeros((n_tasks, n_features))
times = np.linspace(0, 2 * np.pi, n_tasks)
for k in range(n_relevant_features):
    coef[:, k] = np.sin((1.0 + rng.randn(1)) * times + 3 * rng.randn(1))

X = rng.randn(n_samples, n_features)
Y = np.dot(X, coef.T) + rng.randn(n_samples, n_tasks)


coef_lasso_ = np.array([Lasso(alpha=0.5).fit(X, y).coef_ for y in Y.T])
coef_multi_task_lasso_ = MultiTaskRegressor(alpha=1.0).fit(X, Y).coef_

tasks = np.arange(coef.shape[0])
features = np.arange(coef.shape[1])

# Nonzero masks (transpose to: features x tasks)
nz_true  = (coef != 0).T
nz_lasso = (coef_lasso_ != 0).T
nz_mtl   = (coef_multi_task_lasso_ != 0).T

# Shared color scale for coefficient value heatmaps
all_coefs = np.concatenate(
    [coef.ravel(), coef_lasso_.ravel(), coef_multi_task_lasso_.ravel()]
)
vmax = np.percentile(np.abs(all_coefs), 99)
vmin = -vmax

fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)

plots = [
    (coef.T, "Ground Truth"),
    (coef_lasso_.T, "Independent LASSO"),
    (coef_multi_task_lasso_.T, "Multi-Task Lasso"),
]

masks = [
    (nz_true, "Ground Truth"),
    (nz_lasso, "Independent LASSO"),
    (nz_mtl, "Multi-Task Lasso"),
]

# --- Top row: coefficient heatmaps ---
for col, (mat, title) in enumerate(plots):
    ax = axes[0, col]
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_title(f"{title} (Importance)")
    ax.set_xlabel("Task", size=12)
    ax.set_ylabel("Feature", size=12)

# shared colorbar for top row
cbar = fig.colorbar(im, ax=axes[0, :], shrink=0.8)
cbar.set_label("Coefficient value", fontsize=12)

# --- Bottom row: binary selection masks ---
for col, (mask, title) in enumerate(masks):
    ax = axes[1, col]
    ax.imshow(mask, aspect="auto", origin="lower", cmap="gray_r")
    ax.set_title(f"{title} (Selection)")
    ax.set_xlabel("Task", size=12)
    ax.set_ylabel("Feature", size=12)

fig.suptitle(
    "Independent LASSO vs Multi-Task Lasso: Coefficients and Group Sparsity",
    fontsize=16,
)
plt.show()
