import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from nlrs.linear_model import Lasso


def run_adaptive_lasso():
    seed = 42
    np.random.seed(seed)

    n_samples, n_features = 150, 50

    # 1. Base random features
    X = np.random.randn(n_samples, n_features)

    # 2. Set true coefficients: We intentionally make them very large
    # Large coefficients are hurt the most by standard Lasso's shrinkage.
    true_coef = np.zeros(n_features)
    active_indices = [0, 1, 2, 3, 4]
    true_coef[active_indices] = [100.0, 75.0, -50.0, 25.0, -20.0]

    # 3. Introduce Highly Correlated Noise
    # We make features 5, 6, and 7 highly correlated with the active features 0, 1, and 2.
    # Their true coefficient is 0, so they are pure noise, but they look like the true features.
    X[:, 5] = X[:, 0] + np.random.randn(n_samples) * 0.2
    X[:, 6] = X[:, 1] + np.random.randn(n_samples) * 0.2
    X[:, 7] = X[:, 2] + np.random.randn(n_samples) * 0.2

    # 4. Generate Target
    y = X @ true_coef + np.random.randn(n_samples) * 10.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    alpha_val = 15.0

    lasso = Lasso(alpha=alpha_val)
    lasso.fit(X_train, y_train)

    adaptive_lasso = Lasso(alpha=alpha_val, adaptive=True)
    adaptive_lasso.fit(X_train, y_train)

    y_pred_lasso = lasso.predict(X_test)
    y_pred_adaptive = adaptive_lasso.predict(X_test)

    print(
        f"Standard Lasso MSE: {mean_squared_error(y_test, y_pred_lasso):.2f},"
        f" R2: {r2_score(y_test, y_pred_lasso):.4f}"
    )
    print(
        f"Adaptive Lasso MSE: {mean_squared_error(y_test, y_pred_adaptive):.2f},"
        f" R2: {r2_score(y_test, y_pred_adaptive):.4f}"
    )

    # --- Plotting the Results ---
    plt.figure(figsize=(12, 6))

    # We zoom in on the first 15 features to clearly see the active vs noise feature selection
    view_limit = 15
    feature_indices = np.arange(view_limit)

    plt.plot(
        feature_indices, true_coef[:view_limit], 'o',
        label='True Coefficients', markersize=10, color='black', alpha=0.6
    )
    plt.plot(
        feature_indices, lasso.coef_[:view_limit], 'x',
        label='Standard Lasso', markersize=10, markeredgewidth=2
    )
    plt.plot(
        feature_indices, adaptive_lasso.coef_[:view_limit], '^',
        label='Adaptive Lasso', markersize=10
    )

    # Highlight the correlated noise features
    plt.axvspan(4.5, 7.5, color='red', alpha=0.05)

    plt.title('Coefficient Estimation - LASSO vs. Adaptive LASSO', size=20)
    plt.ylabel('Coefficient Value', size=14)
    plt.xticks(feature_indices, size=14)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.4)
    plt.show()


if __name__ == "__main__":
    run_adaptive_lasso()
