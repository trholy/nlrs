import pytest
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from nlrs.linear_model import Ridge

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


@pytest.fixture
def regression_data():
    X, y, _ = make_regression(
        n_samples=100,
        n_features=5,
        noise=0.1,
        random_state=42,
        coef=True,
    )
    return X, y


def test_sklearn_pipeline(regression_data):
    X, y = regression_data
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, solver="SCS"))
    ])
    
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    
    assert preds.shape == (X.shape[0],)
    assert pipeline.named_steps["model"].coef_ is not None


def test_sklearn_grid_search(regression_data):
    X, y = regression_data
    
    model = Ridge(solver="SCS")
    param_grid = {"alpha": [0.1, 1.0, 10.0]}
    
    search = GridSearchCV(model, param_grid, cv=3)
    search.fit(X, y)
    
    assert search.best_params_["alpha"] in param_grid["alpha"]
    assert search.best_estimator_.coef_ is not None


def test_sklearn_randomized_search(regression_data):
    X, y = regression_data
    
    model = Ridge(solver="SCS")
    param_distributions = {"alpha": [0.1, 1.0, 10.0]}
    
    search = RandomizedSearchCV(model, param_distributions, n_iter=2, cv=3, random_state=42)
    search.fit(X, y)
    
    assert search.best_params_["alpha"] in param_distributions["alpha"]
    assert search.best_estimator_.coef_ is not None


@pytest.mark.skipif(not HAS_OPTUNA, reason="Optuna not installed")
def test_optuna_search(regression_data):
    X, y = regression_data
    
    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
        model = Ridge(alpha=alpha, solver="SCS")
        
        scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_squared_error")
        return scores.mean()
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)
    
    assert "alpha" in study.best_params
    assert study.best_value < 0
