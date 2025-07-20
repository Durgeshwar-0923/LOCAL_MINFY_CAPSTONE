# src/utils/optuna_tuner.py
import os
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from src.utils.logger import setup_logger
import warnings, os, atexit, shutil, tempfile
import contextlib
warnings.filterwarnings("ignore", category=UserWarning)

logger = setup_logger(__name__)

# ─── Suppress Common Warnings ──────────────────────────────────────────────────
warnings.filterwarnings("ignore", message="Tcl_AsyncDelete: async handler deleted by the wrong thread")
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')
warnings.filterwarnings("ignore", category=UserWarning, message=".*distutils.*")

os.environ["MPLBACKEND"] = "Agg"
os.environ["DISPLAY"] = ""

# ─── Monkey-Patch GUI Threading Errors ─────────────────────────────────────────
try:
    import tkinter as _tk
    if hasattr(_tk, "Image"):
        _tk.Image.__del__ = lambda self: None
    if hasattr(_tk, "Variable"):
        _tk.Variable.__del__ = lambda self: None
    if hasattr(_tk, "Misc"):
        _tk.Misc.__del__ = lambda self: None
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")

# ─── Temporary Directory for Joblib Stability ──────────────────────────────────
_temp_dir = tempfile.mkdtemp()
os.environ["JOBLIB_TEMP_FOLDER"] = _temp_dir

def cleanup_temp_dir():
    try:
        shutil.rmtree(_temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"Warning during temp cleanup: {e}")
atexit.register(cleanup_temp_dir)

def optimize_model(
    estimator_class,
    param_space_func,
    X,
    y,
    metric="roc_auc",
    timeout=600,
    n_trials=50,
    cv_splits=3,
    random_state=42,
    early_stopping_rounds=None,
    is_classification=True
):
    """
    Optimize model hyperparameters with Optuna.

    Args:
        estimator_class: sklearn-like estimator class
        param_space_func: function(trial) -> dict of hyperparameters
        X: feature DataFrame or array
        y: target array
        metric: "roc_auc", "accuracy", "r2"
        timeout: tuning timeout in seconds
        n_trials: maximum number of trials
        cv_splits: number of CV folds
        random_state: random seed for reproducibility
        early_stopping_rounds: int or None, enable early stopping if supported
        is_classification: if True, uses StratifiedKFold
        
    """

    study = None

    def objective(trial):
        try:
            params = param_space_func(trial)

            if early_stopping_rounds is not None:
                if "n_iter_no_change" in estimator_class().get_params().keys():
                    params["n_iter_no_change"] = early_stopping_rounds
                    params["validation_fraction"] = 0.1
                    params["tol"] = 1e-4

            model = estimator_class(**params)
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state) if is_classification \
                else KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

            scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
            score_mean = scores.mean()
            logger.info(f"Trial {trial.number} {metric} mean: {score_mean:.4f}")
            return score_mean

        except Exception as e:
            logger.error(f"Trial {trial.number} failed with exception: {e}")
            return float("-inf")

    try:
        logger.info(f"🔧 Starting Optuna tuning for {estimator_class.__name__} with metric={metric}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            timeout=timeout,
            n_trials=n_trials,
            callbacks=[optuna.study.MaxTrialsCallback(n_trials, states=(optuna.trial.TrialState.COMPLETE,))],
            show_progress_bar=True,
        )
    except Exception as e:
        logger.error(f"Optuna tuning failed for {estimator_class.__name__} with exception: {e}")
        return {}  # Return empty dict so caller can fallback to default params

    best_params = study.best_trial.params
    logger.info(f"✅ Best trial params for {estimator_class.__name__}: {best_params}")

    os.makedirs("outputs", exist_ok=True)
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f"outputs/optuna_{estimator_class.__name__.lower()}_history.html")
    except Exception as e:
        logger.warning(f"Could not save optimization plot: {e}")

    try:
        df = study.trials_dataframe()
        df.to_csv(f"outputs/{estimator_class.__name__.lower()}_optuna_trials.csv", index=False)
    except Exception as e:
        logger.warning(f"Could not save trials dataframe: {e}")

    try:
        best_model = estimator_class(**best_params)
        best_model.fit(X, y)
        if hasattr(best_model, "feature_importances_"):
            plt.figure(figsize=(10, 6))
            plt.title(f"{estimator_class.__name__} Feature Importances")
            plt.bar(range(len(best_model.feature_importances_)), best_model.feature_importances_)
            plt.xlabel("Feature index")
            plt.ylabel("Importance")
            plt.tight_layout()
            plt.savefig(f"outputs/{estimator_class.__name__.lower()}_feature_importances.png")
            plt.close()
    except Exception as e:
        logger.warning(f"Could not plot feature importances: {e}")

    return best_params
