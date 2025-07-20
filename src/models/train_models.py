# src/models/train_models.py

import warnings, os, atexit, shutil, tempfile, contextlib
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Tcl_AsyncDelete: async handler deleted by the wrong thread")
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')
warnings.filterwarnings("ignore", category=UserWarning, message=".*distutils.*")

os.environ["MPLBACKEND"] = "Agg"
os.environ["DISPLAY"] = ""

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

_temp_dir = tempfile.mkdtemp()
os.environ["JOBLIB_TEMP_FOLDER"] = _temp_dir

def cleanup_temp_dir():
    try:
        shutil.rmtree(_temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"Warning during temp cleanup: {e}")
atexit.register(cleanup_temp_dir)

# ─── Core Libraries ───────────────────────────────────────────────────
import mlflow
import numpy as np
import pandas as pd
import joblib
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

# ─── Progress & Visualization ─────────────────────────────────────────
from rich.console import Console
from rich.table import Table

# ─── Custom Modules ──────────────────────────────────────────────────
from src.utils.logger import setup_logger
from src.utils.metrics import classification_report_dict
from src.models.optuna_tuner import optimize_model
from src.data_processing.preprocessor.target_encoder_wrapper import TargetEncoderWrapper
from src.monitoring.drift_detector import log_drift_report

# ─── Logger & Setup ──────────────────────────────────────────────────
logger = setup_logger(__name__)
console = Console()
client = MlflowClient()
mlflow.set_experiment("Lead_Conversion_Modeling")

for d in ["artifacts", "drift_reports", "outputs", "catboost_logs"]:
    os.makedirs(d, exist_ok=True)

# ─── Model Registry ──────────────────────────────────────────────────
CLASSIFIERS = {
    "LogisticRegression": LogisticRegression,
    "RandomForest": RandomForestClassifier,
    "XGBoost": XGBClassifier,
    "LightGBM": LGBMClassifier,
    "CatBoost": CatBoostClassifier,
    "GradientBoosting": GradientBoostingClassifier,
}

TUNING_SPACE = {
    "LogisticRegression": lambda t: {"C": t.suggest_float("C", 0.01, 10.0)},
    "RandomForest": lambda t: {"n_estimators": t.suggest_int("n_estimators", 50, 200),
                               "max_depth": t.suggest_int("max_depth", 3, 15)},
    "XGBoost": lambda t: {"n_estimators": t.suggest_int("n_estimators", 50, 200),
                          "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3),
                          "max_depth": t.suggest_int("max_depth", 3, 10)},
    "LightGBM": lambda t: {"n_estimators": t.suggest_int("n_estimators", 50, 200),
                           "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2),
                           "num_leaves": t.suggest_int("num_leaves", 20, 150)},
    "CatBoost": lambda t: {"iterations": t.suggest_int("iterations", 50, 200),
                           "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3),
                           "depth": t.suggest_int("depth", 3, 10)},
    "GradientBoosting": lambda t: {"n_estimators": t.suggest_int("n_estimators", 50, 200),
                                    "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3),
                                    "max_depth": t.suggest_int("max_depth", 3, 10)},
}

def safe_start_run(name=None, nested=False):
    if mlflow.active_run():
        mlflow.end_run()
    return mlflow.start_run(run_name=name, nested=nested)

def train_all_models(
    df: pd.DataFrame = None,
    data_path: str = "data/processed/13_final_features.csv",
    experiment_name: str = "Lead_Conversion_Modeling",
    target: str = "Converted",
    test_size: float = 0.2,
    timeout: int = 600,
    cv=5,
    n_trials: int = 20
):
    df = pd.read_csv(data_path)
    logger.info(f"Loaded processed data: {df.shape}")

    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # --- 👇 CHANGE MADE HERE ---
    # Robust FIX for FloatingPointError:
    # The drift detector fails if any numeric column has zero variance (is a constant).
    # This can happen in either the train or test set after a split.
    # This fix ensures we only pass columns that have variance in BOTH sets.
    logger.info("Filtering columns for drift detection to avoid zero-variance error...")
    
    # 1. Isolate numeric columns from both sets
    X_train_numeric = X_train.select_dtypes(include=np.number)
    X_test_numeric = X_test.select_dtypes(include=np.number)
    
    # 2. Find columns with variance (>0 standard deviation) in each set
    train_variant_cols = X_train_numeric.columns[X_train_numeric.std() > 0]
    test_variant_cols = X_test_numeric.columns[X_test_numeric.std() > 0]
    
    # 3. The safe columns are the intersection of the two sets
    safe_numeric_cols = train_variant_cols.intersection(test_variant_cols)
    
    # 4. Reconstruct the full dataframes (numeric + categorical) using only the safe numeric columns
    X_train_drift = pd.concat([X_train[safe_numeric_cols], X_train.select_dtypes(exclude=np.number)], axis=1)
    X_test_drift = pd.concat([X_test[safe_numeric_cols], X_test.select_dtypes(exclude=np.number)], axis=1)
    
    logger.info(f"Passing {X_train_drift.shape[1]} columns to drift detector.")
    log_drift_report(X_train_drift, X_test_drift, dataset_name="train_vs_test")
    # --- END OF CHANGE ---

    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    encoder = TargetEncoderWrapper(cols_to_encode=cat_cols)
    X_train_enc = encoder.fit_transform(X_train.copy(), y_train)
    X_test_enc = encoder.transform(X_test.copy())

    joblib.dump(encoder, "artifacts/final_target_encoder.joblib")
    logger.info(f"💾 Saved final target encoder to artifacts/final_target_encoder.joblib")

    scaler = StandardScaler().fit(X_train_enc)
    X_train_enc = scaler.transform(X_train_enc)
    X_test_enc = scaler.transform(X_test_enc)
    joblib.dump(scaler, "artifacts/feature_scaler.pkl")

    results, models = [], []
    best_auc, best_model, best_run_id = 0.0, None, None

    with safe_start_run("All_Model_Training") as main_run:
        for name, Cls in CLASSIFIERS.items():
            mlflow.set_tag("model_name", name)
            params = optimize_model(Cls, TUNING_SPACE[name], X_train_enc, y_train, n_trials=n_trials)
            model = Cls(**params)
            model.fit(X_train_enc, y_train)
            preds = model.predict(X_test_enc)
            probs = model.predict_proba(X_test_enc)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds),
                "recall": recall_score(y_test, preds),
                "roc_auc": roc_auc_score(y_test, probs),
            }

            mlflow.log_params({f"{name}_{k}": v for k, v in params.items()})
            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics.items()})
            mlflow.sklearn.log_model(model, f"models/{name}")

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_model = model
                best_run_id = main_run.info.run_id

            results.append({"model": name, **metrics})
            models.append((name, model))

        # ─── Train Stacking Model ─────────────────────────────
        stack = StackingClassifier(
            estimators=models,
            final_estimator=LogisticRegression(),
            cv=StratifiedKFold(5),
            n_jobs=1,
            passthrough=True
        )
        stack.fit(X_train_enc, y_train)
        preds = stack.predict(X_test_enc)
        probs = stack.predict_proba(X_test_enc)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
        }

        mlflow.log_metrics({f"Stacking_{k}": v for k, v in metrics.items()})
        mlflow.sklearn.log_model(stack, "models/StackingEnsemble")
        mlflow.set_tag("model_name", "StackingEnsemble")

        if metrics["roc_auc"] > best_auc:
            best_model = stack
            best_run_id = main_run.info.run_id

        results.append({"model": "StackingEnsemble", **metrics})

    # ─── Register Best Model & Archive Old ─────────────────────────────
    if best_model is not None and best_run_id is not None:
        run_model_uri = f"runs:/{best_run_id}/models/{'StackingEnsemble' if isinstance(best_model, StackingClassifier) else name}"
        mv = mlflow.register_model(run_model_uri, "LeadConversionModel")

        for mv_existing in client.search_model_versions(f"name='LeadConversionModel'"):
            if mv_existing.current_stage == "Production" and int(mv_existing.version) != int(mv.version):
                client.transition_model_version_stage(
                    name=mv_existing.name,
                    version=mv_existing.version,
                    stage="Archived"
                )

        client.transition_model_version_stage(
            name="LeadConversionModel",
            version=mv.version,
            stage="Production"
        )

    # ─── Results Table ─────────────────────────────
    df_res = pd.DataFrame(results)
    table = Table(title="Model Comparison")
    table.add_column("Model", style="bold")
    for m in ["accuracy", "precision", "recall", "roc_auc"]:
        table.add_column(m, justify="right")
    for _, r in df_res.iterrows():
        table.add_row(r["model"], *(f"{r[m]:.3f}" for m in ["accuracy", "precision", "recall", "roc_auc"]))
    console.print(table)
    df_res.to_csv("outputs/model_summary.csv", index=False)

    print("\n✅ Final Model Metrics Summary:")
    print(df_res.to_string(index=False))

    return best_model
