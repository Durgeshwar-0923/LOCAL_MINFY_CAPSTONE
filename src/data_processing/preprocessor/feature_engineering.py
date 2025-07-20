import os
import joblib
import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin

# Setup logger (fallback to default if custom logger isn't found)
try:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# Directory to store SHAP, permutation importance and other feature artifacts
ARTIFACT_DIR = "artifacts/feature_engineering"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

SHAP_PLOT_PATH = os.path.join(ARTIFACT_DIR, "shap_feature_importance.png")
SHAP_CSV_PATH = os.path.join(ARTIFACT_DIR, "shap_feature_importance.csv")


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Domain-driven feature engineering to improve model learning.

    Includes:
    - Ordinal encoding for coded categorical features (e.g., '01.Low')
    - Ratio creation to capture user behavior (e.g., time per visit)
    - Flags for contactability
    - Consolidation of sparse one-hot encoded lead sources
    - Country-based segmentation flag
    """

    def __init__(self, ordinal_cols=None, lead_source_dummies=None):
        self.ordinal_cols = ordinal_cols or ['specialization']

        self.lead_source_dummies = lead_source_dummies or [
            'search', 'newspaper_article', 'x_education_forums',
            'newspaper', 'digital_advertisement', 'through_recommendations'
        ]
        self.ordinal_maps_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in self.ordinal_cols:
            if col in X.columns:
                mapping = {}
                for v in X[col].dropna().unique():
                    try:
                        mapping[v] = int(str(v).split('.')[0])  # Extract numerical prefix
                    except Exception:
                        continue
                if mapping:
                    self.ordinal_maps_[col] = mapping
                    logger.info(f"🧠 Learned ordinal mapping for '{col}': {mapping}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        for col, mapping in self.ordinal_maps_.items():
            new_col = f"{col}_ordinal"
            df[new_col] = df[col].map(mapping).astype(float)
            df.drop(columns=[col], inplace=True, errors='ignore')
            logger.info(f"🪄 Transformed '{col}' → '{new_col}' (ordinal encoding)")

        if {'total_time_spent_on_website', 'total_visits'}.issubset(df.columns):
            df['time_per_visit'] = df['total_time_spent_on_website'].fillna(0) / df['total_visits'].replace(0, 1)
            logger.info("🕒 Created 'time_per_visit' = total_time_spent_on_website / total_visits")

        if {'do_not_email', 'do_not_call'}.issubset(df.columns):
            df['no_contact_allowed'] = ((df['do_not_email'] == 1) & (df['do_not_call'] == 1)).astype(int)
            logger.info("📵 Created 'no_contact_allowed' = both do_not_email and do_not_call are 1")

        df['lead_source_channel'] = 'Other_Source'
        for src in self.lead_source_dummies:
            if src in df.columns:
                df.loc[df[src] == 1, 'lead_source_channel'] = src
        df['lead_source_channel'] = df['lead_source_channel'].astype('category')
        df.drop(columns=[c for c in self.lead_source_dummies if c in df.columns], inplace=True)
        logger.info("📦 Merged sparse one-hot lead sources into 'lead_source_channel'")

        if 'country' in df.columns:
            df['is_from_india'] = (df['country'] == 'India').astype(int)
            logger.info("🗺️ Created 'is_from_india' flag")

        return df


def compute_and_save_shap_importance(df: pd.DataFrame, target_col='Converted', top_k=50):
    """
    Compute SHAP feature importance using XGBoost classifier.
    Imputes missing numeric features before training.
    Saves SHAP summary plot and CSV importance to artifacts.
    """
    import shap
    import xgboost as xgb
    import matplotlib.pyplot as plt
    from sklearn.impute import SimpleImputer

    df = df.copy()
    if target_col not in df.columns:
        logger.warning(f"⚠️ Target column '{target_col}' not found in SHAP input.")
        return

    # Select numeric features only (drop target)
    X_all = df.drop(columns=[target_col], errors='ignore').select_dtypes(include=np.number)
    y_all = df[target_col]

    logger.info(f"Numeric features count before SHAP: {X_all.shape[1]}")
    logger.info(f"Missing values per numeric feature before SHAP:\n{X_all.isnull().sum()}")
    logger.info(f"Missing values in target '{target_col}': {y_all.isnull().sum()}")

    # Filter rows with no NaNs in both X and y
    mask = y_all.notna() & X_all.notnull().all(axis=1)
    X = X_all.loc[mask]
    y = y_all.loc[mask]

    if X.empty or y.empty:
        logger.warning("⚠️ SHAP skipped due to empty features or target after filtering.")
        return

    # Impute missing values if any remain (precaution)
    if X.isnull().any().any():
        logger.info("Imputing missing values in numeric features before SHAP computation.")
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    else:
        X_imputed = X

    # Train XGBoost classifier
    model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_imputed, y)

    # Explain with SHAP
    explainer = shap.Explainer(model, X_imputed)
    shap_vals = explainer(X_imputed)

    # Save SHAP summary bar plot for top_k features
    shap.summary_plot(shap_vals[:, :top_k], X_imputed.iloc[:, :top_k], plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PATH)
    plt.close()
    logger.info(f"📈 Saved SHAP plot to {SHAP_PLOT_PATH}")

    # Save mean absolute SHAP values per feature as CSV
    shap_df = pd.DataFrame({
        'feature': X_imputed.columns,
        'shap_importance': np.abs(shap_vals.values).mean(axis=0)
    }).sort_values(by='shap_importance', ascending=False)
    shap_df.to_csv(SHAP_CSV_PATH, index=False)
    logger.info(f"💾 Saved SHAP CSV to {SHAP_CSV_PATH}")


def compute_permutation_importance(df: pd.DataFrame, target_col: str = 'Converted', n_repeats: int = 10, random_state: int = 42):
    """
    Compute permutation importance to estimate generalizable impact of each feature.
    More robust than SHAP for non-tree models.
    """
    from sklearn.inspection import permutation_importance
    from xgboost import XGBClassifier

    df = df.copy()
    X_all = df.drop(columns=[target_col], errors='ignore').select_dtypes(include=np.number)
    y_all = df[target_col]

    # Align indexes and drop NA for target and features
    mask = y_all.notna() & X_all.notnull().all(axis=1)
    X = X_all.loc[mask]
    y = y_all.loc[mask]

    if X.empty:
        logger.warning("No data for permutation importance after filtering; skipping.")
        return

    # Fit model and compute permutation importance
    model = XGBClassifier(n_estimators=100, random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)

    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, scoring='roc_auc')

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values(by="importance_mean", ascending=False)

    path = os.path.join(ARTIFACT_DIR, "permutation_importance.csv")
    importance_df.to_csv(path, index=False)
    logger.info(f"💾 Saved permutation importance to {path}")


def full_feature_engineering_pipeline(df: pd.DataFrame, run_shap_analysis=True, target_col='Converted') -> pd.DataFrame:
    """
    Orchestrates all feature engineering and optional explainability steps.

    Includes:
    - Feature transformation
    - SHAP importance
    - Permutation importance
    """
    fe = FeatureEngineeringTransformer()
    df_eng = fe.fit_transform(df)

    if run_shap_analysis:
        try:
            compute_and_save_shap_importance(df_eng, target_col=target_col)
            compute_permutation_importance(df_eng, target_col=target_col)
        except Exception as e:
            logger.error(f"Explainability step failed: {e}")

    return df_eng
