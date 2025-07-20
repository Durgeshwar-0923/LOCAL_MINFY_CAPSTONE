import os
import joblib
import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin

# Set up a logger
try:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

# Artifact directory for storing imputer parameters (for production reproducibility)
ARTIFACT_DIR = "artifacts/imputation"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Default imputation rules for numerical and categorical features
NUMERIC_IMPUTE_CONFIG = {
    'total_visits': 'median',
    'page_views_per_visit': 'median'
}

CATEGORICAL_IMPUTE_CONFIG = {
    'country': {'strategy': 'constant', 'fill_value': 'Unknown_Country'},
    'specialization': {'strategy': 'constant', 'fill_value': 'Not_Specified'},
    'what_is_your_current_occupation': {'strategy': 'constant', 'fill_value': 'Not_Specified'},
    'what_matters_most_to_you_in_choosing_a_course': {'strategy': 'constant', 'fill_value': 'Not_Specified'},
    'city': {'strategy': 'constant', 'fill_value': 'Unknown_City'},
    'last_activity': {'strategy': 'mode'}  # Mode-based imputation for dynamic behavior fields
}


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Custom imputer that stores imputation strategy during fit
    and uses them consistently during transform.
    Saves learned values to disk for reproducibility in production.
    """

    def __init__(self, numeric_config=None, categorical_config=None, artifact_dir=ARTIFACT_DIR):
        self.numeric_config = numeric_config if numeric_config else NUMERIC_IMPUTE_CONFIG
        self.categorical_config = categorical_config if categorical_config else CATEGORICAL_IMPUTE_CONFIG
        self.artifact_dir = artifact_dir
        self.imputer_values_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()

        # --- Learn numeric imputation values ---
        for col, strategy in self.numeric_config.items():
            if col in df.columns:
                if strategy == 'median':
                    val = df[col].median()
                elif strategy == 'mean':
                    val = df[col].mean()
                elif strategy == 'constant':
                    val = 0  # Safe fallback, customizable
                else:
                    raise ValueError(f"Unknown strategy: {strategy} for column: {col}")
                self.imputer_values_[col] = {'strategy': strategy, 'value': val}
                logger.info(f"🔢 Learned numeric imputation for '{col}': {strategy} = {val}")
            else:
                logger.warning(f"Numeric column '{col}' not found. Skipping.")

        # --- Learn categorical imputation values ---
        for col, config in self.categorical_config.items():
            if col in df.columns:
                strat = config.get('strategy')
                if strat == 'constant':
                    val = config.get('fill_value', 'Missing')
                elif strat == 'mode':
                    val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Missing'
                else:
                    val = config.get('fill_value', 'Missing')
                self.imputer_values_[col] = {'strategy': strat, 'value': val}
                logger.info(f"🔤 Learned categorical imputation for '{col}': {strat} = {val}")
            else:
                logger.warning(f"Categorical column '{col}' not found. Skipping.")

        # Save learned values to disk
        path = os.path.join(self.artifact_dir, 'imputer_values.joblib')
        joblib.dump(self.imputer_values_, path)
        logger.info(f"💾 Saved imputer values to: {path}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Load from disk if not fitted
        if not self.imputer_values_:
            path = os.path.join(self.artifact_dir, 'imputer_values.joblib')
            if os.path.exists(path):
                self.imputer_values_ = joblib.load(path)
                logger.info(f"📥 Loaded imputer values from: {path}")
            else:
                logger.error("❌ No imputer values found. Returning unmodified DataFrame.")
                return df

        # --- Apply imputation based on stored config ---
        for col, info in self.imputer_values_.items():
            if col in df.columns:
                val = info['value']

                # Handle case where category type doesn't contain the value
                if pd.api.types.is_categorical_dtype(df[col]):
                    if val not in df[col].cat.categories:
                        df[col] = df[col].cat.add_categories([val])
                        logger.debug(f"➕ Added '{val}' to category levels for '{col}'")

                # Perform imputation
                df[col] = df[col].fillna(val)
                logger.debug(f"🧩 Imputed missing values in '{col}' with '{val}'")
            else:
                logger.warning(f"Column '{col}' not in DataFrame during transform. Skipping.")

        logger.info("✅ Missing value imputation completed.")
        return df

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


# --- Legacy-compatible helper ---
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple helper for quick imputation.
    """
    imputer = MissingValueImputer()
    return imputer.fit_transform(df)
