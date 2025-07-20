import os
import joblib
import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Logger fallback
try:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class OutlierTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 skewed_columns=None,
                 iqr_threshold=1.5,
                 artifact_dir="artifacts/outliers",
                 apply_log=True,
                 apply_clip=True):
        self.skewed_columns = skewed_columns or []
        self.iqr_threshold = iqr_threshold
        self.artifact_dir = artifact_dir
        self.apply_log = apply_log
        self.apply_clip = apply_clip
        # Use a trailing underscore for attributes learned during 'fit'
        self.clip_bounds_ = {}
        
        os.makedirs(self.artifact_dir, exist_ok=True)
        # Define a single path for the configuration artifact
        self.config_path = os.path.join(self.artifact_dir, "outlier_config.joblib")

    def fit(self, X: pd.DataFrame, y=None):
        """
        Calculates the lower and upper clip bounds for each numeric column
        and saves them to a single configuration file. This is done during training.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if self.apply_clip:
            logger.info("Fitting OutlierTransformer: calculating IQR clip bounds...")
            for col in X.select_dtypes(include=np.number).columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.iqr_threshold * IQR
                upper_bound = Q3 + self.iqr_threshold * IQR
                self.clip_bounds_[col] = (lower_bound, upper_bound)
                logger.info(f"Learned clip bounds for '{col}': [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # --- FIX APPLIED HERE ---
        # Instead of saving one file per column, save the entire dictionary of bounds
        # to a single, predictable artifact file. This is crucial for the prediction service.
        joblib.dump(self.clip_bounds_, self.config_path)
        logger.info(f"💾 Saved all outlier clip bounds to {self.config_path}")
        # --- END OF FIX ---

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies log transformation and clips outliers in the DataFrame based on the
        fitted or loaded bounds. This is used during both training and prediction.
        """
        df = X.copy()

        if self.apply_log and self.skewed_columns:
            for col in self.skewed_columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].clip(lower=0)
                    df[col] = np.log1p(df[col])
                    logger.info(f"log1p transform applied to '{col}'")

        if self.apply_clip:
            # --- FIX APPLIED HERE ---
            # If the transformer hasn't been fitted in the current session (e.g., during prediction),
            # load the saved bounds from the artifact file.
            if not self.clip_bounds_:
                try:
                    self.clip_bounds_ = joblib.load(self.config_path)
                    logger.info(f"📦 Loaded outlier clip bounds from {self.config_path}")
                except FileNotFoundError:
                    logger.error(f"❌ Outlier config not found at {self.config_path}. Cannot apply clipping. Please fit the transformer first.")
                    return df # Return unmodified data if config is missing
            # --- END OF FIX ---

            for col, (lower, upper) in self.clip_bounds_.items():
                if col in df.columns:
                    df[col] = df[col].clip(lower, upper)
                    logger.info(f"Outliers clipped in '{col}' to range: [{lower:.2f}, {upper:.2f}]")

        return df

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Undo log1p transform (if applied) on skewed columns for inference."""
        df = X.copy()

        if self.apply_log and self.skewed_columns:
            for col in self.skewed_columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = np.expm1(df[col])
                    logger.info(f"Inverse log1p applied to '{col}'")

        return df


# --- Legacy functions are preserved as they don't cause issues ---

def transform_skewed_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
            df[col] = np.log1p(df[col])
            logger.info(f"log1p transform applied to '{col}'")
    return df


def clip_outliers(df: pd.DataFrame, iqr_threshold: float = 1.5) -> pd.DataFrame:
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_threshold * IQR
        upper_bound = Q3 + iqr_threshold * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
        logger.info(f"Clipped outliers in '{col}' using IQR threshold {iqr_threshold}")
    return df

def handle_outliers(df: pd.DataFrame, skewed_columns=None, iqr_threshold=1.5) -> pd.DataFrame:
    """
    Wrapper function for outlier handling using OutlierTransformer.
    Used during inference for consistency.
    """
    transformer = OutlierTransformer(
        skewed_columns=skewed_columns,
        iqr_threshold=iqr_threshold,
        apply_log=True,
        apply_clip=True
    )
    transformer.fit(df)
    return transformer.transform(df)
