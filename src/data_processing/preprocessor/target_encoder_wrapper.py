import os
import joblib
import pandas as pd
import logging
from category_encoders import TargetEncoder

# Setup logger for consistent tracking
try:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# Artifact directory to persist the encoder (used in both train and inference)
ARTIFACT_DIR = "artifacts/encoding"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# --- Target Encoding Configuration ---
# WHY: These are known high-cardinality features. One-hot encoding them would lead to a high-dimensional sparse matrix.
# Target encoding uses the average of the target (e.g., conversion rate) for each category,
# which compresses information while preserving signal.
HIGH_CARDINALITY_COLS = [
    'country',
    'lead_source',
    'specialization',
    'last_activity',
    'tags'
]

class TargetEncoderWrapper:
    """
    A simple wrapper for category_encoders.TargetEncoder with:
    - Sklearn-like API
    - Fit/persist/load support
    """

    def __init__(self, cols_to_encode: list, smoothing: float = 20.0):
        """
        smoothing: Controls how much to regularize each category toward the global mean.
        Higher smoothing = less overfitting, but also less category specificity.
        """
        self.cols_to_encode = cols_to_encode
        self.smoothing = smoothing
        self.encoder = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the TargetEncoder to provided columns.
        """
        logger.info(f"🎯 Fitting TargetEncoder for: {self.cols_to_encode}")
        self.encoder = TargetEncoder(
            cols=self.cols_to_encode,
            smoothing=self.smoothing,
            handle_missing='value'  # Treat missing values as a separate category
        )
        self.encoder.fit(X, y)
        logger.info("✅ TargetEncoder fit completed.")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms new data using the previously fitted encoder.
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not fitted. Call fit() before transform().")

        logger.info(f"🔁 Transforming with TargetEncoder: {self.cols_to_encode}")
        return self.encoder.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

def apply_target_encoding(
    df: pd.DataFrame,
    target_col: str = 'Converted',
    is_training: bool = True
) -> pd.DataFrame:
    """
    Main function to apply Target Encoding:

    - During training: fits TargetEncoder and saves it.
    - During inference: loads pre-trained encoder to ensure consistent transformation.

    WHY: Ensures same feature mapping across train/test/inference — critical for preventing leakage.
    """
    logger.info("--- 🚀 Starting Target Encoding ---")
    df_processed = df.copy()
    encoder_path = os.path.join(ARTIFACT_DIR, "target_encoder.joblib")

    # Filter only those columns that exist in the dataset
    existing_cols = [col for col in HIGH_CARDINALITY_COLS if col in df_processed.columns]

    if not existing_cols:
        logger.warning("⚠️ No valid high-cardinality columns found. Skipping target encoding.")
        return df_processed

    # Prepare inputs
    X = df_processed.drop(columns=[target_col], errors='ignore')
    y = df_processed[target_col] if target_col in df_processed.columns else None

    encoder_wrapper = TargetEncoderWrapper(cols_to_encode=existing_cols)

    if is_training:
        # --- Training phase ---
        if y is None:
            logger.error(f"❌ Missing target column '{target_col}' during training. Cannot encode.")
            return df_processed

        # Fit and encode
        X_transformed = encoder_wrapper.fit_transform(X, y)

        # Persist encoder for reproducibility
        joblib.dump(encoder_wrapper.encoder, encoder_path)
        logger.info(f"💾 Saved trained TargetEncoder to: {encoder_path}")
    else:
        # --- Inference phase ---
        try:
            encoder_wrapper.encoder = joblib.load(encoder_path)
            logger.info(f"📦 Loaded existing TargetEncoder from: {encoder_path}")
            X_transformed = encoder_wrapper.transform(X)
        except FileNotFoundError:
            logger.error(f"❌ No saved encoder found at '{encoder_path}'. Returning unencoded data.")
            return df_processed

    # Restore target column if needed
    df_final = X_transformed.copy()
    if target_col in df_processed.columns:
        df_final[target_col] = df_processed[target_col]

    logger.info("✅ Target Encoding finished.")
    return df_final
