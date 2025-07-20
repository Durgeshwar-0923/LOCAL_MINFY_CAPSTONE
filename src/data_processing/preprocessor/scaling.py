import os
import joblib
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# Logger setup
try:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

ARTIFACT_DIR = "artifacts/scaling"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Default scaling column sets (based on EDA insight)
STANDARD_SCALE_COLS = ['total_visits', 'page_views_per_visit']
ROBUST_SCALE_COLS = ['total_time_spent_on_website', 'time_per_visit']
MINMAX_SCALE_COLS = []


def detect_skewed_features(df: pd.DataFrame, threshold: float = 1.0) -> list:
    """
    Detects numeric columns with skewness above a given threshold.
    """
    skewed = df.select_dtypes(include=np.number).apply(lambda x: x.skew()).dropna()
    return skewed[abs(skewed) > threshold].index.tolist()


def visualize_distributions(df_before: pd.DataFrame, df_after: pd.DataFrame, columns: list):
    """
    Save visual comparison of distributions before and after scaling.
    """
    for col in columns:
        if col in df_before.columns and col in df_after.columns:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            df_before[col].hist(bins=30)
            plt.title(f"Before Scaling - {col}")
            plt.subplot(1, 2, 2)
            df_after[col].hist(bins=30)
            plt.title(f"After Scaling - {col}")
            plt.tight_layout()
            plt.savefig(os.path.join(ARTIFACT_DIR, f"{col}_scaling_comparison.png"))
            plt.close()
            logger.info(f"📊 Saved scaling distribution plot for '{col}'")


def apply_feature_scaling(
    df: pd.DataFrame,
    target_col: str = 'converted',
    is_training: bool = True,
    return_scaler: bool = False
):
    """
    Scales features using StandardScaler, RobustScaler, and MinMaxScaler based on EDA and skewness.
    Stores or loads transformers from artifacts/scaling directory.
    """
    logger.info("📏 Starting feature scaling process...")
    df_scaled = df.copy()
    scaler_path = os.path.join(ARTIFACT_DIR, "feature_scaler.joblib")

    # Detect skewed columns
    detected_skewed = detect_skewed_features(df_scaled.drop(columns=[target_col], errors='ignore'))
    logger.info(f"🔍 Detected skewed features (|skew| > 1): {detected_skewed}")

    # Filter known groups from config that exist in data
    standard_cols = [col for col in STANDARD_SCALE_COLS if col in df_scaled.columns and col != target_col]
    robust_cols = [col for col in ROBUST_SCALE_COLS if col in df_scaled.columns and col != target_col]
    minmax_cols = [col for col in MINMAX_SCALE_COLS if col in df_scaled.columns and col != target_col]

    # Assign dynamically skewed columns to RobustScaler if not already assigned
    dynamic_robust_cols = [col for col in detected_skewed if col not in robust_cols + standard_cols + minmax_cols]
    robust_cols += dynamic_robust_cols

    # If no scaling required
    if not (standard_cols or robust_cols or minmax_cols):
        logger.warning("⚠️ No features found for scaling. Returning original DataFrame.")
        return (df_scaled, None) if return_scaler else df_scaled

    # Create transformers
    transformers = []
    if standard_cols:
        transformers.append(('standard_scaler', StandardScaler(), standard_cols))
        logger.info(f"📘 Applying StandardScaler to: {standard_cols}")
    if robust_cols:
        transformers.append(('robust_scaler', RobustScaler(), robust_cols))
        logger.info(f"📙 Applying RobustScaler to: {robust_cols}")
    if minmax_cols:
        transformers.append(('minmax_scaler', MinMaxScaler(), minmax_cols))
        logger.info(f"📗 Applying MinMaxScaler to: {minmax_cols}")

    all_scaled_cols = standard_cols + robust_cols + minmax_cols
    ordered_cols = all_scaled_cols + [col for col in df_scaled.columns if col not in all_scaled_cols]

    df_ordered = df_scaled[ordered_cols]
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

    if is_training:
        transformed_data = preprocessor.fit_transform(df_ordered)
        joblib.dump({'transformer': preprocessor, 'columns': ordered_cols}, scaler_path)
        logger.info(f"💾 Saved scaler to {scaler_path}")
        
        # --- 👇 CHANGE MADE HERE ---
        # This new line saves the list of columns that the scaler was fitted on.
        # This artifact is required by the prediction service.
        joblib.dump(all_scaled_cols, os.path.join(ARTIFACT_DIR, "scaler_columns.joblib"))
        logger.info(f"💾 Saved scaler column list to {os.path.join(ARTIFACT_DIR, 'scaler_columns.joblib')}")
        # --- END OF CHANGE ---

    else:
        try:
            artifact = joblib.load(scaler_path)
            preprocessor = artifact['transformer']
            ordered_cols = artifact['columns']
            df_ordered = df_scaled[ordered_cols]
            transformed_data = preprocessor.transform(df_ordered)
            logger.info(f"📦 Loaded saved scaler from {scaler_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load scaler artifact: {e}")
            return (df_scaled, None) if return_scaler else df_scaled

    df_final = pd.DataFrame(transformed_data, columns=ordered_cols, index=df_scaled.index)
    visualize_distributions(df_ordered, df_final, all_scaled_cols)

    logger.info("✅ Feature scaling complete.")
    return (df_final, preprocessor) if return_scaler else df_final


def inverse_scale_predictions(scaled_preds: np.ndarray, scaler_path: str = None, target_col: str = "adjusted_total_usd") -> np.ndarray:
    """
    Reverse transformation of scaled predictions using saved scaler.
    Only works if target was scaled.
    """
    if scaler_path is None:
        scaler_path = os.path.join(ARTIFACT_DIR, "feature_scaler.joblib")

    try:
        artifact = joblib.load(scaler_path)
        transformer = artifact['transformer']
        columns = artifact['columns']

        # Locate which transformer was used for the target
        for name, trans, cols in transformer.transformers:
            if target_col in cols:
                inverse_scaler = trans
                break
        else:
            raise ValueError(f"Target column '{target_col}' was not scaled.")

        return inverse_scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()

    except Exception as e:
        logger.error(f"❌ Failed to inverse scale predictions: {e}")
        return scaled_preds
