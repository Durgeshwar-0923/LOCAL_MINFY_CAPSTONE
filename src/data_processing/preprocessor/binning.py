import os
import joblib
import pandas as pd
import logging

# Set up a logger for consistent logging
try:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# Directory to save or load binning configurations
ARTIFACT_DIR = "artifacts/binning"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# --- Binning Strategy Configuration ---
# Numeric features with heavy skew are discretized via quantile or uniform binning.
DEFAULT_BINNING_CONFIG = {
    "total_visits": {
        "strategy": "quantile",
        "bins": 4,
        "labels": ["Low_Visits", "Medium_Visits", "High_Visits", "Very_High_Visits"]
    },
    "total_time_spent_on_website": {
        "strategy": "quantile",
        "bins": 4,
        "labels": ["Low_Engagement", "Medium_Engagement", "High_Engagement", "Very_High_Engagement"]
    },
    "page_views_per_visit": {
        "strategy": "quantile",
        "bins": 4,
        "labels": ["Low_Page_Views", "Medium_Page_Views", "High_Page_Views", "Very_High_Page_Views"]
    }
}

class BinningTransformer:
    """
    Transformer for binning numeric features using a predefined configuration.
    Supports fitting on training data (to compute bin edges) and transforming new data
    using saved bin edges, ensuring consistency across train/test splits.
    """
    def __init__(self, config: dict = None, artifact_dir: str = ARTIFACT_DIR):
        self.config = config if config is not None else DEFAULT_BINNING_CONFIG
        self.artifact_dir = artifact_dir
        self.bin_edges = {}  # to store edges after fitting

    def fit(self, df: pd.DataFrame):
        """
        Fit bin edges for each numeric feature based on the chosen strategy.
        Saves computed edges to artifact_dir for later reuse.

        Args:
            df (pd.DataFrame): Training DataFrame.
        """
        for col, params in self.config.items():
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Skipping fit: column '{col}' missing or not numeric.")
                continue

            # Fill true missing (NaN) as zeros to represent no engagement
            series = df[col].fillna(0)
            strategy, bins = params['strategy'], params['bins']

            logger.info(f"Fitting bin edges for '{col}' using {strategy} strategy with {bins} bins.")
            try:
                if strategy == 'quantile':
                    # qcut returns NA bins if duplicates; drop duplicates in edges
                    _, edges = pd.qcut(series, q=bins, retbins=True, duplicates='drop')
                elif strategy == 'uniform':
                    _, edges = pd.cut(series, bins=bins, retbins=True)
                else:
                    raise ValueError(f"Invalid binning strategy '{strategy}' for column '{col}'.")

                self.bin_edges[col] = edges.tolist()

                # Persist configuration
                config_path = os.path.join(self.artifact_dir, f"{col}_binning_config.joblib")
                joblib.dump({'edges': edges, 'labels': params['labels']}, config_path)
                logger.info(f"Saved binning config for '{col}' to {config_path}.")

            except Exception as e:
                logger.error(f"Failed to fit binning for '{col}': {e}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply precomputed bin edges to discretize features in df.
        Loads edges from memory or artifact files if necessary.

        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.DataFrame: DataFrame with new binned columns.
        """
        df_out = df.copy()

        for col, params in self.config.items():
            new_col = f"{col}_binned"
            labels = params['labels']

            # Load edges if not in memory
            if col not in self.bin_edges:
                config_path = os.path.join(self.artifact_dir, f"{col}_binning_config.joblib")
                if os.path.exists(config_path):
                    saved = joblib.load(config_path)
                    edges = saved['edges']
                    self.bin_edges[col] = edges
                    logger.info(f"Loaded existing bin edges for '{col}' from {config_path}.")
                else:
                    logger.warning(f"No binning config found for '{col}', skipping transform.")
                    continue

            edges = self.bin_edges[col]

            # Ensure numeric and fill NaN as zero
            if col not in df_out.columns or not pd.api.types.is_numeric_dtype(df_out[col]):
                logger.error(f"Skipping transform: column '{col}' missing or not numeric.")
                continue
            series = df_out[col].fillna(0)

            # Discretize using saved edges
            try:
                df_out[new_col] = pd.cut(
                    series,
                    bins=edges,
                    labels=labels,
                    include_lowest=True
                ).astype('category')
            except Exception as e:
                logger.error(f"Error during binning transform on '{col}': {e}")
                continue

            # Fill any NaNs in binned column
            if df_out[new_col].isna().any():
                df_out[new_col] = df_out[new_col].cat.add_categories(['Unknown']).fillna('Unknown')
                logger.info(f"Filled NaNs in '{new_col}' with 'Unknown'.")

        logger.info("Completed binning transform.")
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience method to fit on df and immediately transform it.
        """
        self.fit(df)
        return self.transform(df)

# Legacy function for simple pipelines

def apply_feature_binning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy wrapper using BinningTransformer.
    Fits and transforms in one step. Suitable for one-off preprocessing.
    """
    logger.info("--- Applying Feature Binning (legacy) ---")
    transformer = BinningTransformer()
    return transformer.fit_transform(df)
