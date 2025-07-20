import os
import joblib
import pandas as pd
import logging
from sklearn.base import BaseEstimator, TransformerMixin

# Set up logger
try:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

# Directory to store rare label mappings (important for consistent transformation across train/test/inference)
ARTIFACT_DIR = "artifacts/rare_labels"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Default configuration specifying thresholds and replacement labels for rare categories
def default_rare_label_config():
    return {
        'lead_source': {'threshold': 0.01, 'new_name': 'Other'},
        'country': {'threshold': 0.02, 'new_name': 'Other_Country'},
        'specialization': {'threshold': 0.02, 'new_name': 'Other_Specialization'},
        'last_activity': {'threshold': 0.01, 'new_name': 'Other_Activity'}
    }

class RareLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer that replaces rare categories in specified columns with a common 'Other' label.
    
    Purpose:
    - Prevent high cardinality from hurting generalization.
    - Ensure stable encoding post one-hot or label encoding.
    - Avoid data sparsity from rare/unseen categories.
    """

    def __init__(self, config=None, artifact_dir=ARTIFACT_DIR):
        self.config = config if config is not None else default_rare_label_config()
        self.artifact_dir = artifact_dir
        self.rare_labels_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        """
        Identifies rare labels based on frequency threshold for each specified column.
        Saves the list of rare labels for consistent future application.
        """
        df = X.copy()
        for col, params in self.config.items():
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found during fit. Skipping.")
                continue

            # Ensure column is treated as categorical for consistency
            if not pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].astype('category')

            # Compute frequency of each category
            freq = df[col].value_counts(normalize=True, dropna=False)

            # Identify categories with frequency below threshold
            rare = freq[freq < params['threshold']].index.tolist()

            # Save rare labels for this column
            if rare:
                self.rare_labels_[col] = rare
                path = os.path.join(self.artifact_dir, f"{col}_rare_labels.joblib")
                joblib.dump(rare, path)
                logger.info(f"Saved {len(rare)} rare labels for '{col}' to {path}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces rare labels (identified during fit) with the predefined 'Other' value.
        """
        df = X.copy()
        for col, rare in self.rare_labels_.items():
            new_name = self.config[col]['new_name']
            if col not in df.columns:
                logger.warning(f"Column '{col}' missing during transform. Skipping.")
                continue

            # Convert column to categorical if not already
            if not pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].astype('category')

            # Add 'Other' to the list of categories if not already present
            if new_name not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories([new_name])

            # Replace all rare categories with 'Other'
            df[col] = df[col].replace(rare, new_name)

            # Clean up unused categories to avoid clutter
            df[col] = df[col].cat.remove_unused_categories()

            logger.info(f"Replaced {len(rare)} rare labels in '{col}' with '{new_name}'.")

        return df

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

# Convenience function to apply rare label encoding in one step
def apply_rare_label_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shorthand for applying RareLabelEncoder with default config in one go.
    Suitable for use in one-off scripts or exploratory pipelines.
    """
    encoder = RareLabelEncoder()
    return encoder.fit_transform(df)
