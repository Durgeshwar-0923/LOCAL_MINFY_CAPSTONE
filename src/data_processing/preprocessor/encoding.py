import os
import joblib
import pandas as pd
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# Logger setup — fallback to basic config if project logger unavailable
try:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# Where the encoding models/artifacts will be saved
ARTIFACT_DIR = "artifacts/encoding"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Default categorical columns to apply one-hot encoding
DEFAULT_COLUMNS_TO_ENCODE = [
    'lead_origin',
    'lead_source',
    'what_is_your_current_occupation',
    'what_matters_most_to_you_in_choosing_a_course',
    'last_activity',
    'country',
    'specialization',
    'city',
    'tags',
    'lead_source_channel',
    'total_visits_binned',
    'total_time_spent_on_website_binned',
    'page_views_per_visit_binned'
]

# Ordinal columns and their manually defined order
ORDINAL_COLUMNS = ['lead_quality', 'lead_score_band']
ORDINAL_MAPPING = {
    'lead_quality': ['Low', 'Medium', 'High', 'Very High'],
    'lead_score_band': ['00 Low', '01 Medium', '02 High']
}

class OneHotEncodingTransformer(BaseEstimator, TransformerMixin):
    """
    This transformer performs:
    - One-Hot Encoding on nominal categorical features (non-ordered categories).
    - Ordinal Encoding on ordered categorical features.

    It also:
    - Saves the encoder object to disk for inference consistency.
    - Automatically determines and names output features for future integration.
    """

    def __init__(self, 
                 columns=None,
                 ordinal_columns=None,
                 ordinal_mapping=None,
                 artifact_dir=ARTIFACT_DIR,
                 drop_first=True):
        self.columns = columns if columns is not None else DEFAULT_COLUMNS_TO_ENCODE
        self.ordinal_columns = ordinal_columns if ordinal_columns is not None else ORDINAL_COLUMNS
        self.ordinal_mapping = ordinal_mapping if ordinal_mapping is not None else ORDINAL_MAPPING
        self.artifact_dir = artifact_dir
        self.drop_first = drop_first  # Helps prevent multicollinearity
        self.preprocessor = None
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y=None):
        """
        - Identifies relevant categorical columns.
        - Fits appropriate encoders (OneHot or Ordinal).
        - Persists the entire transformer using joblib.
        """
        df = X.copy()
        onehot_cols = [col for col in self.columns if col in df.columns and col not in self.ordinal_columns]
        ordinal_cols = [col for col in self.ordinal_columns if col in df.columns]

        transformers = []

        # One-Hot Encoding for nominal categories
        if onehot_cols:
            logger.info(f"📘 One-Hot Encoding columns: {onehot_cols}")
            transformers.append((
                "onehot",
                OneHotEncoder(handle_unknown='ignore', drop='first' if self.drop_first else None, sparse=False),
                onehot_cols
            ))

        # Ordinal Encoding for ordered categories
        if ordinal_cols:
            logger.info(f"📙 Ordinal Encoding columns: {ordinal_cols}")
            categories = [self.ordinal_mapping[col] for col in ordinal_cols]
            transformers.append((
                "ordinal",
                OrdinalEncoder(categories=categories),
                ordinal_cols
            ))

        # Combine encoders into a unified transformer
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'  # Keep non-categorical features untouched
        )

        # Fit the combined transformer
        self.preprocessor.fit(df)

        # Save transformer for reuse (training/inference consistency)
        artifact_path = os.path.join(self.artifact_dir, 'full_encoding_transformer.joblib')
        joblib.dump(self.preprocessor, artifact_path)
        logger.info(f"💾 Saved combined encoder to {artifact_path}")

        # Extract feature names from the transformer
        try:
            self.feature_names_ = self.preprocessor.get_feature_names_out()
        except Exception as e:
            logger.warning(f"⚠️ Sklearn version may not support get_feature_names_out. Falling back: {e}")
            self.feature_names_ = [f"f_{i}" for i in range(len(self.preprocessor.transform(df)[0]))]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the previously fitted encoders to transform the input data.
        Loads the encoder from disk if not already in memory.
        """
        df = X.copy()

        # Load from disk if missing in memory
        if self.preprocessor is None:
            artifact_path = os.path.join(self.artifact_dir, 'full_encoding_transformer.joblib')
            if os.path.exists(artifact_path):
                self.preprocessor = joblib.load(artifact_path)
                logger.info(f"📦 Loaded encoder from {artifact_path}")
            else:
                logger.error("❌ Encoder artifact not found. Skipping encoding.")
                return df

        # Apply transformation
        transformed_array = self.preprocessor.transform(df)

        # Retrieve or fallback to generic feature names
        try:
            all_feature_names = self.preprocessor.get_feature_names_out()
        except:
            all_feature_names = [f"f_{i}" for i in range(transformed_array.shape[1])]

        # Convert to DataFrame for usability
        df_transformed = pd.DataFrame(transformed_array, columns=all_feature_names, index=df.index)

        # Ensure numeric columns for model compatibility
        df_transformed = df_transformed.apply(pd.to_numeric, errors='ignore')

        logger.info(f"✅ Transformation complete. Output shape: {df_transformed.shape}")
        return df_transformed

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

# Optional one-shot function for quick use outside pipelines
def apply_feature_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply default encoding in a single function call (for notebooks or scripts).
    """
    transformer = OneHotEncodingTransformer()
    return transformer.fit_transform(df)
