import pandas as pd
import logging

# Setup logger for tracking type conversion process
try:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# --- Synchronized configuration with EDA insights ---
# These were found to be continuous numerical features that may require type enforcement and scaling
NUMERIC_COLUMNS = [
    'total_visits',
    'total_time_spent_on_website',
    'page_views_per_visit'
]

# These binary columns contain Yes/No values, which will be converted to 1/0 for modeling compatibility
BINARY_COLUMNS = [
    'do_not_email',
    'do_not_call',
    'search',
    'newspaper_article',
    'x_education_forums',
    'newspaper',
    'digital_advertisement',
    'through_recommendations',
    'a_free_copy_of_mastering_the_interview'
]

# Ordinal mapping helps preserve meaningful order in categories — e.g., High > Medium > Low
ORDINAL_MAPPING = {
    'lead_quality': {
        '01 High': 3,
        '02 Medium': 2,
        '03 Low': 1,
        '04 Very Low': 0
    },
}


def convert_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts column types in the DataFrame to ensure correct data interpretation and efficient modeling.

    Steps:
    - Converts known numeric columns to float, coercing invalid values to NaN.
    - Converts binary categorical columns (Yes/No) to 1/0 with Int64 for downstream model compatibility.
    - Applies ordinal encoding to columns where order matters (e.g., lead_quality).
    - Converts remaining object-type columns to category to save memory and prep for encoding.

    Args:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        pd.DataFrame: Cleaned and type-converted DataFrame.
    """
    logger.info("🧾 Starting column type conversion...")
    df_copy = df.copy()

    # --- Numeric Conversion ---
    # Ensures mathematical operations don't fail due to string/invalid types
    for col in NUMERIC_COLUMNS:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            logger.info(f"🔢 Converted '{col}' to numeric.")
        else:
            logger.warning(f"⚠️ Numeric column '{col}' not found. Skipping.")

    # --- Binary Conversion ---
    # Maps 'Yes' to 1 and 'No' to 0 for model consumption (e.g., logistic regression)
    for col in BINARY_COLUMNS:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].map({'Yes': 1, 'No': 0})
            df_copy[col] = df_copy[col].astype('Int64')  # Keeps NaNs if mapping fails
            logger.info(f"🔘 Converted '{col}' to binary Int64 (0/1).")
        else:
            logger.warning(f"⚠️ Binary column '{col}' not found. Skipping.")

    # --- Ordinal Encoding ---
    # Retains the ranking information in ordered categorical features
    for col, mapping in ORDINAL_MAPPING.items():
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].map(mapping).astype('Int64')
            logger.info(f"🎚️ Ordinal encoded '{col}' using mapping: {mapping}")
        else:
            logger.warning(f"⚠️ Ordinal column '{col}' not found. Skipping.")

    # --- Object to Category Conversion ---
    # Helps reduce memory footprint and prepares for categorical encoding
    object_cols = df_copy.select_dtypes(include=['object']).columns
    for col in object_cols:
        df_copy[col] = df_copy[col].astype('category')
        logger.info(f"📦 Converted '{col}' to category for efficient storage.")

    logger.info("✅ Type conversion complete.")
    return df_copy
