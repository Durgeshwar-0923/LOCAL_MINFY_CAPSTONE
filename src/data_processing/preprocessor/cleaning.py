import os
import re
import numpy as np
import pandas as pd
import logging

# Logger setup
try:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# --- 👇 CHANGE MADE HERE ---
# Added 'Last Activity' and 'Last Notable Activity' to the drop list.
# These columns are high-leakage features as they often contain information
# about whether a lead was already converted, leading to artificially perfect model scores.
ID_COLS = [
    'Prospect ID', 
    'Lead Number', 
    'Last Activity', 
    'Last Notable Activity',
    'Lead Profile', 
    'I agree to pay the amount through cheque'
]
# --- END OF CHANGE ---

CONST_COLS = [
    'Magazine', 'Receive More Updates About Our Courses',
    'Update me on Supply Chain Content', 'Get updates on DM Content',
    'I agree to pay by cheque'
]
HIGH_MISS_COLS = [
    'Asymmetrique Activity Index', 'Asymmetrique Profile Index',
    'Asymmetrique Activity Score', 'Asymmetrique Profile Score'
]
PLACEHOLDERS = ["", "NA", "N/A", "null", "Null", "none", "None", "-", "--", "Select"]
MISSING_THRESHOLD = 0.5
VARIANCE_THRESHOLD = 0.01

# --- Ordinal Category Mapping ---
ORDINAL_MAPPINGS = {
    'lead_quality': {
        '01 Low': 1,
        '02 Medium': 2,
        '03 High': 3
    }
}

def _standardize_name(name: str) -> str:
    name = re.sub(r'[^0-9a-zA-Z]+', '_', name)
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', name)
    name = re.sub(r'__+', '_', name)
    return name.lower().strip('_')

def _map_ordinal_prefixes(df: pd.DataFrame, mappings: dict, verbose: bool = True) -> pd.DataFrame:
    for col, mapping in mappings.items():
        std_col = _standardize_name(col)
        if std_col in df.columns:
            df[std_col] = df[std_col].map(mapping).astype('Int64')
            if verbose:
                logger.info(f"🔢 Mapped ordinal values in '{std_col}' using predefined mapping.")
        else:
            if verbose:
                logger.warning(f"⚠️ Ordinal column '{std_col}' not found. Skipping.")
    return df

def clean_data(df: pd.DataFrame,
               drop_high_missing: bool = True,
               dynamic_var_drop: bool = True,
               keep_cols: list = None,
               verbose: bool = True,
               return_metadata: bool = False) -> pd.DataFrame | tuple:
    """
    Clean the dataset: standardize names, drop known noisy columns,
    replace placeholders, handle leakage, map ordinals.

    Returns:
        - Cleaned DataFrame
        - (optional) Dict with metadata (dropped columns, etc.)
    """
    if verbose:
        logger.info("🧹 Starting data cleaning...")
    df_clean = df.copy()

    # 1. Standardize column names
    orig_cols = df_clean.columns.tolist()
    new_cols = [_standardize_name(col) for col in orig_cols]
    df_clean.columns = new_cols
    if verbose:
        logger.info("🔧 Column names standardized to snake_case.")

    # 2. Drop static columns (ID, constants, high-missing)
    drop_raw = ID_COLS + CONST_COLS
    if drop_high_missing:
        drop_raw += HIGH_MISS_COLS
    drop_snake = {_standardize_name(c) for c in drop_raw}
    to_drop = [c for c in df_clean.columns if c in drop_snake]

    # 2b. High-missing dynamic check
    if drop_high_missing:
        miss_frac = df_clean.isna().mean()
        high_miss = miss_frac[miss_frac > MISSING_THRESHOLD].index.tolist()
        to_drop += high_miss

    # 2c. Low-variance numeric features
    if dynamic_var_drop:
        variances = df_clean.select_dtypes(include=[np.number]).var()
        low_var = variances[variances < VARIANCE_THRESHOLD].index.tolist()
        to_drop += low_var

    # 2d. Respect `keep_cols`
    if keep_cols:
        to_drop = [c for c in to_drop if c not in keep_cols]

    to_drop = sorted(set(to_drop))
    df_clean.drop(columns=to_drop, inplace=True, errors='ignore')
    if verbose and to_drop:
        logger.info(f"🗑️ Dropped {len(to_drop)} columns: {to_drop}")

    # 3. Replace placeholders
    df_clean.replace(to_replace=PLACEHOLDERS, value=np.nan, inplace=True, regex=False)
    if verbose:
        logger.info("✅ Placeholder values replaced with NaN.")

    # 4. Remove duplicates
    before = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    after = len(df_clean)
    if verbose and before != after:
        logger.info(f"🧼 Removed {before - after} duplicate rows.")

    # 5. Drop potential leakage columns
    leak_cols = [c for c in df_clean.columns if re.search(r"(convert|target)", c, re.IGNORECASE) and c != 'converted']
    if leak_cols:
        df_clean.drop(columns=leak_cols, inplace=True, errors='ignore')
        if verbose:
            logger.warning(f"⚠️ Dropped potential leakage columns: {leak_cols}")

    # 6. Map ordinal values
    df_clean = _map_ordinal_prefixes(df_clean, ORDINAL_MAPPINGS, verbose)

    if verbose:
        logger.info("🎯 Data cleaning complete.")

    if return_metadata:
        return df_clean, {
            "dropped_columns": to_drop,
            "original_columns": orig_cols,
            "final_columns": df_clean.columns.tolist(),
            "num_rows_removed": before - after,
        }

    return df_clean
