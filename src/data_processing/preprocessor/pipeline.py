import os
import pandas as pd
import numpy as np
import mlflow
import joblib
import logging
import json

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

# Custom modules
from src.data_processing.preprocessor.cleaning import clean_data
from src.data_processing.preprocessor.type_conversion import convert_column_types
from src.data_processing.preprocessor.missing_imputation import MissingValueImputer
from src.data_processing.preprocessor.outlier_handler import OutlierTransformer
from src.data_processing.preprocessor.feature_engineering import (
    FeatureEngineeringTransformer,
    compute_and_save_shap_importance,
    compute_permutation_importance
)
from src.data_processing.preprocessor.binning import BinningTransformer
from src.data_processing.preprocessor.rare_label_encoder import RareLabelEncoder
from src.data_processing.preprocessor.encoding import OneHotEncodingTransformer
from src.data_processing.preprocessor.clustering import EngagementClusteringTransformer
from src.data_processing.preprocessor.scaling import apply_feature_scaling
from src.data_processing.preprocessor.vif_filter import RFECVTransformer
from src.data_processing.preprocessor.feature_selection import SHAPFeatureSelector

try:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

mlflow.set_experiment("Lead_Conversion_Preprocessing_Boosted")

OUTPUT_DIRS = [
    "outputs/plots", "outputs/distributions", "outputs/importance",
    "outputs/stages", "artifacts", "data/processed", "reports", "artifacts/pipeline"
]
for d in OUTPUT_DIRS:
    os.makedirs(d, exist_ok=True)

SHAP_CSV_PATH = "artifacts/feature_engineering/shap_feature_importance.csv"
SHAP_PLOT_PATH = "artifacts/feature_engineering/shap_feature_importance.png"

PIPELINE_STEPS = [
    ("1_clean", clean_data),                                 # Handle invalid and noisy text or patterns
    ("2_type_convert", convert_column_types),                # Convert to correct dtypes for processing
    ("3_impute", MissingValueImputer()),                     # Handle missing data with statistical imputers
    ("4_outlier", OutlierTransformer()),                     # Detect and cap or remove outliers
    ("5_engineer", FeatureEngineeringTransformer()),         # Create new informative features
    ("6_bin", BinningTransformer()),                          # Bin continuous features into categorical buckets
    ("7_rare_label", RareLabelEncoder()),                     # Merge rare labels to improve signal
    ("8_vif", RFECVTransformer()),                            # Remove multicollinear numerical features
    ("9_encode", OneHotEncodingTransformer()),               # Encode categorical values post-binning
    ("10_scale", apply_feature_scaling),                      # Normalize feature ranges
    ("11_compute_shap", None),                                # Placeholder for SHAP computation step (custom handled)
    ("12_cluster", EngagementClusteringTransformer()),        # Assign cluster labels for segmentation
    ("13_shap_feature_selection", SHAPFeatureSelector())    # Filter least useful features based on SHAP
]

def log_stage(df_before, df_after, step_name):
    rows_before, cols_before = df_before.shape
    rows_after, cols_after = df_after.shape
    logger.info(f"Step '{step_name}': rows {rows_before}->{rows_after}, cols {cols_before}->{cols_after}")
    print(f"[{step_name}] Rows: {rows_before}->{rows_after} | Cols: {cols_before}->{cols_after}")
    if cols_before != cols_after:
        logger.info(f"📉 {cols_before - cols_after} features removed at step '{step_name}'")

def run_preprocessing_pipeline(df: pd.DataFrame, target_col: str = 'Converted', is_training: bool = True):
    run_id = f"preproc_{'train' if is_training else 'infer'}"
    target_series = df[target_col].copy() if (target_col in df.columns and is_training) else None

    with mlflow.start_run(run_name=run_id):
        logger.info(f"🚀 Starting preprocessing MLflow run: {run_id}")
        mlflow.log_param("initial_rows", len(df))
        mlflow.log_param("initial_cols", df.shape[1])

        for step_name, transformer in PIPELINE_STEPS:
            df_before = df.copy()
            logger.info(f"🔧 Applying {step_name}...")

            try:
                if step_name == "10_scale":
                    # Scaling function takes df and params
                    df = transformer(df, target_col=target_col, is_training=is_training)

                elif step_name == "8_vif":
                    numeric_df = df.select_dtypes(include=[np.number])
                    valid_features = numeric_df.loc[:, numeric_df.nunique() > 2]
                    if valid_features.shape[1] > 0:
                        df_vif_filtered = transformer.fit_transform(valid_features)
                        df_non_numeric = df.drop(columns=valid_features.columns, errors='ignore')
                        df = pd.concat([df_vif_filtered, df_non_numeric], axis=1)
                        df = df.loc[:, ~df.columns.duplicated()]
                    else:
                        logger.warning("⚠️ No valid numeric features for VIF. Skipping.")

                elif step_name == "11_compute_shap":
                    # Reattach target column if available before SHAP
                    if target_series is not None:
                        if target_col not in df.columns:
                            df[target_col] = target_series.reindex(df.index).values
                        logger.info(f"ℹ️ Re-attached target column '{target_col}' before SHAP step.")

                        # --- FIX: Convert one-hot encoded columns to numeric before SHAP ---
                        onehot_cols = [col for col in df.columns if col.startswith("onehot__")]
                        if onehot_cols:
                            logger.info(f"🔧 Converting one-hot encoded columns to numeric dtype: "
                                        f"{onehot_cols[:5]}{'...' if len(onehot_cols) > 5 else ''}")
                            df[onehot_cols] = df[onehot_cols].apply(pd.to_numeric, errors='coerce')

                        # Debug logs to verify alignment
                        logger.info(f"Before SHAP step, df shape: {df.shape}, target_col present: {target_col in df.columns}")
                        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                        logger.info(f"DEBUG: Numeric columns before SHAP: {numeric_cols}")
                        logger.info(f"All columns dtypes:\n{df.dtypes.value_counts()}")
                        logger.info(f"Columns and dtypes:\n{df.dtypes}")

                        if target_col in df.columns:
                            logger.info(f"Target value count: {len(df[target_col])}, df row count: {df.shape[0]}")

                        try:
                            compute_and_save_shap_importance(df, target_col=target_col)
                            compute_permutation_importance(df, target_col=target_col)
                            # Log SHAP artifacts to MLflow
                            for artifact_path in [SHAP_CSV_PATH, SHAP_PLOT_PATH]:
                                if os.path.exists(artifact_path):
                                    mlflow.log_artifact(artifact_path, artifact_path="feature_engineering")
                                    logger.info(f"✅ Logged artifact to MLflow: {artifact_path}")
                            # Drop target after SHAP to avoid leakage downstream
                            df = df.drop(columns=[target_col])
                        except Exception as e:
                            logger.warning(f"⚠️ SHAP/Permutation importance step failed: {e}")
                    else:
                        logger.warning(f"⚠️ Target column '{target_col}' not found or None; skipping SHAP step.")

                elif step_name == "13_shap_feature_selection":
                    # Use SHAP CSV artifact if exists, else skip selection gracefully
                    if os.path.exists(SHAP_CSV_PATH):
                        df = transformer.fit_transform(df)
                        logger.info("✅ SHAP feature selection applied successfully.")
                    else:
                        logger.warning(f"⚠️ SHAP feature importance CSV not found at {SHAP_CSV_PATH}. Skipping SHAP feature selection.")

                else:
                    # Default: fit_transform if training else transform
                    if hasattr(transformer, 'fit_transform') and is_training:
                        df = transformer.fit_transform(df)
                    elif hasattr(transformer, 'transform'):
                        df = transformer.transform(df)
                    else:
                        df = transformer(df)

            except Exception as e:
                logger.warning(f"⚠️ Step '{step_name}' failed: {e}")

            log_stage(df_before, df, step_name)

            removed_cols = set(df_before.columns) - set(df.columns)
            if removed_cols:
                removed_path = f"outputs/stages/{step_name}_removed_columns.json"
                with open(removed_path, "w") as f:
                    json.dump(list(removed_cols), f)
                mlflow.log_artifact(removed_path, artifact_path=f"stages/{step_name}")

            stage_path = f"outputs/stages/{step_name}.csv"
            df.to_csv(stage_path, index=False)
            mlflow.log_artifact(stage_path, artifact_path=f"stages/{step_name}")
# --- ✅ Reattach target column before saving final processed data ---
        if target_series is not None:
            df[target_col] = target_series.reindex(df.index).values

        final_path = "data/processed/13_final_features.csv"
        df.to_csv(final_path, index=False)
        mlflow.log_artifact(final_path, artifact_path='processed')

        pipeline_path = 'artifacts/pipeline/preprocessing_pipeline.joblib'
        joblib.dump(PIPELINE_STEPS, pipeline_path)
        mlflow.log_artifact(pipeline_path, artifact_path='pipeline')

        mlflow.log_metric("final_rows", df.shape[0])
        mlflow.log_metric("final_cols", df.shape[1])

        final_shape_str = f"{df.shape[0]} rows × {df.shape[1]} columns"
        logger.info(f"✅ Preprocessing complete. Final dataset shape: {final_shape_str}")
        print(f"\n✅ Preprocessing completed successfully!")
        print(f"🗖 Final dataset size: {final_shape_str}\n")

    return df


def run_pipeline_with_tracking(raw_data_path: str = 'data/raw/Lead Scoring.csv', target_col: str = 'Converted', is_training: bool = True):
    if not os.path.exists(raw_data_path):
        logger.error(f"Raw data path not found: {raw_data_path}")
        return
    df_raw = pd.read_csv(raw_data_path)
    run_preprocessing_pipeline(df_raw, target_col=target_col, is_training=is_training)


def main(raw_path='data/raw/Lead Scoring.csv'):
    if not os.path.exists(raw_path):
        logger.error(f"Raw data missing: {raw_path}")
        return
    df_raw = pd.read_csv(raw_path)
    df_processed = run_preprocessing_pipeline(df_raw, target_col='Converted', is_training=True)
    df_processed.to_csv('data/processed/processed_data.csv', index=False)
    logger.info("Processed data saved.")


if __name__ == '__main__':
    main()
