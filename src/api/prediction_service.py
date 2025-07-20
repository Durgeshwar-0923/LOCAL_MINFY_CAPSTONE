"""
Module: src/api/prediction_service.py
Description: Handles loading of all artifacts and runs the full, multi-stage
             preprocessing pipeline on new data before making a prediction.
"""
import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import logging
import re

# --- Custom Preprocessing Modules ---
from src.data_processing.preprocessor.cleaning import clean_data
from src.data_processing.preprocessor.type_conversion import convert_column_types
from src.data_processing.preprocessor.missing_imputation import MissingValueImputer
from src.data_processing.preprocessor.outlier_handler import OutlierTransformer
from src.data_processing.preprocessor.feature_engineering import FeatureEngineeringTransformer
from src.data_processing.preprocessor.binning import BinningTransformer
from src.data_processing.preprocessor.rare_label_encoder import RareLabelEncoder
from src.data_processing.preprocessor.encoding import OneHotEncodingTransformer
from src.data_processing.preprocessor.clustering import EngagementClusteringTransformer
from src.data_processing.preprocessor.scaling import apply_feature_scaling
from src.data_processing.preprocessor.vif_filter import RFECVTransformer as VIFTransformer
from src.data_processing.preprocessor.feature_selection import SHAPFeatureSelector
from src.data_processing.preprocessor.target_encoder_wrapper import TargetEncoderWrapper
from sklearn.preprocessing import StandardScaler
from src.monitoring.drift_detector import log_drift_report


# --- Logger Setup ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Schema Definition ---
# --- CHANGE MADE HERE ---
# Removed the 'Converted' target column from the expected schema for prediction.
EXPECTED_RAW_SCHEMA = {
    'Prospect ID': 'prospect_id', 'Lead Number': 'lead_number', 'Lead Origin': 'lead_origin',
    'Lead Source': 'lead_source', 'Do Not Email': 'do_not_email', 'Do Not Call': 'do_not_call',
    'TotalVisits': 'total_visits',
    'Total Time Spent on Website': 'total_time_spent_on_website',
    'Page Views Per Visit': 'page_views_per_visit', 'Last Activity': 'last_activity',
    'Country': 'country', 'Specialization': 'specialization',
    'How did you hear about X Education': 'how_did_you_hear_about_x_education',
    'What is your current occupation': 'what_is_your_current_occupation',
    'What matters most to you in choosing a course': 'what_matters_most_to_you_in_choosing_a_course',
    'Search': 'search', 'Magazine': 'magazine', 'Newspaper Article': 'newspaper_article',
    'X Education Forums': 'x_education_forums', 'Newspaper': 'newspaper',
    'Digital Advertisement': 'digital_advertisement', 'Through Recommendations': 'through_recommendations',
    'Receive More Updates About Our Courses': 'receive_more_updates_about_our_courses',
    'Tags': 'tags', 'Lead Quality': 'lead_quality',
    'Update me on Supply Chain Content': 'update_me_on_supply_chain_content',
    'Get updates on DM Content': 'get_updates_on_dm_content', 'Lead Profile': 'lead_profile',
    'City': 'city', 'Asymmetrique Activity Index': 'asymmetrique_activity_index',
    'Asymmetrique Profile Index': 'asymmetrique_profile_index',
    'Asymmetrique Activity Score': 'asymmetrique_activity_score',
    'Asymmetrique Profile Score': 'asymmetrique_profile_score',
    'I agree to pay the amount through cheque': 'i_agree_to_pay_the_amount_through_cheque',
    'A free copy of Mastering The Interview': 'a_free_copy_of_mastering_the_interview',
    'Last Notable Activity': 'last_notable_activity'
}
# --- END OF CHANGE ---

# --- Artifact Paths ---
ARTIFACT_PATHS = {
    "model": "models:/LeadConversionModel/Production", "imputer": "artifacts/imputation/imputer_values.joblib",
    "outlier_config": "artifacts/outliers/outlier_config.joblib", "vif_selected_features": "artifacts/vif/vif_selected_features.joblib",
    "encoder": "artifacts/encoding/full_encoding_transformer.joblib", "initial_scaler": "artifacts/scaling/feature_scaler.joblib",
    "shap_selected_features_csv": "artifacts/feature_engineering/shap_feature_importance.csv",
    "final_target_encoder": "artifacts/final_target_encoder.joblib", "final_scaler": "artifacts/feature_scaler.pkl"
}

def _standardize_input_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    def standardize(name): return re.sub(r'[^0-9a-zA-Z]+', '_', str(name)).lower().strip('_')
    raw_to_snake = {standardize(k): v for k, v in EXPECTED_RAW_SCHEMA.items()}
    df_renamed = raw_df.copy()
    df_renamed.columns = [standardize(col) for col in df_renamed.columns]
    final_df = pd.DataFrame()
    expected_snake_cols = list(EXPECTED_RAW_SCHEMA.values())
    for standardized_raw_name, final_snake_name in raw_to_snake.items():
        if standardized_raw_name in df_renamed.columns: final_df[final_snake_name] = df_renamed[standardized_raw_name]
    for col in expected_snake_cols:
        if col not in final_df.columns: final_df[col] = np.nan
    final_df = final_df[expected_snake_cols]
    return final_df

# --- Pre-process reference data for drift check ---
REFERENCE_DATA_PATH = os.path.join("data", "raw", "Lead Scoring.csv")
PROCESSED_REFERENCE_DF = None
if os.path.exists(REFERENCE_DATA_PATH):
    logger.info("Pre-processing reference data for drift detection...")
    raw_ref_df = pd.read_csv(REFERENCE_DATA_PATH)
    ref_df_std = _standardize_input_columns(raw_ref_df)
    ref_df_cleaned = clean_data(ref_df_std, verbose=False)
    PROCESSED_REFERENCE_DF = convert_column_types(ref_df_cleaned)
    logger.info("✅ Pre-processed reference data is ready.")

def _load_all_artifacts():
    logger.info("Loading all prediction artifacts...")
    artifacts = {}
    for name, path in ARTIFACT_PATHS.items():
        try:
            if not os.path.exists(path) and not name == "model":
                 logger.warning(f"⚠️ Artifact not found at path: {path}. Setting to None.")
                 artifacts[name] = None
                 continue
            if name == "model":
                artifacts[name] = mlflow.pyfunc.load_model(model_uri=path)
                logger.info(f"✅ Successfully loaded model from {path}")
            elif name == "shap_selected_features_csv":
                artifacts[name] = pd.read_csv(path)
                logger.info(f"✅ Successfully loaded artifact: {name}")
            else:
                artifacts[name] = joblib.load(path)
                logger.info(f"✅ Successfully loaded artifact: {name}")
        except Exception as e:
            logger.error(f"❌ Failed to load artifact: {name} from path: {path}. Error: {e}")
            artifacts[name] = None
    return artifacts

ALL_ARTIFACTS = _load_all_artifacts()

def preprocess_input_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Starting preprocessing for input data with shape {raw_df.shape}")
    
    df = _standardize_input_columns(raw_df)

    # Drift check is now on cleaned data
    df_cleaned = clean_data(df, verbose=False)
    df_typed = convert_column_types(df_cleaned)

    if PROCESSED_REFERENCE_DF is not None:
        logger.info("Performing prediction-time data drift check on cleaned data...")
        try:
            log_drift_report(PROCESSED_REFERENCE_DF, df_typed, "prediction_vs_training_cleaned")
        except Exception as e:
            logger.warning(f"⚠️ Could not perform prediction-time drift check. Error: {e}")
    else:
        logger.warning("⚠️ Processed reference dataframe for drift check not found. Skipping.")

    df = df_typed # Continue pipeline with the type-converted data
    
    imputer = MissingValueImputer(); imputer.imputer_values_ = ALL_ARTIFACTS["imputer"]
    df = imputer.transform(df)
    
    outlier_transformer = OutlierTransformer(); df = outlier_transformer.transform(df)
    
    fe_transformer = FeatureEngineeringTransformer(); fe_transformer.fit(df); df = fe_transformer.transform(df)
    
    binning_transformer = BinningTransformer(); df = binning_transformer.transform(df)
    
    rare_label_encoder = RareLabelEncoder(); df = rare_label_encoder.transform(df)
    
    vif_features_to_keep = ALL_ARTIFACTS["vif_selected_features"]
    if vif_features_to_keep:
        numeric_cols = df.select_dtypes(include=np.number)
        vif_candidates = numeric_cols.loc[:, numeric_cols.nunique() > 2].columns
        cols_to_drop = [col for col in vif_candidates if col not in vif_features_to_keep]
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    encoder = ALL_ARTIFACTS["encoder"]
    if encoder:
        if hasattr(encoder, 'feature_names_in_'):
             expected_cols = encoder.feature_names_in_
             for col in expected_cols:
                 if col not in df.columns: df[col] = np.nan
        encoded_data = encoder.transform(df)
        feature_names = encoder.get_feature_names_out()
        df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)

    initial_scaler_artifact = ALL_ARTIFACTS["initial_scaler"]
    if initial_scaler_artifact:
        scaler_transformer = initial_scaler_artifact['transformer']
        scaler_cols = initial_scaler_artifact['columns']
        for col in scaler_cols:
            if col not in df.columns: df[col] = 0
        df_to_scale = df[scaler_cols]
        scaled_data = scaler_transformer.transform(df_to_scale)
        df_scaled = pd.DataFrame(scaled_data, columns=scaler_cols, index=df.index)
        df_unscaled = df.drop(columns=scaler_cols, errors='ignore')
        df = pd.concat([df_unscaled, df_scaled], axis=1)

    shap_importance_df = ALL_ARTIFACTS["shap_selected_features_csv"]
    if shap_importance_df is not None:
        top_features = shap_importance_df["feature"].head(50).tolist()
        cols_to_keep = [col for col in df.columns if col in top_features]
        df = df[cols_to_keep]
    
    final_target_encoder = ALL_ARTIFACTS["final_target_encoder"]
    if final_target_encoder:
        df = final_target_encoder.transform(df)

    final_scaler = ALL_ARTIFACTS["final_scaler"]
    if final_scaler:
        if hasattr(final_scaler, 'feature_names_in_'):
            expected_cols = final_scaler.feature_names_in_
            df = df.reindex(columns=expected_cols, fill_value=0)
        df_final = pd.DataFrame(final_scaler.transform(df), columns=df.columns, index=df.index)
    else:
        df_final = df

    logger.info(f"✅ Preprocessing fully complete. Final model-ready shape: {df_final.shape}")
    return df_final


def predict_probabilities(raw_df: pd.DataFrame) -> np.ndarray:
    model = ALL_ARTIFACTS.get("model")
    if model is None:
        raise RuntimeError("Model could not be loaded. The prediction service cannot proceed.")
    
    df_prepped = preprocess_input_data(raw_df)
    
    if hasattr(model, 'metadata') and model.metadata and model.metadata.signature:
        model_features = model.metadata.get_input_schema().input_names()
        df_prepped = df_prepped.reindex(columns=model_features, fill_value=0)
        logger.info("Reordered and aligned columns to match model's expected schema.")

    predictions = model.predict(df_prepped)
    
    if isinstance(predictions, pd.DataFrame):
        probs = predictions.iloc[:, 0].values
    elif isinstance(predictions, np.ndarray):
        if predictions.ndim == 2 and predictions.shape[1] == 2:
            probs = predictions[:, 1]
        else:
            probs = predictions.flatten()
    else:
        raise TypeError(f"Unexpected prediction output type: {type(predictions)}")

    logger.info("✅ Prediction successful.")
    return probs
