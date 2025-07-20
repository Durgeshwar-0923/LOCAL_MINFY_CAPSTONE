import os
import uuid
import pandas as pd
import flask
import boto3
import json
import tarfile
import traceback
import mlflow
import re
import numpy as np
import joblib 

from werkzeug.utils import secure_filename
from src.utils.logger import setup_logger

# --- This is the key fix: Import all the individual preprocessor components ---
from src.data_processing.preprocessor.cleaning import clean_data
from src.data_processing.preprocessor.type_conversion import convert_column_types
from src.data_processing.preprocessor.missing_imputation import MissingValueImputer
from src.data_processing.preprocessor.outlier_handler import OutlierTransformer
from src.data_processing.preprocessor.feature_engineering import FeatureEngineeringTransformer
from src.data_processing.preprocessor.binning import BinningTransformer
from src.data_processing.preprocessor.rare_label_encoder import RareLabelEncoder
from src.data_processing.preprocessor.encoding import OneHotEncodingTransformer
from src.data_processing.preprocessor.vif_filter import RFECVTransformer as VIFTransformer
from src.data_processing.preprocessor.target_encoder_wrapper import TargetEncoderWrapper
from sklearn.preprocessing import StandardScaler

# --- Initial Setup ---
logger = setup_logger(__name__)
app = flask.Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = os.urandom(24)

# --- Configuration ---
S3_BUCKET_NAME = "flaskcapstonebucket"
MODEL_S3_KEY = "models/LeadConversionModel/model.tar.gz"
MODEL_PATH = "/tmp/model" 

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
PRED_FOLDER = os.path.join(PROJECT_ROOT, "predictions")
REFERENCE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Lead Scoring.csv")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PRED_FOLDER, exist_ok=True)

# --- Schema Standardization (No longer includes 'Converted') ---
EXPECTED_RAW_SCHEMA = {
    'Prospect ID': 'prospect_id', 'Lead Number': 'lead_number', 'Lead Origin': 'lead_origin',
    'Lead Source': 'lead_source', 'Do Not Email': 'do_not_email', 'Do Not Call': 'do_not_call',
    'TotalVisits': 'totalvisits', 'Total Time Spent on Website': 'total_time_spent_on_website',
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

def _standardize_input_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    def standardize(name): return re.sub(r'[^0-9a-zA-Z]+', '_', str(name)).lower().strip('_')
    df_renamed = raw_df.copy()
    df_renamed.columns = [standardize(col) for col in df_renamed.columns]
    raw_to_snake = {standardize(k): v for k, v in EXPECTED_RAW_SCHEMA.items()}
    final_df = pd.DataFrame()
    expected_cols = list(EXPECTED_RAW_SCHEMA.values())
    for standardized_user_col in df_renamed.columns:
        if standardized_user_col in raw_to_snake:
            final_snake_name = raw_to_snake[standardized_user_col]
            final_df[final_snake_name] = df_renamed[standardized_user_col]
    for col in expected_cols:
        if col not in final_df.columns:
            final_df[col] = np.nan
    return final_df[expected_cols]

# --- Preprocessing Pipeline Class ---
class PreprocessingPipeline:
    def __init__(self, artifact_path='/tmp/model/artifacts'):
        self.artifact_path = artifact_path
        self.artifacts = self._load_all_artifacts()

    def _load_all_artifacts(self):
        logger.info(f"Loading all preprocessing artifacts from: {self.artifact_path}")
        artifacts = {}
        # Single file artifacts
        single_file_map = {
            "imputer": "imputation/imputer_values.joblib", "outlier_config": "outliers/outlier_config.joblib",
            "vif_selected_features": "vif/vif_selected_features.joblib", "encoder": "encoding/full_encoding_transformer.joblib",
            "initial_scaler": "scaling/feature_scaler.joblib", "final_target_encoder": "final_target_encoder.joblib",
            "final_scaler": "feature_scaler.pkl"
        }
        for name, rel_path in single_file_map.items():
            full_path = os.path.join(self.artifact_path, rel_path)
            try:
                artifacts[name] = joblib.load(full_path)
                logger.info(f"✅ Successfully loaded artifact: {name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load artifact: {name} from path: {full_path}. Setting to None.")
                artifacts[name] = None
        
        # Directory-based artifacts (for binning and rare labels)
        artifacts["binning_configs"] = {}
        binning_path = os.path.join(self.artifact_path, "binning")
        if os.path.exists(binning_path):
            for file in os.listdir(binning_path):
                name = file.replace("_binning_config.joblib", "")
                artifacts["binning_configs"][name] = joblib.load(os.path.join(binning_path, file))
        
        artifacts["rare_label_configs"] = {}
        rare_label_path = os.path.join(self.artifact_path, "rare_labels")
        if os.path.exists(rare_label_path):
            for file in os.listdir(rare_label_path):
                name = file.replace("_rare_labels.joblib", "")
                artifacts["rare_label_configs"][name] = joblib.load(os.path.join(rare_label_path, file))

        return artifacts

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = _standardize_input_columns(raw_df)
        df = clean_data(df, verbose=False)
        df = convert_column_types(df)
        
        imputer = MissingValueImputer(); imputer.imputer_values_ = self.artifacts.get("imputer", {})
        df = imputer.transform(df)
        
        outlier_transformer = OutlierTransformer(); outlier_transformer.clip_bounds_ = self.artifacts.get("outlier_config", {})
        df = outlier_transformer.transform(df)
        
        fe_transformer = FeatureEngineeringTransformer(); fe_transformer.fit(df); df = fe_transformer.transform(df)
        
        # --- FIX APPLIED HERE: Manually set the loaded artifacts ---
        binning_transformer = BinningTransformer(); 
        binning_transformer.bin_edges = self.artifacts.get("binning_configs", {})
        df = binning_transformer.transform(df)
        
        rare_label_encoder = RareLabelEncoder(); 
        rare_label_encoder.rare_labels_ = self.artifacts.get("rare_label_configs", {})
        df = rare_label_encoder.transform(df)
        # --- END OF FIX ---
        
        vif_features_to_keep = self.artifacts.get("vif_selected_features")
        if vif_features_to_keep:
            numeric_cols = df.select_dtypes(include=np.number).columns
            vif_candidates = numeric_cols.loc[:, numeric_cols.nunique() > 2].columns
            cols_to_drop = [col for col in vif_candidates if col not in vif_features_to_keep]
            df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        encoder = self.artifacts.get("encoder")
        if encoder:
            df['converted'] = 0 
            encoded_data = encoder.transform(df)
            feature_names = encoder.get_feature_names_out()
            df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)
            cols_to_drop = [col for col in df.columns if 'converted' in col]
            df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        initial_scaler_artifact = self.artifacts.get("initial_scaler")
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
        
        final_target_encoder = self.artifacts.get("final_target_encoder")
        if final_target_encoder:
            df = final_target_encoder.transform(df)

        final_scaler = self.artifacts.get("final_scaler")
        if final_scaler:
            df_final = pd.DataFrame(final_scaler.transform(df), columns=df.columns, index=df.index)
        else:
            df_final = df
        logger.info(f"✅ Preprocessing fully complete.")
        return df_final

# --- Load Model and Preprocessor at Startup ---
def download_and_load_model():
    # ... (This logic is correct)
    try:
        os.makedirs(MODEL_PATH, exist_ok=True)
        local_tar_path = os.path.join(MODEL_PATH, "model.tar.gz")
        logger.info(f"Downloading model from s3://{S3_BUCKET_NAME}/{MODEL_S3_KEY}...")
        s3_client = boto3.client("s3")
        s3_client.download_file(S3_BUCKET_NAME, MODEL_S3_KEY, local_tar_path)
        logger.info("✅ Model downloaded successfully.")
        logger.info(f"Unpacking {local_tar_path}...")
        with tarfile.open(local_tar_path, "r:gz") as tar:
            tar.extractall(path=MODEL_PATH)
        logger.info("✅ Model unpacked successfully.")
        model_artifact_name = "StackingEnsemble"
        mlflow_model_path = os.path.join(MODEL_PATH, model_artifact_name)
        logger.info(f"Loading MLflow model from: {mlflow_model_path}")
        model = mlflow.pyfunc.load_model(mlflow_model_path)
        logger.info("✅ MLflow model loaded.")
        artifact_path = os.path.join(MODEL_PATH, "artifacts")
        logger.info(f"Loading preprocessor from: {artifact_path}")
        preprocessor = PreprocessingPipeline(artifact_path=artifact_path)
        logger.info("✅ Preprocessor loaded.")
        return model, preprocessor
    except Exception as e:
        logger.error("❌ FAILED TO LOAD MODEL AND PREPROCESSOR ON STARTUP.")
        traceback.print_exc()
        return None, None

model, preprocessor = download_and_load_model()

# --- Flask App Routes ---
# The rest of the file does not need to change.
@app.route("/")
def home():
    return flask.render_template("index.html", model_name="LeadConversionModel", model_loaded=(model is not None))

@app.route("/upload", methods=["POST"])
def upload():
    if not model or not preprocessor:
        flask.flash("Model is not loaded. Cannot process requests. Check server logs.", "danger")
        return flask.redirect(flask.url_for('home'))
    if 'file' not in flask.request.files:
        flask.flash("No file part in the request.", "warning")
        return flask.redirect(flask.url_for('home'))
    file = flask.request.files['file']
    if file.filename == '':
        flask.flash("No file selected for uploading.", "warning")
        return flask.redirect(flask.url_for('home'))
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        try:
            raw_df = pd.read_csv(file_path)
            processed_df = preprocessor.transform(raw_df)
            if hasattr(model, 'metadata') and model.metadata.signature:
                model_features = model.metadata.get_input_schema().input_names()
                processed_df = processed_df.reindex(columns=model_features, fill_value=0)
            predictions = model.predict(processed_df)
            probabilities = predictions.iloc[:, 0].tolist() if isinstance(predictions, pd.DataFrame) else predictions.tolist()
            results_df = raw_df.copy()
            results_df['Lead_Conversion_Probability'] = [f"{p:.2%}" for p in probabilities]
            results_df['Lead_Converted_Prediction'] = (pd.Series(probabilities) > 0.5).astype(int)
            result_id = uuid.uuid4().hex[:8]
            result_file = f"prediction_{result_id}_{filename}"
            result_path = os.path.join(PRED_FOLDER, result_file)
            results_df.to_csv(result_path, index=False)
            table_headers = results_df.columns.tolist()
            table_rows = results_df.head(100).values.tolist()
            return flask.render_template(
                "result.html", table_headers=table_headers, table_rows=table_rows,
                result_file=result_file, model_name="LeadConversionModel", num_records=len(results_df)
            )
        except Exception as e:
            logger.error(f"An error occurred during prediction: {e}", exc_info=True)
            flask.flash(f"An error occurred during processing: {e}", "danger")
            return flask.redirect(flask.url_for('home'))
    else:
        flask.flash("Invalid file type. Please upload a CSV file.", "danger")
        return flask.redirect(flask.url_for('home'))

@app.route("/sample")
def sample():
    return flask.send_file(REFERENCE_DATA_PATH, as_attachment=True)

@app.route("/download/<filename>")
def download(filename):
    safe_filename = secure_filename(filename)
    file_path = os.path.join(PRED_FOLDER, safe_filename)
    if os.path.exists(file_path):
        return flask.send_file(file_path, as_attachment=True)
    else:
        flask.flash("File not found.", "danger")
        return flask.redirect(flask.url_for('home'))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
