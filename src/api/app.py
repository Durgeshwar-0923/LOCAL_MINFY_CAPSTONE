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

# --- Configuration ---
S3_BUCKET_NAME = "flaskcapstonebucket"
MODEL_S3_KEY = "models/LeadConversionModel/model.tar.gz"
MODEL_PATH = "/tmp/model" # A temporary directory on the EC2 server

# --- This class now lives inside app.py to avoid import errors ---
class PreprocessingPipeline:
    def __init__(self, artifact_path='/tmp/model/artifacts'):
        self.artifact_path = artifact_path
        self.artifacts = self._load_all_artifacts()

    def _load_all_artifacts(self):
        logger.info(f"Loading all preprocessing artifacts from: {self.artifact_path}")
        artifacts = {}
        artifact_map = {
            "imputer": "imputation/imputer_values.joblib",
            "outlier_config": "outliers/outlier_config.joblib",
            "vif_selected_features": "vif/vif_selected_features.joblib",
            "encoder": "encoding/full_encoding_transformer.joblib",
            "initial_scaler": "scaling/feature_scaler.joblib",
            "final_target_encoder": "final_target_encoder.joblib",
            "final_scaler": "feature_scaler.pkl"
        }
        for name, rel_path in artifact_map.items():
            full_path = os.path.join(self.artifact_path, rel_path)
            try:
                artifacts[name] = joblib.load(full_path)
                logger.info(f"✅ Successfully loaded artifact: {name}")
            except Exception as e:
                logger.error(f"❌ Failed to load artifact: {name} from path: {full_path}. Error: {e}")
                artifacts[name] = None
        return artifacts

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()
        df = clean_data(df, verbose=False)
        df = convert_column_types(df)
        
        imputer = MissingValueImputer(); imputer.imputer_values_ = self.artifacts["imputer"]
        df = imputer.transform(df)
        
        outlier_transformer = OutlierTransformer(); outlier_transformer.clip_bounds_ = self.artifacts["outlier_config"]
        df = outlier_transformer.transform(df)
        
        fe_transformer = FeatureEngineeringTransformer(); fe_transformer.fit(df); df = fe_transformer.transform(df)
        
        binning_transformer = BinningTransformer(); df = binning_transformer.transform(df)
        
        rare_label_encoder = RareLabelEncoder(); df = rare_label_encoder.transform(df)
        
        vif_features_to_keep = self.artifacts["vif_selected_features"]
        if vif_features_to_keep:
            numeric_cols = df.select_dtypes(include=np.number)
            vif_candidates = numeric_cols.loc[:, numeric_cols.nunique() > 2].columns
            cols_to_drop = [col for col in vif_candidates if col not in vif_features_to_keep]
            df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        encoder = self.artifacts["encoder"]
        if encoder:
            encoded_data = encoder.transform(df)
            feature_names = encoder.get_feature_names_out()
            df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)

        initial_scaler_artifact = self.artifacts["initial_scaler"]
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
        
        final_target_encoder = self.artifacts["final_target_encoder"]
        if final_target_encoder:
            df = final_target_encoder.transform(df)

        final_scaler = self.artifacts["final_scaler"]
        if final_scaler:
            df_final = pd.DataFrame(final_scaler.transform(df), columns=df.columns, index=df.index)
        else:
            df_final = df

        logger.info(f"✅ Preprocessing fully complete. Final model-ready shape: {df_final.shape}")
        return df_final

# --- Load Model and Preprocessor at Startup ---
def download_and_load_model():
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
@app.route("/")
def home():
    return flask.render_template("index.html", model_name="LeadConversionModel", model_loaded=(model is not None))

@app.route("/upload", methods=["POST"])
def upload():
    # ... (This part of your code does not need to change)
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
        try:
            raw_df = pd.read_csv(file.stream)
            processed_df = preprocessor.transform(raw_df)
            if hasattr(model, 'metadata') and model.metadata.signature:
                model_features = model.metadata.get_input_schema().input_names()
                processed_df = processed_df.reindex(columns=model_features, fill_value=0)
            predictions = model.predict(processed_df)
            probabilities = predictions.iloc[:, 0].tolist() if isinstance(predictions, pd.DataFrame) else predictions.tolist()
            results_df = raw_df.copy()
            results_df['Lead_Conversion_Probability'] = [f"{p:.2%}" for p in probabilities]
            results_df['Lead_Converted_Prediction'] = (pd.Series(probabilities) > 0.5).astype(int)
            table_headers = results_df.columns.tolist()
            table_rows = results_df.head(100).values.tolist()
            return flask.render_template(
                "result.html",
                table_headers=table_headers,
                table_rows=table_rows,
                model_name="LeadConversionModel",
                num_records=len(results_df)
            )
        except Exception as e:
            logger.error(f"An error occurred during prediction: {e}", exc_info=True)
            flask.flash(f"An error occurred during processing: {e}", "danger")
            return flask.redirect(flask.url_for('home'))
    else:
        flask.flash("Invalid file type. Please upload a CSV file.", "danger")
        return flask.redirect(flask.url_for('home'))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
