import os
import uuid
import pandas as pd
import flask
import boto3
import json
import tarfile
import traceback
import mlflow

from werkzeug.utils import secure_filename
from src.utils.logger import setup_logger
# We need to import the PreprocessingPipeline class so we can load the artifact
from src.data_processing.preprocessor.preprocessor import PreprocessingPipeline 

# --- Initial Setup ---
logger = setup_logger(__name__)
# Tell Flask where to find the HTML templates
app = flask.Flask(__name__, template_folder='templates')

# --- Configuration ---
# S3 path to your packaged model
S3_BUCKET_NAME = "flaskcapstonebucket"
MODEL_S3_KEY = "models/LeadConversionModel/model.tar.gz"

# Local path where the model will be downloaded and unzipped inside the server
MODEL_PATH = "/tmp/model"

# --- Load Model and Preprocessor at Startup ---
def download_and_load_model():
    """
    Downloads the model package from S3, unzips it, and loads the
    MLflow model and preprocessor artifacts into memory.
    """
    try:
        # Create a directory to store the model
        os.makedirs(MODEL_PATH, exist_ok=True)
        local_tar_path = os.path.join(MODEL_PATH, "model.tar.gz")

        # Download from S3 using the EC2 instance's IAM Role
        logger.info(f"Downloading model from s3://{S3_BUCKET_NAME}/{MODEL_S3_KEY}...")
        s3_client = boto3.client("s3")
        s3_client.download_file(S3_BUCKET_NAME, MODEL_S3_KEY, local_tar_path)
        logger.info("✅ Model downloaded successfully.")

        # Unzip the tar.gz file
        logger.info(f"Unpacking {local_tar_path}...")
        with tarfile.open(local_tar_path, "r:gz") as tar:
            tar.extractall(path=MODEL_PATH)
        logger.info("✅ Model unpacked successfully.")

        # Load the MLflow model
        # The model is in a subdirectory named after its artifact name
        model_artifact_name = "StackingEnsemble"
        mlflow_model_path = os.path.join(MODEL_PATH, model_artifact_name)
        logger.info(f"Loading MLflow model from: {mlflow_model_path}")
        model = mlflow.pyfunc.load_model(mlflow_model_path)
        logger.info("✅ MLflow model loaded.")

        # Load the preprocessor
        artifact_path = os.path.join(MODEL_PATH, "artifacts")
        logger.info(f"Loading preprocessor from: {artifact_path}")
        preprocessor = PreprocessingPipeline(artifact_path=artifact_path)
        logger.info("✅ Preprocessor loaded.")
        
        return model, preprocessor

    except Exception as e:
        logger.error("❌ FAILED TO LOAD MODEL AND PREPROCESSOR ON STARTUP.")
        traceback.print_exc()
        return None, None

# Load the model when the application starts
model, preprocessor = download_and_load_model()

# --- Flask App Routes ---
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
        try:
            raw_df = pd.read_csv(file.stream)
            logger.info(f"Read {len(raw_df)} rows from the uploaded file.")
            
            # Preprocess the data using the loaded preprocessor
            processed_df = preprocessor.transform(raw_df)

            # Re-align columns to match model signature
            if hasattr(model, 'metadata') and model.metadata.signature:
                model_features = model.metadata.get_input_schema().input_names()
                processed_df = processed_df.reindex(columns=model_features, fill_value=0)

            # Make predictions
            predictions = model.predict(processed_df)
            probabilities = predictions.iloc[:, 0].tolist() if isinstance(predictions, pd.DataFrame) else predictions.tolist()

            # Add results to the original DataFrame for display
            results_df = raw_df.copy()
            results_df['Lead_Conversion_Probability'] = [f"{p:.2%}" for p in probabilities]
            results_df['Lead_Converted_Prediction'] = (pd.Series(probabilities) > 0.5).astype(int)
            logger.info("Successfully generated predictions.")

            # Prepare data for rendering in the HTML template
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

# --- You do not need the /sample or /download routes for this version ---

if __name__ == "__main__":
    # For production, we will use Gunicorn, not this.
    # This is only for local testing.
    app.run(debug=True, host='0.0.0.0', port=8000)
