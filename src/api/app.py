"""
Module: src/api/app.py
Description: Main Flask application file. Handles HTTP requests, file uploads,
             and orchestrates the prediction process by calling the prediction_service.
"""
import os
import uuid
import pandas as pd
from flask import Flask, request, render_template, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename

# --- Custom Service Imports ---
from src.api.prediction_service import predict_probabilities, ALL_ARTIFACTS
from src.utils.logger import setup_logger

# --- Initial Setup ---
logger = setup_logger(__name__)
app = Flask(__name__)

# --- FIX APPLIED HERE: Absolute Path Configuration ---
# Get the absolute path of the project's root directory.
# This ensures that no matter where the script is run from, the paths are always correct.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
PRED_FOLDER = os.path.join(PROJECT_ROOT, "predictions")
REFERENCE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Lead Scoring.csv")
# --- END OF FIX ---

MODEL_NAME = os.getenv("MODEL_NAME", "LeadConversionModel")
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "a_default_secret_key_for_development")

# Create necessary directories at startup.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PRED_FOLDER, exist_ok=True)


@app.route("/")
def home():
    model_loaded = ALL_ARTIFACTS.get("model") is not None
    if not model_loaded:
        logger.error("Health Check FAILED: Model or artifacts could not be loaded.")
        flash("Error: The prediction model or its artifacts could not be loaded. Please check the server logs.", "danger")
    else:
        logger.info("Health Check PASSED: Model and artifacts are loaded.")
    return render_template("index.html", model_name=MODEL_NAME, model_loaded=model_loaded)


@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        flash("No file part in the request.", "warning")
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        flash("No file selected for uploading.", "warning")
        return redirect(url_for('home'))

    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        # The UPLOAD_FOLDER path is now absolute and correct.
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        logger.info(f"File '{filename}' uploaded successfully.")

        try:
            raw_df = pd.read_csv(file_path)
            logger.info(f"Read {len(raw_df)} rows from the uploaded file.")
            
            probabilities = predict_probabilities(raw_df)

            results_df = raw_df.copy()
            results_df['Lead_Conversion_Probability'] = [f"{p:.2%}" for p in probabilities]
            results_df['Lead_Converted_Prediction'] = (probabilities > 0.5).astype(int)
            logger.info("Successfully generated predictions and probabilities.")

            result_id = uuid.uuid4().hex[:8]
            result_file = f"prediction_{result_id}_{filename}"
            # The PRED_FOLDER path is now absolute and correct.
            result_path = os.path.join(PRED_FOLDER, result_file)
            results_df.to_csv(result_path, index=False)
            logger.info(f"Saved prediction results to '{result_path}'")

            table_headers = results_df.columns.tolist()
            table_rows = results_df.head(100).values.tolist()
            
            return render_template(
                "result.html",
                table_headers=table_headers,
                table_rows=table_rows,
                result_file=result_file,
                model_name=MODEL_NAME,
                num_records=len(results_df)
            )

        except Exception as e:
            logger.error(f"An error occurred during prediction: {e}", exc_info=True)
            flash(f"An error occurred during processing: {e}", "danger")
            return redirect(url_for('home'))
    else:
        flash("Invalid file type. Please upload a CSV file.", "danger")
        return redirect(url_for('home'))

@app.route("/sample")
def sample():
    logger.info("Providing sample data file for download.")
    return send_file(REFERENCE_DATA_PATH, as_attachment=True)


@app.route("/download/<filename>")
def download(filename):
    logger.info(f"Processing download request for file: {filename}")
    safe_filename = secure_filename(filename)
    # The PRED_FOLDER path is now absolute and correct.
    file_path = os.path.join(PRED_FOLDER, safe_filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        flash("File not found.", "danger")
        return redirect(url_for('home'))


if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 8000)))
