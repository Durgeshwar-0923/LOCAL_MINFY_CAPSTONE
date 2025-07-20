# File: src/monitoring/drift_detector.py

import os
import json
import re
from pathlib import Path
from datetime import datetime

import mlflow
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def sanitize_mlflow_key(key: str) -> str:
    """
    Sanitize metric names to comply with MLflow requirements:
    Only alphanumerics, dashes, underscores, periods, spaces, and slashes allowed.
    """
    return re.sub(r"[^a-zA-Z0-9_\-./ ]", "_", key)


def log_drift_report(
    reference_data,
    current_data,
    dataset_name: str = "train_vs_test",
) -> None:
    """
    📊 Generates an Evidently report (data drift + summary), saves HTML and JSON,
    logs them as MLflow artifacts, and extracts key drift and dataset metrics.

    Args:
        reference_data (pd.DataFrame): Reference/historical dataset.
        current_data (pd.DataFrame): New or test dataset to compare.
        dataset_name (str): Identifier used as prefix for saved files and metrics.
    """
    # 0️⃣ Select common columns
    common_cols = set(reference_data.columns).intersection(current_data.columns)
    if not common_cols:
        logger.warning(f"No common columns between reference and {dataset_name}; skipping report.")
        return
    ref = reference_data.loc[:, sorted(common_cols)]
    cur = current_data.loc[:, sorted(common_cols)]

    # 1️⃣ Create Evidently report
    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    result = report.run(reference_data=ref, current_data=cur)

    # 2️⃣ Prepare save directory
    save_dir = Path("evidently_reports")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 3️⃣ Save HTML & JSON with timestamp
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    html_path = save_dir / f"evidently_{dataset_name}_{ts}.html"
    json_path = save_dir / f"evidently_{dataset_name}_{ts}.json"
    result.save_html(str(html_path))
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(result.json())
    logger.info(f"Saved Evidently reports to {html_path} and {json_path}.")

    # 4️⃣ Log artifacts to MLflow
    mlflow.log_artifact(str(html_path), artifact_path="evidently")
    mlflow.log_artifact(str(json_path), artifact_path="evidently")
    logger.info(f"Logged Evidently artifacts under 'evidently/'.")

    # 5️⃣ Load JSON and extract metrics
    with open(json_path, "r", encoding="utf-8") as fp:
        report_json = json.load(fp)
    metrics_list = report_json.get("metrics", [])

    # 6️⃣ Log overall drift metrics
    drift_entry = next(
        (m for m in metrics_list if m.get("metric_id", "").startswith("DriftedColumnsCount")),
        None,
    )
    if drift_entry:
        count = drift_entry["value"]["count"]
        share = drift_entry["value"]["share"]
        mlflow.log_metric(f"{dataset_name}__drifted_columns_count", float(count))
        mlflow.log_metric(f"{dataset_name}__drifted_columns_share", float(share))
        logger.info(f"Drifted columns: {count} ({share:.2%})")

    # 7️⃣ Log dataset size metrics
    row_entry = next((m for m in metrics_list if m.get("metric_id") == "RowCount()"), None)
    col_entry = next((m for m in metrics_list if m.get("metric_id") == "ColumnCount()"), None)
    if row_entry:
        mlflow.log_metric(f"{dataset_name}__row_count", float(row_entry["value"]))
    if col_entry:
        mlflow.log_metric(f"{dataset_name}__column_count", float(col_entry["value"]))

    # 8️⃣ Log per-column drift scores (with sanitized names)
    for m in metrics_list:
        mid = m.get("metric_id", "")
        if mid.startswith("ValueDrift(column="):
            col = mid.split("=")[1].rstrip(")")
            val = m.get("value")
            if isinstance(val, (int, float)):
                safe_col = sanitize_mlflow_key(col)
                mlflow.log_metric(f"{dataset_name}__drift_{safe_col}", float(val))

    logger.info(f"✅ All drift & dataset metrics for '{dataset_name}' logged to MLflow.")
