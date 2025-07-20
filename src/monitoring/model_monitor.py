# File: src/monitoring/model_monitor.py

import pandas as pd
import mlflow
from src.monitoring.drift_detector import log_drift_report
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def monitor_model_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    dataset_name: str = "train_vs_current",
    experiment_name: str = "Drift_Monitoring"
) -> bool:
    """
    Performs data drift check and logs Evidently report to MLflow.

    Args:
        reference_data (pd.DataFrame): Historical (reference) dataset.
        current_data (pd.DataFrame): New or incoming dataset.
        dataset_name (str): Identifier for dataset comparison.
        experiment_name (str): MLflow experiment name.

    Returns:
        drift_detected (bool): True if dataset drift is detected.
    """
    logger.info("📊 Starting model drift monitoring...")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"Drift_{dataset_name}"):
        run_id = mlflow.active_run().info.run_id
        report_path, drift_detected = log_drift_report(
            reference_data=reference_data,
            current_data=current_data,
            dataset_name=dataset_name,
            run_id=run_id
        )

        if drift_detected:
            logger.warning(f"⚠️ Drift detected in {dataset_name}. Report: {report_path}")
        else:
            logger.info(f"✅ No significant drift detected in {dataset_name}.")

        return drift_detected
