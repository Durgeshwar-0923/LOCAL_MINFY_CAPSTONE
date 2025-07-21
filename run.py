# main.py

import os
import pandas as pd
import mlflow

from sqlalchemy import create_engine, exc

# --- Updated Imports for the New Pipeline Structure ---
from src.config.config import Paths, DatabaseConfig
from src.data_ingestion.data_loader import DataLoader
from src.data_processing.preprocessor.pipeline import run_pipeline_with_tracking
from src.data_processing.eda import run_sweetviz, eda_summary
from src.models.train_models import train_all_models
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ✅ Add: Function to load CSV to PostgreSQL
def load_csv_to_postgres(
    csv_path: str = os.path.join("data", "raw", "Lead Scoring.csv"),
    table_name: str = None,
    if_exists_policy: str = "replace",
    chunksize: int = 1000
) -> None:
    """
    Loads data from a CSV file into a PostgreSQL table in chunks.
    """
    try:
        config = DatabaseConfig()
        if table_name is None:
            table_name = config.table_name

        config.ensure_database_exists()

        engine = create_engine(config.db_url)
        logger.info(f"📤 Loading CSV: {csv_path} → PostgreSQL table: {table_name}")

        csv_iterator = pd.read_csv(csv_path, iterator=True, chunksize=chunksize)

        for i, chunk in enumerate(csv_iterator):
            policy = if_exists_policy if i == 0 else "append"
            chunk.to_sql(table_name, engine, if_exists=policy, index=False)
            logger.info(f"📝 Wrote chunk {i+1} to table {table_name}")

        logger.info(f"✅ Data from {csv_path} loaded into PostgreSQL successfully.")

    except FileNotFoundError:
        logger.error(f"❌ File not found at path: {csv_path}")
    except exc.SQLAlchemyError as e:
        logger.error(f"❌ A database error occurred: {e}")
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred: {e}")

def main():
    """
    Main orchestrator for the entire Lead Conversion ML Pipeline.
    """
    # --- 0. Load CSV into PostgreSQL ---
    logger.info("🚀 Loading raw CSV data into PostgreSQL...")
    load_csv_to_postgres(if_exists_policy="replace")
    logger.info("✅ PostgreSQL data load complete.")

    # --- 1. Configuration and Setup ---
    paths = Paths()

    local_mlflow_uri = "file:" + os.path.join(os.getcwd(), "mlruns")
    mlflow.set_tracking_uri(local_mlflow_uri)
    logger.info(f"MLflow tracking URI set to: {local_mlflow_uri}")

    EDA_EXPERIMENT_NAME = "Lead_Conversion_EDA"
    TRAINING_EXPERIMENT_NAME = "Lead_Conversion_Modeling"
    TARGET_COLUMN = "Converted"

    paths.PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    paths.REPORTS.mkdir(parents=True, exist_ok=True)

    # --- 2. Data Loading ---
    logger.info("🔄 Connecting to the database and loading raw data...")
    loader = DataLoader(DatabaseConfig())
    df_raw = loader.load_data()

    staged_raw_data_path = paths.RAW_DATA / "staged_lead_scoring_data.csv"
    df_raw.to_csv(staged_raw_data_path, index=False)
    logger.info(f"📁 Raw data staged at: {staged_raw_data_path}")

    # --- 3. EDA ---
    mlflow.set_experiment(EDA_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="EDA Report"):
        eda_summary(df_raw, target_col=TARGET_COLUMN)
        eda_path = paths.REPORTS / "eda_report.html"
        run_sweetviz(df_raw, target_col=TARGET_COLUMN, output_path=str(eda_path))
        mlflow.log_artifact(str(eda_path), artifact_path="EDA")
        logger.info("✅ EDA complete and logged.")

    # --- 4. Preprocessing Pipeline ---
    logger.info("🛠️ Running preprocessing pipeline...")
    run_pipeline_with_tracking(
        raw_data_path=str(staged_raw_data_path),
        is_training=True
    )
    logger.info("✅ Preprocessing complete.")

    # --- 5. Model Training ---
    final_processed_path = paths.PROCESSED_DATA / "13_final_features.csv"
    if not final_processed_path.exists():
        logger.error(f"❌ Final preprocessed file missing: {final_processed_path}")
        return

    logger.info(f"📂 Loading processed data: {final_processed_path}")
    processed_df = pd.read_csv(final_processed_path)

    mlflow.set_experiment(TRAINING_EXPERIMENT_NAME)
    logger.info("🚀 Starting training pipeline using Optuna + Stacking...")

    best_model = train_all_models(
        df=processed_df,
        target=TARGET_COLUMN,
        experiment_name=TRAINING_EXPERIMENT_NAME,
        n_trials=15,
        timeout=900,
        cv=5
    )

    logger.info(f"✅ Training completed. Best model: {best_model}")
    logger.info("🎉 Lead Conversion ML pipeline execution complete.")

if __name__ == "__main__":
    main()
