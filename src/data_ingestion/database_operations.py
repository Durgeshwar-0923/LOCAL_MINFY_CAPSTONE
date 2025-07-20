import pandas as pd
from sqlalchemy import create_engine, exc
import os
from src.config.config import DatabaseConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_csv_to_postgres(
    csv_path: str = os.path.join("data", "raw", "Lead Scoring.csv"),
    table_name: str = None,
    if_exists_policy: str = "replace",
    chunksize: int = 1000
) -> None:
    """
    Loads data from a CSV file into a PostgreSQL table in chunks.

    Args:
        csv_path (str): The path to the CSV file.
        table_name (str): The name of the target database table.
        if_exists_policy (str): How to behave if the table already exists.
                                Options: 'replace', 'append', 'fail'.
        chunksize (int): The number of rows to write in each batch.
    """
    try:
        config = DatabaseConfig()
        if table_name is None:
            table_name = config.table_name

        # Only ensure the database exists; let pandas handle the table schema!
        config.ensure_database_exists()

        engine = create_engine(config.db_url)
        logger.info(f"📤 Loading CSV: {csv_path} → PostgreSQL table: {table_name}")

        # Use a chunksize iterator to handle large files
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

# Example usage:
if __name__ == "__main__":
    load_csv_to_postgres(if_exists_policy="replace")  # or "append" after first load
