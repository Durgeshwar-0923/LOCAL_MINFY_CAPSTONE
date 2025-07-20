import pandas as pd
from sqlalchemy import create_engine
from src.config.config import DatabaseConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataLoader:
    """Handles loading data from the PostgreSQL database."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        try:
            self.engine = create_engine(config.db_url)
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            raise

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the specified table in the PostgreSQL database.
        
        This now constructs a query that explicitly uses the schema and
        quotes the table name to ensure case-sensitivity is respected.
        """
        logger.info(f"Preparing to load data from: {self.config.schema_name}.{self.config.table_name}")
        # The corrected, robust query with schema and quoted table name
        query = f'SELECT * FROM "{self.config.schema_name}"."{self.config.table_name}"'
        
        try:
            with self.engine.connect() as connection:
                df = pd.read_sql(query, connection)
            logger.info(f"✅ Successfully loaded {len(df)} rows from table '{self.config.table_name}'.")
            return df
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            raise