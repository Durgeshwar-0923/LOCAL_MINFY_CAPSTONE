import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote_plus

# It's good practice to load environment variables at the start
from dotenv import load_dotenv
load_dotenv()

@dataclass
class Paths:
    ROOT: Path = Path(__file__).parent.parent.parent
    DATA: Path = ROOT / "data"
    MODELS: Path = ROOT / "models"
    REPORTS: Path = ROOT / "reports"
    RAW_DATA: Path = DATA / "raw"
    PROCESSED_DATA: Path = DATA / "processed"

@dataclass
class DatabaseConfig:
    # These values are now set based on your provided info
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", 5432))
    database: str = os.getenv("DB_NAME", "Lead_db")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASS", "Minfy@Durgesh")
    # The schema, which is 'public' for standard PostgreSQL
    schema_name: str = "public"
    # The EXACT, case-sensitive table name
    table_name: str = "Lead_data"

    @property
    def db_url(self) -> str:
        """Constructs the SQLAlchemy connection URI correctly."""
        password_safe = quote_plus(self.password)
        return f"postgresql://{self.user}:{password_safe}@{self.host}:{self.port}/{self.database}"

@dataclass
class MLConfig:
    target_column: str = "converted"
    random_state: int = 42
    test_size: float = 0.2

@dataclass
class Paths:
    ROOT: Path = Path(__file__).parent.parent.parent
    DATA: Path = ROOT / "data"
    MODELS: Path = ROOT / "models"
    REPORTS: Path = ROOT / "reports"

    RAW_DATA: Path = DATA / "raw"
    PROCESSED_DATA: Path = DATA / "processed"
    DRIFT_REPORTS: Path = DATA / "drift_reports"

@dataclass
class MLOpsConfig:
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    airflow_dag_folder: Path = Paths.ROOT / "airflow" / "dags"
