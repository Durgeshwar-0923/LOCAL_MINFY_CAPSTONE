# 1. Import modules
from src.data_ingestion.data_loader import DataLoader
from src.config.config import DatabaseConfig
from src.data_processing.eda import eda_summary, plot_distributions

# 2. Load data from PostgreSQL
loader = DataLoader(DatabaseConfig())
df = loader.load_data()

# 3. Run EDA
eda_summary(df)
plot_distributions(df, columns=['total_compensation'])  # or any relevant columns
