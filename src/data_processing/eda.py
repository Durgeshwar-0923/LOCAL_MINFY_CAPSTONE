import os
import pandas as pd
import sweetviz as sv
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def run_sweetviz(df: pd.DataFrame, target_col: str, output_path: str = "reports/eda_report.html") -> None:
    """
    Generate an automated EDA report using Sweetviz, focused on the target variable.
    """
    logger.info("🧪 Starting Sweetviz EDA report generation...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Analyze the dataframe with respect to the target feature for more insightful plots
    report = sv.analyze(df, target_feat=target_col)
    
    report.show_html(output_path)
    logger.info(f"✅ Sweetviz report saved to: {output_path}")


def eda_summary(df: pd.DataFrame, target_col: str) -> None:
    """
    Display a comprehensive summary of the dataset in the logs.
    """
    logger.info("📊 Running detailed EDA summary...")

    logger.info(f"Shape of the dataset: {df.shape}")
    logger.info(f"Full column list: {df.columns.tolist()}")
    
    # --- Missing Values Summary ---
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({'count': missing_values, 'percentage': missing_percent})
    logger.info(f"Missing values summary:\n{missing_df[missing_df['count'] > 0].sort_values(by='percentage', ascending=False)}")

    # --- Data Types Summary ---
    logger.info(f"Data types summary:\n{df.dtypes.value_counts()}")

    # --- Target Variable Distribution ---
    if target_col in df.columns:
        logger.info(f"🎯 Target variable ('{target_col}') distribution:\n{df[target_col].value_counts(normalize=True)}")

    # --- Numeric and Categorical Summaries ---
    numeric_df = df.select_dtypes(include=['number'])
    categorical_df = df.select_dtypes(include=['object', 'category'])

    if not numeric_df.empty:
        logger.info(f"🧮 Numeric Features Summary:\n{numeric_df.describe().T}")
    if not categorical_df.empty:
        logger.info(f"🔤 Categorical Features Summary:\n{categorical_df.describe(include='all').T}")


def plot_distributions(
    df: pd.DataFrame,
    numeric_cols: list = None,
    categorical_cols: list = None,
    save: bool = True,
    save_dir: str = "reports/plots"
) -> None:
    """
    Plot and save histograms for numeric columns and bar charts for categorical columns.
    """
    logger.info("📈 Plotting feature distributions...")

    if save:
        os.makedirs(save_dir, exist_ok=True)

    # --- Plot Numeric Distributions ---
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f"Distribution of {col}", fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        if save:
            path = os.path.join(save_dir, f"numeric_{col}_dist.png")
            plt.savefig(path, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved numeric distribution plot for '{col}' to '{path}'")
        else:
            plt.show()

    # --- Plot Categorical Distributions ---
    if categorical_cols is None:
        # Select top 10 most frequent object/category columns to avoid clutter
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in categorical_cols:
        if df[col].nunique() < 50: # Only plot for columns with a reasonable number of categories
            plt.figure(figsize=(10, 6))
            # Use countplot for categorical data
            sns.countplot(y=df[col], order=df[col].value_counts().index)
            plt.title(f"Distribution of {col}", fontsize=14)
            plt.xlabel("Count", fontsize=12)
            plt.ylabel(col, fontsize=12)
            plt.tight_layout()
            if save:
                path = os.path.join(save_dir, f"categorical_{col}_dist.png")
                plt.savefig(path, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved categorical distribution plot for '{col}' to '{path}'")
            else:
                plt.show()