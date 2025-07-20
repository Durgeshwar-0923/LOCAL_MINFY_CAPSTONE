### ✅ Updated vif_filter.py
import os
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

ARTIFACT_PATH = "artifacts/vif"
os.makedirs(ARTIFACT_PATH, exist_ok=True)

class RFECVTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold
        self.dropped_features_ = []
        self.selected_features_ = []
        self.imputer_ = SimpleImputer(strategy='median')

    def _filter_valid_numeric_columns(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        valid_cols = [
            col for col in numeric_cols
            if df[col].nunique() > 1 and df[col].std() > 0
        ]
        if not valid_cols:
            logger.warning("⚠️ No valid numerical features with >2 unique values for VIF calculation.")
        return valid_cols

    def fit(self, X: pd.DataFrame, y=None):
        logger.info(f"📉 Fitting RFECVTransformer with VIF threshold: {self.threshold}")
        df = X.copy()
        df.columns = [col.split("__")[-1] if "__" in col else col for col in df.columns]
        numeric_cols = self._filter_valid_numeric_columns(df)
        if not numeric_cols:
            self.selected_features_ = []
            return self

        df_numeric = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df_numeric_imputed = pd.DataFrame(
            self.imputer_.fit_transform(df_numeric),
            columns=numeric_cols,
            index=df.index
        )

        iteration = 0
        while True:
            vif_data = pd.DataFrame()
            vif_data["feature"] = df_numeric_imputed.columns
            try:
                vif_data["VIF"] = [
                    variance_inflation_factor(df_numeric_imputed.values, i)
                    for i in range(df_numeric_imputed.shape[1])
                ]
            except np.linalg.LinAlgError as e:
                logger.error(f"❌ VIF calculation failed: {e}")
                break

            max_vif = vif_data["VIF"].max()
            if max_vif <= self.threshold or df_numeric_imputed.shape[1] == 1:
                logger.info("✅ All features below VIF threshold.")
                break

            feature_to_drop = vif_data.sort_values("VIF", ascending=False)["feature"].iloc[0]
            self.dropped_features_.append(feature_to_drop)
            df_numeric_imputed.drop(columns=[feature_to_drop], inplace=True)
            logger.info(f"🔻 Dropped '{feature_to_drop}' (VIF = {max_vif:.2f}) at iteration {iteration}")
            iteration += 1

        self.selected_features_ = df_numeric_imputed.columns.tolist()

        vif_data_final = vif_data[~vif_data["feature"].isin(self.dropped_features_)]
        vif_csv = os.path.join(ARTIFACT_PATH, "vif_selected_features.csv")
        vif_joblib = os.path.join(ARTIFACT_PATH, "vif_selected_features.joblib")
        vif_data_final.to_csv(vif_csv, index=False)
        joblib.dump(self.selected_features_, vif_joblib)

        logger.info(f"📄 Final VIF summary saved to: {vif_csv}")
        logger.info(f"💾 VIF-selected features saved to: {vif_joblib}")

        if MLFLOW_AVAILABLE:
            mlflow.log_artifact(vif_csv, artifact_path="vif")
            mlflow.log_artifact(vif_joblib, artifact_path="vif")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if not self.selected_features_:
            logger.warning("⚠️ No selected features found — returning original DataFrame.")
            return df
        df.columns = [col.split("__")[-1] if "__" in col else col for col in df.columns]
        final_cols = self.selected_features_ + [
            col for col in df.columns
            if col not in self.selected_features_ and col not in self.dropped_features_
        ]
        logger.info(f"✅ VIF filtering complete. Selected {len(self.selected_features_)} features.")
        return df[final_cols]

    def get_feature_names_out(self):
        return self.selected_features_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)