import os
import joblib
import pandas as pd
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# Set up logger
try:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

ARTIFACT_DIR = "artifacts/feature_selection"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

class RFECVTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 target_col: str = 'Converted',
                 estimator=None,
                 cv_splits: int = 5,
                 scoring: str = 'roc_auc',
                 step: int = 1,
                 n_jobs: int = -1,
                 artifact_dir: str = ARTIFACT_DIR):
        self.target_col = target_col
        self.estimator = estimator if estimator is not None else LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        )
        self.cv = StratifiedKFold(n_splits=cv_splits)
        self.scoring = scoring
        self.step = step
        self.n_jobs = n_jobs
        self.artifact_dir = artifact_dir
        self.selector_ = None
        self.selected_features_ = None

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        if self.target_col not in df.columns:
            logger.error(f"Target column '{self.target_col}' not in DataFrame. Fit aborted.")
            return self

        y = df[self.target_col]
        X_feat = df.drop(columns=[self.target_col])

        logger.info("Fitting RFECV selector...")
        self.selector_ = RFECV(
            estimator=self.estimator,
            step=self.step,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs
        )
        self.selector_.fit(X_feat, y)

        selector_path = os.path.join(self.artifact_dir, "rfecv_selector.joblib")
        joblib.dump(self.selector_, selector_path)
        logger.info(f"Saved RFECV selector to {selector_path}")

        self.selected_features_ = list(X_feat.columns[self.selector_.support_])
        features_path = os.path.join(self.artifact_dir, "selected_features.joblib")
        joblib.dump(self.selected_features_, features_path)
        logger.info(f"Saved selected features list ({len(self.selected_features_)} features) to {features_path}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if self.selected_features_ is None:
            features_path = os.path.join(self.artifact_dir, "selected_features.joblib")
            if os.path.exists(features_path):
                self.selected_features_ = joblib.load(features_path)
                logger.info(f"Loaded selected features from {features_path}")
            else:
                logger.error("Selected features artifact not found; skipping RFECV transform.")
                return df

        cols_to_keep = [f for f in self.selected_features_ if f in df.columns]
        if self.target_col in df.columns:
            cols_to_keep.append(self.target_col)

        df_reduced = df[cols_to_keep]
        logger.info(f"Transformed DataFrame to {len(cols_to_keep)-1} features + target; new shape {df_reduced.shape}")
        return df_reduced

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

def select_features_rfe(df: pd.DataFrame, target_col: str = 'Converted', is_training: bool = True) -> pd.DataFrame:
    transformer = RFECVTransformer(target_col=target_col)
    if is_training:
        return transformer.fit_transform(df)
    return transformer.transform(df)

class SHAPFeatureSelector:
    def __init__(self, shap_csv_path: str = r"artifacts/feature_engineering/shap_feature_importance.csv", top_k: int = 50):
        self.shap_csv_path = shap_csv_path
        self.top_k = top_k

    def transform(self, df: pd.DataFrame, target_col: str = 'Converted') -> pd.DataFrame:
        if not os.path.exists(self.shap_csv_path):
            logger.warning(f"⚠️ SHAP file not found at {self.shap_csv_path}. Skipping SHAP filtering.")
            return df

        shap_df = pd.read_csv(self.shap_csv_path).sort_values("shap_importance", ascending=False)
        top_features = shap_df["feature"].head(self.top_k).tolist()

        if target_col in df.columns and target_col not in top_features:
            top_features.append(target_col)

        removed = [col for col in df.columns if col not in top_features]
        if removed:
            logger.info(f"🔍 SHAP Selector: Retained {len(top_features)} features | Removed {len(removed)} features.")

        return df[top_features]

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)
