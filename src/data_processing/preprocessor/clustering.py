import os
import joblib
import pandas as pd
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

ARTIFACT_DIR = "artifacts/clustering"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

class EngagementClusteringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, n_clusters=4, random_state=42, artifact_dir=ARTIFACT_DIR):
        self.features = features or [
            'total_visits',
            'total_time_spent_on_website',
            'time_per_visit'
        ]
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.artifact_dir = artifact_dir
        self.scaler = None
        self.kmeans = None

    def _resolve_feature_names(self, columns):
        matched = []
        for f in self.features:
            matched.extend([col for col in columns if col.endswith(f)])
        return matched

    def fit(self, X: pd.DataFrame, y=None):
        cols = self._resolve_feature_names(X.columns)
        if not cols:
            logger.error("❌ No valid clustering features found. Aborting fit.")
            return self

        X_cluster = X[cols].copy().fillna(X[cols].median())
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_cluster)

        scaler_path = os.path.join(self.artifact_dir, "engagement_scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"📦 Saved scaler → {scaler_path}")

        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init='auto', random_state=self.random_state)
        self.kmeans.fit(X_scaled)

        kmeans_path = os.path.join(self.artifact_dir, "engagement_kmeans.joblib")
        joblib.dump(self.kmeans, kmeans_path)
        logger.info(f"📦 Saved KMeans → {kmeans_path} (Inertia={self.kmeans.inertia_:.2f})")

        cluster_df = pd.DataFrame(X_scaled, columns=cols)
        cluster_df['cluster'] = self.kmeans.labels_
        cluster_df.to_csv(os.path.join(self.artifact_dir, "cluster_assignments.csv"), index=False)
        logger.info("📝 Logged clustering assignments and centroids.")

        if MLFLOW_AVAILABLE:
            mlflow.log_artifact(scaler_path, artifact_path="clustering")
            mlflow.log_artifact(kmeans_path, artifact_path="clustering")
            mlflow.log_artifact(os.path.join(self.artifact_dir, "cluster_assignments.csv"), artifact_path="clustering")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        cols = self._resolve_feature_names(df.columns)
        if not cols:
            logger.warning("⚠️ No clustering features present in transform input.")
            return df

        if self.scaler is None:
            scaler_path = os.path.join(self.artifact_dir, "engagement_scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"📤 Loaded scaler from {scaler_path}")
            else:
                logger.error("❌ Scaler artifact missing — skipping cluster transform.")
                return df

        if self.kmeans is None:
            kmeans_path = os.path.join(self.artifact_dir, "engagement_kmeans.joblib")
            if os.path.exists(kmeans_path):
                self.kmeans = joblib.load(kmeans_path)
                logger.info(f"📤 Loaded KMeans from {kmeans_path}")
            else:
                logger.error("❌ KMeans model missing — skipping cluster transform.")
                return df

        X_cluster = df[cols].copy().fillna(df[cols].median())
        X_scaled = self.scaler.transform(X_cluster)
        cluster_labels = self.kmeans.predict(X_scaled)
        df["engagement_cluster"] = pd.Categorical(cluster_labels)
        logger.info("✅ Added 'engagement_cluster' label to data.")

        return df

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
