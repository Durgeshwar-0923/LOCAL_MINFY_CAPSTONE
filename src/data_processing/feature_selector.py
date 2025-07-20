import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def select_features(df: pd.DataFrame, target: str, k: int = 10):
    X = df.drop(columns=[target])
    y = df[target]
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    selected = importances.nlargest(k).index.tolist()
    return selected
