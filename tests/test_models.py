import pandas as pd
import numpy as np
from src.models.model_trainer import train_models

def test_model_trainer_runs():
    X = pd.DataFrame(np.random.rand(20, 3), columns=["a", "b", "c"])
    y = np.random.rand(20)
    model = train_models(X, y)
    assert hasattr(model, "predict")