import pandas as pd
from src.data_processing.preprocessor import DataPreprocessor

def test_preprocessor_handles_missing():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
    pre = DataPreprocessor()
    result = pre.preprocess(df)
    assert result.isnull().sum().sum() == 0