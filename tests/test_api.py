from src.api.app import app

def test_api_predict():
    client = app.test_client()
    response = client.post("/predict", json={"a": 1, "b": 2, "c": 3})
    assert response.status_code == 200