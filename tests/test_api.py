from fastapi.testclient import TestClient
import os
from src.main import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post('/model/predict', json={"prefix": "The world"})
    assert response.status_code == 200
    data = response.json()
    assert "text" in data, "Response should contain the original text."
    assert "predictions" in data, "Response should contain predictions."
    assert isinstance(data['predictions'], list), "Predictions should be a list."
