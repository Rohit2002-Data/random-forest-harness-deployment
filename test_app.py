from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_home_has_prediction_form():
    response = client.get("/")
    assert response.status_code == 200
    assert "Random Forest Prediction" in response.text
    for feature in ["Open", "High", "Low", "Adj Close", "Volume"]:
        assert feature in response.text


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction():
    response = client.post(
        "/predict",
        json={"Open": 100.0, "High": 105.0, "Low": 98.0, "Adj Close": 102.0, "Volume": 1000000},
    )
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], (int, float))
