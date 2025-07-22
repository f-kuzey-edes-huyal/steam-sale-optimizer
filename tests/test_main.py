import pytest
from fastapi.testclient import TestClient
from main_new import app

client = TestClient(app)

# Sample input matching your Pydantic model structure
sample_game_data = {
    "game_id": 123,
    "name": "Test Game",
    "release_date": "2023-01-01",
    "total_reviews": 1000,
    "positive_percent": 75,
    "genres": "Action, Adventure",
    "tags": "Multiplayer;Co-op",
    "current_price": "$20.00",
    "discounted_price": "$10.00",
    "owners": "50000..100000",
    "days_after_publish": 365,
    "review": "Great game, loved it!",
    "owner_min": 50000,
    "owner_max": 100000,
    "owners_log_mean": 11.5
}

def test_reload_model():
    response = client.post("/reload_model")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] == "Model reloaded successfully."

def test_predict():
    response = client.post("/predict", json=sample_game_data)
    assert response.status_code == 200
    json_resp = response.json()
    assert "predicted_discount_pct" in json_resp
    assert isinstance(json_resp["predicted_discount_pct"], float)
