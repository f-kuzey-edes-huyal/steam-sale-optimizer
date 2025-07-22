import pytest
from fastapi.testclient import TestClient
import pandas as pd

from main import app, parse_price, preprocess_input, GameData
from unittest.mock import patch, MagicMock

client = TestClient(app)

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

def test_parse_price():
    assert parse_price("$20.00") == 20.0
    assert parse_price("USD 15") == 15.0
    assert parse_price("  $ 30.5 ") == 30.5
    assert parse_price("invalid") is None

@patch("main.tfidf")
@patch("main.svd")
@patch("main.mlb_genres")
@patch("main.mlb_tags")
@patch("main.competitor_transformer")
def test_preprocess_input(mock_competitor, mock_mlb_tags, mock_mlb_genres, mock_svd, mock_tfidf):
    # Mock the transformers to return predictable outputs
    mock_tfidf.transform.return_value = [[0.1, 0.2]]
    mock_svd.transform.return_value = [[0.3]]
    mock_mlb_genres.transform.return_value = [[1, 0]]
    mock_mlb_genres.classes_ = ['Action', 'Adventure']
    mock_mlb_tags.transform.return_value = [[0, 1]]
    mock_mlb_tags.classes_ = ['Multiplayer', 'Co-op']
    mock_competitor.transform.return_value = pd.DataFrame({'competitor_pricing': [0.5], **{k:0 for k in ['total_reviews','positive_percent','current_price','discounted_price','owners_log_mean','days_after_publish']}})

    data = GameData(**sample_game_data)
    df = preprocess_input(data)

    # Should contain all expected columns
    expected_cols = ['total_reviews','positive_percent','current_price','discounted_price','owners_log_mean','days_after_publish',
                     'review_score','competitor_pricing','Action','Adventure','Multiplayer','Co-op']
    for col in expected_cols:
        assert col in df.columns

def test_reload_model():
    response = client.post("/reload_model")
    assert response.status_code == 200
    assert response.json() == {"message": "Model reloaded successfully."}

@patch("main.model")
@patch("main.preprocess_input")
def test_predict(mock_preprocess_input, mock_model):
    mock_preprocess_input.return_value = pd.DataFrame([[1,2,3,4,5]])
    mock_model.predict.return_value = [0.25]

    response = client.post("/predict", json=sample_game_data)
    assert response.status_code == 200
    json_resp = response.json()
    assert "predicted_discount_pct" in json_resp
    assert isinstance(json_resp["predicted_discount_pct"], float)
    assert json_resp["predicted_discount_pct"] == 0.25
