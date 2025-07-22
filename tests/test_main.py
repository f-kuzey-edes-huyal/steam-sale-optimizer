import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_models():
    with patch("main.joblib.load") as mock_load:
        # Create mocks for each loaded model/component with expected methods
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]  # prediction returns list with float
        
        mock_tfidf = MagicMock()
        mock_tfidf.transform.return_value = "tfidf_matrix"

        mock_svd = MagicMock()
        mock_svd.transform.return_value = [0.8]  # e.g., array with one score

        mock_mlb_genres = MagicMock()
        mock_mlb_genres.transform.return_value = [[1,0]]
        mock_mlb_genres.classes_ = ['Action', 'Adventure']

        mock_mlb_tags = MagicMock()
        mock_mlb_tags.transform.return_value = [[1,0]]
        mock_mlb_tags.classes_ = ['Multiplayer', 'Co-op']

        mock_competitor_transformer = MagicMock()
        mock_competitor_transformer.transform.return_value = MagicMock()
        mock_competitor_transformer.transform.return_value.reset_index.return_value = MagicMock()

        # joblib.load returns these mocks in order they are called
        mock_load.side_effect = [
            mock_model,
            mock_tfidf,
            mock_svd,
            mock_mlb_genres,
            mock_mlb_tags,
            mock_competitor_transformer
        ]

        yield
