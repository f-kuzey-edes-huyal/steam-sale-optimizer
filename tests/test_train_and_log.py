import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.train_and_log import mean_absolute_percentage_error, parse_price, transform_review_column


def test_parse_price_valid():
    print("Running test_parse_price_valid...")
    assert parse_price("$20.00") == 20.00
    assert parse_price("USD 15") == 15.00
    assert parse_price("  $100 ") == 100.00
    print("test_parse_price_valid passed!")


def test_parse_price_invalid():
    print("Running test_parse_price_invalid...")
    assert pd.isna(parse_price("Free"))
    assert pd.isna(parse_price(None))
    print("test_parse_price_invalid passed!")


def test_transform_review_column():
    print("Running test_transform_review_column...")
    df = pd.DataFrame({"review": ["This is great", "Bad game", "Loved it", ""]})
    df_transformed = transform_review_column(df.copy(), seed=42)
    assert "review_score" in df_transformed.columns
    assert not df_transformed["review_score"].isna().any()
    print("test_transform_review_column passed!")


def test_mape():
    print("Running test_mape...")
    y_true = [100, 200, 300]
    y_pred = [110, 190, 310]
    mape = mean_absolute_percentage_error(y_true, y_pred)
    assert round(mape, 2) == pytest.approx(6.11, rel=1e-2)
    print("test_mape passed!")