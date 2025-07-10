import os
import time
import pandas as pd
import numpy as np
import joblib
import psycopg
import datetime
import pytz
import logging
import csv
from dotenv import load_dotenv

from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnQuantileMetric,
    ColumnValueRangeMetric,
    ColumnCorrelationsMetric
)
from evidently import ColumnMapping

# Load environment variables
load_dotenv()

# Paths
DATA_PATH = "data/combined4.csv"
DRIFTED_DATA_PATH = "data/drifted_data.csv"

# Model paths
MODEL_PATH = 'models/discount_model_pipeline.pkl'
TFIDF_PATH = 'models/tfidf_vectorizer.pkl'
SVD_PATH = 'models/svd_transform.pkl'
MLB_GENRES_PATH = 'models/mlb_genres.pkl'
MLB_TAGS_PATH = 'models/mlb_tags.pkl'
COMPETITOR_TRANS_PATH = 'models/competitor_pricing_transformer.pkl'

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# PostgreSQL settings
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'monitoring_db')


class CompetitorPricingTransformer:
    def __init__(self):
        self.tag_price_map = {}

    def fit(self, df):
        tag_prices = {}
        for tags, price in zip(df['tags'], df['current_price']):
            for tag in tags:
                tag_prices.setdefault(tag, []).append(price)
        self.tag_price_map = {tag: np.median(prices) for tag, prices in tag_prices.items()}
        return self

    def transform(self, df):
        def competitor_price_for_tags(tags):
            prices = [self.tag_price_map.get(tag, np.nan) for tag in tags]
            prices = [p for p in prices if not np.isnan(p)]
            return np.mean(prices) if prices else np.nan

        df['competitor_pricing'] = df['tags'].apply(competitor_price_for_tags)
        median_price = df['current_price'].median()
        df['competitor_pricing'] = df['competitor_pricing'].fillna(median_price)
        return df


# Load models and transformers
model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)
svd = joblib.load(SVD_PATH)
mlb_genres = joblib.load(MLB_GENRES_PATH)
mlb_tags = joblib.load(MLB_TAGS_PATH)
competitor_transformer = joblib.load(COMPETITOR_TRANS_PATH)


def get_connection():
    return psycopg.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB,
        autocommit=True
    )


def preprocess_monitoring_data(df):
    df = df.dropna(subset=[
        'total_reviews', 'positive_percent', 'genres', 'tags',
        'current_price', 'discounted_price', 'owners_log_mean', 'days_after_publish'
    ])

    def parse_price(x):
        try:
            return float(str(x).replace('$', '').replace('USD', '').strip())
        except Exception:
            return np.nan

    df['current_price'] = df['current_price'].apply(parse_price)
    df['discounted_price'] = df['discounted_price'].apply(parse_price)
    df = df.dropna(subset=['current_price', 'discounted_price'])

    df['discount_pct'] = 1 - (df['discounted_price'] / df['current_price'])

    df['genres'] = df['genres'].fillna('').apply(lambda x: [g.strip() for g in str(x).split(',') if g.strip()])
    df['tags'] = df['tags'].fillna('').apply(lambda x: [t.strip() for t in str(x).split(';') if t.strip()])

    if 'review' not in df.columns:
        df['review'] = ''
    else:
        df['review'] = df['review'].fillna('')

    if len(df) == 0:
        return df

    tfidf_matrix = tfidf.transform(df['review'])
    df['review_score'] = svd.transform(tfidf_matrix).flatten()

    df = competitor_transformer.transform(df)

    df['genres'] = df['genres'].apply(lambda lst: [g for g in lst if g in mlb_genres.classes_])
    df['tags'] = df['tags'].apply(lambda lst: [t for t in lst if t in mlb_tags.classes_])

    genres_encoded = pd.DataFrame(
        mlb_genres.transform(df['genres']), columns=mlb_genres.classes_, index=df.index
    )
    tags_encoded = pd.DataFrame(
        mlb_tags.transform(df['tags']), columns=mlb_tags.classes_, index=df.index
    )

    df = pd.concat([df, genres_encoded, tags_encoded], axis=1)
    df = df.drop(columns=['genres', 'tags', 'review'], errors='ignore')

    features = [
        'total_reviews', 'positive_percent', 'current_price', 'discounted_price',
        'owners_log_mean', 'days_after_publish', 'review_score', 'competitor_pricing'
    ] + list(mlb_genres.classes_) + list(mlb_tags.classes_)

    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df


def load_reference_data():
    df = pd.read_csv(
        DATA_PATH,
        quotechar='"',
        escapechar='\\',
        encoding='utf-8',
        quoting=csv.QUOTE_MINIMAL
    )
    ref_df = df[df['days_after_publish'] <= 5].copy()

    if ref_df.empty:
        logging.warning("Reference data empty, fallback to first 100 rows")
        ref_df = df.head(100).copy()

    ref_df = preprocess_monitoring_data(ref_df)
    if not ref_df.empty:
        ref_df['prediction'] = model.predict(ref_df)
    return ref_df


def load_batch(day: int):
    df = pd.read_csv(
        DRIFTED_DATA_PATH,
        quotechar='"',
        escapechar='\\',
        encoding='utf-8',
        quoting=csv.QUOTE_MINIMAL
    )
    lower = day * 5
    upper = (day + 1) * 5
    batch = df[(df['days_after_publish'] >= lower) & (df['days_after_publish'] <= upper)].copy()
    if batch.empty:
        return None
    batch = preprocess_monitoring_data(batch)
    if not batch.empty:
        batch['prediction'] = model.predict(batch)
    return batch


def run_evidently_report(ref_df, current_df):
    if ref_df.empty or current_df.empty:
        return None

    ref_df = ref_df.loc[:, ~ref_df.columns.duplicated()]
    current_df = current_df.loc[:, ~current_df.columns.duplicated()]

    common_cols = sorted(set(ref_df.columns) & set(current_df.columns))
    if 'prediction' in common_cols:
        common_cols.remove('prediction')

    def get_non_constant_cols(ref_df, cur_df, cols):
        return [col for col in cols if ref_df[col].nunique(dropna=False) > 1 and cur_df[col].nunique(dropna=False) > 1]

    common_cols = get_non_constant_cols(ref_df, current_df, common_cols)

    if 'prediction' not in ref_df.columns:
        ref_df['prediction'] = 0
    if 'prediction' not in current_df.columns:
        current_df['prediction'] = 0

    ref_df = ref_df.reindex(columns=common_cols + ['prediction'], fill_value=0)
    current_df = current_df.reindex(columns=common_cols + ['prediction'], fill_value=0)

    numerical_features = [col for col in common_cols if pd.api.types.is_numeric_dtype(ref_df[col])]
    numerical_features = get_non_constant_cols(ref_df, current_df, numerical_features)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Suppress pandas correlation constant input warning by message filter instead of import
        warnings.filterwarnings("ignore", message="An input array is constant; the correlation coefficient is not defined.")

        report = Report(metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ColumnDriftMetric(column_name='prediction'),
            ColumnQuantileMetric(column_name='current_price', quantile=0.5),
            ColumnValueRangeMetric(column_name='positive_percent'),
            ColumnCorrelationsMetric(column_name='current_price')
        ])

        report.run(
            reference_data=ref_df,
            current_data=current_df,
            column_mapping=ColumnMapping(
                prediction='prediction',
                numerical_features=numerical_features
            )
        )

    return report.as_dict(), ref_df, current_df


def store_metrics(metrics: dict, timestamp: datetime.datetime, cursor):
    drift_score = metrics['metrics'][2]['result'].get('drift_score', 0)
    num_drifted_cols = metrics['metrics'][0]['result'].get('number_of_drifted_columns', 0)
    share_missing = metrics['metrics'][1]['result']['current'].get('share_of_missing_values', 0)
    median_current_price = metrics['metrics'][3]['result'].get('quantile', 0)
    positive_range = metrics['metrics'][4]['result'].get('value_range', 0)
    corr_df = metrics['metrics'][5]['result'].get('correlations', {})

    if corr_df:
        corr_df = pd.DataFrame(corr_df)
        corr_df = corr_df.apply(pd.to_numeric, errors='coerce')
        np.fill_diagonal(corr_df.values, 0)
        mean_abs_corr = np.abs(corr_df.values).mean()
    else:
        mean_abs_corr = 0

    cursor.execute(
        """
        INSERT INTO model_monitoring_metrics (
            timestamp,
            prediction_drift,
            num_drifted_columns,
            share_missing_values,
            median_current_price,
            positive_percent_range,
            mean_abs_corr
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            timestamp,
            drift_score,
            num_drifted_cols,
            share_missing,
            median_current_price,
            positive_range,
            mean_abs_corr
        )
    )


def prepare_db():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_monitoring_metrics (
                    timestamp TIMESTAMP PRIMARY KEY,
                    prediction_drift FLOAT,
                    num_drifted_columns INTEGER,
                    share_missing_values FLOAT,
                    positive_percent_range FLOAT
                )
            """)

            columns_to_check = {
                "median_current_price": "FLOAT",
                "mean_abs_corr": "FLOAT"
            }

            for col, col_type in columns_to_check.items():
                cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name='model_monitoring_metrics' AND column_name=%s
                """, (col,))
                if cur.fetchone() is None:
                    logging.info(f"Adding missing column '{col}' to model_monitoring_metrics table.")
                    cur.execute(f"ALTER TABLE model_monitoring_metrics ADD COLUMN {col} {col_type};")


def save_visual_report(ref_df, current_df, day: int):
    ref_df = ref_df.loc[:, ~ref_df.columns.duplicated()]
    current_df = current_df.loc[:, ~current_df.columns.duplicated()]

    common_cols = sorted(set(ref_df.columns) & set(current_df.columns))
    if 'prediction' in common_cols:
        common_cols.remove('prediction')

    if 'prediction' not in ref_df.columns:
        ref_df['prediction'] = 0
    if 'prediction' not in current_df.columns:
        current_df['prediction'] = 0

    ref_df = ref_df.reindex(columns=common_cols + ['prediction'], fill_value=0)
    current_df = current_df.reindex(columns=common_cols + ['prediction'], fill_value=0)

    numerical_features = [col for col in common_cols if pd.api.types.is_numeric_dtype(ref_df[col])]

    report = Report(metrics=[
        DatasetDriftMetric(),
        ColumnDriftMetric(column_name='prediction')
    ])

    report.run(
        reference_data=ref_df,
        current_data=current_df,
        column_mapping=ColumnMapping(
            prediction='prediction',
            numerical_features=numerical_features
        )
    )
    os.makedirs("reports", exist_ok=True)
    report.save_html(f"reports/report_day_{day}.html")


if __name__ == "__main__":
    logging.info("Starting model monitoring...")
    prepare_db()
    reference_data = load_reference_data()

    SEND_INTERVAL = 0  # Adjust as needed

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                day = 0
                while True:
                    batch = load_batch(day)
                    if batch is None:
                        logging.info(f"No data for day {day}, stopping monitoring.")
                        break

                    result = run_evidently_report(reference_data, batch)
                    if result is None:
                        logging.warning(f"Skipping day {day} because data empty.")
                        day += 1
                        continue

                    metrics, ref_df_for_report, batch_for_report = result

                    timestamp = datetime.datetime.now(pytz.UTC)
                    store_metrics(metrics, timestamp, cur)

                    drift_score = metrics['metrics'][2]['result'].get('drift_score', 0)
                    num_drifted = metrics['metrics'][0]['result'].get('number_of_drifted_columns', 0)
                    missing_pct = metrics['metrics'][1]['result']['current'].get('share_of_missing_values', 0)

                    logging.info(f"Day {day} | Drift: {drift_score:.3f} | Drifted cols: {num_drifted} | Missing: {missing_pct:.2%}")

                    save_visual_report(ref_df_for_report, batch_for_report, day)

                    day += 1
                    time.sleep(SEND_INTERVAL)

    except Exception as e:
        logging.error(f"Monitoring failed: {e}")
        raise
