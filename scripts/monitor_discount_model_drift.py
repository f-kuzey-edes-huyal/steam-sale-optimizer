import pandas as pd
import datetime
import time
import logging
import psycopg
import joblib
import pytz
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from evidently import ColumnMapping

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
SEND_TIMEOUT = 10  # seconds

# Load model
model = joblib.load('models/discount_model_pipeline.pkl')

# Load reference data
reference_data = pd.read_csv('data/reference_discount.csv')

# Define the columns (you can adjust this to match your actual list)
num_features = ['total_reviews', 'positive_percent', 'current_price', 'discounted_price',
                'owners_log_mean', 'days_after_publish', 'review_score', 'competitor_pricing']
column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    target=None
)

# Define SQL
CREATE_TABLE_SQL = """
DROP TABLE IF EXISTS discount_model_metrics;
CREATE TABLE discount_model_metrics (
    timestamp TIMESTAMP,
    prediction_drift FLOAT,
    num_drifted_columns INTEGER,
    share_missing_values FLOAT
)
"""

def prep_db():
    with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if not res.fetchall():
            conn.execute("CREATE DATABASE test;")
    with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
        conn.execute(CREATE_TABLE_SQL)

def get_current_batch(i):
    full_data = pd.read_csv("data/new_discount_data.csv")
    full_data = full_data.sort_values(by='days_after_publish')
    day_i = full_data[full_data['days_after_publish'] == i].copy()
    if day_i.empty:
        return None
    day_i['prediction'] = model.predict(day_i[num_features])
    return day_i

def run_drift_report(ref_data, curr_data):
    report = Report(metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ])
    report.run(reference_data=ref_data, current_data=curr_data, column_mapping=column_mapping)
    return report.as_dict()

def insert_metrics_to_db(metrics_dict, timestamp, cursor):
    prediction_drift = metrics_dict['metrics'][0]['result']['drift_score']
    num_drifted_columns = metrics_dict['metrics'][1]['result']['number_of_drifted_columns']
    share_missing = metrics_dict['metrics'][2]['result']['current']['share_of_missing_values']

    cursor.execute(
        """
        INSERT INTO discount_model_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values)
        VALUES (%s, %s, %s, %s)
        """,
        (timestamp, prediction_drift, num_drifted_columns, share_missing)
    )

def batch_monitoring():
    prep_db()
    last_send = datetime.datetime.now(pytz.UTC) - datetime.timedelta(seconds=SEND_TIMEOUT)
    with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
        for i in range(0, 30):  # Simulate 30 days
            with conn.cursor() as cursor:
                batch = get_current_batch(i)
                if batch is None:
                    logging.warning(f"No data found for batch {i}. Skipping.")
                    continue

                drift_report = run_drift_report(reference_data, batch)
                insert_metrics_to_db(drift_report, last_send, cursor)
                logging.info(f"Drift logged for batch {i}")

            now = datetime.datetime.now(pytz.UTC)
            elapsed = (now - last_send).total_seconds()
            if elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - elapsed)
            last_send += datetime.timedelta(seconds=SEND_TIMEOUT)

if __name__ == "__main__":
    batch_monitoring()
