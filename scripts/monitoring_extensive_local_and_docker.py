import os
import time
import json
import requests
import pandas as pd
import numpy as np
import logging
import joblib
import psycopg2
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric, CorrelationMetric, ColumnQuantileMetric

# Logging config
logging.basicConfig(level=logging.INFO)

# Monitoring config from environment
MONITOR_HOST = os.getenv("MONITOR_HOST", "localhost")
MONITOR_PORT = os.getenv("MONITOR_PORT", 5432)
MONITOR_USER = os.getenv("MONITOR_USER", "postgres")
MONITOR_PASSWORD = os.getenv("MONITOR_PASSWORD", "example")
MONITOR_DB = os.getenv("MONITOR_DB", "monitoring")

REFERENCE_DATA_PATH = "data/monitoring_reference.parquet"
CURRENT_DATA_PATH = "data/monitoring_current.parquet"
PREPROCESSOR_PATH = "models/preprocessor.b"

TARGET_COLUMN = "discounted"

# Load reference data
reference_data = pd.read_parquet(REFERENCE_DATA_PATH)
reference_data[TARGET_COLUMN] = reference_data[TARGET_COLUMN].astype(int)

# Load preprocessing pipeline
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Get connection (with retry logic)
def get_connection(retries=3, delay=5):
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(
                host=MONITOR_HOST,
                port=MONITOR_PORT,
                user=MONITOR_USER,
                password=MONITOR_PASSWORD,
                dbname=MONITOR_DB
            )
            conn.autocommit = True
            return conn
        except psycopg2.OperationalError as e:
            logging.warning(f"DB connection failed (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(delay)
    logging.error("Could not connect to PostgreSQL after retries.")
    raise psycopg2.OperationalError("Database connection failed.")

# Prepare DB
def prepare_db():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_monitoring_metrics (
                    timestamp TIMESTAMP PRIMARY KEY,
                    prediction_drift FLOAT,
                    num_drifted_columns INTEGER,
                    share_missing_values FLOAT,
                    positive_percent_range FLOAT,
                    median_current_price FLOAT,
                    mean_abs_corr FLOAT
                )
            """)

# Load current data
def load_current_data():
    try:
        current_data = pd.read_parquet(CURRENT_DATA_PATH)
        current_data[TARGET_COLUMN] = current_data[TARGET_COLUMN].astype(int)
        return current_data
    except Exception as e:
        logging.error(f"Failed to load current data: {e}")
        return None

# Run monitoring logic
def run_monitoring(current_data):
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        ColumnDriftMetric(column=TARGET_COLUMN),
        ColumnQuantileMetric(column="current_price", quantile=0.5),
        CorrelationMetric()
    ])

    report.run(reference_data=reference_data, current_data=current_data)
    return report

# Extract metrics from report
def extract_metrics(report, current_data):
    try:
        metrics = report.as_dict()
        prediction_drift = metrics['metrics'][0]['result']['dataset_drift']
        num_drifted_columns = metrics['metrics'][0]['result']['number_of_drifted_columns']
        share_missing_values = metrics['metrics'][1]['result']['current']['share_of_missing_values']
        positive_percent_range = (
            current_data[TARGET_COLUMN].sum() / len(current_data)
            if len(current_data) > 0 else 0
        )
        median_current_price = float(current_data["current_price"].median())
        corr_matrix = current_data.corr(numeric_only=True).abs()
        mean_abs_corr = (
            corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            .stack()
            .mean()
        )

        return {
            "prediction_drift": prediction_drift,
            "num_drifted_columns": num_drifted_columns,
            "share_missing_values": share_missing_values,
            "positive_percent_range": positive_percent_range,
            "median_current_price": median_current_price,
            "mean_abs_corr": mean_abs_corr
        }

    except Exception as e:
        logging.error(f"Error extracting metrics: {e}")
        return None

# Store metrics in DB
def store_metrics(timestamp, metrics):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO model_monitoring_metrics (
                    timestamp, prediction_drift, num_drifted_columns,
                    share_missing_values, positive_percent_range,
                    median_current_price, mean_abs_corr
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (timestamp) DO NOTHING;
            """, (
                timestamp,
                metrics["prediction_drift"],
                metrics["num_drifted_columns"],
                metrics["share_missing_values"],
                metrics["positive_percent_range"],
                metrics["median_current_price"],
                metrics["mean_abs_corr"]
            ))

# Main loop
def main_loop():
    prepare_db()
    while True:
        logging.info("Checking for new monitoring data...")
        current_data = load_current_data()
        if current_data is not None:
            report = run_monitoring(current_data)
            timestamp = datetime.now()
            metrics = extract_metrics(report, current_data)
            if metrics:
                store_metrics(timestamp, metrics)
                logging.info(f"Metrics stored for timestamp {timestamp}")
        else:
            logging.warning("No valid current data found.")
        time.sleep(60)

if __name__ == "__main__":
    main_loop()
