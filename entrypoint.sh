#!/bin/bash

echo "Initializing Airflow DB..."
airflow db init

echo "Creating admin user..."
airflow users create \
  --username "${AIRFLOW_USER}" \
  --password "${AIRFLOW_PASSWORD}" \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com || true

echo "Starting Airflow webserver..."
exec airflow webserver
