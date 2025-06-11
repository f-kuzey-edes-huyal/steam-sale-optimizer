FROM apache/airflow:2.9.0-python3.10

USER root

RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean

USER airflow