FROM apache/airflow:2.9.0-python3.10

USER root

# Install system dependencies required by Chrome & Selenium
RUN apt-get update && apt-get install -y \
    wget unzip curl gnupg2 libxi6 libgconf-2-4 \
    libnss3 libxss1 libasound2 fonts-liberation \
    xdg-utils lsb-release libu2f-udev xvfb \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Switch to airflow user BEFORE installing Python dependencies
USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
