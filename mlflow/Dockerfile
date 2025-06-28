FROM python:3.10-slim

# Set workdir
WORKDIR /mlflow

# Avoid prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow
RUN pip install --no-cache-dir mlflow==2.12.1

# Default command
CMD ["mlflow"]
