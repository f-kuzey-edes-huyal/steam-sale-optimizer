FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /mlflow

# Avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow
RUN pip install --no-cache-dir mlflow==2.12.1

# Expose MLflow UI port
EXPOSE 5000

# Run MLflow server with backend DB and artifact store paths
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow_db/mlflow.db", \
     "--default-artifact-root", "/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
