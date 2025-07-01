FROM python:3.10-slim

# Set working directory
WORKDIR /mlflow

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install useful tools (curl, ps, bash)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    procps \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow
RUN pip install --no-cache-dir mlflow==2.12.1

# Expose the MLflow UI port
EXPOSE 5000

# ✅ Proper CMD with shell
CMD ["/bin/sh", "-c", "mlflow server --backend-store-uri sqlite:////mlflow_db/mlflow.db --default-artifact-root /mlruns --host 0.0.0.0 --port 5000"]