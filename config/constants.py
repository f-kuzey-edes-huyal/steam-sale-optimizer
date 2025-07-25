#MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_TRACKING_URI ="http://mlflow:5000"
MLFLOW_TRACKING_URI_local ="http://localhost:5000"
#MLFLOW_TRACKING_URI_air =  "file:/opt/airflow/mlruns"
EXPERIMENT_NAME = "Experiment: Included Competitor Pricing and Review Score"
EXPERIMENT_NAME2 = "Experiment: No Competitor Pricing or Review Score"
EXPERIMENT_NAME3 = "Experiment: Apache Airflow"
DATA_PATH = "data/combined4.csv"      
DRIFTED_DATA_PATH = "data/drifted_data.csv"  
SEED = 42
MODEL_PATH = "/opt/airflow/models/discount_model_pipeline.pkl"

