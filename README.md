
## 🧩 Problem Definition

Steam is one of the most popular digital distribution platforms for PC gaming, known not only for its vast library of games but also for its seasonal discounts—an exciting time for gamers and gift-givers alike. While discounts drive traffic and increase sales, they also present a strategic pricing challenge:

- Too __steep a discount__ might undermine revenue potential.

- Too __small a discount__ might deter potential buyers altogether.

In this project, I aim to develop a data-driven pipeline to suggest optimal discount percentages for Steam games by analyzing publicly available Steam data and enriched third-party sources.

Game developers and publishers often face uncertainty when deciding on discount strategies. The ideal discount should balance increased sales volume with preserved revenue margins. However, Steam does not directly provide ownership or purchase volume data, which is critical for such analysis.

To tackle this, I created an __end-to-end MLOps pipeline__ that:

- Scrapes game pricing, rating, and tag-related data directly from the Steam Store.

- Enriches the dataset using SteamSpy, a third-party service that estimates number of owners (a proxy for sales volume).

- Used four regressor models with different hyperparameters. __Tracked__ all experiments with __MLflow__ for the case with and without user reviews.
  
- Best model based on mean absolute error is __registered__ with __MLflow__.
  
- __Apache Airflow__ used to orchestrate all steps. __One DAG handles everything__. Isn’t that cool? From scraping to cleaning to deployment, it’s all there. You just gotta trigger one file and the whole pipeline runs.


- Used __Evidently__ and __Grafana__ for __drift monitoring__.

- Model deployed to the cloud using __Microsoft Azure: Cloud Computing Services__

- __CI/CD pipeline__ set up with GitHub Actions. __Unit__ and __integration tests__ applied


- Infrastructure handled with __Terraform__. The repo is cloned into Azure Cloud, and all containers are built there to repeat the experiment on the cloud platform.



## 📊 Data Overview

This Python script [steam_scraper_new_trial.py](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/scripts/steam_scraper_new_trial.py) includes functions to scrape the most important game-level features  from the Steam store: https://store.steampowered.com/. 

```"game_id", "name", "release_date", "release_year" "total_reviews", "positive_percent" "developer", "publisher", "genres", "tags", "platforms" "current_price", "discounted_price", "discount_percent"```

The script [steamspy_scrape.py](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/scripts/steamspy_scrape.py) contains functions to extract additional metadata from SteamSpy, using the endpoint: https://steamspy.com/api.php?request=appdetails&appid={appid}
The most critical feature retrieved here is "owners", which indicates the estimated number of game owners. 

Additionally, the [review_scraper.py](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/scripts/review_scraper.py) script scrapes user reviews from the Steam platform. These reviews will later be used in a multimodal analysis, combining both tabular features and textual data.

Together, this pipeline integrates three sources of data using the functions defined in:

```python scripts\main_scraper1_new_features.py```

By running the code below, you will combine the reviews ([reviews.csv](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/data/reviews.csv), the number of owners scraped from SteamSpy ([steamspy_data.csv](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/data/steamspy_data.csv), and other game features using an SQL query ([steamdata.csv](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/data/steamdata.csv). Additionally, the query will exclude free-to-play games, as they are not suitable for our analysis.

Before running the script, make sure to execute your [SQL schema](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/pgadmin.sql) (e.g., via pgAdmin or a similar tool) to create the necessary database and grant access permissions.

```python scripts\load_combine_exclude_freegames.py```

### 🛠️ Automated Data Collection and Merging Using Apache Airflow

Before accessing the Airflow UI, make sure you have built and started the Docker containers using ```docker-compose up --build```. Then, you can visit http://localhost:8080 to view and trigger your DAGs in the Apache Airflow UI.

[The DAG](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/dags/scrape_and_combine_steam_csvs_dag.py) file orchestrates the data scraping and combination process by calling two scripts: [airflow_main_scraper1_new.py](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/scripts/airflow_main_scraper1_new.py) and [load_and_combine_new.py](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/scripts/load_and_combine_new.py). These scripts scrape the required game data, combine three datasets using SQL, and exclude free-to-play games from the final output.

The figure below shows how to trigger your DAG.

<p align="center">
  <img src="https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/figures/trigger_gag_scrape.png?raw=true" alt="Trigger the DAG" width="800"/>
</p>

The figure below shows a successfully run DAG.

<p align="center">
  <img src="https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/figures/scrape_bysteam.png?raw=true" alt="Successfully runned image" width="800"/>
</p>

## 🧪 Experiment Tracking with MLflow

First, activate your virtual environment in your terminal and run the command below to start the MLflow UI:
```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Next, open a new terminal window, activate your virtual environment again, and run your experiment tracking code.



```
python scripts\train_last.py
```

Access your MLflow experiments by navigating to http://127.0.0.1:5000 in your browser to visualize and manage your experiment tracking

<img src="https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/figures/exp_tracking_optuna.png" alt="Alt text" width="800"/>

I observed a __14% improvement__ in mean absolute error for the small training sample size when including competitor pricing and review scores, compared to the experiment where these features were not included.  

__Don't forget to activate the virtual environment (```venv\Scripts\activate```) when working locally instead of using Docker containers.__

## 🗃️ Model Registry

Instead of selecting the model for registration within the training script, you can register the model separately using ```mlflow.register_model```.

While your MLflow tracking server is running, open a new terminal and execute the code below to register your model.

```
python scripts\model_registry_final.py
```

By clicking the link http://127.0.0.1:5000/#/models, you can view your registered models.

<img src="https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/figures/model_registry_im.png" alt="Alt text" width="800"/>


Note: Model registration is not the same as deployment, but it allows you to prepare the model for future deployment or serving.

## 🛠️ Orchestrating Experiment Tracking and Model Registry with Apache Airflow

To orchestrate experiment tracking and model registry, trigger the [dag_experiment_tracking_model_registry.py](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/dags/dag_experiment_tracking_model_rgistry..py) script, which defines the DAG with ```dag_id="discount_model_training_pipeline_new```".

Besides triggering it from the Apache Airflow UI, you can also run it directly from the Docker container using the following line:

```docker exec -it steam-sale-optimizer-airflow-scheduler-1 airflow dags trigger discount_model_training_pipeline_new```


## 📈 Monitoring Data Drift 

Run the following code to perform monitoring:

```docker exec -it steam-sale-optimizer-postgres-1 psql -U postgres```

```CREATE DATABASE monitoring_db;```

```docker exec -it steam-sale-optimizer-airflow-scheduler-1 python scripts/monitoring_extensive_graphs.py```



### Adminer Login Instructions

Go to: [http://localhost:8081](http://localhost:8081)

Use the credentials from your `.env` file:

- System: PostgreSQL
- Server: `postgres`
- Username: `${POSTGRES_USER}`
- Password: `${POSTGRES_PASSWORD}`
- Database: `${POSTGRES_DB}`

![Adminer Screenshot](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/figures/adminer_scrrenshot.png?raw=true)

### 🔧 Accessing Grafana


Open your Grafana service: [http://localhost:3000](http://localhost:3000)  
Log in using the credentials you set in your `.env` file **before building the Docker images**:

- **Username**: ```admin```
- **Password**: ```admin```

The [new_drift.json](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/dashboards/new_derift.json) file provides the Grafana dashboard layout. When you visit http://localhost:3000, you will see the dashboard as shown below.

You will see the dashboard by going to: Home → Dashboards → Model Monitoring → Model Monitoring Dashboard.

![Grafana Dashboard](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/figures/grafana_dashboorrd.png?raw=true)




## 📦 Model Deployment

To deploy the model, you should run the code below.

```uvicorn main_new:app --reload```

 Then, test your model locally by:
 
 ```python test.py```

<p align="center">
  <img src="https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/raw/main/figures/test_locally.png" alt="Local Test Result" width="600"/>
</p>

<p align="center"><i>Figure: Testing the FastAPI model locally</i></p>

I have created a lighter Docker image to deploy my model using a separate Dockerfile. To build and run this Dockerfile, and then push the image to Docker Hub, follow the steps below:

```docker build -t fkuzeyedeshuyal/deployment-d -f Dockerfile.deployment .```

```docker run -p 8000:80 fkuzeyedeshuyal/deployment-d```

```python test.py```

```docker login```

```docker push fkuzeyedeshuyal/deployment-d```

### 🚀 Deployment to the Cloud (Azure)

The following steps show how to deploy the FastAPI application using Azure CLI:

```az login```

```az group create --name myResourceGroup --location westeurope```

```az appservice plan create --name myAppServicePlan --resource-group myResourceGroup --sku B1 --is-linux```

```az webapp create --resource-group myResourceGroup --plan myAppServicePlan --name kuzey-ml-app --deployment-container-image-name fkuzeyedeshuyal/deployment-d:latest```

```az webapp show --resource-group myResourceGroup --name kuzey-ml-app --query defaultHostName -o tsv```

This will give you the URL: https://kuzey-ml-app.azurewebsites.net. By adding /docs to the end of the URL — like this:
👉 [https://kuzey-ml-app.azurewebsites.net/docs] — you can view the interactive FastAPI Swagger UI for your deployed app.

The model was successfully deployed to Azure App Service using a custom Docker image.




<p align="center">
  <img src="https://raw.githubusercontent.com/f-kuzey-edes-huyal/steam-sale-optimizer/main/figures/azure_deployment1.png" alt="Azure Test Screenshot" width="600"/>
</p>



<p align="center">
  <img src="https://raw.githubusercontent.com/f-kuzey-edes-huyal/steam-sale-optimizer/main/figures/azure_test.png" alt="Azure Test Screenshot" width="800"/>
</p>

> ⚠️ **Important:**  
> Don't forget to delete the Azure resources you created after your work is complete to avoid additional charges.

You can do this by running:

 ```az group delete --name myResourceGroup --yes --no-wait```

### 🚀 Orchestrate Deployment with Apache Airflow

I prepared a DAG file named [fastapi_deployment_dag.py](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/dags/fastapi_deployment_dag.py) with the URL set to "http://fastapi-app:80/predict". After triggering the DAG from the Apache Airflow UI, I ran the test script inside the Docker container to verify the FastAPI deployment.

```docker exec -it steam-sale-optimizer-airflow-scheduler-1 bash```

```python scripts/test_docker.py```

## 🚀 Full Orchestration with Apache Airflow

I used the script named [you_have_to_live_before_you_die_young_dag.py](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/dags/you_have_to_live_before_you_die_young_dag.py). This DAG now needs to be triggered.

```docker exec -it steam-sale-optimizer-airflow-scheduler-1 airflow dags trigger you_have_to_live_before_you_die_young_dag```

The figure below demonstrates the successful execution of the full orchestration DAG.

<p align="center">
  <img src="https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/figures/full_orchestration.png" alt="Azure Test Screenshot" width="900"/>
</p>


docker stop $(docker ps -q)

docker start $(docker ps -a -q)

```docker compose up -d```

Note: While working on this pipeline, I mistakenly mounted some files incorrectly while trying to orchestrate it using Apache Airflow. Each time I did this, I had to run docker-compose down and rebuild the containers, which forced me to reinstall all the packages — countless times. This was very inefficient, especially during periods of poor internet connectivity.

Thanks to discussions with Ulaş Huyal, I found a better approach. With the code snippet below, I no longer need to reinstall all packages every time I fix a mount issue:



```docker-compose down --volumes --remove-orphans && docker-compose up --build -d```

## 🧪🔗 Unit & Integration Testing

Added __unit tests__ to verify core utility functions including price parsing, review text transformation, and mean absolute percentage error calculation to ensure data processing and metric computations are accurate.

Run the code below to successfully apply the unit tests

```pytest -s tests/test_train_and_log.py```

The unit test file (__test_main.py__) provided with a code line below,  checks  two critical API endpoints in your FastAPI app. The test_reload_model function ensures the __/reload_model__ route returns a success message and status code 200, verifying the model is reloaded properly. The __test_predict__ function submits a sample game data payload to the __/predict__ endpoint and asserts that a float value for "predicted_discount_pct" is returned, confirming that the prediction pipeline is functioning as expected. 


```pytest -s tests/test_main.py```

The DAG named ```fastapi_deployment_dag_integration_tests.py``` applies __integration testing__ by automatically verifying the FastAPI deployment in the pipeline. After deploying or reloading the model on the FastAPI server, the DAG sends real HTTP requests with test data to the /predict endpoint. It checks the response status and output correctness to ensure the entire system — from model loading to prediction serving — works as expected end-to-end. This helps catch issues early by testing the integration of all components involved in model deployment and serving.

I have also applied an integration test for the full orchestration DAG named __```you_have_to_live_before_you_die_young_dag.py```__.

![Full Orchestration DAG Screenshot](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/figures/full_orcgestration_with_integration_test.png)



## ⚙️📦 Terraform Configuration for Azure Infrastructure

When deploying multiple containers to the cloud, __Terraform__ is extremely helpful for provisioning and managing your infrastructure. Here are three important points from my experience using Terraform:

- First, when creating the environment to construct containers via Terraform, I cloned my GitHub repository. However, I needed to copy my local .env file separately because it was not included in the repository.

- Second, the Docker version installed on the Azure virtual machine was outdated. I had to remove the old version and then install the correct, updated Docker version to ensure compatibility.

- Third, I encountered errors related to Docker credentials. To fix this, I removed the Docker credential helpers and created a separate markdown file inside the Docker folder to document these changes.

Although I mainly worked outside of cloud platforms before, I demonstrated that I can build and manage containerized infrastructure on cloud platforms using Terraform.

The main Terraform code for using the Azure platform looks like that, but I could not stay with it as it was not as easy as just five lines. I have prepared another [README file](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/terraform/terraform.md) inside the terraform folder.

| Command                 | Description                    |
|-------------------------|--------------------------------|
| `az login`              | Log in to your Azure account   |
| `cd terraform`          | Change directory to terraform  |
| `terraform init`        | Initialize the Terraform config|
| `terraform plan`        | Preview the changes to apply   |
| `terraform apply --auto-approve` | Apply changes without prompt  |


Use your VM’s public IP and the ports you mapped in your docker-compose file. For example:

Airflow Webserver: ```http://<your-vm-public-ip>:8080```

MLflow: ```http://<your-vm-public-ip>:5000```

Adminer: ```http://<your-vm-public-ip>:8081```

Grafana: ```http://<your-vm-public-ip>:3000```

FastAPI app: ```http://<your-vm-public-ip>:8082```



```terraform destroy --auto-approve```



## 🚀 CI/CD Pipelines with GitHub Actions

This project uses GitHub Actions for automated testing, __linting__, and deployment. Every push or pull request to the `main` branch triggers a pipeline that builds Docker containers, runs unit tests for the FastAPI app, and checks code quality.

## 🐳 How to Run Docker Containers

Follow the steps below to run Docker containers. You do not need to activate a virtual environment for Docker experiments.

```git clone https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer.git```

```cd steam-sale-optimizer```

```docker-compose up --build```

## 🛠️ How to Set Up the Environment for Local Runs

```git clone https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer.git```

```cd steam-sale-optimizer```

 ```venv\scripts\activate```
