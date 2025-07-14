
A machine learning pipeline for optimizing game discount strategies using Steam reviews, tags, and competitor pricing. Designed for data-driven revenue maximization in the gaming industry.

## 🧩 Problem Definition

## 📊 Data Overview


## 🧪 Experiment Tracking with MLflow

First, activate your virtual environment in your terminal and run the command below to start the MLflow UI:
```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Next, open a new terminal window, activate your virtual environment again, and run your experiment tracking code.



```
python scripts\train_optuna_hyperparameter_mlflow_reviews_competitor_pricing_change_criterion_mean_absolute.py
```

Access your MLflow experiments by navigating to http://127.0.0.1:5000 in your browser to visualize and manage your experiment tracking

<img src="https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/figures/exp_tracking_optuna.png" alt="Alt text" width="800"/>

I observed a __14% improvement__ in mean absolute error for the small training sample size when including competitor pricing and review scores, compared to the experiment where these features were not included.  


## 🗃️ Model Registry

Instead of selecting the model for registration within the training script, you can register the model separately using ```mlflow.register_model```.

While your MLflow tracking server is running, open a new terminal and execute the code below to register your model.

```
python scripts\model_registry_final.py
```



Note: Model registration is not the same as deployment, but it allows you to prepare the model for future deployment or serving.

## 🛠️ Orchestrating Experiment Tracking with Apache Airflow

To perform experiment tracking orchestration, run ```docker-compose up --build```, then navigate to http://localhost:8080 and trigger the relevant DAG. If you want to orchestrate your experiment tracking with Apache Airflow, the related [DAG file](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/dags/airflow_dag_train.py) is provided. You need to trigger the tasks, as shown in the image below. 

<img src="https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/figures/click_trigger_training.png" width="800">

If your code runs successfully at each step, you will see bold green indicators, like those shown in the image below.


<img src="https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/figures/success_dag.png" width="800">

By the way, this figure might give the impression that I got everything right on the first try, but in reality, I had many unsuccessful attempts before finally running this DAG script successfully.




## 📈 Monitoring Data Drift 
```python scripts\monitoring_extensive.py```

```docker exec -it steam-sale-optimizer-postgres-1 psql -U postgres```

```CREATE DATABASE monitoring_db;```

```docker exec -it steam-sale-optimizer-airflow-scheduler-1 python scripts/monitoring_extensive.py```



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

- **Username**: `${GRAFANA_USER}`
- **Password**: `${GRAFANA_PASSWORD}`

⚠️ Ensure your `.env` file is present in the same directory as your `docker-compose.yml` when you run:
```
 docker compose up --build
 ```


## 📦 Model Deployment

To deploy the model, you should run the code below.

```uvicorn main:app --reload```

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
  <img src="https://raw.githubusercontent.com/f-kuzey-edes-huyal/steam-sale-optimizer/main/figures/azure_test.png" alt="Azure Test Screenshot" width="600"/>
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

docker stop $(docker ps -q)

docker start $(docker ps -a -q)

```docker compose up -d```

Note: While working on this pipeline, I mistakenly mounted some files incorrectly while trying to orchestrate it using Apache Airflow. Each time I did this, I had to run docker-compose down and rebuild the containers, which forced me to reinstall all the packages — countless times. This was very inefficient, especially during periods of poor internet connectivity.

Thanks to discussions with Ulaş Huyal, I found a better approach. With the code snippet below, I no longer need to reinstall all packages every time I fix a mount issue:



```docker-compose down --volumes --remove-orphans && docker-compose up --build -d```


## Terraform 

When deploying multiple containers to the cloud, __Terraform__ is extremely helpful for provisioning and managing your infrastructure.

```az login```

```terraform init```

```terraform plan```

```terraform apply --auto-approve```

```terraform destroy --auto-approve```



docker system prune -a
docker compose down
docker compose build --no-cache
docker compose up





```pip install -e . ```

```docker-compose up --build```

```docker-compose build```
```docker-compose up -d```

```
*** Found local files:
***   * /opt/airflow/logs/dag_id=scrape_and_combine_steam_csvs_dag/run_id=manual__2025-06-12T14:47:06.328632+00:00/task_id=scrape_steam_data/attempt=1.log
[2025-06-12, 14:47:10 UTC] {local_task_job_runner.py:120} ▶ Pre task execution logs
[2025-06-12, 14:47:10 UTC] {logger.py:11} INFO - ====== WebDriver manager ======
[2025-06-12, 14:47:11 UTC] {logger.py:11} INFO - Get LATEST chromedriver version for google-chrome
[2025-06-12, 14:47:11 UTC] {taskinstance.py:441} ▶ Post task execution logs
```


```
docker exec -it steam-sale-optimizer-airflow-scheduler-1 /bin/bash

google-chrome --version

chromedriver --version

```

Proxy settings caused an issue while scraping data via Airflow from SteamSpy.
I adjusted the proxy settings in the review-fetching script.

I had to adjust my scripts to run my DAG file successfully with Apache Airflow orchestration. The main change was setting up the proxy settings for the driver. Without these settings, the requests were blocked or failed because Airflow runs in a different environment where internet access goes through a proxy.

#important links

[https://cookiecutter-data-science.drivendata.org/]

[steam scraping](https://medium.com/@thekareemyusuf/building-a-dataset-of-steam-games-with-web-scraping-2abb02409f08)
[airflow](https://medium.com/@mrunmayee.dhapre/ml-pipeline-in-airflow-71ca7e1f03ba)
[airflow2](https://medium.com/@mohamadhasan.sarvandani/learning-apache-airflow-with-simple-examples-c1b05b4761b0)
[ I'm getting an error while importing a module from another folder.](https://www.reddit.com/r/learnpython/comments/10l5j6t/cant_import_class_from_a_module_in_another_folder/)

[another suggestion])https://stackoverflow.com/questions/73166298/cant-do-python-imports-from-another-dir)

[https://steamapi.xpaw.me/#IPlayerService/GetOwnedGames]

[Scraping information of all games from Steam with Python](https://medium.com/codex/scraping-information-of-all-games-from-steam-with-python-6e44eb01a299)

[https://www.reddit.com/r/gamedev/comments/x0qs4z/we_gathered_data_about_54000_games_in_steam_and/]

[https://www.gamedeveloper.com/business/genre-viability-on-steam-and-other-trends---an-analysis-using-review-count]

[https://medium.com/thedeephub/postgresql-integration-with-python-a-simple-guide-34b675e4bffd]
[https://db-engines.com/en/ranking]
[https://neon.com/postgresql/postgresql-python/query]

[https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-mlflow-models-online-endpoints?view=azureml-api-2&tabs=cli]

[https://medium.com/@ab.vancouver.canada/monitoring-data-drift-with-evidently-ai-and-grafana-a-comprehensive-guide-169bff90f48c]

??? ```ERROR: Could not install packages due to an OSError: [WinError 32] The process cannot access the file because it is being used by another process: 'C:\\Users\\Kuzey\\AppData\\Local\\Temp\\pip-unpack-lwymme3e\\future-1.0.0-py3-none-any.whl'```

# Notes for Myself

- I have to change my data scraping criteria. Selecting only the most recently published games may not be the best approach. I need to rethink it.
- There are many free games. From what I’ve learned, some of them use different strategies like making the game free to play but charging players for in-game items such as clothes or weapons. Others earn money through ads, especially on mobile platforms.
- Can I scrape how many copies were sold for each game ID? If I can't get this information, can I find something that is strongly related to the number of copies sold? I can try to find a dataset that includes the number of copies sold. Then, I can look for a feature that has a high correlation with it using the features I already collected.
- <img src="https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/motivation.png?raw=true" width="300">
- Reviews are in multiple languages, so we need to find a way to handle that. Also, consider how to use these reviews effectively. Should we use a multimodal model that processes both text and other features, or convert the reviews into numerical scores like sentiment polarity?
-  I will need the ```reviewer_id``` for database normalization, as I aim to include 20 reviews per game.
-  For my initial analysis, I plan to use Steam Spy, which provides a range of values for the owners column. I aim to calculate the log mean of these values and use it in my analysis. From what I understand from the blog I read, Steam now makes user libraries private by default. Because of this, the values from Steam Spy may introduce some uncertainty into my analysis.

A good way to improve this approach is to use the number of reviewers. For example, I might assume that for every 1 review, there are about 70 game owners. However, as far as I understand, this review ratio depends on the type of game. After I get my pipeline fully working, I will focus on developing a better method.

Actually, I think Ulas could really help at this point. Both his Steam experience and way of thinking would be very useful! I’m a bit worried that bringing him in now might slow things down because I need to make a lot of adjustments and additions to the project, but at some point, I believe we can find a better way by discussing with him.


