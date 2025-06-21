
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
python scripts/train_base_mlflow_experiment_tracking.py
```

Access your MLflow experiments by navigating to http://127.0.0.1:5000 in your browser to visualize and manage your experiment tracking

<img src="https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/figures/exp_tracking_optuna.png" alt="Alt text" width="800"/>


## 🗃️ Model Registry

Instead of selecting the model for registration within the training script, you can register the model separately using ```mlflow.register_model```.

While your MLflow tracking server is running, open a new terminal and execute the code below to register your model.

```
python scripts\select_and_register_best_model.py
```

Note: Model registration is not the same as deployment, but it allows you to prepare the model for future deployment or serving.




## 📈 Monitoring Data Drift 

## 📦 Model Deployment

## 🚀 Full Orchestration with Apache Airflow









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


