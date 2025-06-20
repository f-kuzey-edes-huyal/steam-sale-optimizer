def get_search_space(trial):
    model_type = trial.suggest_categorical("model_type", ["RandomForest", "LightGBM", "ExtraTrees", "LinearSVR"])
    
    if model_type == "RandomForest":
        return {
            "model_type": model_type,
            "rf_n_estimators": trial.suggest_int("rf_n_estimators", 50, 500, step=25),
            "rf_max_depth": trial.suggest_int("rf_max_depth", 5, 50),
            "rf_min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
        }

    elif model_type == "LightGBM":
        return {
            "model_type": model_type,
            "lgbm_n_estimators": trial.suggest_int("lgbm_n_estimators", 50, 500, step=25),
            "lgbm_max_depth": trial.suggest_int("lgbm_max_depth", 3, 30),
            "lgbm_learning_rate": trial.suggest_float("lgbm_learning_rate", 0.001, 0.3, log=True),
            "lgbm_num_leaves": trial.suggest_int("lgbm_num_leaves", 20, 100),
        }

    elif model_type == "ExtraTrees":
        return {
            "model_type": model_type,
            "et_n_estimators": trial.suggest_int("et_n_estimators", 50, 500, step=25),
            "et_max_depth": trial.suggest_int("et_max_depth", 5, 50),
            "et_min_samples_split": trial.suggest_int("et_min_samples_split", 2, 20),
        }

    elif model_type == "LinearSVR":
        return {
            "model_type": model_type,
            "svr_epsilon": trial.suggest_float("svr_epsilon", 1e-4, 0.1, log=True),
            "svr_C": trial.suggest_float("svr_C", 0.01, 100, log=True),
        }
