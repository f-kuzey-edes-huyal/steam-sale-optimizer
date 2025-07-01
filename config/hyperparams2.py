def get_search_space(trial):
    model_type = trial.suggest_categorical("model_type", ["RandomForest", "LightGBM", "ExtraTrees", "LinearSVR"])

    if model_type == "RandomForest":
        return {
            "model_type": model_type,
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=25),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }

    elif model_type == "LightGBM":
        return {
            "model_type": model_type,
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        }

    elif model_type == "ExtraTrees":
        return {
            "model_type": model_type,
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=25),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }

    elif model_type == "LinearSVR":
        return {
            "model_type": model_type,
            "epsilon": trial.suggest_float("epsilon", 1e-4, 0.1, log=True),
            "C": trial.suggest_float("C", 0.01, 100, log=True),
        }
