import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.sklearn
import json
from itertools import product

target_vars = 'NOxGT'
model_name = 'LightGBM'

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
mlflow.set_tracking_uri(f"file:///{os.path.join(parent_dir, 'mlruns')}")
mlflow.set_experiment(experiment_name = f"AirQuality_{target_vars}_Prediction")

df_train = pd.read_pickle(parent_dir + r"\data\processed\feature_engineering_train_dataset.pkl")
df_valid = pd.read_pickle(parent_dir + r"\data\processed\feature_engineering_validation_dataset.pkl")

param_grid = {
    "n_estimators": [500, 1000, 2000],              # Boosting rounds
    "learning_rate": [0.01, 0.05, 0.1],             # Smaller = more accurate
}

keys = list(param_grid.keys())
combinations = list(product(*param_grid.values()))

for combo in combinations:
    params = dict(zip(keys, combo))
    # Set X_train, y_train
    X_train = df_train.drop(columns = {"date_time",target_vars})
    y_train = df_train[target_vars]
    # Set X_test, y_test
    X_valid = df_valid.drop(columns = {"date_time",target_vars})
    y_valid = df_valid[target_vars]
    # Column order for test data prediction
    column_order = list(X_train.columns)
    input_example = X_train.head(1)
    # Modeling
    model = lgb.LGBMRegressor(**params, n_jobs=-1)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
    )
    train_pred = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, train_pred))
    # Test Score
    y_pred = model.predict(X_valid)
    mae_validation = mean_absolute_error(y_valid, y_pred)
    rmse_validation = np.sqrt(mean_squared_error(y_valid, y_pred))

    # Save artifacts to temp filenames
    column_order_path = f"column_order_for_{target_vars}.json"
    with open(column_order_path, "w") as f:
        json.dump(column_order, f)
    input_example = X_train.head(1)
    run_name = f"{model_name}_{target_vars}_" + "_".join([f"{k}={v}" for k, v in params.items()])

    # Log results
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("run_type", f"{model_name}")
        mlflow.set_tag("target", target_vars)
        # 2. Log parameters (input setup)
        mlflow.log_param("model", model_name)
        mlflow.log_param("param_name", params)
        # Log train performance for monitor overfitting
        mlflow.log_metric("mae_train", mae_train)
        mlflow.log_metric("rmse_train", rmse_train)
        # Log validation performance for decision making
        mlflow.log_metric("mae_validation", mae_validation)
        mlflow.log_metric("rmse_validation", rmse_validation)
        # Log scaler and column order as artifacts
        mlflow.log_artifact(column_order_path)
        # Log model
        mlflow.sklearn.log_model(model, artifact_path=f"{model_name}_for_{target_vars}", input_example=input_example)
    os.remove(column_order_path)
