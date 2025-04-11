import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.sklearn
import json
import joblib
from itertools import product

target_vars = 'NO2GT'
model_name = 'ElasticNetRegression'

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
mlflow.set_tracking_uri(f"file:///{os.path.join(parent_dir, 'mlruns')}")
mlflow.set_experiment(experiment_name = f"AirQuality_{target_vars}_Prediction")

list_alpha = [0.01, 0.1, 1.0, 10]
list_l1_ratio = [0.1, 0.5, 0.9]

df_train = pd.read_pickle(parent_dir + r"\data\processed\feature_engineering_train_dataset.pkl")
df_valid = pd.read_pickle(parent_dir + r"\data\processed\feature_engineering_validation_dataset.pkl")

combinations = list(product(list_alpha, list_l1_ratio))

for alpha, l1_ratio in combinations:

    # Set X_train, y_train
    X_train = df_train.drop(columns = {"date_time",target_vars})
    y_train = df_train[target_vars]
    # Set X_test, y_test
    X_valid = df_valid.drop(columns = {"date_time",target_vars})
    y_valid = df_valid[target_vars]
    last_col_scale_index = 213+1
    columns_to_scale = [a_col for a_col in X_train.iloc[:,:last_col_scale_index].columns]
    scaler = StandardScaler()

    # Fit and transform only selected columns
    X_train_scaled = scaler.fit_transform(X_train[columns_to_scale])
    X_train[columns_to_scale] = X_train_scaled

    X_valid_scaled = scaler.transform(X_valid[columns_to_scale])
    X_valid[columns_to_scale] = X_valid_scaled

    # Save the means and standard deviations for coefficient adjustment
    feature_means = scaler.mean_
    feature_stds = scaler.scale_
    # Column order for test data prediction
    column_order = list(X_train.columns)
    # Modeling
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=88)
    model.fit(X_train, y_train)
    # Get standardized coefficients
    standardized_coefficients = model.coef_
    intercept = model.intercept_
    # Assuming standardized_coefficients includes all features
    scaled_coefs = standardized_coefficients[:last_col_scale_index]
    unscaled_coefs = standardized_coefficients[last_col_scale_index:]
    # Adjust only the scaled coefficients
    original_scaled_coefs = scaled_coefs / feature_stds
    # Combine back the full coefficient vector
    original_coefficients = np.concatenate([original_scaled_coefs, unscaled_coefs])
    # Adjust intercept
    original_intercept = intercept - np.sum(original_scaled_coefs * feature_means)
    # Train Score
    train_pred = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, train_pred))
    # Test Score
    y_pred = model.predict(X_valid)
    mae_validation = mean_absolute_error(y_valid, y_pred)
    rmse_validation = np.sqrt(mean_squared_error(y_valid, y_pred))
    # Get coefficient
    df_coef = pd.DataFrame({"feature": X_train.columns,"coefficient": original_coefficients})
    df_coef = pd.concat([df_coef,pd.DataFrame([{"feature": "INTERCEPT", "coefficient": original_intercept}])], ignore_index=True)
    coef_path = f"coefficients_{target_vars}.csv"
    df_coef.to_csv(coef_path, index=False)
    # Save artifacts to temp filenames
    scaler_path = f"scaler_for_{target_vars}.pkl"
    column_order_path = f"column_order_for_{target_vars}.json"
    joblib.dump(scaler, scaler_path)
    with open(column_order_path, "w") as f:
        json.dump(column_order, f)
    input_example = X_train.head(1)
    # Log results
    with mlflow.start_run(run_name=f"{model_name}_{target_vars}_a{alpha}_l1{l1_ratio}"):
        mlflow.set_tag("run_type", f"{model_name}")
        mlflow.set_tag("target", target_vars)
        # 2. Log parameters (input setup)
        mlflow.log_param("model", "Elastic Net LinearRegression")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("features", ", ".join(X_train.columns))
        # Log train performance for monitor overfitting
        mlflow.log_metric("mae_train", mae_train)
        mlflow.log_metric("rmse_train", rmse_train)
        # Log validation performance for decision making
        mlflow.log_metric("mae_validation", mae_validation)
        mlflow.log_metric("rmse_validation", rmse_validation)
        # Log scaler and column order as artifacts
        mlflow.log_artifact(scaler_path)
        mlflow.log_artifact(column_order_path)
        # Log model
        mlflow.sklearn.log_model(model, artifact_path=f"{model_name}_for_{target_vars}", input_example=input_example)
        # Log coefficient
        mlflow.log_artifact(coef_path)
    os.remove(scaler_path)
    os.remove(column_order_path)
    os.remove(coef_path)