import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.sklearn
import json
import joblib

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(experiment_name = "AirQuality_Simple_Regression")

df_train = pd.read_pickle(parent_dir + r"\data\processed\feature_engineering_train_dataset.pkl")
df_valid = pd.read_pickle(parent_dir + r"\data\processed\feature_engineering_validation_dataset.pkl")

list_target_vars = ['COGT','C6H6GT','NOxGT','NO2GT']

for a_target in list_target_vars:
    # Set X_train, y_train
    X_train = df_train.drop(columns = {"date_time",a_target})
    y_train = df_train[a_target]
    # Set X_test, y_test
    X_valid = df_valid.drop(columns = {"date_time",a_target})
    y_valid = df_valid[a_target]

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
    model = LinearRegression()
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
    coef_path = f"coefficients_{a_target}.csv"
    df_coef.to_csv(coef_path, index=False)

    # Save artifacts to temp filenames
    scaler_path = f"scaler_for_{a_target}.pkl"
    column_order_path = f"column_order_for_{a_target}.json"

    joblib.dump(scaler, scaler_path)

    with open(column_order_path, "w") as f:
        json.dump(column_order, f)

    # Log results
    with mlflow.start_run(run_name="SimpleRegression_%s"%a_target):
        mlflow.set_tag("run_type", "SimpleRegression_Baseline")
        mlflow.set_tag("target", a_target)
        # 2. Log parameters (input setup)
        mlflow.log_param("model", "Simple LinearRegression")
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
        mlflow.sklearn.log_model(model, artifact_path=f"SimpleRegressionModel_for_{a_target}")
        # Log coefficient
        mlflow.log_artifact(coef_path)

    os.remove(scaler_path)
    os.remove(column_order_path)
    os.remove(coef_path)