from confluent_kafka import Consumer
import json
import pandas as pd
import numpy as np
import logging
import os
import pickle

# Configuration
KafkaTopic = "uci_air_quality_data"
LogFile = "consumer.log"
is_first_message = True

# Logging setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

# Define log file path relative to script location
currect_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(currect_dir)
log_path = os.path.join(currect_dir, 'consumer.log')

# Create file handler for logging and set formatter
file_handler = logging.FileHandler(log_path, mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Kafka Consumer Config
conf = {
    'bootstrap.servers': '127.0.0.1:9092',
    'group.id': 'air_quality_test_001',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': False
}
consumer = Consumer(conf)
consumer.subscribe([KafkaTopic])

logger.info(f"Subscribed to topic: {KafkaTopic}")
logger.info("Starting Kafka consumer...")

# Load historical context
df_valid = pd.read_pickle(parent_dir + r"\phase_3_model_prediction\cleaned_validation_dataset.pkl")
df_valid = df_valid.tail(25)

# Output collection
list_date_time = []
list_target_val = []
list_baseline_predict_val = []
list_predict_val = []
list_actual_val = []

print("consumer.py is running...")

# Consuming Data from Kafka
try:  
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            logger.error(f"Kafka error: {msg.error()}")
            continue
        try:
            value = msg.value().decode('utf-8')
            data = json.loads(value)
            if data.get('end_of_stream'):
                logger.info("End-of-stream marker received.")
                break
            df_msg = pd.DataFrame([data['message']])
            df_msg['date_time'] = df_msg['Date'] + " " + df_msg['Time']
            df_msg['Date'] = pd.to_datetime(df_msg['Date'], format='%Y-%m-%d')
            df_msg['date_time'] = pd.to_datetime(df_msg['date_time'], format='%Y-%m-%d %H:%M:%S')
            timestamp_str = pd.to_datetime(df_msg['date_time'].values[0]).strftime("%Y%m%d%H%M%S")
            df_msg['day_of_week'] = df_msg['Date'].apply(lambda x : x.strftime("%A"))
            df_msg.columns = df_msg.columns.str.replace('(', '', regex=False).str.replace(')', '', regex=False)
            df_msg = df_msg.drop(columns = {"NMHCGT",'PT08.S2NMHC','Date','Time'})
            for a_col in df_msg.columns:
                if a_col not in ['date_time','day_of_week']:
                    df_msg['%s'%a_col] = np.where(df_msg['%s'%a_col]==-200,np.nan,df_msg['%s'%a_col])  
            # First Transaction
            if is_first_message:
                is_first_message = False
                df_test = pd.concat([df_valid,df_msg], axis = 0)
            else:
                df_test = pd.concat([df_test,df_msg], axis = 0)
            # Filling Null Value
            for a_col in df_test.columns:
                if a_col not in ['date_time','day_of_week']:
                    df_test['%s'%a_col] = df_test[a_col].ffill()

            df_test = df_test.reset_index(drop = True)
            
            # Feature Engineering
            # Rolling statistics
            rolling_features = {}
            for a_col in df_test.columns:
                if a_col not in ['date_time', 'day_of_week']:
                    for a_lag in range(2, 25):
                        roll_mean = df_test[a_col].rolling(window=a_lag).mean().shift(1)
                        roll_std = df_test[a_col].rolling(window=a_lag).std().shift(1)
                        mean_col_name = f'rolling_{a_col}_mean_{a_lag}'
                        std_col_name = f'rolling_{a_col}_std_{a_lag}'
                        rolling_features[mean_col_name] = roll_mean
                        rolling_features[std_col_name] = roll_std
            df_rolling = pd.DataFrame(rolling_features)
            df_dataprep = pd.concat([df_test, df_rolling], axis=1)
            ## Lagged Features
            lagged_features = []
            list_target_vals = ['COGT','C6H6GT','NOxGT','NO2GT']
            for a_target in list_target_vals:
                for a_lag in range(1, 25):
                    col_name1 = f"{a_target}_lag_{a_lag}"
                    lagged = df_test[a_target].shift(a_lag).rename(col_name1)
                    lagged_features.append(lagged)
            df_lags = pd.concat(lagged_features, axis=1)
            df_dataprep = pd.concat([df_dataprep, df_lags], axis=1)
            df_dataprep = df_dataprep.dropna().reset_index(drop = True)
            ## Dummy Variables
            df_dataprep['hour'] = df_dataprep['date_time'].dt.hour
            df_dataprep = pd.get_dummies(df_dataprep, columns=['hour','day_of_week'])
            list_hour = [a_hour for a_hour in range(24)]
            list_hour_col = ["hour_%s"%a_col for a_col in list_hour if "hour_%s"%a_col not in df_dataprep.columns]
            # Create dummy columns
            for a_col in list_hour_col:
                df_dataprep[a_col] = False
            list_day_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            list_day_of_week_col = ["day_of_week_%s"%a_col for a_col in list_day_of_week if "day_of_week_%s"%a_col not in df_dataprep.columns]
            for a_col in list_day_of_week_col:
                df_dataprep[a_col] = False        
            df_dataprep = df_dataprep.drop(columns = {"hour_0", 'day_of_week_Sunday'})
            df_dataprep = df_dataprep.astype({col: 'int' for col in df_dataprep.columns if df_dataprep[col].dtype == 'bool'})
            df_dataprep = df_dataprep.tail(1)

            # Prediction
            list_target_cols = ['COGT','C6H6GT','NOxGT','NO2GT']
            for a_target in list_target_cols:
                X_test = df_dataprep.drop(columns = {"date_time",a_target})
                y_test = df_dataprep[a_target]
                last_col_scale_index = 611+1
                columns_to_scale = [a_col for a_col in X_test.iloc[:,:last_col_scale_index].columns]
                with open(parent_dir + r"\phase_3_model_prediction\scaler_for_%s.pkl"%a_target, "rb") as f:
                    scaler = pickle.load(f)
                # Fit and transform only selected columns
                X_test_scaled = scaler.transform(X_test[columns_to_scale])
                X_test[columns_to_scale] = X_test_scaled
                # Sort Column 
                with open(parent_dir + r"\phase_3_model_prediction\column_orderfor_%s.json"%a_target, "r") as f:
                    column_order = json.load(f)
                X_test = X_test[column_order]
                # Import Model
                with open(parent_dir + r"\phase_3_model_prediction\regression_model_for_%s.pkl"%a_target, "rb") as f:
                    globals()['baseline_model_%s'%a_target] = pickle.load(f)
                with open(parent_dir + r"\phase_3_model_prediction\elasticnet_model_for_%s.pkl"%a_target, "rb") as f:
                    globals()['model_%s'%a_target] = pickle.load(f)
                # Predicting
                baseline_prediction = globals()['baseline_model_%s'%a_target].predict(X_test)
                prediction = globals()['model_%s'%a_target].predict(X_test)
                # Gather output
                list_date_time.append(str(df_dataprep['date_time'].values[0]))
                list_target_val.append(a_target)
                list_baseline_predict_val.append(baseline_prediction[0])
                list_predict_val.append(prediction[0])
                list_actual_val.append(df_dataprep['%s'%a_target].values[0])

            # Export related material for performance monitoring
            with open(parent_dir + r"\phase_3_model_prediction\Model_Performance_Monitoring\performance_date_time_asof_%s.json"%(timestamp_str), "w") as f:
                json.dump(list_date_time, f)
            with open(parent_dir + r"\phase_3_model_prediction\Model_Performance_Monitoring\performance_target_val_asof_%s.json"%(timestamp_str), "w") as f:
                json.dump(list_target_val, f)
            with open(parent_dir + r"\phase_3_model_prediction\Model_Performance_Monitoring\performance_baseline_predict_val_asof_%s.json"%(timestamp_str), "w") as f:
                json.dump(list_baseline_predict_val, f)
            with open(parent_dir + r"\phase_3_model_prediction\Model_Performance_Monitoring\performance_predict_val_asof_%s.json"%(timestamp_str), "w") as f:
                json.dump(list_predict_val, f)
            with open(parent_dir + r"\phase_3_model_prediction\Model_Performance_Monitoring\performance_actual_val_asof_%s.json"%(timestamp_str), "w") as f:
                json.dump(list_actual_val, f)
        except Exception as e:
            logger.exception(f"Exception during message processing: {e}")

finally:
    consumer.close()
    logger.info("Consumer closed.")