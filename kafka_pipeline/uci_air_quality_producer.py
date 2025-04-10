import os
from ucimlrepo import fetch_ucirepo
from confluent_kafka import Producer
import json
import pandas as pd
import time
import logging

# Configuration
TestDataSetCutOff = "2004-12-31"
KafkaTopic = "uci_air_quality_data"
KafkaBroker = "127.0.0.1:9092"
LogFile = "producer.log"

# Logging setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

# Define log file path relative to script location
log_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(log_dir, 'producer.log')

# Create file handler for logging and set formatter
file_handler = logging.FileHandler(log_path, mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Kafka Producer Configuration
conf = {'bootstrap.servers': KafkaBroker}
producer = Producer(conf)

print("producer.py is running...")

# Data Preprocessing
def get_air_quality_test_dataset():
    air_quality = fetch_ucirepo(id=360)
    if air_quality is None or air_quality.data is None:
        raise ValueError("Failed to fetch Air Quality dataset from UCI.")

    df = air_quality.data.features.copy()

    required_columns = ['Date', 'Time']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    df['date_time'] = pd.to_datetime(df['Date'] + " " + df['Time'], errors='coerce')
    df = df.sort_values(by='date_time').reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    df_test = df[df['Date'] > TestDataSetCutOff].reset_index(drop=True)
    df_test['Date'] = df_test['Date'].astype(str)
    df_test = df_test.drop(columns=['date_time'])

    logger.info(f"Raw dataset shape: {df.shape}")
    logger.info(f"Filtered test dataset shape: {df_test.shape}")
    return df_test

# Kafka Callback
def delivery_report(err, msg):
    if err is not None:
        logger.error(f"Delivery failed: {err}")
    else:
        logger.info(f"Delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

# Message Sender
def send_message():
    try:
        logger.info(f"Kafka Topic: {KafkaTopic}")
        logger.info(f"Kafka Broker: {KafkaBroker}")

        df_test = get_air_quality_test_dataset()
        logger.info(f"Start sending {df_test.shape[0]} records to topic '{KafkaTopic}'")

        for i in range(df_test.shape[0]):
            record = df_test.iloc[i].to_dict()
            message = {'Record Number': i, 'message': record}
            producer.produce(KafkaTopic, value=json.dumps(message), callback=delivery_report)
            producer.poll(0)
            time.sleep(0.05)

        # Send end-of-stream marker
        end_msg = {'Record Number': -1, 'message': {}, 'end_of_stream': True}
        producer.produce(KafkaTopic, value=json.dumps(end_msg), callback=delivery_report)
        producer.poll(0)
        logger.info("Sent end-of-stream marker.")

    except Exception as e:
        logger.exception(f"Exception occurred while sending messages: {e}")
    finally:
        producer.flush()
        logger.info("Producer flushed and closed.")

if __name__ == '__main__':
    send_message()