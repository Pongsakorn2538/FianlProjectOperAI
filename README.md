# Real-Time Air Quality Monitoring System

This project presents an end-to-end, production-ready air quality monitoring system designed to process, predict, and monitor pollution levels using real-time data streams and machine learning. The system is built on a robust MLOps pipeline, integrating modern technologies such as Kafka, Docker, Kubernetes, MLflow, and Evidently to ensure scalability, consistency, and transparency across the model lifecycle.

---

## Prerequisites
1. **Python**  
    Please refer to https://realpython.com/installing-python/

2. **Python Library**  
    Please refer to requirements.txt and the instruction from https://packaging.python.org/en/latest/tutorials/installing-packages/

3. **Kafka Set Up**  
    Please refer to Kafka Setup Description.pdf in documents folder.

4. **ML Flow Set Up**  
    Please refer to ML Flow Environment Set Up.pdf in documents folder.

---

## Test Kafka on Local Environment
After setting up Kafka and ML flow,
1. Start Kafka server. Please refer to Kafka Setup Description.pdf in documents folder.
2. Run producer.py by using the following command: python producer.py
3. Run consumer.py by using the following command: python consumer.py

---

## Test ML Flow on Local Environment
1. Start ML flow. Please refer to ML Flow Environment Set Up.pdf in documents folder.
2. Runing the each model in the folder mlflow_experiments
    - COGT_SimpleRegression.py
    - COGT_ElasticNet_Regression.py
    - COGT_LightGBM.py
    - C6H6GT_SimpleRegression.py
    - C6H6GT_ElasticNet_Regression.py
    - C6H6GT_LightGBM.py
    - NOxGT_SimpleRegression.py
    - NOxGT_ElasticNet_Regression.py
    - NOxGT_LightGBM.py
    - NO2GT_SimpleRegression.py
    - NO2GT_ElasticNet_Regression.py
    - NO2GT_LightGBM.py