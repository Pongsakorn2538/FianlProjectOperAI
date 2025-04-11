Code for serving your trained model as an API

- app.py or main.py: FastAPI or Flask app
- predict.py: loads and predicts using the saved model
- model_loader.py: handles loading scaler, model, and column order
- Dockerfile: optional Dockerfile for this service