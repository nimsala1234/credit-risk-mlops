# Credit Risk Prediction & MLOps Pipeline

Educational credit default prediction prototype using:
SQL ingestion, logistic regression, gradient boosting,
validation-based model selection, held-out evaluation,
MLflow tracking, FastAPI and Docker.

## Dataset
Yeh, I. (2009). Default of Credit Card Clients.
UCI Machine Learning Repository.
https://doi.org/10.24432/C55S3H
License: CC BY 4.0.

## Run
pip install -r requirements.txt

python train.py

uvicorn api:app --reload

Open http://localhost:8000/docs

## Experiment tracking
mlflow ui --backend-store-uri sqlite:///mlflow.db

## Drift report
python drift.py batch.csv

## Docker inference
docker build -t credit-risk .
docker run -p 8000:8000 -v "$PWD/artifacts:/app/artifacts:ro" credit-risk

## Limitations
Historical dataset; random stratified holdout, not temporal validation.
Mean-shift monitoring is a simple diagnostic, not a validated drift detector.
Fairness, calibration and external validation require further study.
Not intended for actual lending decisions.
