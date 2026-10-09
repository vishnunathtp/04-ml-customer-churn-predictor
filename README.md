# 04-ml-customer-churn-predictor

Supervised machine learning pipeline for telecom customer churn prediction using Scikit-Learn and RandomForestClassifier.

## Pipeline Architecture
- **ColumnTransformer**: Normalizes numeric telemetry and hot-encodes categorical variables.
- **RandomForestClassifier**: Ensemble model outputting binary predictions and churn probability scores.
- **Serialization**: Model artifacts persisted with `joblib`.

## Running the Project
```bash
pip install -r requirements.txt
python train.py
python predict.py
```
