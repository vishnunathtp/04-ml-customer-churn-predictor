import pandas as pd
import joblib
import os

def predict_single(customer_data: dict):
    model_path = 'churn_model.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found. Run train.py first.")

    model = joblib.load(model_path)
    df = pd.DataFrame([customer_data])
    churn_prob = model.predict_proba(df)[0][1]
    prediction = int(churn_prob >= 0.5)

    return {
        "churn_prediction": prediction,
        "churn_probability": round(churn_prob, 4),
        "status": "High Risk" if prediction == 1 else "Safe"
    }

if __name__ == '__main__':
    sample = {
        'tenure_months': 3,
        'monthly_charges': 85.0,
        'total_charges': 255.0,
        'contract_type': 'Month-to-Month',
        'paperless_billing': 'Yes'
    }
    print("Test Inference Sample:", sample)
    # If model is trained:
    if os.path.exists('churn_model.joblib'):
        print("Result:", predict_single(sample))
    else:
        print("Please run train.py to train and serialize churn_model.joblib")
