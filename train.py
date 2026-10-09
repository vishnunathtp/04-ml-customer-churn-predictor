import pandas as pd
import joblib
from model_pipeline import create_churn_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train():
    df = pd.read_csv('dataset/sample.csv')
    X = df.drop(columns=['customer_id', 'churn'])
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = create_churn_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Training Complete.")
    print("Evaluation Accuracy:", accuracy_score(y_test, y_pred))

    joblib.dump(pipeline, 'churn_model.joblib')
    print("Saved trained model to churn_model.joblib")

if __name__ == '__main__':
    train()
