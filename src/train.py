import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
data = load_breast_cancer()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow experiment
mlflow.set_experiment("confidence_collapse_experiment")

with mlflow.start_run():

    # model = LogisticRegression(max_iter=500)
    model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, preds)

    # confidence = max probability per prediction
    confidence = np.max(probs, axis=1)
    avg_confidence = np.mean(confidence)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("avg_confidence", avg_confidence)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # save model
    joblib.dump(model, "model.pkl")

print("Training complete")
