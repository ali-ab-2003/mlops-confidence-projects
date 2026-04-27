from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.noise import add_gaussian_noise
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def evaluate(noise_level):

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Add noise to test set only
    X_test_noisy = add_gaussian_noise(X_test, noise_level)

    # model = LogisticRegression(max_iter=500)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    model.fit(X_train, y_train)

    preds = model.predict(X_test_noisy)
    probs = model.predict_proba(X_test_noisy)

    accuracy = accuracy_score(y_test, preds)
    confidence = np.max(probs, axis=1)

    return accuracy, np.mean(confidence)

