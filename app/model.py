"""Model training and loading.

Trains a small image classifier on the scikit-learn digits dataset
(8x8 grayscale images, 10 classes). Swap this out for your own model:
the rest of the service does not care what the model is, only that it
exposes .predict() and .predict_proba().
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_PATH = Path(__file__).parent / "model.joblib"
N_FEATURES = 64  # 8x8 image flattened


def train(save: bool = True) -> float:
    """Train the classifier and return test accuracy."""
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, clf.predict(X_test))

    if save:
        joblib.dump(clf, MODEL_PATH)

    return float(accuracy)


def load():
    """Load the trained model, training it first if it does not exist."""
    if not MODEL_PATH.exists():
        train(save=True)
    return joblib.load(MODEL_PATH)


def predict(model, pixels: list[float]) -> dict:
    """Run inference on a single flattened 8x8 image."""
    if len(pixels) != N_FEATURES:
        raise ValueError(f"expected {N_FEATURES} values, got {len(pixels)}")

    x = np.array(pixels, dtype=float).reshape(1, -1)
    probabilities = model.predict_proba(x)[0]
    predicted = int(np.argmax(probabilities))

    return {
        "prediction": predicted,
        "confidence": round(float(probabilities[predicted]), 4),
    }


if __name__ == "__main__":
    print(f"test accuracy: {train():.3f}")
