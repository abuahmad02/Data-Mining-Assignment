"""
Ass-1: Digit Recognition using Supervised Learning
Algorithm used: Logistic Regression
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_data():
    """Load the digits dataset."""
    digits = load_digits()
    x = pd.DataFrame(digits.data)
    y = pd.Series(digits.target, name="digit")
    return digits, x, y


def train_model(x_train, y_train):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42)
    )
    model.fit(x_train, y_train)
    return model


def save_sample_image(digits):
    plt.figure(figsize=(3, 3))
    plt.imshow(digits.images[0], cmap="gray")
    plt.title(f"Sample Digit: {digits.target[0]}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("sample_digit.png")
    plt.close()


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    print("Ass-1: Digit Recognition using Logistic Regression")
    print("-" * 55)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def main():
    digits, x, y = load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = train_model(x_train, y_train)
    save_sample_image(digits)
    evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    main()