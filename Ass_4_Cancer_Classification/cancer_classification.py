"""
Ass-4: Cancer Dataset Classification with Confusion Matrix
Algorithm used: Logistic Regression
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_data():
    cancer = load_breast_cancer()
    x = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = pd.Series(cancer.target, name="diagnosis")
    return cancer, x, y


def train_model(x_train, y_train):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42)
    )
    model.fit(x_train, y_train)
    return model


def save_confusion_matrix_plot(cm, target_names):
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names
    )
    display.plot(cmap="Blues")
    plt.title("Cancer Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()


def evaluate_model(model, x_test, y_test, target_names):
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)

    print("Ass-4: Cancer Dataset Classification using Logistic Regression")
    print("-" * 65)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion Matrix:")
    print(cm)

    print("\nConfusion Matrix Explanation:")
    print(f"Malignant correctly predicted as malignant: {cm[0][0]}")
    print(f"Malignant incorrectly predicted as benign: {cm[0][1]}")
    print(f"Benign incorrectly predicted as malignant: {cm[1][0]}")
    print(f"Benign correctly predicted as benign: {cm[1][1]}")

    save_confusion_matrix_plot(cm, target_names)


def main():
    cancer, x, y = load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = train_model(x_train, y_train)
    evaluate_model(model, x_test, y_test, cancer.target_names)


if __name__ == "__main__":
    main()