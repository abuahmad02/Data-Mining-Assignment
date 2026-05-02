"""
Ass-2: Naive Bayes Classification using an Online Dataset
Algorithm used: Gaussian Naive Bayes
"""

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def load_data():
    wine = load_wine()
    x = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="wine_class")
    return wine, x, y


def train_model(x_train, y_train):
    model = GaussianNB()
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test, target_names):
    """Evaluate the trained model on test data."""
    y_pred = model.predict(x_test)

    print("Ass-2: Naive Bayes Classification on Wine Dataset")
    print("-" * 55)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def main():
    wine, x, y = load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = train_model(x_train, y_train)
    evaluate_model(model, x_test, y_test, wine.target_names)


if __name__ == "__main__":
    main()