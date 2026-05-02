"""
Ass-3: K-Nearest Neighbors on IRIS Dataset
Algorithm used: K-Nearest Neighbors
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_data():
    iris = load_iris()
    x = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")
    return iris, x, y


def train_and_compare_k_values(x_train, x_test, y_train, y_test):
    k_values = [1, 3, 5, 7, 9]
    accuracy_scores = {}
    models = {}

    for k in k_values:
        model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=k)
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy_scores[k] = accuracy_score(y_test, y_pred)
        models[k] = model

    return accuracy_scores, models


def save_accuracy_plot(accuracy_scores):
    """Save a line plot comparing accuracy values for different K values."""
    plt.figure(figsize=(6, 4))
    plt.plot(list(accuracy_scores.keys()), list(accuracy_scores.values()), marker="o")
    plt.xlabel("K Value")
    plt.ylabel("Accuracy")
    plt.title("KNN Accuracy Comparison on IRIS Dataset")
    plt.xticks(list(accuracy_scores.keys()))
    plt.ylim(0.8, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("knn_accuracy_comparison.png")
    plt.close()


def evaluate_best_model(best_k, best_model, x_test, y_test, target_names):
    y_pred = best_model.predict(x_test)

    print("Ass-3: KNN Classification on IRIS Dataset")
    print("-" * 55)
    print("Best K Value:", best_k)
    print("Best Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def main():
    iris, x, y = load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    accuracy_scores, models = train_and_compare_k_values(
        x_train,
        x_test,
        y_train,
        y_test
    )

    print("Accuracy Comparison for Different K Values:")
    for k, accuracy in accuracy_scores.items():
        print(f"K = {k}: Accuracy = {accuracy:.4f}")

    best_k = max(accuracy_scores, key=accuracy_scores.get)
    best_model = models[best_k]

    save_accuracy_plot(accuracy_scores)
    print()
    evaluate_best_model(best_k, best_model, x_test, y_test, iris.target_names)


if __name__ == "__main__":
    main()