# Ass-3: K-Nearest Neighbors on IRIS Dataset

## Objective

The objective of this project is to classify iris flowers using the K-Nearest Neighbors algorithm and compare accuracy for different values of K.

## Dataset Description

- Dataset: IRIS dataset
- Source: Scikit-learn built-in datasets
- Total samples: 150
- Classes: Setosa, Versicolor, Virginica
- Features:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- Target: Iris flower species

## Technologies/Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Methodology

1. Load the IRIS dataset using `sklearn.datasets.load_iris`.
2. Convert the dataset into a pandas DataFrame.
3. Split the data into training and testing sets.
4. Apply feature scaling using `StandardScaler`.
5. Train KNN models using different values of K.
6. Compare the accuracy for each K value.
7. Select the best K value based on test accuracy.
8. Evaluate the best model using classification report and confusion matrix.


## Results Summary

The accuracy comparison helps identify which K value performs best on the test data. KNN works well on the IRIS dataset because the dataset is small, clean, and contains clearly separated flower classes.

## Conclusion

This project demonstrates KNN classification and shows how changing the value of K can affect model accuracy. The IRIS dataset is suitable for understanding basic supervised classification.