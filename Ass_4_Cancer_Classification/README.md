# Ass-4: Cancer Dataset Classification with Confusion Matrix

## Objective

The objective of this project is to classify breast cancer tumors as malignant or benign using a supervised machine learning algorithm and explain the test results using a confusion matrix.

## Dataset Description

- Dataset: Breast Cancer Wisconsin Diagnostic dataset
- Source: Scikit-learn built-in datasets
- Total samples: 569
- Classes:
  - Malignant
  - Benign
- Features: 30 numerical features computed from digitized images of breast mass cells, such as radius, texture, perimeter, area, smoothness, compactness, and concavity
- Target: Tumor diagnosis class

## Technologies/Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Methodology

1. Load the breast cancer dataset using `sklearn.datasets.load_breast_cancer`.
2. Convert the dataset into a pandas DataFrame.
3. Split the data into training and testing sets.
4. Apply feature scaling using `StandardScaler`.
5. Train a Logistic Regression classifier.
6. Predict tumor classes for the test data.
7. Evaluate the model using accuracy, classification report, and confusion matrix.
8. Explain the confusion matrix in simple terms.


## Results Summary

The model usually gives high accuracy because the breast cancer dataset is well-structured and suitable for binary classification. The confusion matrix is important because it shows how many malignant and benign cases were correctly or incorrectly predicted.

## Confusion Matrix Explanation

For this dataset:

- True Negative: Malignant cases correctly predicted as malignant
- False Positive: Malignant cases incorrectly predicted as benign
- False Negative: Benign cases incorrectly predicted as malignant
- True Positive: Benign cases correctly predicted as benign

## Conclusion

This project demonstrates binary classification using Logistic Regression. The confusion matrix helps understand model performance beyond accuracy, especially for medical datasets where incorrect predictions can be serious.