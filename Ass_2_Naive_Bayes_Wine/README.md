# Ass-2: Naive Bayes Classification using an Online Dataset

## Objective

The objective of this project is to apply the Naive Bayes classification algorithm on a publicly available dataset and evaluate its performance.

## Dataset Description

- Dataset: Wine Recognition dataset
- Source: UCI Machine Learning Repository, available through scikit-learn as `load_wine`
- Total samples: 178
- Classes: 3 types of wine
- Features: 13 chemical analysis features such as alcohol, malic acid, ash, magnesium, flavanoids, color intensity, and proline
- Target: Wine class label

## Technologies/Libraries Used

- Python
- NumPy
- Pandas
- Scikit-learn

## Methodology

1. Load the Wine dataset using `sklearn.datasets.load_wine`.
2. Convert the dataset into a pandas DataFrame.
3. Split the data into training and testing sets.
4. Train a Gaussian Naive Bayes classifier.
5. Predict the wine class for test samples.
6. Evaluate the model using accuracy, classification report, and confusion matrix.


## Results Summary

Naive Bayes performs well on the Wine dataset because the features are numerical and the classes are clearly separated. The confusion matrix shows the number of correct and incorrect predictions for each wine class.

## Conclusion

This project demonstrates how Naive Bayes can be used for multi-class classification. Gaussian Naive Bayes is simple, fast, and suitable for academic classification tasks.