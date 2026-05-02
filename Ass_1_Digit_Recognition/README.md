# Ass-1: Digit Recognition using Supervised Learning

## Objective

The objective of this project is to recognize handwritten digits using a supervised machine learning algorithm. The model is trained on labeled digit images and predicts the digit class for new test samples.

## Dataset Description

- Dataset: `load_digits` dataset from scikit-learn
- Source: Scikit-learn built-in datasets
- Total samples: 1,797
- Classes: Digits from 0 to 9
- Features: 64 numerical pixel values representing an 8 x 8 grayscale image
- Target: Digit label from 0 to 9

## Technologies/Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Methodology

1. Load the digits dataset using `sklearn.datasets.load_digits`.
2. Convert the feature data into a pandas DataFrame for readability.
3. Split the dataset into training and testing sets.
4. Apply feature scaling using `StandardScaler`.
5. Train a Logistic Regression model.
6. Predict digit labels for the test data.
7. Evaluate the model using accuracy, classification report, and confusion matrix.


## Results Summary

The Logistic Regression model gives high accuracy on the digits dataset because the dataset is clean and well-labeled. The classification report shows precision, recall, and F1-score for each digit class. The confusion matrix shows how many digits were correctly and incorrectly classified.

## Conclusion

This project demonstrates supervised learning for digit recognition. Logistic Regression performs well on the scikit-learn digits dataset and is suitable for a simple academic machine learning assignment.