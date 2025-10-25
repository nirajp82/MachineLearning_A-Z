# Data Preprocessing: Importance of Training-Test Split in ML Model Evaluation

## Overview
Splitting a dataset into **training** and **test** sets is a fundamental step in the machine learning workflow. This process ensures that model evaluation is based on unseen data, providing an unbiased measure of performance and generalization ability.

---

## Why the Training-Test Split Matters
When building a machine learning model, it’s essential to evaluate how well it performs on data it hasn’t encountered before.  
Without a proper split, a model might appear to perform well simply because it has memorized patterns from the same data used for training — a problem known as **overfitting**.  

A training-test split prevents this by separating data into two parts:
- **Training Set:** Used to fit and train the model.
- **Test Set:** Used exclusively for evaluation after the model is trained.

---

## Example: Predicting Car Sale Prices
Consider a scenario where the goal is to predict **car sale prices** based on features such as **mileage** and **age**.

Suppose the dataset contains **20 cars**:
- **Training Set (80%)** → 16 cars  
- **Test Set (20%)** → 4 cars  

The model learns relationships (e.g., between mileage, age, and price) using the training data.  
Once trained, it is then applied to the test set to predict sale prices for cars it hasn’t seen before.

---

## Applying the Model to the Test Set
When predictions are made on the test data:
- The model uses only what it learned from the training data.
- The true prices of the test cars are already known but hidden from the model during training.
- Comparing the **predicted prices** with the **actual prices** reveals how accurately the model generalizes.

---

## Evaluating Model Performance
Evaluation metrics such as **Mean Squared Error (MSE)**, **R² Score**, or **Mean Absolute Error (MAE)** can be used to quantify prediction accuracy.  
A good model should perform well not only on training data but also on the test set.  
If there is a significant performance gap, it may indicate overfitting or the need for additional data or feature tuning.

---

## Summary
Splitting data into training and test sets is a cornerstone of reliable machine learning model evaluation.  
It ensures that re
