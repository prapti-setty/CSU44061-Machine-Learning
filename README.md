# CSU44061-Machine-Learning
College Assignment aimed at getting experience with the entire machine learning pipeline

## Table of Contents
1. [About](#About)
1. [Method to Solve](#Solve)
1. [Other things tried](#Other)

## About

Task: Given a training dataset with many features and an income, predict, for some given test dataset the income.

## Solve

Used libraries: Scikit-learn, numpy, math & pandas

Method:
Step 1: Some preprocessing - drop Nan in training & fill nan in test with 0 or unknown \n
Step 2: Simple Linear regression with only single column (Age) - results - 150k \n
Step 3: Linear regression with multiple columns (all numerical columns used here) \n
Step 4: Label encode categorical columns like University Degree and gender as these have fewer variables \n
Step 5: One hot encode above columns \n
Step 6: One hot encode all categorical columns \n
Step 6: Try ridge & lasso regression ( Ridge = better, lets use ridge) \n
Step 7: Replace test Nans with meaningful data (Means) \n
Step 8: Try Random Forest regressor - performs best so far but results "look" different to Ridge (i.e differences of a few thousand in salaries) \n
Step 9: Try mean of Random Forest and Ridge - performs good \n

#Testing

Used train_test_split to split data 0.2 - 0.8 \n
When testing Locally, train without validation data by just changing what the model is fit with

## Other

- Tried Scaling (MinMax and Standard) data - Helps for ridge, but not significantly for Random Forest Regressor 
- Tried Normalizing data - similar to above
- Tried adding polynomial_features but increases run time significantly
