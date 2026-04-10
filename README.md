#### 🚗 Car Price Prediction - End-to-End Machine Learning Project

This project focuses on predicting car prices using Machine Learning and building a complete end-to-end ML pipeline.

## 📌 Overview

The goal of this project is to analyze a car dataset and build regression models to accurately predict the price of a car based on different features. The project follows a full ML pipeline including data preprocessing, model training, evaluation, and selection.

## ⚙️ Algorithms Used

During the experimentation phase (in Jupyter Notebook), multiple algorithms were tested to evaluate their performance:

Linear Regression
K-Nearest Neighbors (KNN)
Decision Tree
Random Forest
AdaBoost
Gradient Boosting
# 🔍 Model Selection Strategy

All the above models were initially implemented and evaluated in a Jupyter Notebook to compare their performance.

Based on evaluation metrics, the top 3 performing models were selected:

Decision Tree
Random Forest
Gradient Boosting

These models showed better performance compared to others.

## 🏆 Final Model Selection

From the selected top 3 models, the final model is chosen based on the R² Score (coefficient of determination).

The model with the highest R² score is used as the final model for prediction in the main pipeline.

## 🔄 Workflow
Data Cleaning & Preprocessing
Feature Engineering
Model Training (multiple models in Jupyter Notebook)
Model Evaluation & Comparison
Selection of Top 3 Models
Final Model Selection based on R² Score
Prediction using the best model
# 🎯 Objective

To build a robust and scalable ML pipeline that can effectively predict car prices by selecting the best-performing model through proper evaluation.

# 🚀 Future Scope
Hyperparameter tuning
Deployment using Flask
Integration with real-time data

# 💡 This project demonstrates how to experiment with multiple models, compare their performance, and select the best one for production use.