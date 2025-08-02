# martial_status_prediction

This project implements a machine learning model to predict a person's marital status based on various features. The goal is to predict the `marital-status` label, which is a multi-class classification problem. The project follows the standard machine learning life cycle, including data exploration, preprocessing, model training, evaluation, and hyperparameter tuning.

## Project Overview

The census dataset used (`censusData.csv`) contains demographic and economic attributes of individuals, such as age, education, occupation, and hours worked per week. The goal is to predict whether a person is "Married," "Previously Married," or "Not Currently Married."

This project was completed as part of the Cornell's **Break Through Tech AI Program**.

## Project Files

- **`data.zip`**: Contains the dataset used for training and testing in a zip file.
- **`marital_status_modeling.ipynb`**: Jupyter Notebook with detailed code for data preprocessing, model training, evaluation, and improvement.
- **`marital_status_model.pkl`**: Saved model file (Pickle format) for predictions on new data.

## Installation and Setup

To run this project, clone the repository, extract `data.zip` file, and install the necessary dependencies.

Example:
```bash
git clone https://github.com/yourusername/marital_status_prediction.git
cd ML-Marital-Status-Prediction
pip install -r requirements.txt
```

Run jupyter notebook with the following:

```bash
jupyter notebook
```

## Key Features

- **Logistic Regression Model** for multi-class classification.
- **GridSearchCV** for hyperparameter tuning to find optimal `C`.
- **Feature Engineering**: One-hot encoding for categorical variables, normalization for numerical features, and outlier handling.
- **Model Evaluation** using metrics like Accuracy, Log Loss, AUC, and Precision-Recall curves.

## Steps Involved

1. **Data Exploration**:
    - Inspecting the dataset for missing values, duplicates, and outliers.
    - Visualizing relationships between features and the target variable (`marital-status`).

2. **Data Preprocessing**:
    - Removing irrelevant features (e.g., `education` and `fnlwgt`).
    - Handling missing values by filling numerical features with mean/median values and categorical features with 'Unknown'.
    - One-hot encoding categorical variables (e.g., `workclass`, `occupation`, `native-country`).
    - Normalizing numerical features and addressing class imbalance.

3. **Modeling**:
    - Using **Logistic Regression** as the primary machine learning model.
    - Hyperparameter tuning using **GridSearchCV** to find the best `C` value.
    - Feature selection using **SelectKBest** to optimize model performance.

4. **Model Evaluation**:
    - Evaluating model performance using **Accuracy**, **Log Loss**, **ROC Curve**, **Precision-Recall Curve**, and **Confusion Matrix**.
    - Using cross-validation and AUC-ROC to assess the model's generalization ability.

5. **Model Persistence**:
    - Saving the trained model using **Pickle** to use it for future predictions.

## Future Work

- Experiment with different algorithms such as **Random Forest** or **XGBoost** to compare performance.
- Apply **SMOTE** (Synthetic Minority Over-sampling Technique) for handling class imbalance.
- Evaluate the model on additional metrics like **F1-Score** and **ROC AUC**.
