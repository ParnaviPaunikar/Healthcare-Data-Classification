# Healthcare Data Classification Framework

## Diabetes Risk Assessment Using Machine Learning

This project develops a machine learning system for classifying patient records based on diabetes risk.

## Objective

The main objective is to develop a healthcare data classification framework that can analyze patient health parameters and classify the record into a diabetes risk category.

## Dataset

The dataset contains healthcare-related patient attributes:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome

The `Outcome` column is the target variable.

- 0 = Lower risk classification
- 1 = Higher risk classification

## Machine Learning Model

Logistic Regression is used as the classification algorithm.

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the healthcare dataset.
2. Identified invalid zero values in selected medical attributes.
3. Replaced invalid zero values with missing values.
4. Filled missing values using median imputation.
5. Separated input features and target variable.
6. Split the dataset into training and testing sets.
7. Applied StandardScaler for feature scaling.

## Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

## Web Application

A Streamlit web application was developed to allow users to enter patient information and obtain a model-based diabetes risk classification.

The application also displays an estimated risk probability.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Project Structure

```text
Healthcare-Data-Classification
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data
│   └── diabetes.csv
│
├── model
│   ├── disease_risk_model.pkl
│   └── scaler.pkl
│
└── venv