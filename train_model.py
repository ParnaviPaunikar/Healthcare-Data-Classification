import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("data/diabetes.csv")

print("Healthcare Dataset Loaded Successfully!")
print("-----------------------------------------")


# ==========================================
# 2. DATA PREPROCESSING
# ==========================================

columns_with_invalid_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for column in columns_with_invalid_zero:
    data[column] = data[column].replace(0, pd.NA)
    data[column] = data[column].fillna(data[column].median())


print("Data preprocessing completed.")


# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = data.drop("Outcome", axis=1)
y = data["Outcome"]


# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split Completed!")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==========================================
# 5. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed.")


# ==========================================
# 6. CREATE MACHINE LEARNING MODEL
# ==========================================

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

model.fit(X_train_scaled, y_train)

print("\nLogistic Regression Model Trained Successfully!")


# ==========================================
# 8. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test_scaled)


# ==========================================
# 9. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)


print("\n=========================================")
print("MODEL PERFORMANCE")
print("=========================================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))


print("\nConfusion Matrix:")
print(cm)


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 10. SAVE MODEL AND SCALER
# ==========================================

import joblib

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/disease_risk_model.pkl")

joblib.dump(scaler, "model/scaler.pkl")

print("\n=========================================")
print("MODEL SAVING")
print("=========================================")

print("Model saved as:")
print("model/disease_risk_model.pkl")

print("\nScaler saved as:")
print("model/scaler.pkl")

print("\nMachine Learning Model Completed Successfully!")