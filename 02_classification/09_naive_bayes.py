# Naive Bayes Classifier — Gaussian NB
# Based on Bayes theorem with assumption of feature independence
# Dataset: Iris species classification

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("../data/iris.csv")
print("Dataset shape:", df.shape)
print("Classes:", df["species"].unique())

X = df.drop("species", axis=1)
y = df["species"]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Train Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

# Evaluate
labels = ["setosa", "versicolor", "virginica"]
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=labels))
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Class probabilities for one sample
proba = gnb.predict_proba(X_test[:1])
print("Class probabilities for first test sample:")
for label, prob in zip(gnb.classes_, proba[0]):
    print(f"  {label}: {prob:.4f}")
