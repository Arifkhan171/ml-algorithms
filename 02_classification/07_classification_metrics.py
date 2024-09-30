# Classification Metrics — Evaluating Classifier Performance
# Accuracy, Confusion Matrix, Precision, Recall, F1 Score
# Dataset: Diabetes dataset

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              precision_score, recall_score,
                              f1_score, classification_report)

# Load dataset
df = pd.read_csv("../data/diabetes.csv")
print("Dataset shape:", df.shape)
print(df.head())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train models
dt = DecisionTreeClassifier(random_state=42)
lr = LogisticRegression(max_iter=1000)

dt.fit(X_train, y_train)
lr.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
y_pred_lr = lr.predict(X_test)

# ── Metrics comparison ──────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("DECISION TREE")
print("=" * 50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_dt):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_dt)}")

print("\n" + "=" * 50)
print("LOGISTIC REGRESSION")
print("=" * 50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_lr):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_lr)}")

print("\nFull Classification Report (LR):")
print(classification_report(y_test, y_pred_lr))
