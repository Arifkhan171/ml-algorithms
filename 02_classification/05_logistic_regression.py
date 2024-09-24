# Logistic Regression — Binary and Multiclass Classification
# Datasets: Heart disease (binary), Iris (multiclass)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              recall_score, precision_score)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ── Binary Classification — Heart Disease ──────────────────────────────────
print("=" * 50)
print("BINARY CLASSIFICATION — Heart Disease Dataset")
print("=" * 50)

df = pd.read_csv("../data/heart.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, :-1], df.iloc[:, -1],
    test_size=0.2, random_state=42)

lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier()

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_dt = dt.predict(X_test)

print(f"\nLogistic Regression accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Decision Tree accuracy:       {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"\nLR Confusion Matrix:\n{confusion_matrix(y_test, y_pred_lr)}")
print(f"LR Recall:    {recall_score(y_test, y_pred_lr):.4f}")
print(f"LR Precision: {precision_score(y_test, y_pred_lr):.4f}")

# ── Multiclass Classification — Iris ───────────────────────────────────────
print("\n" + "=" * 50)
print("MULTICLASS CLASSIFICATION — Iris Dataset")
print("=" * 50)

df2 = pd.read_csv("../data/iris.csv")
encoder = LabelEncoder()
df2["species"] = encoder.fit_transform(df2["species"])

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    df2.iloc[:, :-1], df2.iloc[:, -1],
    test_size=0.2, random_state=42)

lr2 = LogisticRegression(multi_class="multinomial", max_iter=1000)
lr2.fit(X_train2, y_train2)
y_pred2 = lr2.predict(X_test2)

print(f"\nMulticlass LR accuracy: {accuracy_score(y_test2, y_pred2):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test2, y_pred2)}")
print(f"Recall (macro):    {recall_score(y_test2, y_pred2, average='macro'):.4f}")
print(f"Precision (macro): {precision_score(y_test2, y_pred2, average='macro'):.4f}")
