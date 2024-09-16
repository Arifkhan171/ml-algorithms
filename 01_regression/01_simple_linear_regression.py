# Simple Linear Regression
# Predicts a continuous value using one input feature
# Dataset: Placement data (CGPA vs Package)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("../data/placement.csv")
print("Dataset shape:", df.shape)
print(df.head())

X = df.iloc[:, 0:1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluate
y_pred = lr.predict(X_test)
print(f"\nR2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"Coefficient (slope): {lr.coef_[0]:.4f}")
print(f"Intercept: {lr.intercept_:.4f}")

# Visualize
plt.scatter(df.iloc[:, 0], df.iloc[:, -1], label="Actual Data")
plt.plot(X_train, lr.predict(X_train), color="red", label="Regression Line")
plt.xlabel("CGPA")
plt.ylabel("Package (LPA)")
plt.title("Simple Linear Regression — CGPA vs Package")
plt.legend()
plt.tight_layout()
plt.show()

# ── Custom Implementation from Scratch ──────────────────────────────────────
class SimpleLinearRegression:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X_train, y_train):
        num = sum((X_train[i] - X_train.mean()) *
                  (y_train[i] - y_train.mean())
                  for i in range(len(X_train)))
        den = sum((X_train[i] - X_train.mean()) ** 2
                  for i in range(len(X_train)))
        self.m = num / den
        self.b = y_train.mean() - self.m * X_train.mean()

    def predict(self, X_test):
        return self.m * X_test + self.b

X_vals = df.iloc[:, 0].values
y_vals = df.iloc[:, 1].values
X_tr, X_te, y_tr, y_te = train_test_split(
    X_vals, y_vals, test_size=0.2, random_state=2)

custom_lr = SimpleLinearRegression()
custom_lr.fit(X_tr, y_tr)
print(f"\nCustom LR — slope: {custom_lr.m:.4f}, intercept: {custom_lr.b:.4f}")
