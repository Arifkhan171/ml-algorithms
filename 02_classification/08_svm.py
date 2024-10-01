# Support Vector Machine (SVM)
# Finds the optimal hyperplane that maximizes margin between classes
# Dataset: Diabetes dataset

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("../data/diabetes.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── SVM with different kernels ──────────────────────────────────────────────
kernels = ["linear", "rbf", "poly", "sigmoid"]
print("SVM accuracy with different kernels:")
for kernel in kernels:
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    print(f"  kernel={kernel:8s} → accuracy={accuracy_score(y_test, y_pred):.4f}")

# ── Best SVM configuration ──────────────────────────────────────────────────
svm_best = SVC(kernel="linear", gamma=0.5, C=5.0)
# gamma = distance of influence of support vectors
# C     = error tolerance (higher C = fewer misclassifications allowed)
svm_best.fit(X_train_scaled, y_train)
y_pred_best = svm_best.predict(X_test_scaled)
print(f"\nBest SVM (linear, gamma=0.5, C=5.0):")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"  Support vectors: {svm_best.n_support_}")
