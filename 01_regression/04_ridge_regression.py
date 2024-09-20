# Ridge Regression (L2 Regularization)
# Adds penalty to large coefficients to prevent overfitting
# Compared with standard Linear Regression and SGD Regressor

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
data = load_diabetes()
X, y = data.data, data.target
print(f"Dataset shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=45)

# ── 1. Standard Linear Regression ──────────────────────────────────────────
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(f"\nLinear Regression R2:  {r2_score(y_test, y_pred):.4f}")

# ── 2. Ridge Regression (L2 penalty) ───────────────────────────────────────
ridge = Ridge(alpha=0.01)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
print(f"Ridge Regression R2:   {r2_score(y_test, y_pred):.4f}")
print(f"Ridge intercept:       {ridge.intercept_:.4f}")

# ── 3. SGD Regressor with L2 penalty ───────────────────────────────────────
sgd = SGDRegressor(penalty="l2", max_iter=500,
                   eta0=0.1, learning_rate="constant", alpha=0.001)
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)
print(f"SGD Regressor R2:      {r2_score(y_test, y_pred):.4f}")

# ── 4. Ridge with sparse_cg solver ─────────────────────────────────────────
ridge2 = Ridge(alpha=0.01, max_iter=500, solver="sparse_cg")
ridge2.fit(X_train, y_train)
y_pred = ridge2.predict(X_test)
print(f"Ridge (sparse_cg) R2:  {r2_score(y_test, y_pred):.4f}")

# Alpha comparison
print("\nRidge R2 across different alpha values:")
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
    r = Ridge(alpha=alpha)
    r.fit(X_train, y_train)
    score = r2_score(y_test, r.predict(X_test))
    print(f"  alpha={alpha:6.3f} → R2={score:.4f}")
