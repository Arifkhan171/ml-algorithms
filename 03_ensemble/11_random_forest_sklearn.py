# Random Forest — Sklearn Implementation + GridSearchCV
# Dataset: Heart disease classification

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("../data/heart.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── Default Random Forest ───────────────────────────────────────────────────
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"Default RF accuracy:       {accuracy_score(y_test, y_pred):.4f}")
print(f"Cross-val accuracy (10cv): "
      f"{np.mean(cross_val_score(rf, X, y, cv=10)):.4f}")

# ── RF with max_samples ─────────────────────────────────────────────────────
rf2 = RandomForestClassifier(max_samples=0.75, random_state=42)
rf2.fit(X_train, y_train)
y_pred2 = rf2.predict(X_test)
print(f"\nRF (max_samples=0.75):     {accuracy_score(y_test, y_pred2):.4f}")

# ── GridSearchCV for best hyperparameters ───────────────────────────────────
param_grid = {
    "n_estimators": [20, 60, 100],
    "max_depth":    [2, 8, None],
    "max_features": [0.2, 0.6, 1.0],
    "max_samples":  [0.5, 0.75, 1.0]
}

print("\nRunning GridSearchCV (this may take a minute)...")
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5, verbose=0, n_jobs=-1)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score:   {grid_search.best_score_:.4f}")

# Feature importance
importances = pd.Series(rf.feature_importances_, index=X.columns)
print("\nTop 5 important features:")
print(importances.sort_values(ascending=False).head())
