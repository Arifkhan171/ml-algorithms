# Bagging Classifier — Bootstrap Aggregation
# Trains multiple models on bootstrap samples and aggregates predictions
# Types: Bagging, Pasting, Random Subspaces, Random Patches

import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Generate dataset
X, y = make_classification(n_samples=10000, n_features=10,
                             n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Base Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print(f"Single Decision Tree: {accuracy_score(y_test, dt.predict(X_test)):.4f}")

# ── 1. Bagging (bootstrap=True) ─────────────────────────────────────────────
bg = BaggingClassifier(estimator=DecisionTreeClassifier(),
                        n_estimators=500, max_samples=0.25,
                        bootstrap=True, random_state=42)
bg.fit(X_train, y_train)
print(f"Bagging (500 trees):  {accuracy_score(y_test, bg.predict(X_test)):.4f}")

# ── 2. Pasting (bootstrap=False) ────────────────────────────────────────────
bg_paste = BaggingClassifier(estimator=DecisionTreeClassifier(),
                               n_estimators=50, max_samples=0.25,
                               bootstrap=False, n_jobs=-1, random_state=42)
bg_paste.fit(X_train, y_train)
print(f"Pasting (50 trees):   {accuracy_score(y_test, bg_paste.predict(X_test)):.4f}")

# ── 3. Random Subspaces (column sampling only) ──────────────────────────────
bg_sub = BaggingClassifier(estimator=DecisionTreeClassifier(),
                            n_estimators=50, max_samples=1.0,
                            max_features=0.5, bootstrap_features=True,
                            n_jobs=-1, random_state=42)
bg_sub.fit(X_train, y_train)
print(f"Random Subspaces:     {accuracy_score(y_test, bg_sub.predict(X_test)):.4f}")

# ── 4. GridSearchCV for best parameters ─────────────────────────────────────
params = {"n_estimators": [50, 100], "max_samples": [0.25, 0.5],
          "bootstrap": [True, False], "max_features": [0.5, 1.0]}
gs = GridSearchCV(BaggingClassifier(random_state=42),
                   params, cv=3, n_jobs=-1)
gs.fit(X_train, y_train)
print(f"\nGridSearch best score:  {gs.best_score_:.4f}")
print(f"GridSearch best params: {gs.best_params_}")
