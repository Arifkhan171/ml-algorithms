# Stacking Ensemble — Stack Multiple Models with a Meta-Learner
# Base models make predictions → meta-learner learns from their predictions
# Dataset: Heart disease classification

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier,
                               GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("../data/heart_stacking.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=8)

# Base estimators (level 0)
base_estimators = [
    ("rf",   RandomForestClassifier(n_estimators=10, random_state=42)),
    ("knn",  KNeighborsClassifier(n_neighbors=10)),
    ("gbdt", GradientBoostingClassifier(random_state=42))
]

# Meta-learner (level 1)
meta_learner = LogisticRegression(max_iter=1000)

# Stacking classifier
stacking = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=10
)

stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)

print("Base estimator scores:")
for name, estimator in base_estimators:
    estimator.fit(X_train, y_train)
    score = accuracy_score(y_test, estimator.predict(X_test))
    print(f"  {name}: {score:.4f}")

print(f"\nStacking accuracy: {accuracy_score(y_test, y_pred):.4f}")
cv_score = np.mean(cross_val_score(stacking, X, y, cv=5))
print(f"Stacking CV score: {cv_score:.4f}")
