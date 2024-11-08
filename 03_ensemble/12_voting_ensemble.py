# Voting Ensemble — Combine Multiple Classifiers
# Hard voting: majority vote | Soft voting: average probabilities
# Dataset: Iris and synthetic classification

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("../data/iris.csv")
df["species"] = LabelEncoder().fit_transform(df["species"])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Base classifiers
clf1 = LogisticRegression(max_iter=1000)
clf2 = RandomForestClassifier(random_state=42)
clf3 = DecisionTreeClassifier(random_state=42)

estimators = [("lr", clf1), ("rf", clf2), ("dt", clf3)]

# Individual scores
print("Individual classifier scores (10-fold CV):")
for name, clf in estimators:
    score = np.mean(cross_val_score(clf, X, y, cv=10, scoring="accuracy"))
    print(f"  {name}: {score:.4f}")

# Soft voting (uses predict_proba — better than hard voting)
vc_soft = VotingClassifier(estimators=estimators, voting="soft")
soft_score = np.mean(cross_val_score(vc_soft, X, y, cv=10, scoring="accuracy"))
print(f"\nSoft Voting accuracy: {soft_score:.4f}")

# Hard voting
vc_hard = VotingClassifier(estimators=estimators, voting="hard")
hard_score = np.mean(cross_val_score(vc_hard, X, y, cv=10, scoring="accuracy"))
print(f"Hard Voting accuracy: {hard_score:.4f}")

# Weighted voting — find best weights
print("\nSearching best weights for soft voting:")
best_score, best_weights = 0, None
for i in range(1, 4):
    for j in range(1, 4):
        for k in range(1, 4):
            vc = VotingClassifier(estimators=estimators,
                                   voting="soft", weights=[i, j, k])
            score = np.mean(cross_val_score(vc, X, y, cv=10))
            if score > best_score:
                best_score, best_weights = score, [i, j, k]

print(f"Best weights: {best_weights}, score: {best_score:.4f}")
