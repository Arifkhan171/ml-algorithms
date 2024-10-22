# Random Forest — Manual Implementation
# Shows how Random Forest works: row sampling + feature sampling + aggregation
# Dataset: Synthetic classification data

import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate dataset
X, y = make_classification(n_features=5, n_redundant=0,
                            n_informative=5, n_clusters_per_class=1,
                            random_state=42)
df = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4", "col5"])
df["target"] = y

# ── Sampling functions ──────────────────────────────────────────────────────
def sample_rows(df, percent):
    """Row sampling with replacement (bootstrap)"""
    return df.sample(int(percent * len(df)), replace=True)

def sample_features(df, percent):
    """Column sampling without replacement"""
    cols = random.sample(df.columns.tolist()[:-1],
                          int(percent * (len(df.columns) - 1)))
    new_df = df[cols].copy()
    new_df["target"] = df["target"]
    return new_df

def combined_sample(df, row_pct, col_pct):
    """Both row and column sampling"""
    return sample_features(sample_rows(df, row_pct), col_pct)

# ── 3-tree manual random forest ─────────────────────────────────────────────
random.seed(42)
trees = []
for i in range(3):
    sample = combined_sample(df, row_pct=0.5, col_pct=0.8)
    X_s = sample.iloc[:, :-1]
    y_s = sample["target"]
    tree = DecisionTreeClassifier(random_state=i)
    tree.fit(X_s, y_s)
    trees.append((tree, list(X_s.columns)))
    print(f"Tree {i+1} trained on {X_s.shape[0]} samples, "
          f"features: {list(X_s.columns)}")

# Majority vote prediction
X_test_sample = df.iloc[:10, :-1]
predictions = np.array([
    tree.predict(X_test_sample[cols])
    for tree, cols in trees
])
majority_vote = np.apply_along_axis(
    lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
print(f"\nMajority vote predictions: {majority_vote}")
print(f"Actual labels:             {df.iloc[:10, -1].values}")
