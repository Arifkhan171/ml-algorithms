# Decision Tree Classifier
# Dataset: Wind/Weather data — predict whether to play or not

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load and encode dataset
df = pd.read_csv("../data/wind_data.csv")
print("Dataset:\n", df)

# Encode categorical columns
for col in ["outlook", "humidity", "windy", "play"]:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.iloc[:, :3]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2)

# ── Gini criterion ──────────────────────────────────────────────────────────
dt_gini = DecisionTreeClassifier(criterion="gini")
dt_gini.fit(X_train, y_train)
print(f"\nGini accuracy:    {dt_gini.score(X_test, y_test):.4f}")
print(f"Predictions:      {dt_gini.predict(X_test)}")

# ── Entropy criterion ───────────────────────────────────────────────────────
dt_entropy = DecisionTreeClassifier(criterion="entropy")
dt_entropy.fit(X_train, y_train)
print(f"Entropy accuracy: {dt_entropy.score(X_test, y_test):.4f}")

# ── Key hyperparameters explained ──────────────────────────────────────────
print("""
Key Decision Tree Hyperparameters:
  criterion      = 'gini' or 'entropy' (split quality measure)
  max_depth      = maximum tree depth (controls overfitting)
  min_samples_leaf = minimum samples required at leaf node
  max_features   = number of features considered at each split
  max_leaf_nodes = maximum number of leaf nodes
""")
