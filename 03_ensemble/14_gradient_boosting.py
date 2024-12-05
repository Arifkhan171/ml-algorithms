# Gradient Boosting — Sequential Ensemble Learning
# Each tree corrects the residual errors of the previous tree
# Builds trees sequentially unlike Random Forest (parallel)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

np.random.seed(42)

# ── Synthetic dataset ────────────────────────────────────────────────────────
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)

# ── Step-by-step gradient boosting ──────────────────────────────────────────
df = pd.DataFrame({"X": X.ravel(), "y": y})
df["pred1"] = df["y"].mean()           # initial prediction = mean
df["res1"]  = df["y"] - df["pred1"]   # residuals

# Tree 1 fits residuals
tree1 = DecisionTreeRegressor(max_leaf_nodes=8, random_state=42)
tree1.fit(df[["X"]], df["res1"])
df["pred2"] = df["pred1"] + tree1.predict(df[["X"]])
df["res2"]  = df["y"] - df["pred2"]

# Tree 2 fits residuals of tree 1
tree2 = DecisionTreeRegressor(max_leaf_nodes=8, random_state=42)
tree2.fit(df[["X"]], df["res2"])
df["pred3"] = df["pred2"] + tree2.predict(df[["X"]])

print("Step-by-step R2 scores:")
print(f"  After mean (step 0): {r2_score(y, df['pred1']):.4f}")
print(f"  After tree 1:        {r2_score(y, df['pred2']):.4f}")
print(f"  After tree 2:        {r2_score(y, df['pred3']):.4f}")

# ── Sklearn GradientBoostingRegressor ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

for n_trees in [10, 50, 100, 200]:
    gbr = GradientBoostingRegressor(n_estimators=n_trees,
                                     learning_rate=0.1,
                                     max_depth=3, random_state=42)
    gbr.fit(X_train, y_train)
    score = r2_score(y_test, gbr.predict(X_test))
    print(f"  n_estimators={n_trees:3d} → R2={score:.4f}")
