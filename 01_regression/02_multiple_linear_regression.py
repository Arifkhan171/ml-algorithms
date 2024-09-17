# Multiple Linear Regression
# Predicts a continuous value using multiple input features
# Uses synthetic data with 2 features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Generate synthetic dataset
X, y = make_regression(n_samples=100, n_features=2,
                        n_informative=2, noise=50, random_state=42)
df = pd.DataFrame({"feature1": X[:, 0], "feature2": X[:, 1], "target": y})

print("Dataset shape:", df.shape)
print(df.describe().round(2))

X_train, X_test, y_train, y_test = train_test_split(
    df.drop("target", axis=1), df["target"],
    test_size=0.2, random_state=42)

# Train model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Evaluation metrics
print(f"\nR2 Score:  {r2_score(y_test, y_pred):.4f}")
print(f"MSE:       {mean_squared_error(y_test, y_pred):.4f}")
print(f"MAE:       {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Intercept: {lr.intercept_:.4f}")
print(f"Coefs:     {lr.coef_}")

# Visualize actual vs predicted
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", label="Perfect Prediction")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Multiple Linear Regression — Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()
