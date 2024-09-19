# Gradient Descent — How Linear Regression Learns
# Shows step-by-step how the intercept converges to optimal value

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Small dataset to visualize gradient descent
X, y = make_regression(n_samples=4, n_features=1,
                        n_informative=1, noise=80, random_state=13)

# sklearn solution (the target we converge to)
lr = LinearRegression()
lr.fit(X, y)
y_ols = lr.predict(X)
print(f"OLS intercept: {lr.intercept_:.4f}, slope: {lr.coef_[0]:.4f}")

# ── Manual Gradient Descent on intercept (slope fixed) ─────────────────────
m  = 78.35   # fixed slope
b  = -100    # starting intercept
lr_rate = 0.01
epochs = 100

plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Data points", zorder=5)
plt.plot(X, y_ols, color="red", linewidth=2, label="OLS solution")

for i in range(epochs):
    loss_slope = -2 * np.sum(y - m * X.ravel() - b)
    b = b - (lr_rate * loss_slope)
    y_iter = m * X + b
    if i % 20 == 0:
        plt.plot(X, y_iter, alpha=0.4,
                 label=f"Epoch {i}: b={b:.2f}")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Gradient Descent — Intercept Converging to Optimal Value")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

print(f"\nFinal intercept after {epochs} epochs: {b:.4f}")

# ── Gradient Descent on larger dataset ─────────────────────────────────────
X2, y2 = make_regression(n_samples=100, n_features=1,
                           n_informative=1, noise=20, random_state=13)
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=2)

lr2 = LinearRegression()
lr2.fit(X2_train, y2_train)
y2_pred = lr2.predict(X2_test)
print(f"R2 Score on 100-sample dataset: {r2_score(y2_test, y2_pred):.4f}")
