# K-Means Clustering — Unsupervised Learning
# Groups data into K clusters by minimizing within-cluster sum of squares
# Uses Elbow Method to find optimal K

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# Generate synthetic clustered data
X, y = make_blobs(n_samples=1000, centers=6,
                   n_features=2, random_state=23)

plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", alpha=0.6)
plt.title("Original Data (6 clusters)")
plt.tight_layout()
plt.show()

X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Elbow Method — find optimal K ───────────────────────────────────────────
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    km.fit(X_train)
    wcss.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker="o")
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.title("Elbow Method — Find Optimal K")
plt.grid()
plt.tight_layout()
plt.show()

print("WCSS values:")
for k, w in enumerate(wcss, 1):
    print(f"  K={k}: {w:.2f}")

# ── Train with optimal K ─────────────────────────────────────────────────────
optimal_k = 6
kmeans = KMeans(n_clusters=optimal_k, init="k-means++",
                random_state=42, n_init=10)
labels = kmeans.fit_predict(X_train)

plt.figure(figsize=(6, 4))
plt.scatter(X_train[:, 0], X_train[:, 1], c=labels, cmap="tab10", alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200, c="black", marker="X", label="Centroids")
plt.title(f"K-Means Clustering (K={optimal_k})")
plt.legend()
plt.tight_layout()
plt.show()
