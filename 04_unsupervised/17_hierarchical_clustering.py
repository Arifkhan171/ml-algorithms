# Hierarchical Clustering — Agglomerative Clustering
# Builds a tree of clusters (dendrogram) bottom-up
# Dataset: Shopping data — customer segmentation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("../data/shopping_data.csv")
print("Dataset shape:", df.shape)
print(df.head())

# Use Annual Income and Spending Score
X = df.iloc[:, 3:5].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Dendrogram — visualize cluster hierarchy ─────────────────────────────────
plt.figure(figsize=(12, 6))
plt.title("Dendrogram — Hierarchical Clustering")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
dend = shc.dendrogram(shc.linkage(X_scaled, method="ward"))
plt.tight_layout()
plt.show()

# ── Agglomerative Clustering ─────────────────────────────────────────────────
n_clusters = 5   # determined from dendrogram
cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
labels = cluster.fit_predict(X_scaled)

print(f"\nNumber of clusters: {n_clusters}")
print(f"Cluster labels: {np.unique(labels)}")
print(f"Cluster sizes:  {np.bincount(labels)}")

plt.figure(figsize=(8, 5))
colors = ["red", "blue", "green", "purple", "orange"]
for i in range(n_clusters):
    plt.scatter(X[labels == i, 0], X[labels == i, 1],
                c=colors[i], label=f"Cluster {i + 1}", alpha=0.7)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title("Hierarchical Clustering — Customer Segments")
plt.legend()
plt.tight_layout()
plt.show()
