# ML Algorithms

A complete, structured repository covering all essential Machine Learning algorithms — from regression to ensemble methods and unsupervised learning — implemented with real datasets and proper evaluation metrics.

---

## 📌 About This Repository

This repository covers 17 ML algorithms organized by category. Each script is self-contained, uses real datasets, and includes evaluation metrics with results. Built as part of a hands-on Machine Learning journey over 2+ years.

---

## 📁 Repository Structure

```
ml-algorithms/
│
├── 01_regression/
│   ├── 01_simple_linear_regression.py    # OLS + custom implementation
│   ├── 02_multiple_linear_regression.py  # R2, MSE, MAE evaluation
│   ├── 03_gradient_descent.py            # Manual gradient descent visualization
│   └── 04_ridge_regression.py           # L2 regularization, SGD, alpha tuning
│
├── 02_classification/
│   ├── 05_logistic_regression.py         # Binary + multiclass classification
│   ├── 06_decision_tree.py               # Gini vs Entropy criterion
│   ├── 07_classification_metrics.py      # Accuracy, Precision, Recall, F1, CM
│   ├── 08_svm.py                         # Multiple kernels + feature scaling
│   └── 09_naive_bayes.py                 # GaussianNB + class probabilities
│
├── 03_ensemble/
│   ├── 10_random_forest_manual.py        # RF from scratch (sampling + voting)
│   ├── 11_random_forest_sklearn.py       # RF + GridSearchCV + feature importance
│   ├── 12_voting_ensemble.py             # Hard/soft voting + weight tuning
│   ├── 13_bagging.py                     # Bagging, Pasting, Random Subspaces
│   ├── 14_gradient_boosting.py           # Manual + sklearn GBM + learning rate
│   └── 15_stacking.py                    # Base models + meta-learner
│
├── 04_unsupervised/
│   ├── 16_kmeans_clustering.py           # Elbow method + centroid visualization
│   └── 17_hierarchical_clustering.py    # Dendrogram + customer segmentation
│
├── data/
│   ├── placement.csv                     # Simple linear regression
│   ├── diabetes.csv                      # Classification (SVM, LR, metrics)
│   ├── heart.csv                         # Logistic regression, Random Forest
│   ├── iris.csv                          # Multiclass classification, Naive Bayes
│   ├── wind_data.csv                     # Decision tree
│   ├── heart_stacking.csv                # Stacking ensemble
│   └── shopping_data.csv                 # Hierarchical clustering
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧠 Algorithms Covered

| # | Algorithm | Type | Dataset | Key Metric |
|---|-----------|------|---------|-----------|
| 01 | Simple Linear Regression | Regression | Placement | R2 Score |
| 02 | Multiple Linear Regression | Regression | Synthetic | R2, MSE, MAE |
| 03 | Gradient Descent | Optimization | Synthetic | Visual convergence |
| 04 | Ridge Regression | Regression | Diabetes (sklearn) | R2 Score |
| 05 | Logistic Regression | Classification | Heart, Iris | Accuracy, Recall |
| 06 | Decision Tree | Classification | Wind data | Accuracy |
| 07 | Classification Metrics | Evaluation | Diabetes | F1, Precision, Recall |
| 08 | SVM | Classification | Diabetes | Accuracy |
| 09 | Naive Bayes | Classification | Iris | Accuracy |
| 10 | Random Forest (Manual) | Ensemble | Synthetic | Accuracy |
| 11 | Random Forest (Sklearn) | Ensemble | Heart | Accuracy, CV |
| 12 | Voting Ensemble | Ensemble | Iris | CV Score |
| 13 | Bagging | Ensemble | Synthetic | Accuracy |
| 14 | Gradient Boosting | Ensemble | Synthetic | R2 Score |
| 15 | Stacking | Ensemble | Heart | Accuracy |
| 16 | K-Means Clustering | Unsupervised | Synthetic | WCSS (Elbow) |
| 17 | Hierarchical Clustering | Unsupervised | Shopping | Dendrogram |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)

---

## 🚀 How to Run

**1. Clone the repository:**
```bash
git clone https://github.com/Arifkhan171/ml-algorithms.git
cd ml-algorithms
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run any algorithm:**
```bash
python 01_regression/01_simple_linear_regression.py
python 03_ensemble/11_random_forest_sklearn.py
python 04_unsupervised/16_kmeans_clustering.py
```

---

## 👤 Author

**Arif Khan**
Final Year CS Student | University of Loralai | ML • Deep Learning • Agentic AI

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/arif-khan-71a711376)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github)](https://github.com/Arifkhan171)

---

> ⭐ Part of a complete Data Science & AI portfolio — from ML Algorithms to Agentic AI systems.
