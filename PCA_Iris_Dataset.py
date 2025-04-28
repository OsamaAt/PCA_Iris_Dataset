# Step 1: Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Step 2: Load Data
data = load_iris()
X, y = data.data, data.target
target_names = data.target_names

# Step 3: Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Step 4: Plot PCA Result
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title("PCA on Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Class Label")
plt.grid(True)

# Optional: Add Class Names Legend
handles, labels = scatter.legend_elements()
plt.legend(handles, target_names, title="Classes")

plt.show()

# Step 5: Explained Variance
explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance Ratio per Component: {explained_variance}")
print(f"Total Explained Variance: {np.sum(explained_variance):.2f}")

# Step 6: Plot Explained Variance (Optional Bonus)
plt.figure(figsize=(6,4))
plt.bar(["PC1", "PC2"], explained_variance, color='skyblue')
plt.title("Explained Variance by Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.show()
