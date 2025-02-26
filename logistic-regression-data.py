import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data for logistic regression
num_samples = 200

# Generate two classes of data
x1_class0 = np.random.normal(loc=2, scale=1, size=(num_samples // 2, 2))
x1_class1 = np.random.normal(loc=5, scale=1, size=(num_samples // 2, 2))

# Create labels
y_class0 = np.zeros((num_samples // 2, 1))
y_class1 = np.ones((num_samples // 2, 1))

# Combine into dataset
X = np.vstack((x1_class0, x1_class1))
y = np.vstack((y_class0, y_class1)).flatten()

# Convert to DataFrame (Optional: if you want to save or inspect it)
df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df["Label"] = y

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Class 0", alpha=0.6, edgecolors='k')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Class 1", alpha=0.6, edgecolors='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("Synthetic Data for Logistic Regression")
plt.grid(True)
plt.show()
