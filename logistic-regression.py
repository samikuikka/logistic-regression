import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
num_samples = 200

# ---------------------------
# Data Generation
# --------------------------- 
x1_class0 = np.random.normal(loc=2, scale=1, size=(num_samples // 2, 2))
x1_class1 = np.random.normal(loc=5, scale=1, size=(num_samples // 2, 2))

y_class0 = np.zeros((num_samples // 2, 1))
y_class1 = np.ones((num_samples // 2, 1))

X = np.vstack((x1_class0, x1_class1))
y = np.vstack((y_class0, y_class1)).flatten()

N = len(X)
train_ratio = 0.8
train_size = int(N * train_ratio)

indices = np.random.permutation(N)
train_idx, test_idx = indices[:train_size], indices[train_size:]

# ---------------------------
# Data Preprocessing
# ---------------------------
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

mean_X_train = np.mean(X_train[:, 1:], axis=0)  
std_X_train = np.std(X_train[:, 1:], axis=0)   

X_train[:, 1:] = (X_train[:, 1:] - mean_X_train) / std_X_train
X_test[:, 1:] = (X_test[:, 1:] - mean_X_train) / std_X_train


# ---------------------------
# Gradient Ascent Setup
# ---------------------------
learning_rate = 0.01
num_iterations = 1000
weights = np.zeros(X_train.shape[1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for i in range(num_iterations):
    preds = sigmoid(np.dot(X_train, weights))
    gradient = np.dot((y_train - preds), X_train) 
    weights += learning_rate * gradient

# ---------------------------
# Evaluation
# ---------------------------
test_preds = sigmoid(np.dot(X_test, weights)) 
predicted_labels = (test_preds >= 0.5).astype(int)

accuracy = np.mean(predicted_labels == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# ---------------------------
# Visualization
# ---------------------------
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]  # shape (40000, 2)
grid_points_scaled = (grid_points - mean_X_train) / std_X_train  # shape (40000,2)
grid_points_scaled = np.hstack([np.ones((grid_points_scaled.shape[0], 1)), grid_points_scaled])  # shape (40000,3)

Z = sigmoid(np.dot(grid_points_scaled, weights))
Z = Z.reshape(xx.shape)  

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.7)
plt.colorbar(label='Probability of Class 1')

plt.scatter(X_test[:, 1] * std_X_train[0] + mean_X_train[0], 
            X_test[:, 2] * std_X_train[1] + mean_X_train[1], 
            c=y_test, edgecolors='k', marker='o', label='Test Data')

plt.scatter((X_test[predicted_labels == 1, 1] * std_X_train[0] + mean_X_train[0]),
            (X_test[predicted_labels == 1, 2] * std_X_train[1] + mean_X_train[1]),
            color='yellow', s=50, marker='x', label='Predicted Class 1')

plt.title("Logistic Regression Decision Boundary (Fixed)")
plt.xlabel("Feature 1 (Original Scale)")
plt.ylabel("Feature 2 (Original Scale)")
plt.legend()
plt.show()
