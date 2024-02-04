import numpy as np 
import matplotlib.pyplot as plt

# Generate some example data
X = np.linspace(0, 10, 100)  # Input features (in this case, a single feature)
y = 3* X + 2 + np.random.normal(size=X.size)  # Target variable

# Add a column of ones to X for the bias term (intercept)
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Solve the normal equation to find the coefficients (theta)
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Predict values using the calculated coefficients
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]y_pred = X_test_b.dot(theta)

# Plot the original data points and the regression line
plt.scatter(X, y, label="Training Data", color="blue")
plt.plot(X_test, y_pred, label="Regression Line", color="red")
plt.xlabel("X")plt.ylabel("y")plt.legend()plt.show()

# Print the calculated coefficients (intercept and slope)
print("Intercept:", theta[0])print("Coefficient:", theta[1])
