import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
np.random.seed(3)
# Generating the curve
x = np.linspace(0, 1, 400)
y = np.sin(2*np.pi * x) # Generating and plotting the blue circles
x_points = np.linspace(0, 1, 10)
y_points = np.sin(2*np.pi * x_points) + np.random.normal(0,0.2,10)
plt.scatter(x_points, y_points, color='blue', edgecolors='black')
polynomial_features = PolynomialFeatures(degree=3)
x_poly_train = polynomial_features.fit_transform(x_points.reshape(-1, 1))
x_poly_test = polynomial_features.transform(x.reshape(-1, 1))
# Training the model
model = LinearRegression().fit(x_poly_train, y_points) # Predicting
y_poly_pred_test = model.predict(x_poly_test) # Setting axis labels
plt.xlabel('x')
plt.ylabel('t') # Displaying the grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.axhline(y=0, color='k', linewidth=0.8)
plt.axvline(x=0, color='k', linewidth=0.8)
plt.plot(x, y_poly_pred_test, c = 'r')
# Adjusting the y-axis limit
plt.ylim(-1.5, 1.5)
plt.legend()
# Displaying the plot
plt.show()
