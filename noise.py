import numpy as np
import matplotlib.pyplot as plt
from model import gradient_descent

np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)   
Y = 3*X.flatten() + 5 + np.random.randn(50)*2

X = (X - X.mean(axis=0)) / X.std(axis=0)
mean = Y.mean()
std = Y.std()
Y = (Y - mean) / std

w, b, costs = gradient_descent(X, Y)

Y_pred = (np.dot(X, w) + b) * std + mean

plt.scatter(X, Y*std + mean, label="Data")
plt.plot(X, Y_pred, color='red', label="Fitted Line")

plt.title("Noisy Data Fit")
plt.legend()
plt.show()