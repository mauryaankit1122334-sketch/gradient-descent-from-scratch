from data import get_data
from model import gradient_descent
import matplotlib.pyplot as plt

X, Y = get_data()

X = (X - X.mean(axis=0)) / X.std(axis=0)

mean = Y.mean()
std = Y.std()
Y = (Y - mean) / std

_,_,cost_small = gradient_descent(X, Y, lr=0.01)
_,_,cost_large = gradient_descent(X, Y, lr=0.1)

plt.plot(cost_small, label="Small LR (0.01)")
plt.plot(cost_large, label="Large LR (0.1)")

plt.yscale('log')   

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend()
plt.title("Learning Rate Comparison")
plt.show()