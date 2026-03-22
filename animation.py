import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from data import get_data

X, Y = get_data()  


mean1 = Y.mean()
std1 = Y.std()
X = (X - X.mean(axis=0)) / X.std(axis=0)
Y = (Y - mean1) / std1

n_samples, n_features = X.shape


w = np.zeros(n_features)
b = 0   
lr = 0.01

costs = []

fig, ax = plt.subplots()
line, = ax.plot([], [])

ax.set_title("Cost vs Iterations")
ax.set_xlabel("Iterations")
ax.set_ylabel("Cost")

ax.set_yscale('log')  
def compute_cost():
    y_pred = (np.dot(X, w) + b)
    return (1/n_samples) * np.sum((Y - y_pred)**2)

def update(frame):
    global w, b

    y_pred = np.dot(X, w) + b   

    dw = (-2/n_samples) * np.dot(X.T, error)
    db = (-2/n_samples) * np.sum(error)

    w -= lr * dw
    b -= lr * db

    cost = compute_cost()
    costs.append(cost)

    line.set_data(range(len(costs)), costs)

    ax.relim()
    ax.autoscale_view()

    return line,
ani = FuncAnimation(fig, update, frames=50, interval=200)
plt.show()