import numpy as np
def compute_cost(X, Y, m, b):
    n = len(X)
    y_pred = m * X + b
    return (1/n) * np.sum((Y - y_pred)**2)


def gradient_descent(X, Y, lr=0.01, epochs=100):
    n_samples, n_features = X.shape
    
    w = np.zeros(n_features)
    b = 0
    costs = []

    for _ in range(epochs):
        y_pred = np.dot(X, w) + b 

        error = Y - y_pred

        
        dw = (-2/n_samples) * np.dot(X.T, error)
        db = (-2/n_samples) * np.sum(error)

        w -= lr * dw
        b -= lr * db

        cost = (1/n_samples) * np.sum(error**2)
        costs.append(cost)

    return w, b, costs
