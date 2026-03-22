import numpy as np
from data import get_data

X, Y = get_data()   

X = (X - X.mean(axis=0)) / X.std(axis=0)
X = X[:, :-1]   

mean = Y.mean()
std = Y.std()
Y = (Y - mean) / std

# SGD
def SGD(X, Y, lr=0.01, epochs=50):
    n_samples, n_features = X.shape
    
    w = np.zeros(n_features)
    b = 0

    for _ in range(epochs):
        #  Shuffle data (important)
        perm = np.random.permutation(n_samples)
        X_shuffled = X[perm]
        Y_shuffled = Y[perm]

        for i in range(n_samples):
            xi = X_shuffled[i]
            yi = Y_shuffled[i]

            y_pred = np.dot(w, xi) + b
            error = yi - y_pred   

            dw = -2 * xi * error
            db = -2 * error

            w -= lr * dw
            b -= lr * db

    return w, b


# Mini Batch 
def mini_batch_GD(X, Y, lr=0.01, batch_size=32, epochs=50):
    n_samples, n_features = X.shape
    
    w = np.zeros(n_features)
    b = 0

    for _ in range(epochs):
        #  Shuffle data
        perm = np.random.permutation(n_samples)
        X_shuffled = X[perm]
        Y_shuffled = Y[perm]

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]

            y_pred = np.dot(X_batch, w) + b
            error = Y_batch - y_pred

            dw = (-2/len(X_batch)) * np.dot(X_batch.T, error)
            db = (-2/len(X_batch)) * np.sum(error)

            w -= lr * dw
            b -= lr * db

    return w, b


#  Train Models
w_sgd, b_sgd = SGD(X, Y)
w_mb, b_mb = mini_batch_GD(X, Y)

 
def predict(X, w, b, mean, std):
    Y_pred_norm = np.dot(X, w) + b
    return Y_pred_norm * std + mean


Y_pred_sgd = predict(X, w_sgd, b_sgd, mean, std)
Y_pred_mb = predict(X, w_mb, b_mb, mean, std)

Y_real = Y * std + mean   


#  Metrics 
def evaluate(Y_true, Y_pred):
    mse = np.mean((Y_true - Y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(Y_true - Y_pred))
    r2 = 1 - (np.sum((Y_true - Y_pred)**2) / np.sum((Y_true - np.mean(Y_true))**2))
    
    return mse, rmse, mae, r2


mse_sgd, rmse_sgd, mae_sgd, r2_sgd = evaluate(Y_real, Y_pred_sgd)
mse_mb, rmse_mb, mae_mb, r2_mb = evaluate(Y_real, Y_pred_mb)

print("\n SGD Results")
print("Weights:", w_sgd)
print("Bias:", b_sgd)
print("MSE:", mse_sgd)
print("RMSE:", rmse_sgd)
print("MAE:", mae_sgd)
print("R2:", r2_sgd)

print("\n Mini Batch Results")
print("Weights:", w_mb)
print("Bias:", b_mb)
print("MSE:", mse_mb)
print("RMSE:", rmse_mb)
print("MAE:", mae_mb)
print("R2:", r2_mb)