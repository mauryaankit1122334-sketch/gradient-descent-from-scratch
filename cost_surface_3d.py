import numpy as np
import matplotlib.pyplot as plt
from data import get_data

X, Y = get_data()


X = X[:, 0]


X = (X - X.mean()) / X.std()
Y = (Y - Y.mean()) / Y.std()

m_vals = np.linspace(-1, 3, 50)
b_vals = np.linspace(-1, 3, 50)

M, B = np.meshgrid(m_vals, b_vals)
Z = np.zeros_like(M)

for i in range(len(m_vals)):
    for j in range(len(b_vals)):
        y_pred = M[i, j] * X + B[i, j]
        Z[i, j] = (1/len(X)) * np.sum((Y - y_pred)**2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(M, B, Z, cmap='viridis')

ax.set_xlabel('m')
ax.set_ylabel('b')
ax.set_zlabel('Cost')

plt.title("3D Cost Surface")
plt.show()