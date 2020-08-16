import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn.utils import check_random_state

n_samples = 200
generator = check_random_state(100)
t = 1.5 * np.pi * (1 + 2 * generator.rand(1, n_samples))
y = 21 * generator.rand(1, n_samples)

t_args = np.argsort(t)[0]
y_args = np.argsort(y)[0]

X = np.array([np.array([t_, y_]) for t_ in np.squeeze(t) for y_ in np.squeeze(y)])
X = X.reshape((n_samples, n_samples, 2))
print(X.shape)
t_indx = t_args[np.arange(n_samples//4, 3*n_samples//4)]
y_indx = y_args[np.arange(n_samples//3, 2*n_samples//3)][:, np.newaxis]
print(y_indx.shape)
X[t_indx, y_indx, :] = np.zeros((y_indx.shape[0], t_indx.shape[0], 2))
print(X.shape)

t_ = X[:, 0, 0]
y_ = X[0, :, 1]
plt.scatter(t_, y_, c=t_)
# plt.scatter(t,y,c=t)
plt.show()