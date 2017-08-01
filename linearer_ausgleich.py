import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.9, 2.1])

# y = Ap mit A = [[x 1]] und p = [[m], [n]]
A = np.vstack([x, np.ones(len(x))]).T
m, n = np.linalg.lstsq(A, y)[0]
print m, n

# oder polyfit
m_poly, n_poly = np.polyfit(x, y, 1)
print m_poly, n_poly

plt.figure()
plt.plot(x, y, 'o', label='Daten', markersize=10)
plt.plot(x, m*x + n, 'r', label='Linearer Ausgleich')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc=4)
plt.tight_layout()
plt.show()
