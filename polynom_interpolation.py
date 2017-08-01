import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

f = lambda x: 1.0/(1 + x**2)

def build_matrix(degree, n_samples):
    exponent = np.arange(0, degree+1)
    x = np.linspace(-5, 5, n_samples)

    A = x[:,np.newaxis] ** exponent[np.newaxis,:]
    b = f(x)
    return A, b


plt.figure()
for i in range(2, 11):
    A, b = build_matrix(i, 40)
    c = scipy.linalg.lstsq(A, b)[0]
    # from highest to lowest degree
    y = np.polyval(c[::-1], x)
    plt.plot(x, y, label=r"$n = %d$" % i)

plt.plot(x, f(x), "k", linewidth=2, label=r"$\frac{1}{1+x^2}$")
plt.xlim(-5, 5)
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')
plt.legend()
plt.tight_layout()
plt.show()