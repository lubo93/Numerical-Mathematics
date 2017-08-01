import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**5 + x - 1
fp = lambda x: 5*x**4 + 1

zero = 0.75487770177
def newton_method(x0, tol):
    delta = 1.0*f(x0)/fp(x0)
    err = []
    err.append(abs(x0-zero))
    while delta > tol:
        x0 -= delta
        delta = f(x0)/fp(x0)
        err.append(abs(x0-zero))
    return x0, err

x0, err = newton_method(1, 10**-4)

iterations =  np.linspace(1,len(err),len(err))
print np.polyfit(np.log(iterations[0:3]), np.log(err[0:3]), 1)

plt.figure()
plt.plot(err)
plt.plot(iterations, 0.1*iterations**-2)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel('Iterationen')
plt.ylabel('Fehler')
plt.tight_layout()
plt.show()