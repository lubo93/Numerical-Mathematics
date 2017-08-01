from scipy.optimize import newton

f = lambda x: x**5 + x - 1

x0 = 1
print newton(f, x0)