f = lambda x: x**5 + x - 1
fp = lambda x: 5*x**4 + 1

def newton_method(x0, tol):
    delta = 1.0*f(x0)/fp(x0)
    while delta > tol:
        x0 -= delta
        delta = f(x0)/fp(x0)
    return x0

print newton_method(1, 10**-4)