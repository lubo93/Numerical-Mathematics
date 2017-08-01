from numpy import array
from scipy.linalg import norm, solve

F = lambda x: array([x[0]**2-x[1]**4, x[0]-x[1]**3])
DF = lambda x: array([[2*x[0],-4*x[1]**3],[1,-3*x[1]**2]])
x = array([0.7, 0.7])
tol = 10**(-10)
s = solve(DF(x),F(x))
x -= s

while norm(s) > tol*norm(x):
    s = solve(DF(x),F(x))
    x -= s

print x
