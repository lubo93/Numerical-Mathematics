from numpy import *
from matplotlib.pyplot import *
from scipy.linalg import expm, solve
import matplotlib.pyplot as plt

# Exponentielles Euler-Verfahren
def expEV(n_steps, t0, tf, y0, f, Df):
    
    t_arr, h = linspace(t0, tf, n_steps+1, retstep=True)
    y = []
    y.append(y0)
    for i in range(n_steps):
        DF = Df(y[-1])
        y.append(y[-1] + dot(expm(h*DF) - eye(*DF.shape), solve(DF, f(y[-1]))))

    return t_arr, asarray(y)

# Jacobimatrix
Df = lambda y: array([[-2*y[0]/y[1], (y[0]/y[1])**2 + log(y[1]) +1],
                      [-1.0, 0.0]])
# Funktion
f = lambda y: array([-y[0]**2/y[1] + y[1]*log(y[1]), -y[0]])

# Analytische Loesung
sol = lambda t: array([-cos(t)*exp(sin(t)), exp(sin(t))]).T

# Anfangswert
y0 = array([-1, 1])

t0 = 0
tf = 6
nsteps = 40
ts, y = expEV(nsteps, t0, tf, y0, f, Df)

t_ex = linspace(t0, tf, 1000)
y_ex = sol(t_ex)

plt.figure()
plt.plot(t_ex, y_ex[:,0], 'r', label=r'Analytical $y_1$')
plt.plot(t_ex, y_ex[:,1], 'g', label=r'Analytical $y_2$')
plt.plot(ts, y[:,0],'r-x', label=r'Exp. Euler $y_1$')
plt.plot(ts, y[:,1],'g-x', label=r'Exp. Euler $y_2$')
plt.legend(loc='best')
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.tight_layout()
plt.show()

