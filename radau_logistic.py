from numpy import exp, linspace, array, asarray, zeros_like, zeros, sum
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

f = lambda t,y: array([5*y*(1-y)])
logistic = lambda x: 1./(1+9*exp(-5.*x))
 
def CL(t0, y0, N, T, A, b, c, n_stages):
    y_CL = []
    t_CL, h = linspace(t0, T, N+1, retstep=True)
    y_CL.append(y0)
    for i in range(N):
        y_CL.append(CL_step(y_CL[-1], t_CL[i], h, A, b, c, n_stages))
    
    return t_CL, asarray(y_CL)
        
def CL_step(y_old, t, h, A, b, c, n_stages):

    s = n_stages

    def F(dydt):
        dydt_star = zeros_like(dydt)
        for i in range(s):
            y_star = y_old + h*sum(A[i,:]*dydt[:,:], axis=1)
            dydt_star[:,i] =  f(t + c[i]*h, y_star)

        return dydt - dydt_star


    initial_guess = zeros((y_old.size, s))
    for i in range(s):
        initial_guess[:,i] = f(t, y_old)

    k = nicer_fsolve(F, initial_guess)
    
    y_new = y_old + h*sum(b*k, axis=1)
    return y_new


def nicer_fsolve(F, initial_guess):
    shape = initial_guess.shape
    initial_guess = initial_guess.reshape((-1,))
    result = fsolve(lambda x: F(x.reshape(shape)).reshape((-1)), initial_guess)

    return result.reshape(shape)
    

# Radau + plot

N = 100
T = 3.0
y0 = array([ 0.1])
t0 = 0  

A = array([[5./12  , -1./12], 
              [3./4, 1./4]])
b = array([3./4, 1./4])
c = sum(A, axis=1)
n_stages = A.shape[0]
        
t_RA, y_RA = CL(t0, y0, N, T, A, b, c, n_stages)

plt.figure()
plt.title(r'Logistic Equation Radau', fontsize=26.)
plt.plot(t_RA, y_RA[:,0], 'o', label=r'Radau')
plt.plot(t_RA, logistic(t_RA), label=r'exact')
plt.xlabel(r'$t$')
plt.ylabel(r'$y(t)$')
plt.legend(loc=4)
plt.tight_layout()
plt.show() 