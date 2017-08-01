from numpy import sqrt, linspace, array, asarray, zeros_like, zeros, sum
from scipy.optimize import fsolve
from scipy.special import airy
import matplotlib.pyplot as plt

f = lambda t,y: array([y[1], t*y[0]])
    
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
    

# Kollokation + plot

N = 500
T = -40
y0 = array([ 0.35502805388781723926,
            -0.25881940379280679841])
t0 = 0  

A = array([[0.25,  0.25 - sqrt(3)/6.0],
              [0.25 + sqrt(3)/6.0,  0.25]])
b = array([0.5, 0.5])
c = sum(A, axis=1)
n_stages = A.shape[0]
        
t_CL4, y_CL4 = CL(t0, y0, N, T, A, b, c, n_stages)
exact = asarray(airy(t_CL4))[0,:]

plt.figure()
plt.title('Airy function Collocation', fontsize=26.)
plt.plot(t_CL4, y_CL4[:,0], 'o',label='CL4')
plt.plot(t_CL4, exact, label='exact')
plt.xlabel('t')
plt.ylabel('Ai(t)')
plt.legend(loc=3)
plt.tight_layout()
plt.show() 