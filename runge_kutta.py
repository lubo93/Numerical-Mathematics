from numpy import cos, array, asarray, linspace, polyfit, log10
from numpy.linalg import norm
import matplotlib.pyplot as plt

f1 = lambda t,y: array([y[1], -y[0]])
f2 = lambda t,y: array([y[1], -10*y[1]-10025*y[0]+10025])

# Die England Formel Runge Kutta Methode
def runge_kutta_england(f, t0, tend, y0, N):
    t_england, h = linspace(t0, tend, N+1, retstep=True)
    y_england = []
    y_england.append(y0)
   
    for i in range(N):
        k1 = f(t_england[i], y_england[-1])
        k2 = f(t_england[i]+0.5*h, y_england[-1]+0.5*h*k1)
        k3 = f(t_england[i]+0.5*h, y_england[-1]+0.25*h*k1+0.25*h*k2)
        k4 = f(t_england[i]+h, y_england[-1]-h*k2+2*h*k3)
        
        y_england.append(y_england[-1]+h*(1./6*k1+2./3*k3+1./6*k4))
    
    return t_england, asarray(y_england)
    


t0 = 0    
tend = 10
y0 = array([1,0])     
N = 100  
t_england_f1, y_england_f1 = runge_kutta_england(f1, t0, tend, y0, N)

# Plotte die Loesung fuer Gleichung f1, um den Code zu testen
plt.figure()
plt.plot(t_england_f1, y_england_f1[:,0], 'o',label=r'England scheme')
plt.plot(t_england_f1, cos(t_england_f1), label=r'exact')
plt.xlabel(r'$t$')
plt.ylabel(r'$y(t)$')
plt.legend(loc=5)
plt.tight_layout()
plt.show()

# Nun bestimmen wir die Konvergenzordnung
err = []
N_arr = array(range(6,100))

for N in N_arr:
    t_england_f1, y_england_f1 = runge_kutta_england(f1, t0, tend, y0, N)
    err.append(norm(cos(t_england_f1)-y_england_f1[:,0]))

m, n = polyfit(log10(1./N_arr), log10(err), 1)
print "Konvergenzordnung: ", m

plt.figure()
plt.plot(1./N_arr, (1./N_arr)**m*10**n, 'k-', alpha=0.4, linewidth=8.)
plt.plot(1./N_arr, err, 'o')
plt.text(0.5*10**(-1),10**(-3),r'Error$\sim$(Step size)$^{3.5}$', fontsize=26)
plt.xlabel(r'Step size')
plt.ylabel(r'Error')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc=5)
plt.tight_layout()
plt.show()

# Loesung fuer Gleichung f2
t0 = 0    
tend = 1
y0 = array([2,95])     
N = 30
t_england_f2, y_england_f2 = runge_kutta_england(f2, t0, tend, y0, N)

# Plotte die Loesung fuer Gleichung f1, um den Code zu testen
plt.figure()
plt.plot(t_england_f2, y_england_f2[:,0], '-o',label=r'England scheme')
plt.xlabel(r'$t$')
plt.ylabel(r'$y(t)$')
plt.legend(loc=1)
plt.tight_layout()
plt.show()

# Stabilitaetsfunktion
S = lambda z: 1./54*(3+z)*(6+z)*(3+2*z)

# z ist eine komplexe Zahl mit z=x+iy
# wir schauen was im Bereich -8,..,8 fuer x und
# -3...3 fuer y passiert und wann |S(z)|<1

plt.figure()
for x in linspace(-8,0,100):
    for y in linspace(-3,3,100):
        z = x + 1j*y
        if abs(S(z)) < 1:
            plt.plot(x,y,'LightBlue', marker='s', markersize=10, markeredgecolor='None', zorder=0)

plt.hlines(0,-8,1,color='k',linewidth=2., zorder=1)   
plt.vlines(0,-3,3,color='k',linewidth=2., zorder=1)  
plt.xlim([-8,1])   
plt.xlabel(r'$\Re(z)$')
plt.ylabel(r'$\Im(z)$')
plt.legend(loc=1)
plt.tight_layout()
plt.show()


