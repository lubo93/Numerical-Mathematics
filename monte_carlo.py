from numpy import *
import matplotlib.pyplot as plt

# Definiere Runge-Funktion
f = lambda x: 1./(1+25*x**2)
Ianalytisch = lambda a, b: 1./5*arctan(5*b)-1./5*arctan(5*a)

# Definiere Normalverteilung
g = lambda x, sigma: 1./sqrt(2*pi*sigma**2)*exp(-0.5*x**2/sigma**2)

def int_uniform(F, N, a, b):
    x = random.random(N)
    # Intervalltransformation
    x = a + (b-a)*x
    # Auswertung
    y = 1.0*(b-a)*f(x)
     
    return mean(y), std(y)**2

def int_normal(F, N, a, b):
    mu = 0
    sigma = 0.28
    
    number_points = 0
    xt = []
    # Wir wohlen genau N Punkte im Intervall [a,b]    
    while number_points != N:    
        x = random.normal(mu, sigma, N-number_points)    
        ind = where(logical_and(x >= a, x <= b))
        xt.extend(x[ind])
        number_points = len(xt)
    
    xt = asarray(xt)    
    y = f(xt)/g(xt, sigma)
     
    return mean(y), std(y)**2
    
a = -1.0
b = 1.0

print("Ohne Importance-Sampling:")
# Monte-Carlo ohne Importance-Sampling
for k in range(2,6):
    I, var = int_uniform(f, 10**k, a, b)
    err = abs(I-Ianalytisch(a,b))
    print("N=%f, Integral=%0.5f, Fehler=%0.5f, Varianz=%0.5f" % (10**k, I, err, var))
print("\n")
# Plot fuer den Vergleich mit der Normalverteilung

t = linspace(-1, 1, 100)

plt.figure()
plt.plot(t, f(t), label='Runge-Funktion')
plt.plot(t, g(t, 0.1), label='Normalverteilung $\sigma=0.1$')
plt.plot(t, g(t, 0.2), label='Normalverteilung $\sigma=0.2$')
plt.plot(t, g(t, 0.28), label='Normalverteilung $\sigma=0.28$')
plt.plot(t, g(t, 0.35), label='Normalverteilung $\sigma=0.35$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=1)
plt.tight_layout()
plt.show()

print("Mit Importance-Sampling:")
# Monte-Carlo mit Importance-Sampling
for k in range(2,6):
    I, var = int_normal(f, 10**k, a, b)
    err = abs(I-Ianalytisch(a,b))
    print("N=%f, Integral=%0.5f, Fehler=%0.5f, Varianz=%0.5f" % (10**k, I, err, var)) 
