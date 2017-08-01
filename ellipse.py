from numpy import *
from numpy.linalg import lstsq, norm, solve
import matplotlib.pyplot as plt

# Datenpunkte
x = array([-0.1, 6.7, 6.7, -0.2, -7.3, -6.5])
y = array([3.1, 1.4, -1.0, -2.6, -1.1, 1.3])

x2 = sum(x**2)
y2 = sum(y**2)
x4 = sum(x**4)
y4 = sum(y**4)

# Definiere F, DF, Gradient und Hessematrix von Phi
F = lambda z: (x/z[0])**2+(y/z[1])**2-1
DF = lambda z: -2*array([x**2/z[0]**3, y**2/z[1]**3]).T

gradPhi = lambda z: array([sum(F(z)*DF(z)[:,0]), sum(F(z)*DF(z)[:,1])])

HPhi = lambda z: array([[sum((DF(z)**2)[:,0] + F(z)*(6*x**2/z[0]**3)), sum(DF(z)[:,0]*DF(z)[:,1])],
                                 [sum(DF(z)[:,0]*DF(z)[:,1]), sum((DF(z)**2)[:,1] + F(z)*(6*y**2/z[1]**3))]])

z_final = array([7.51014307, 2.90380614])

def ellipse_gauss(F, DF, z, maxit=50, abstol=1e-6):
    delta_norm = []
    for i in range(maxit):
        s = lstsq(DF(z), F(z))[0]
        z -= s
        
        # Konvergenzstudie
        delta_norm.append(norm(z-z_final))
        if norm(s) < abstol:
            return z, asarray(delta_norm)
    
    print "Keine Konvergenz"
    
def newton(gradPhi, HPhi, z, maxit=50, abstol=1e-6):
    delta_norm = []
    for i in range(maxit):
        s = solve(HPhi(z), gradPhi(z))
        z -= s

        # Konvergenzstudie
        delta_norm.append(norm(z-z_final))
        if norm(s) < abstol:
            return z, asarray(delta_norm)
    
    print "Keine Konvergenz"
    
# Startwert fuer Parametervektor
z = array([1., 1.])

# Zuerst die Loesung fuer das Gauss-Newton-Verfahren
z_gauss, convergence_arr_gauss = ellipse_gauss(F, DF, z)

print z_gauss
t = linspace(0,2*pi,100)
x_gauss = z_gauss[0]*cos(t)
y_gauss = z_gauss[1]*sin(t)


# Jetzt die Loesung fuer das Newton-Verfahren
z = array([1., 1.])
z_newton, convergence_arr_newton = newton(gradPhi, HPhi, z)
print z_newton

x_newton = z_newton[0]*cos(t)
y_newton = z_newton[1]*sin(t)

plt.figure()
plt.plot(x, y,'bo', label='Datenpunkte')
plt.plot(x_gauss, y_gauss, '-', label='Gauss-Newton-Verfahren')
plt.plot(x_newton, y_newton, '-', label='Newton-Verfahren')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend(loc='center')
plt.tight_layout()
plt.show()

# Konvergenzplot
plt.figure()
plt.plot(convergence_arr_gauss,'bo', label='Gauss-Newton')
plt.plot(convergence_arr_newton,'ro', label='Newton')
plt.xlabel('Iterations')
plt.ylabel('Correction')
plt.legend(loc=1)
plt.tight_layout()
plt.show()