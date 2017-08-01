from numpy import *
from scipy import integrate

f = lambda x: exp(10*cos(10*x))

# Adapative Quadratur mit Trapez und Simpsonregel
def adaptive_quadrature(f,M,rtol,abstol):
    
    # Step size
    h = diff(M)
    # Mittelpunktwerte
    mp = 0.5*( M[:-1]+M[1:] )
    # Funktionswerte
    fx = f(M); fm = f(mp)
    
    # Lokale Trapezregel
    trp_loc = h*( fx[:-1]+fx[1:] )/2
    # Lokale Simpsonregel
    simp_loc= h*( fx[:-1]+4*fm+fx[1:] )/6

    I = sum(simp_loc)
    
    # Lokale Fehlerschaetzung
    est_loc = abs(simp_loc - trp_loc)
    # Totaler Fehler
    err_tot = sum(est_loc)
    
    if err_tot > rtol*abs(I) and err_tot > abstol:
        
        refcells = nonzero( est_loc > 0.9*err_tot/size(est_loc) )[0]
        I = adaptive_quadrature(f,sort(append(M,mp[refcells])),rtol,abstol)

    return I

M = linspace(0,1,11)
rtol = 1e-6; abstol = 1e-10
I = adaptive_quadrature(f,M,rtol,abstol)
exact,error = integrate.quad(f,M[0],M[-1])
print "Adapative quadrature:",I, "Exact result: ",exact
print 'Error: ',abs(I-exact)