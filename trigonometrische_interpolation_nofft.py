from numpy import arange, pi, outer, exp, linspace, fft, conj, sign
from numpy.linalg import solve

def trigpolycoeff(t, y):
    N = y.shape[0]
    if N%2 == 1:
        n = (N-1.)/2.
        M = exp(2*pi*1j*outer(t[:-1],arange(-n,n)))
    else:
        n = N/2.
        M = exp(2*pi*1j*outer(t[:-1],arange(-n+1,n)))
    c = solve(M, y[:-1])
    return c

f = lambda x: 0.5 * sign(x+0.5)-0.5 * sign(x-0.5)
        
N = 1024
x = linspace(-1,1,N)
y = f(x)

direkt = trigpolycoeff(x,y)
viafft = fft.fftshift(fft.fft(y[:-1])/(N-1))
viaIfft = conj(fft.fftshift(fft.ifft(y[:-1])))
print abs(direkt-viafft).max(), abs(direkt-viaIfft).max()