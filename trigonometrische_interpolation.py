from numpy import zeros, fft, real, linspace, sign
import matplotlib.pyplot as plt

def evaliptrig(y,N):
    n = len(y)
    if (n%2) == 0:
        c = fft.ifft(y)
        a = zeros(N, dtype=complex)
        a[:n/2] = c[:n/2]
        a[N-n/2:] = c[n/2:]
        v = fft.fft(a)
        return v
    else: raise TypeError, 'odd length'

f = lambda x: 0.5 * sign(x+0.5)-0.5 * sign(x-0.5)
        
N = 4096
x_N = linspace(-1,1,N)
x = linspace(-1,1,256)
y = f(x)
v = real(evaliptrig(y,N))

plt.figure()
plt.plot(x_N, f(x_N), linewidth=2., label=r'Step function')
plt.plot(x_N, v, linewidth=2., label=r'Trigonometric interpolation')
plt.ylim([-0.2,1.2])
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')
plt.legend(loc=5)
plt.tight_layout()
plt.show()