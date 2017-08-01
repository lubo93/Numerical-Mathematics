import numpy as np
import matplotlib.pyplot as plt

def lstsq_QR(A, b):
    Q,R = np.linalg.qr(A)
    Qb = np.dot(Q.T,b)
    y = np.linalg.solve(R,Qb)
    return y  
    
def lstsq_SVD(A, b, eps=1e-6):
    U,s,Vh = np.linalg.svd(A)
    r = 1+np.where(s/s[0]>eps)[0].max() # numerical rank 
    y = np.dot(Vh[:r,:].T, np.dot(U[:,:r].T,b)/s[:r] )
    return y      
    
x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.9, 2.1])

# y = Ap mit A = [[x 1]] und p = [[m], [n]]
A = np.vstack([x, np.ones(len(x))]).T
m, n = np.linalg.lstsq(A, y)[0]
print "numpy lstsq: ", m, n

# with SVD
m_svd, n_svd = lstsq_SVD(A, y)
print "svd lstsq: ", m_svd, n_svd

# with QR
m_qr, n_qr = lstsq_QR(A, y)
print "qr lstsq: ", m_qr, n_qr

# oder polyfit
m_poly, n_poly = np.polyfit(x, y, 1)
print "polyfit: ", m_poly, n_poly

plt.figure()
plt.plot(x, y, 'o', label='Daten', markersize=10)
plt.plot(x, m*x + n, 'r', label='Linearer Ausgleich')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc=4)
plt.tight_layout()
plt.show()