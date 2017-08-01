from numpy import arctan

f = lambda x: arctan(x)
fp = lambda x: 1./(1. + x**2)

tol = 1e-5
x0 = 10
maxit = 100
for i in range(maxit):
    
    # Bestimme lambda in jedem Iterationsschritt
    # Verwende den natuerlichen Monotonietest
    lamb = 1.
    delta=f(x0)/fp(x0)
    flag = 1
    while flag:
        lamb *= 0.5
        if abs(f(x0-lamb*delta)/fp(x0)) <= (1-lamb/2.)*abs(delta):
            flag = 0
    # Gedaempftes Newton Verfahren
    x1 = x0-lamb*delta
    #Fehlerschranke (Maschinengenauigkeit)
    if abs(x1) > 1e-14:
        err = abs((x1-x0)/x1)
        if err < tol:
                print 'Ergebnis:' , x1  ,'\t','Iterationen:',i
                break
        else:
            err = abs(x1-x0)
            if err < tol:
                print 'Ergebnis' , x1  ,'\t','Iterationen:',i
                break
            
    x0 = x1