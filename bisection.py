def f(x):
    return x**5 + x -1
    
def bisection(a,b,tol):
    c = (a+b)/2.0
    while (b-a)/2.0 > tol:
        if f(c) == 0:
            return c
        elif f(a)*f(c) < 0:
            b = c
        else:
		a = c
        c = (a+b)/2.0	
    return c
 
print bisection(0,5,10**-10)