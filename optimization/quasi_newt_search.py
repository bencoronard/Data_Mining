from autograd import numpy as np
from autograd import grad

## Davidon-Fletcher-Powell (DFP) ##
def dfp(
        D0, x1, x2,
        grad_x1, grad_x2):
  
    I = np.eye(x1.size)
    p = x2 - x1
    q = grad_x2 - grad_x1
    pq = sum(p*q)
    PP = np.outer(p,p)
    Dq = np.matmul(D0,q)
    C = PP/pq - np.matmul(np.outer(Dq,q),D0)/sum(q*Dq)

    return C

## Broyden-Fletcher-Goldfarb-Shanno (BFGS) ##
def bfgs(
        D0, x1, x2,
        grad_x1, grad_x2):
  
    p = x2 - x1
    q = grad_x2 - grad_x1
    pq = sum(p*q)
    QQ = np.outer(q,q)
    Dp = np.matmul(D0,p)
    C = QQ/pq - np.matmul(np.outer(Dp,p),D0)/sum(p*Dp)

    return C

## Broyden Family ##
def broyden(
        phi, D0, x1, x2,
        grad_x1, grad_x2):
  
    return (1 - phi)*dfp(D0,x1,x2,grad_x1,grad_x2) + phi*bfgs(D0,x1,x2,grad_x1,grad_x2)

def armijo_search(
        obj_func, x0, eps=0.2,
        alpha=1.1, max_iter=100):

    f_x = grad(obj_func)
    def lin_approx(x):
        return obj_func(0.0) + x*eps*f_x(0.0)
    x = x0
    k = 0
    if obj_func(x) <= lin_approx(x):
        while obj_func(x) <= lin_approx(x) and k <= max_iter:
            x_prev = x
            x = alpha*x
            k += 1
        step = x_prev
    else:
        while obj_func(x) > lin_approx(x) and k <= max_iter:
            x_prev = x
            x = x/alpha
            k += 1
        step = x_prev

    return step

def quasi_newt_search(
        obj_func, x0, D0, phi,
        tol=1e-6, max_iter=100):
  
    f_x = grad(obj_func)
    x = x0
    D = D0
    gradient = f_x(x)
    k = 1
    while np.linalg.norm(gradient) >= tol and k <= max_iter:
        descend = np.linalg.solve(D,-gradient)
        step = armijo_search(lambda t: obj_func(x + t*descend),0.1)
        x_k = x + step*descend
        grad_x = gradient
        gradient = f_x(x_k)
        C = broyden(phi,D,x,x_k,grad_x,gradient)
        D += C
        x = x_k
        k += 1
        
    return x_k