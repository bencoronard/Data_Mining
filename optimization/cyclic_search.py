import numpy as np
from autograd import grad

def newton_search(
        obj_func, x0,
        tol=1e-6, max_iter=100):

    f_x = grad(obj_func)
    f_xx = grad(f_x)
    x = x0
    dx = tol + 1
    k = 0
    while abs(dx) > tol and k < max_iter:
        dx = -f_x(x)/f_xx(x)
        x = x + dx
        k += 1

    return x

def cyclic_search(
        obj_func, x0,
        tol=1e-6, max_iter=100):
  
    dimension = len(x0)
    direction = np.eye(dimension)
    x = x0
    x_i = x0
    delta = tol + 1
    k = 0
    while delta > tol and k <= max_iter:
        for i in np.arange(dimension):
            step = newton_search(lambda t: obj_func(x_i + t*direction[i]),0.0)
            x_i = x_i + step*direction[i]
        x_prev = x
        x = x_i
        delta = np.amax(abs(x - x_prev))
        k += 1
        
    return x