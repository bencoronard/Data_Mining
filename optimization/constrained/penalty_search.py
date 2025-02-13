from autograd import numpy as np
from autograd import grad, hessian

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

def penalty_search(
        obj_func, constrain, x0,
        tol=1e-6, max_iter=100):

    MULTIPLIER = 2
    penalty_factor = 1
    x = x0
    k = 1
    while penalty_factor*constrain(x) > tol and k < max_iter:
        penalty_fn = lambda t: obj_func(t) + penalty_factor*constrain(t)
        descend = -grad(penalty_fn)(x)
        step = newton_search(lambda t: penalty_fn(x + t*descend),0.1)
        x = x + step*descend
        penalty_factor = MULTIPLIER*penalty_factor
        k += 1

    return x