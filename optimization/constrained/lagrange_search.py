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

def lagrange_search(
        obj_func, constrain, x0, v0,
        tol=1e-6, max_iter=100):

    MULTIPLIER = 10
    def lagrangian(x,v,q):
        return obj_func(x) + sum(v*constrain(x)) + q*sum(constrain(x)**2)
    gradient = grad(lagrangian)
    x = x0
    v = v0
    q = 1
    k = 1
    while np.linalg.norm(gradient(x,v,q)) > tol and k < max_iter:
        x = newton_search(lambda t: lagrangian(t,v,q),x)
        v = v + q*constrain(x)
        q = MULTIPLIER*q
        k += 1

    return x