from autograd import numpy as np
from autograd import grad, hessian

def lev_search(
        obj_func, x0, zeta0,
        tol=1e-6, max_iter=100):

    f_x = grad(obj_func)
    f_xx = hessian(obj_func)
    n = x0.size
    I = np.eye(n)
    x = x0
    zeta = zeta0
    descend = -f_x(x)
    k = 1
    while np.linalg.norm(descend) >= tol and k <= max_iter :
        Dk = np.linalg.solve(f_xx(x) + zeta*I,descend)
        xk = x + Dk
        if obj_func(x) <= obj_func(xk):
            zeta = 10*zeta
        else:
            x = xk
            zeta = zeta/10
        k += 1
        descend = -f_x(x)

    return x 