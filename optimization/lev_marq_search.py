from autograd import numpy as np
from autograd import grad, hessian

def lev_marq_search(
        obj_func, x0, zeta0,
        tol=1e-6, max_iter=100):

    f_x = grad(obj_func)
    f_xx = hessian(obj_func)
    x = x0
    zeta = zeta0
    descend = -f_x(x)
    k = 1
    while np.linalg.norm(descend) >= tol and k <= max_iter :
        H = f_xx(x)
        D = np.diag(np.diag(H))
        Dk = np.linalg.solve(H + zeta*D,descend)
        xk = x + Dk
        if obj_func(x) <= obj_func(xk):
            zeta = 10*zeta
        else:
            x = xk
            zeta = zeta/10
        k += 1
        descend = -f_x(x)

    return x