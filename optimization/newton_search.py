from autograd import numpy as np
from autograd import grad, hessian

def newton_search(
        obj_func, x0,
        tol=1e-6, max_iter=100):
  
    f_x = grad(obj_func)
    f_xx = hessian(obj_func)
    x = x0
    descend = -f_x(x)
    k = 1
    while np.linalg.norm(descend) >= tol and k <= max_iter:
        x = x + np.linalg.solve(f_xx(x),descend)
        k += 1
        descend = -f_x(x)

    return x