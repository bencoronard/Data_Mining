from autograd import numpy as np
from autograd import grad

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

def find_direction(
        gradient, A, D0):
  
    n_constrain, n_var = A.shape
    a = np.concatenate((np.concatenate((D0,A.T),axis=1),
                        np.concatenate((A,np.zeros((n_constrain,n_constrain))),axis=1)),
                       axis=0)
    b = np.concatenate((-gradient,np.zeros((n_constrain,1)).flatten()))
    x = np.linalg.solve(a,b)
    direction = x[0:n_var]
    Dd = np.matmul(D0,direction)
    if np.sum(Dd != np.zeros((n_var,1)),axis=None) != 0:
        direction = direction/np.sqrt(sum(direction*Dd))

    return direction

def proj_steep_descent(
        obj_func, A, b, x0,
        tol=1e-6, max_iter=100):
  
    f_x = grad(obj_func)
    n_var = len(x0)
    I = np.eye(n_var)
    d = np.ones(n_var)
    d = d/np.linalg.norm(d)
    x = x0
    gradient = f_x(x)
    k = 0
    while abs(sum(gradient*d)) > tol and k < max_iter:
        d = find_direction(gradient,A,I)
        step = armijo_search(lambda t: obj_func(x + t*d),0.1)
        x = x + step*d
        gradient = f_x(x)
        k += 1

    return x