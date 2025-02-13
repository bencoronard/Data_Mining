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