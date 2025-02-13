import numpy as np

def hooke_disc_search(
        obj_func, x0, d, step,
        tol=1e-6, max_iter=100):

    dimension = len(x0)
    direction = np.eye(dimension)
    x = x0
    x_i = x0
    delta = tol + 1
    k = 0
    while delta > tol and d > tol and k <= max_iter:
        for i in np.arange(dimension):
            dummy = x_i + d*direction[i]
            if obj_func(dummy) < obj_func(x_i):
                x_i = dummy
            else:
                dummy = x_i - d*direction[i]
                if obj_func(dummy) < obj_func(x_i):
                    x_i = dummy
        if obj_func(x_i) < obj_func(x):
            x_prev = x
            x = x_i
            descend = x - x_prev
            x_i = x + step*descend
        else:
            d = d/2
            x_i = x
        delta = np.amax(abs(x - x_prev))
        k += 1

    return x