import numpy as np

def gen_three_points(func, x0):

    x1 = 0
    if func(x0) >= func(x1):
        x3 = x0
        x2 = (x1 + x3)/2
        while func(x1) < func(x2):
            x3 = x2
            x2 = (x1 + x3)/2
    else:
        x2 = x0
        x3 = 2*x2 - x1
        while func(x2) > func(x3):
            x2 = x3
            x3 = 2*x2 - x1

    return x1,x2,x3

def quadratic_fit_search(
        obj_func, x0,
        tol=1e-6, max_iter=100):
  
    x1,x2,x3 = gen_three_points(obj_func,x0)
    k = 1
    while x3 - x1 > tol and k < max_iter:
        A = np.array([[x1**2,x1,1], [x2**2,x2,1], [x3**2,x3,1]])
        b = np.array([obj_func(x1), obj_func(x2), obj_func(x3)])
        x = np.linalg.solve(A,b)
        xb = -x[1]/(2*x[0])
        if xb == x2:
            if (x3 - x2) > (x2 - x1):
                xb = x2 + tol/2
            else:
                xb = x2 - tol/2
        if xb > x2:
            if obj_func(xb) >= obj_func(x2):
                x3 = xb
            else:
                x1 = x2
                x2 = xb
        else:
            if obj_func(xb) >= obj_func(x2):
                x1 = xb
            else:
                x3 = x2
                x2 = xb
        k += 1
        
    return x2