import numpy as np

def fibonacci_search(
        obj_func, left_bound,
        right_bound, tol=1e-6):

    last_fib = (right_bound - left_bound)/tol
    fib_seq = np.array([1,1])
    n = 1
    while fib_seq[n] < last_fib:
        fib_seq = np.append(fib_seq, fib_seq[n] + fib_seq[n-1])
        n += 1

    xL = left_bound
    xR = right_bound
    k = 1
    xLi = xL + (fib_seq[n-k-1]/fib_seq[n-k+1])*(xR - xL)
    xRi = xL + (fib_seq[n-k]/fib_seq[n-k+1])*(xR - xL)
    while k < n:
        if obj_func(xLi) < obj_func(xRi):
            xR = xRi
        else:
            xL = xLi
        xLi = xL + (fib_seq[n-k-1]/fib_seq[n-k+1])*(xR - xL)
        xRi = xL + (fib_seq[n-k]/fib_seq[n-k+1])*(xR - xL)
        k += 1
    eps = (xR - xL)/10
    xRi = xLi + eps
    if obj_func(xLi) < obj_func(xRi):
        xR = xRi
    else:
        xL = xLi

    return (xL + xR)/2