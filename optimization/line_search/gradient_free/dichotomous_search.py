def dichotomous_search(
        obj_func, left_bound, right_bound,
        eps=1e-3, tol=1e-6, max_iter=100):

    xL = left_bound
    xR = right_bound
    xMid = (xL + xR)/2
    xLi = xMid - eps
    xRi = xMid + eps
    k = 0
    while abs(xLi - xRi) > tol and k < max_iter:
        if obj_func(xLi) < obj_func(xRi):
            xR = xRi
        else:
            xL = xLi
        xMid = (xL + xR)/2
        xLi = xMid - eps
        xRi = xMid + eps
        k += 1
    
    return xMid