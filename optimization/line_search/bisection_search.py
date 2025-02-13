from autograd import grad

def bisection_search(
        obj_func, left_bound,
        right_bound, tol=1e-6):
                     
    length = (right_bound - left_bound)*0.5
    n = 1
    while length > tol:
        length = length*0.5
        n += 1
    
    xL = left_bound
    xR = right_bound
    f_x = grad(obj_func)
    k = 1
    while k <= n:
        xMid = (xL + xR)/2
        slope = f_x(xMid)
        if slope > 0:
            xR = xMid
        elif slope < 0:
            xL = xMid
        else:
            xL = xMid
            xR = xMid
            k = n
        k += 1

    return (xL + xR)/2