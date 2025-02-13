import numpy as np
import math

GR = (math.sqrt(5) + 1)/2

def golden_section(
        obj_func, left_bound, right_bound,
        tol=1e-6, max_iter=100):

    xL = left_bound
    xR = right_bound
    xLi = xR - (xR - xL)/GR
    xRi = xL + (xR - xL)/GR
    k = 0
    while abs(xLi - xRi) > tol and k < max_iter:
        if obj_func(xLi) < obj_func(xRi):
            xR = xRi
        else:
            xL = xLi
        xLi = xR - (xR - xL)/GR
        xRi = xL + (xR - xL)/GR
        k += 1
    
    return (xLi + xRi)/2