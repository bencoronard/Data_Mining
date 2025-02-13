import numpy as np

def simplex_search(
        obj_func, x0, alpha, beta,
        gamma, tol=1e-6, max_iter=100):

    # Generate initial simplex from x0
    c = 1
    n = len(x0) # dimension
    a = c/(n*np.sqrt(2))*(np.sqrt(n+1) + n - 1)
    b = c/(n*np.sqrt(2))*(np.sqrt(n+1) - 1)
    D = a*np.eye(n) + b*(np.ones((n,n)) - np.eye(n))
    X = np.array(np.array(x0).reshape(n,1)).dot(np.ones((1,n+1)))
    X[:,1:n+1] = X[:,1:n+1] + D

    # The method starts here.
    Y = f(X)
    dummy = np.sort(Y)
    xH, xh, xL = X[:,Y==dummy[n]].flatten(), X[:,Y==dummy[n-1]].flatten(), X[:,Y==dummy[0]].flatten()
    xC = (np.sum(X,axis=1) - xH)/n
    yH, yh, yL, yC = f(xH), f(xh), f(xL), f(xC)
    stdErr = np.sqrt((sum((Y - yC)**2) - (yH - yC)**2)/n)
    k = 1
    while stdErr >= tol and k <= max_iter:
        xR = xC + alpha*(xC - xh)
        yR = f(xR)
        if yR >= yL and yR <= yh:
            xh = xR
            yh = yR
        elif yR < yL:
            xE = xC + gamma*(xR - xC)
            yE = f(xE)
            if yE < yR:
                xh = xE
                yh = yE
            else:
                xh = xR
                yh = yR
        else:
            if yR < yh:
                xh = xR
                yh = yR
            xCO = xC + beta*(xh - xC)
            yCO = f(xCO)
            if yCO <= yh:
                xh = xCO
                yh = yCO
            else:
                X = (X + xL)/2
                Y = f(X)
        dummy = np.sort(Y)
        xH, xh, xL = X[:,Y==dummy[n]].flatten(), X[:,Y==dummy[n-1]].flatten(), X[:,Y==dummy[0]].flatten()
        xC = (np.sum(X,axis=1) - xH)/n
        yH, yh, yL, yC = f(xH), f(xh), f(xL), f(xC)
        stdErr = np.sqrt((sum((Y - yC)**2) - (yH - yC)**2)/n)
        k += 1

    return xL