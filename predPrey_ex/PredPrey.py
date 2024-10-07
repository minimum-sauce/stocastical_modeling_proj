import numpy as np

def stochPredPrey():
    M = np.array([[-1,1],
                [0, -1],
                [1, 0]])
    return M

def propPredPrey(X, coeff):
    alpha=coeff[0]
    beta=coeff[1]
    gamma=coeff[2]

    w = np.array([beta*X[0]*X[1],gamma*X[1], alpha*X[0]])
    return w
