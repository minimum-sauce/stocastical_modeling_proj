import numpy as np

def predpreyODE(t, Y, coeff):
    y1, y2 = Y
    alfa, beta, gamma = coeff
    Yprim = [alfa*y1 - beta*y1*y2, beta*y1*y2 - gamma*y2] 
    return np.array(Yprim)