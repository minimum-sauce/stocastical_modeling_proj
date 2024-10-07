#==================================
# Gillespies algoritm
# prop - propensities (1D numpy-array)
# stoch - Stoichiometry matrix (2D numpy-array)
# X0 - initial state (list, tuplet or numpy-array)
# tspan - silulation time interval
# coeff - model parameters
#==================================
import numpy as np
import random


def SSA(prop, stoch, X0, tspan, coeff):
    # prop  - propensities
    # stoch - stiochiometry vector
    # X0    - Initial state vector
    tvec = np.zeros(1)
    tvec[0] = tspan[0]
    Xarr = np.zeros([1, len(X0)])
    Xarr[0, :] = X0
    t = tvec[0]
    X = X0
    sMat = stoch()
    while t < tspan[1]:
        r1, r2 = np.random.uniform(0, 1, size=2)  # Find two random numbers on uniform distr.
        re = prop(X, coeff)
        cre = np.cumsum(re)
        a0 = cre[-1]
        if a0 < 1e-12:
            break

        tau = np.random.exponential(scale=1 / a0)  # Random number exponential distribution
        cre = cre / a0
        r = 0
        while cre[r] < r2:
            r += 1

        t += tau
        # if new time is larger than final time, skip last calculation
        if t > tspan[1]:
            break

        tvec = np.append(tvec, t)
        X = X + sMat[r, :]
        Xarr = np.vstack([Xarr, X])

    # If iterations stopped before final time, add final time and no change
    if tvec[-1] < tspan[1]:
        tvec = np.append(tvec, tspan[1])
        Xarr = np.vstack([Xarr, X])

    return tvec, Xarr
