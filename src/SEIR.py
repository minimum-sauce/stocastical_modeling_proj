#!/usr/bin/env python

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from gillespie import SSA

S = 0
E = 1
I = 2
R = 3

incubation = 1.0
alpha = 1.0 / incubation
beta = 0.3
gamma = 1 / 7
t_span = np.array([0, 120])
N = 1000
suseptable = 5
y0 = np.array([N - suseptable, 0, suseptable, 0])


def stoch():
    m = np.array([[-1, 1, 0, 0],
                  [0, -1, 1, 0],
                  [0, 0, -1, 1]])
    return m


def propensities(X, coeff):
    alpha = coeff[0]
    beta = coeff[1]
    gamma = coeff[2]
    N = coeff[3]

    w = np.array([
        beta * X[S] * X[I] / N,
        alpha * X[E],
        gamma * X[I]
    ])
    return w


X0 = (y0[0], y0[1], y0[2], y0[3])

coeff = (alpha, beta, gamma, N)

t, X = SSA(propensities, stoch, X0, t_span, coeff)

plt.figure(2)

plt.plot(t, X[:, S], label="S(t)")
plt.plot(t, X[:, E], label="E(t)")
plt.plot(t, X[:, I], label="I(t)")
plt.plot(t, X[:, R], label="R(t)")

plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.show()
