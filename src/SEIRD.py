#!/usr/bin/env python

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from gillespie import SSA

S = 0
E = 1
I = 2
R = 3
D = 4

incubation = 10.0
alpha = 1.0 / incubation
beta = 0.3
gamma = 1.0 / 7.0
micro = 0.05
t_span = np.array([0, 220])

N = 1000.0
infected = 0.0
exposed = 7.0
recovered = 0.0
dead = 0.0
suseptable = N - infected - exposed - recovered - dead
y0 = np.array([suseptable, exposed, infected, recovered, dead])


def stoch():
    m = np.array([[-1, 1, 0, 0, 0],
                  [0, -1, 1, 0, 0],
                  [0, 0, -1, 1, 0],
                  [0, 0, -1, 0, 1]])
    return m


def propensities(X, coeff):
    alpha = coeff[0]
    beta = coeff[1]
    gamma = coeff[2]
    micro = coeff[3]
    N = coeff[4]

    w = np.array([
        beta * X[S] * X[I] / N,
        alpha * X[E],
        gamma * X[I],
        micro * X[I],
    ])
    return w


X0 = (y0[0], y0[1], y0[2], y0[3], y0[4])

coeff = (alpha, beta, gamma, micro, N)

t, X = SSA(propensities, stoch, X0, t_span, coeff)

plt.figure(2)

plt.plot(t, X[:, S], label="S(t)")
plt.plot(t, X[:, E], label="E(t)")
plt.plot(t, X[:, I], label="I(t)")
plt.plot(t, X[:, R], label="R(t)")
plt.plot(t, X[:, D], label="D(t)")

plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.show()
