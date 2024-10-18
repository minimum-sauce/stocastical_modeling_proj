#!/usr/bin/env python

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from gillespie import SSA

# ----------------------------------- #
# Deterministic modeling
# ----------------------------------- #

S = 0
I = 1
R = 2


def deltaS(y, beta, population):
    return -beta * y[I] / population * y[S]


def deltaI(y, gamma, beta, population):
    return beta * y[I] / population * y[S] - gamma * y[I]


def deltaR(y, gamma):
    return gamma * y[I]


def ode(t, y, gamma, beta, population):
    return np.array([deltaS(y, beta, population),
                     deltaI(y, gamma, beta, population),
                     deltaR(y, gamma)])


beta = 0.3
gamma = 1 / 7
t_span = np.array([0, 120])
N = 1000
infected = 5
y0 = np.array([N - infected, infected, 0])

sol = solve_ivp(ode, t_span, y0, args=(gamma, beta, N))

plt.figure(1)

plt.plot(sol.t, sol.y[S], label="S(t)")
plt.plot(sol.t, sol.y[I], label="I(t)")
plt.plot(sol.t, sol.y[R], label="R(t)")

plt.xlabel("t")
plt.ylabel("y")
plt.legend()


# ----------------------------------- #
# Stochastical modeling
# ----------------------------------- #

def stoch():
    m = np.array([[-1, 1, 0],
                  [0, -1, 1]])
    return m


def propensities(X, coeff):
    beta = coeff[0]
    gamma = coeff[1]
    N = coeff[2]

    w = np.array([
        beta * X[S] * X[I] / N,
        gamma * X[I]
    ])
    return w


X0 = (y0[0], y0[1], y0[2])

coeff = (beta, gamma, N)


plt.figure(2)

colors = ['#1f77b4',  # A pleasant blue
          '#ff7f0e',  # A soft orange
          '#2ca02c',  # A calm green
          '#d62728',  # A muted red
          '#9467bd',  # A subtle purple
          '#8c564b',  # A gentle brown
          '#e377c2']  # A light pink

for _ in range(6):
    t, X = SSA(propensities, stoch, X0, t_span, coeff)

    plt.plot(t, X[:, S], color=colors[0], label="S(t)" if _ == 0 else "")
    #plt.plot(t, X[:, E], color=colors[1], label="E(t)" if _ == 0 else "")
    plt.plot(t, X[:, I], color=colors[2], label="I(t)" if _ == 0 else "")
    plt.plot(t, X[:, R], color=colors[3], label="R(t)" if _ == 0 else "")
    #plt.plot(t, X[:, D], color=colors[4], label="D(t)" if _ == 0 else "")
    #plt.plot(t, X[:, V1], color=colors[5], label="V1(t)" if _ == 0 else "")
    #plt.plot(t, X[:, V2], color=colors[6], label="V2(t)" if _ == 0 else "")

plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.show()
