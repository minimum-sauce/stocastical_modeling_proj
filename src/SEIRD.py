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

incubation = 5.0
alpha = 1.0 / incubation
beta = 0.5
gamma = 1.0 / 7.0
micro = 0.08
t_span = np.array([0, 220])

N = 1000.0
infected = 0.0
exposed = 5.0
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
    plt.plot(t, X[:, E], color=colors[1], label="E(t)" if _ == 0 else "")
    plt.plot(t, X[:, I], color=colors[2], label="I(t)" if _ == 0 else "")
    plt.plot(t, X[:, R], color=colors[3], label="R(t)" if _ == 0 else "")
    plt.plot(t, X[:, D], color=colors[4], label="D(t)" if _ == 0 else "")
    #plt.plot(t, X[:, V1], color=colors[5], label="V1(t)" if _ == 0 else "")
    #plt.plot(t, X[:, V2], color=colors[6], label="V2(t)" if _ == 0 else "")

plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.show()
