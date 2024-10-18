#!/usr/bin/env python

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from gillespie import SSA

S = 0
E = 1
I = 2
R = 3

incubation = 10.0
alpha = 1.0 / incubation
beta = 0.3
gamma = 1 / 7
t_span = np.array([0, 220])
N = 1000
infected = 5
y0 = np.array([N - infected, 0, infected, 0])


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
    #plt.plot(t, X[:, D], color=colors[4], label="D(t)" if _ == 0 else "")
    #plt.plot(t, X[:, V1], color=colors[5], label="V1(t)" if _ == 0 else "")
    #plt.plot(t, X[:, V2], color=colors[6], label="V2(t)" if _ == 0 else "")

plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.show()
