#!/usr/bin/env python

from logging import log
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from gillespie import SSA
import math

S = 0
E = 1
I = 2
R = 3
D = 4
V1 = 5
V2 = 6


def stoch():
                  # 0  1   2   3   4   5    6
    m = np.array([[-1,+1,  0,  0,  0,  0,  0],  # suseptible -> exposed
                  [-1, 0,  0,  0,  0, +1,  0],  # suseptible -> vacc_1
                  [0, -1, +1,  0,  0,  0,  0],  # exposed -> infected
                  [0,  0, -1, +1,  0,  0,  0],  # infected -> Recovered
                  [0,  0, -1,  0, +1,  0,  0],  # infected -> dead
                  [0, +1,  0,  0,  0, -1,  0],  # vacc1 -> exposed
                  [0,  0,  0,  0,  0, -1, +1],  # vacc1 -> vacc2
                  [0,  0,  0, +1,  0, -1,  0],  # vacc1 -> immune
                  [0, +1,  0,  0,  0,  0, -1],  # vacc2 -> exposed
                  [0,  0,  0, +1,  0,  0, -1]  # vacc2 -> imune
                  ])
    return m


def propensities(Y, coeff):
    suseptible_exposed = coeff[0]
    suseptible_vacc1 = coeff[1]
    exposed_infected = coeff[2]
    infected_recovered = coeff[3]
    infected_dead = coeff[4]
    vacc1_vacc2 = coeff[5]
    vacc1_exposed = coeff[6]
    vacc1_immune = coeff[7]
    vacc2_exposed = coeff[8]
    vacc2_immune = coeff[9]
    N = coeff[10]

    return np.array([
        suseptible_exposed * Y[S] * Y[I] / N,  # suseptible -> exposed
        suseptible_vacc1,  # suseptible -> vacc_1
        exposed_infected * Y[E],  # exposed -> infected
        infected_recovered * Y[I],  # infected -> Recovered
        infected_dead * Y[I],  # infected -> dead
        vacc1_exposed * Y[V1],  # vacc1 -> exposed
        vacc1_vacc2 * Y[V1],  # vacc1 -> vacc2
        vacc1_immune * Y[V1],  # vacc1 -> immune
        vacc2_exposed * Y[V2],  # vacc2 -> exposed
        vacc2_immune * Y[V2],  # vacc2 -> recovered
    ])


# ----------------- #
# propencity coefs  #
# ----------------- #
N = 1_000
incubation = 5
suseptible_exposed = 0.3
suseptible_vacc1 = N * 0.002
exposed_infected = 1.0 / incubation
infected_recovered = 1.0 / 7.0
infected_dead = 0.01
vacc1_vacc2 = 0.90
vacc1_exposed = 0.09
vacc1_immune = 0.01
vacc2_immune = 0.95
vacc2_exposed = 0.05
# vacc1_vacc2 = 0.49
# vacc1_exposed = 0.5
# vacc1_immune = 0.01
# vacc2_immune = 0.5
# vacc2_exposed = 0.5

# ----------------- #
# initial values    #
# ----------------- #
recovered = 0
exposed = 5
dead = 0
infected = 0
first_vaccination = 0
second_vaccination = 0
suseptible = N - dead - recovered - infected - exposed - first_vaccination - second_vaccination

coeff = (
    suseptible_exposed,
    suseptible_vacc1,
    exposed_infected,
    infected_recovered,
    infected_dead,
    vacc1_vacc2,
    vacc1_exposed,
    vacc1_immune,
    vacc2_exposed,
    vacc2_immune,
    N)

Y0 = (suseptible, exposed, infected, recovered, dead, first_vaccination, second_vaccination)
t_span = np.array([0, 120])


plt.figure(1)

colors = ['#1f77b4',  # A pleasant blue
          '#ff7f0e',  # A soft orange
          '#2ca02c',  # A calm green
          '#d62728',  # A muted red
          '#9467bd',  # A subtle purple
          '#8c564b',  # A gentle brown
          '#e377c2']  # A light pink

for _ in range(20):
    t, X = SSA(propensities, stoch, Y0, t_span, coeff)

    plt.plot(t, X[:, S], color=colors[0], label="S(t)" if _ == 0 else "")
    plt.plot(t, X[:, E], color=colors[1], label="E(t)" if _ == 0 else "")
    plt.plot(t, X[:, I], color=colors[2], label="I(t)" if _ == 0 else "")
    plt.plot(t, X[:, R], color=colors[3], label="R(t)" if _ == 0 else "")
    plt.plot(t, X[:, D], color=colors[4], label="D(t)" if _ == 0 else "")
    plt.plot(t, X[:, V1], color=colors[5], label="V1(t)" if _ == 0 else "")
    plt.plot(t, X[:, V2], color=colors[6], label="V2(t)" if _ == 0 else "")

plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.show()
