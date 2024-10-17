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
IM = 7


def stoch():
                  # 0  1   2  3  4  5  6
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
        suseptible_vacc1 * math.log10(Y[S]),  # suseptible -> vacc_1
        exposed_infected * Y[E],  # exposed -> infected
        infected_recovered * Y[I],  # infected -> Recovered
        infected_dead * Y[I],  # infected -> dead
        vacc1_vacc2 * Y[V1],  # vacc1 -> vacc2
        vacc1_exposed * Y[V1],  # vacc1 -> exposed
        vacc1_immune * Y[V1],  # vacc1 -> immune
        vacc2_exposed * Y[V2],  # vacc2 -> exposed
        vacc2_immune * Y[V2],  # vacc2 -> recovered
    ])


# ----------------- #
# propencity coefs  #
# ----------------- #
incubation = 5.0 
suseptible_exposed = 0.4
suseptible_vacc1 = 0.5
exposed_infected = 1.0 / incubation
infected_recovered = 1.0 / 10.0
infected_dead = 0.1
vacc1_vacc2 = 1.0
vacc1_immune = 0.7
vacc1_exposed = 0.06
vacc2_immune = 0.95
vacc2_exposed = 0.03
N = 10_000
# recovered_vacc1 = 2.3 
# recovered_vacc2 = 1.2

# ----------------- #
# initial values    #
# ----------------- #
recovered = 0
exposed = 0
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

t, X = SSA(propensities, stoch, Y0, t_span, coeff)

plt.figure(2)

plt.plot(t, X[:, S], label="S(t)")
plt.plot(t, X[:, E], label="E(t)")
plt.plot(t, X[:, I], label="I(t)")
plt.plot(t, X[:, R], label="R(t)")
plt.plot(t, X[:, D], label="D(t)")
plt.plot(t, X[:, V1], label="V1(t)")
plt.plot(t, X[:, V2], label="V2(t)")

plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.show()
