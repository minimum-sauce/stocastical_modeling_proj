#!/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants

beta = 0.4          # Smittspridning
gamma = 1.0 / 10    # Tillfrisknande
alpha = 1.0 / 5.0    # Inkubationstid
my = 0.01            # Dödlighet (1%) NOT TRUUUUE
        # recovered_vacc1 * Y[R],  # Recovered -> vacc1 
        # recovered_vacc2, #* Y[R],  # Recovered -> vacc2
v1 = 0.01
v2 = 0.07
v1_r = 0.7
v2_r = 0.95


total = 10000
infekterade = 0
resistanta = 0
exponerade = 0
vaccinerad1 = 0
vaccinerad2 = 0 
döda = 0
mottagliga = total - infekterade - döda - resistanta - vaccinerad1 - vaccinerad2



t0 = 0
t1 = 120

Y0 = (mottagliga, exponerade, infekterade, resistanta, döda, vaccinerad1, vaccinerad2)


def SSA(prop, stoch, X0, tspan, coeff):
    # prop  - propensities
    # stoch - stoichiometry vector
    # Initial state vector
    tvec = np.zeros(1)
    tvec[0] = tspan[0]
    Xarr = np.zeros([1, len(X0)])
    Xarr[0, :] = X0
    t = tvec[0]
    X = X0
    sMat = stoch()

    while t < tspan[1]:
        # Generate two random numbers from a uniform distribution
        r1, r2 = np.random.uniform(0, 1, size=2)

        # Compute propensities and cumulative sum
        re = prop(X, coeff)
        cre = np.cumsum(re)
        a0 = cre[-1]

        # Stop if no reactions can occur
        if a0 < 1e-12:
            break

        # Generate time increment tau from exponential distribution
        tau = np.random.exponential(scale=1 / a0)
        cre = cre / a0  # Normalize cumulative sum

        # Determine the reaction channel
        r = 0
        while cre[r] < r2:
            r += 1

        t += tau

        # If new time exceeds final time, exit loop
        if t > tspan[1]:
            break

        # Update time and state
        tvec = np.append(tvec, t)
        X = X + sMat[r, :]
        Xarr = np.vstack([Xarr, X])

    # If simulation stopped before reaching the final time, append the final time
    if tvec[-1] < tspan[1]:
        tvec = np.append(tvec, tspan[1])
        Xarr = np.vstack([Xarr, X])

    return tvec, Xarr


def propensities(X, params):

    beta, gamma = params
    S, E, I, R, D, V1, V2 = X

    # Sannolikheten för smittspridning
    smittsamhet = beta * S * I / total

    exp_to_infected = E*alpha

    # Sannolikheten för tillfrisknande
    infected_to_resistant = gamma * I

    # Sannolikhet för dödlighet
    infected_to_dead = my * I

    mottaglig_to_vaccinated1 = v1

    vaccinated1_to_resistant = V1 * v1_r

    vaccinated1_to_vaccinated2 = v2 * V1

    vaccinated2_to_resistant = V2 * v2_r

    vaccinated1_to_exposed = V1 * (1-v1_r)
    
    vaccinated2_to_exposed = V2 * (1-v2_r)



    return np.array([smittsamhet, exp_to_infected, infected_to_resistant, 
    infected_to_dead, mottaglig_to_vaccinated1, vaccinated1_to_resistant, 
    vaccinated1_to_vaccinated2, vaccinated2_to_resistant, vaccinated1_to_exposed, 
    vaccinated2_to_exposed])


def stoch():        # S   E   I  R  D  V1 V2 
    return np.array([[-1, +1, 0, 0, 0, 0, 0], #Mottaglig till exponerad
                     [0, -1, +1, 0, 0, 0, 0], #Exponerad till infekterad
                     [0,  0, -1, +1, 0, 0, 0], # Infekterad till frisk(Resistant)
                     [0,  0, -1, 0, +1, 0, 0], # Infekterad till död
                     [-1, 0,  0, 0, 0, +1, 0], #Mottaglig till Vaccinerad 1
                     [0,  0,  0, +1, 0, -1, 0],# Vaccinerad 1 till Immun
                     [0,  0,  0, +1, 0, -1, 1],# Vaccinerad 1 till vaccinerad 2
                     [0,  0,  0, +1, 0, 0, -1],# Vaccinerad 2 till Immun
                     [0,  +1,  0, 0, 0, -1, 0],# Vaccinerad 1 till exponerad
                     [0,  +1,  0, 0, 0, 0, -1] # Vaccinerad 2 till exponerad
                     ])


t_span = (t0, t1)

tvec, Xarr = SSA(propensities, stoch, Y0, t_span, [beta, gamma])

# Plotta resultaten
plt.plot(tvec, Xarr[:, 0], label='Mottagliga (S)')
plt.plot(tvec, Xarr[:, 1], label='Exponerade (E)')
plt.plot(tvec, Xarr[:, 2], label='Infekterade (E)')
plt.plot(tvec, Xarr[:, 3], label='Resistenta (R)')
plt.plot(tvec, Xarr[:, 4], label='Döda (D)')
plt.plot(tvec, Xarr[:, 5], label='Vaccinerade (V1)')
plt.plot(tvec, Xarr[:, 6], label='Vaccinerade (V2)')
plt.xlabel('Tid (dagar)')
plt.ylabel('Antal personer')
plt.legend()
plt.title('Covid-19 modell med Gillespies algoritm')
plt.grid()
plt.show()
