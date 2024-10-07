import PredPrey as pp
import PredPreyODE as ppODE
from gillespie import SSA
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

pred0 = 1000
prey0 = 500
X0 = (prey0, pred0)
alfa = 10
beta = 0.01
gamma = 3
coeff = (alfa, beta, gamma)
t0 = 0
t_end = 10
tspan = (t0, t_end)

t, X = SSA(pp.propPredPrey, pp.stochPredPrey, X0, tspan, coeff)

times = np.arange(t0, t_end, 0.01)
sol = solve_ivp(ppODE.predpreyODE, tspan, X0, t_eval=times, args=(coeff,))

plt.figure(1)
plt.plot(t, X[:, 0], 'b-', label="Prey")
plt.plot(t, X[:, 1], 'r-', label="Predator")
ax1 = plt.gca()
xmin, xmax, ymin, ymax = ax1.axis()
ax1.set(xlim=(0, xmax), ylim=(0, ymax))
plt.title("Pred-Prey, stochastic")
plt.legend()

plt.figure(2)
plt.plot(sol.t, sol.y[0], "b-", label="Prey")
plt.plot(sol.t, sol.y[1], "r-", label="Predator")
ax2 = plt.gca()
# xmin, xmax, ymin, ymax = ax2.axis()
ax2.set(xlim=(0, xmax), ylim=(0, ymax))
plt.legend
plt.title("Pred-Prey, deterministic")
plt.show()
