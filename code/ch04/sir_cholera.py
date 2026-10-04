# SIR model for a cholera outbreak and the final-size relation
# Applied ODE & PDE with Python, Ch. 4 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

N = 100_000                  # population of the town
gamma = 1 / 5                # recovery rate: mean infectious period 5 days
R0 = 2.0                     # basic reproduction number (assumed)
beta = R0 * gamma            # transmission rate per day


def sir(t, z):
    S, I, R = z
    new_inf = beta * S * I / N
    return [-new_inf, new_inf - gamma * I, gamma * I]


def peak(t, z):              # dI/dt = 0  <=>  S = N / R0
    return z[0] - N / R0


sol = solve_ivp(sir, (0, 200), [N - 10, 10, 0], events=peak, dense_output=True,
                rtol=1e-8, atol=1e-6)
tp = sol.t_events[0][0]
S_end = sol.y[0, -1]
print(f"beta = {beta:.2f}/day, gamma = {gamma:.2f}/day, R0 = {R0}")
print(f"epidemic peak on day {tp:.1f} with I = {sol.sol(tp)[1]:.0f} infectious")
print(f"fraction never infected (ODE, t = 200): {S_end / N:.4f}")
# final-size relation: ln(s_inf) = R0 (s_inf - 1), with s_inf in (0, 1/R0)
s_inf = brentq(lambda s: np.log(s) - R0 * (s - 1), 1e-6, 1 / R0)
print(f"fraction never infected (final size eq.): {s_inf:.4f}")
print(f"total cases ~ {N * (1 - s_inf):.0f}; herd immunity threshold 1 - 1/R0 = {1 - 1 / R0:.2f}")

tt = np.linspace(0, 200, 801)
S, I, R = sol.sol(tt) / 1000
plt.figure(figsize=(7, 3.8))
plt.plot(tt, S, label="S susceptible")
plt.plot(tt, I, label="I infectious")
plt.plot(tt, R, label="R recovered")
plt.axvline(tp, color="gray", ls=":")
plt.xlabel("time (days)")
plt.ylabel("people (thousands)")
plt.legend()
plt.tight_layout()
plt.savefig("ch04_sir_cholera.pdf", bbox_inches="tight")
plt.close()
