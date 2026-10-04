# The SIR model of an epidemic
# Applied ODE & PDE with Python, Ch. 1 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

N = 50000.0                  # population of the town
gamma = 1.0 / 5.0            # recovery rate: infectious for 5 days on average
R0 = 2.5                     # basic reproduction number
beta = R0 * gamma            # transmission rate

def sir(t, z):
    """Right-hand side of S' = -beta S I/N, I' = beta S I/N - gamma I, R' = gamma I."""
    S, I, R = z
    new = beta * S * I / N
    return [-new, new - gamma * I, gamma * I]

z0 = [N - 10, 10, 0]
t = np.linspace(0, 150, 1501)
sol = solve_ivp(sir, (0, 150), z0, t_eval=t, rtol=1e-8, atol=1e-6)
S, I, R = sol.y
ipk = np.argmax(I)
print(f"beta = {beta:.3f} per day, gamma = {gamma:.3f} per day, R0 = {R0}")
print(f"peak of the epidemic : day {t[ipk]:.1f}, {I[ipk]:.0f} infectious")
print(f"theory: peak when S = N/R0 = {N/R0:.0f};  simulated S = {S[ipk]:.0f}")
print(f"total ever infected  : {R[-1] + I[-1]:.0f} ({100*(R[-1] + I[-1])/N:.1f}%)")
# final-size relation  ln(s) = R0 (s - 1)  for s = S(inf)/N in (0, 1)
s_inf = brentq(lambda s: np.log(s) - R0 * (s - 1), 1e-6, 0.999)
print(f"final-size relation  : {100*(1 - s_inf):.1f}% infected")
print(f"S + I + R at t = 150 : {S[-1] + I[-1] + R[-1]:.6f}")

plt.figure(figsize=(7, 4))
for comp, lab in zip((S, I, R), ("S susceptible", "I infectious", "R removed")):
    plt.plot(t, comp / 1000, label=lab)
plt.xlabel("t (days)"); plt.ylabel("people (thousands)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("ch01_sir.pdf", bbox_inches="tight")
plt.close()
