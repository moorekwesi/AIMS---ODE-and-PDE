# Starter code: SIRS model with births, vaccination and seasonal transmission
# Applied ODE & PDE with Python, Ch. 13 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

N = 1_000_000                  # population (kept constant: births = deaths)
gamma = 1 / 5                  # recovery rate (1/day)
mu = 1 / (60 * 365)            # birth and death rate (1/day)
omega = 1 / (2 * 365)          # waning of immunity (about two years)
beta0, amp = 0.5, 0.3          # mean transmission rate and seasonal amplitude


def model(t, z, nu):
    """nu = vaccination rate of susceptibles (1/day)."""
    S, I, R = z
    beta = beta0 * (1 + amp * np.cos(2 * np.pi * t / 365))   # rainy-season forcing
    inf = beta * S * I / N
    return [mu * N - inf - (nu + mu) * S + omega * R,
            inf - (gamma + mu) * I,
            gamma * I + nu * S - (mu + omega) * R]


years = 10
tt = np.linspace(0, 365 * years, 365 * years + 1)
z0 = [0.5 * N, 100, 0.5 * N - 100]
print(f"R0 = beta0/(gamma+mu) = {beta0 / (gamma + mu):.2f}")
print("annual maximum of I (thousands), years 1..10")
plt.figure(figsize=(8, 3.6))
for nu in [0.0, 1 / 1000, 1 / 365]:
    sol = solve_ivp(model, (tt[0], tt[-1]), z0, t_eval=tt, args=(nu,),
                    method="LSODA", rtol=1e-8, atol=1e-6)
    I = sol.y[1]
    peaks = [I[365 * k:365 * (k + 1)].max() / 1000 for k in range(years)]
    print(f"nu = {nu:.4f}: " + " ".join(f"{p:5.1f}" for p in peaks))
    plt.semilogy(tt / 365, np.maximum(I, 1e-2), label=f"vaccination rate nu = {nu:.4f}/day")
plt.ylim(1e-2, 1e5)
plt.xlabel("time (years)")
plt.ylabel("infectious I")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("ch13_sir_seasonal.pdf", bbox_inches="tight")
plt.close()
