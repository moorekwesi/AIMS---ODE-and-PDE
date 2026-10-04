# The Lorenz system: sensitive dependence on initial conditions
# Applied ODE & PDE with Python, Ch. 4 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0


def lorenz(t, u):
    x, y, z = u
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


T = 40.0
tt = np.linspace(0, T, 8001)
opts = dict(t_eval=tt, method="DOP853", rtol=1e-12, atol=1e-12)
u1 = solve_ivp(lorenz, (0, T), [1.0, 1.0, 1.0], **opts).y
u2 = solve_ivp(lorenz, (0, T), [1.0 + 1e-8, 1.0, 1.0], **opts).y
dist = np.linalg.norm(u1 - u2, axis=0)

print("   t     |u1(t) - u2(t)|")
for t in [0, 5, 10, 15, 20, 25, 30, 35]:
    print(f"{t:5.0f}     {dist[int(t * 200)]:.3e}")
mask = (tt > 12) & (tt < 28)              # fit before the separation saturates
lam = np.polyfit(tt[mask], np.log(dist[mask]), 1)[0]
print(f"growth rate of log(distance) on [12, 28] ~ {lam:.2f} (largest Lyapunov exponent ~ 0.9)")
print(f"first time the distance exceeds 1: t = {tt[np.argmax(dist > 1)]:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(u1[0], u1[2], lw=0.4)
ax[0].set_xlabel("x")
ax[0].set_ylabel("z")
ax[0].set_title("Lorenz attractor (x, z)")
ax[1].semilogy(tt, dist)
ax[1].set_xlabel("t")
ax[1].set_ylabel("distance between solutions")
plt.tight_layout()
plt.savefig("ch04_lorenz.pdf", bbox_inches="tight")
plt.close()
