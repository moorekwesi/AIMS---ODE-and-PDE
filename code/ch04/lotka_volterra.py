# Lotka-Volterra predator-prey model and its conserved quantity
# Applied ODE & PDE with Python, Ch. 4 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

a, b, c, d = 1.0, 0.1, 1.5, 0.075           # prey growth, predation, predator death, conversion


def lv(t, z):
    x, y = z                                 # x = prey, y = predators
    return np.array([a * x - b * x * y, -c * y + d * x * y])


V = lambda x, y: d * x - c * np.log(x) + b * y - a * np.log(y)   # first integral
z0, T = np.array([10.0, 5.0]), 30.0

# Euler with h = 0.01 versus solve_ivp (RK45) with two tolerances
h, z = 0.01, z0.copy()
for n in range(int(round(T / h))):
    z = z + h * lv(n * h, z)
print("drift |V(T) - V(0)| of the conserved quantity on [0, 30]:")
print(f"  Euler, h = 0.01           : {abs(V(*z) - V(*z0)):.3e}")
for rtol in [1e-3, 1e-9]:
    s = solve_ivp(lv, (0, T), z0, rtol=rtol, atol=1e-3 * rtol)
    print(f"  RK45, rtol = {rtol:.0e} ({s.t.size - 1:4d} steps): "
          f"{abs(V(*s.y[:, -1]) - V(*z0)):.3e}")
print(f"equilibrium (c/d, a/b) = ({c / d:.1f}, {a / b:.1f})")

s = solve_ivp(lv, (0, T), z0, rtol=1e-9, atol=1e-12, dense_output=True)
tt = np.linspace(0, T, 1500)
x, y = s.sol(tt)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(tt, x, label="prey x")
ax[0].plot(tt, y, label="predators y")
ax[0].set_xlabel("t")
ax[0].legend()
for x0 in [6.0, 10.0, 14.0]:
    xs, ys = solve_ivp(lv, (0, 12), [x0, 5.0], rtol=1e-9, atol=1e-12,
                       t_eval=np.linspace(0, 12, 1500)).y
    ax[1].plot(xs, ys)
ax[1].plot(c / d, a / b, "k*", ms=10)
ax[1].set_xlabel("prey x")
ax[1].set_ylabel("predators y")
plt.tight_layout()
plt.savefig("ch04_lotka_volterra.pdf", bbox_inches="tight")
plt.close()
