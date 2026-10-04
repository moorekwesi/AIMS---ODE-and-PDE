# Predator-prey dynamics: the Lotka-Volterra system
# Applied ODE & PDE with Python, Ch. 1 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

a, b, c, d = 1.0, 0.5, 0.75, 0.25      # prey growth, predation, predator death, conversion

def lv(t, z):
    """x' = a x - b x y (prey),  y' = -c y + d x y (predators)."""
    x, y = z
    return [a*x - b*x*y, -c*y + d*x*y]

def V(x, y):
    """Conserved quantity of the Lotka-Volterra system."""
    return d*x - c*np.log(x) + b*y - a*np.log(y)

t = np.linspace(0, 30, 3001)
sol = solve_ivp(lv, (0, 30), [5.0, 1.0], t_eval=t, rtol=1e-10, atol=1e-12)
x, y = sol.y
Vt = V(x, y)
print(f"equilibrium (c/d, a/b) = ({c/d:.2f}, {a/b:.2f})")
print(f"prey     : min {x.min():.3f}, max {x.max():.3f}")
print(f"predator : min {y.min():.3f}, max {y.max():.3f}")
print(f"variation of V along the orbit = {Vt.max() - Vt.min():.2e}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(t, x, label="prey x(t)"); ax[0].plot(t, y, "--", label="predators y(t)")
ax[0].set_xlabel("t"); ax[0].legend(); ax[0].grid(alpha=0.3)
for x0 in (3.5, 5.0, 7.0):
    s = solve_ivp(lv, (0, 15), [x0, 1.0], rtol=1e-10, atol=1e-12, max_step=0.01)
    ax[1].plot(s.y[0], s.y[1])
ax[1].plot(c/d, a/b, "ko")
ax[1].set_xlabel("prey x"); ax[1].set_ylabel("predators y"); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ch01_lotka_volterra.pdf", bbox_inches="tight")
plt.close()
