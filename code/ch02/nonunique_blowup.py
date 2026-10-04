# Non-uniqueness for y' = y^(1/3) and blow-up for y' = y^2
# Applied ODE & PDE with Python, Ch. 2 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

# (a) y' = y^(1/3), y(0) = 0: y = 0 and y = (2(t-c)/3)^(3/2) for t > c
t = np.linspace(0, 3, 400)
ax1.plot(t, 0 * t, "k", lw=2, label="y = 0")
for c in (0.0, 0.5, 1.0, 1.5):
    y = np.where(t > c, (2 * np.maximum(t - c, 0) / 3) ** 1.5, 0.0)
    ax1.plot(t, y, label=f"c = {c}")
    ax1.plot(t, -y, color=ax1.lines[-1].get_color(), ls="--")
ax1.set_title("y' = y^(1/3), y(0) = 0")
ax1.set_xlabel("t")
ax1.legend(fontsize=8)

# (b) y' = y^2, y(0) = y0 > 0: y = y0/(1 - y0 t) blows up at t* = 1/y0
for y0 in (0.5, 1.0, 2.0):
    ts = np.linspace(0, 1 / y0 - 0.02, 300)
    ax2.plot(ts, y0 / (1 - y0 * ts), label=f"y0 = {y0}, t* = {1/y0:.1f}")
ax2.set_ylim(0, 30)
ax2.set_xlim(0, 2.2)
ax2.set_title("y' = y^2: blow-up in finite time")
ax2.set_xlabel("t")
ax2.legend(fontsize=8)
plt.tight_layout()
plt.savefig("ch02_nonunique_blowup.pdf", bbox_inches="tight")
plt.close()

# what does a numerical solver do near the blow-up time t* = 1?
sol = solve_ivp(lambda t, y: y**2, (0, 2), [1.0], rtol=1e-8, atol=1e-10)
print("solver status :", sol.status)
print("message       :", sol.message)
print(f"last time     : {sol.t[-1]:.10f}")
print(f"last value    : {sol.y[0, -1]:.4e}")
