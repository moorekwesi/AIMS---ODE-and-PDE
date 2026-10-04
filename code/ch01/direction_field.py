# Direction fields, isoclines and solution curves
# Applied ODE & PDE with Python, Ch. 1 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def direction_field(ax, f, tlim, ylim, n=21):
    """Draw short unit segments of slope f(t, y) on an n x n grid."""
    T, Y = np.meshgrid(np.linspace(*tlim, n), np.linspace(*ylim, n))
    S = f(T, Y)
    L = np.sqrt(1 + S**2)                     # normalise (1, S) to unit length
    ax.quiver(T, Y, 1/L, S/L, angles="xy", color="gray", width=0.003,
              headwidth=0, headlength=0, headaxislength=0, pivot="middle")
    ax.set_xlim(tlim); ax.set_ylim(ylim); ax.set_xlabel("t"); ax.set_ylabel("y")

def escape(t, y):                             # stop when the solution leaves the window
    return abs(y[0]) - 4.0
escape.terminal = True

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))

# (a) y' = t - y, isoclines y = t - m, exact solution y = t - 1 + C e^{-t}
f1 = lambda t, y: t - y
direction_field(ax[0], f1, (-2, 3), (-3, 3))
tt = np.linspace(-2, 3, 200)
for m in (-1, 0, 1, 2):
    ax[0].plot(tt, tt - m, ":", color="tab:red", lw=1)
for C in (-2, -1, 0, 0.5, 1):
    ax[0].plot(tt, tt - 1 + C*np.exp(-tt), lw=1.6)
ax[0].set_title("y' = t - y   (dotted: isoclines)")

# (b) y' = y^2 - t, no elementary solution formula: integrate numerically
f2 = lambda t, y: y**2 - t
direction_field(ax[1], f2, (-2, 4), (-3, 3))
for y0 in np.linspace(-3, 1.5, 10):
    sol = solve_ivp(lambda t, y: f2(t, y), (-2, 4), [y0], events=escape,
                    max_step=0.02, rtol=1e-8)
    ax[1].plot(sol.t, sol.y[0], lw=1.4)
ax[1].set_title("y' = y^2 - t")
plt.tight_layout()
plt.savefig("ch01_direction_fields.pdf", bbox_inches="tight")
plt.close()
