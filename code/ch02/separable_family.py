# Integral curves of a separable equation and where solutions end
# Applied ODE & PDE with Python, Ch. 2 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

# y' = (1 + 3x^2)/(2y)  <=>  y^2 - x - x^3 = C  (level curves of G)
x = np.linspace(-1.6, 1.6, 400)
yv = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, yv)
G = Y**2 - X - X**3

fig, ax = plt.subplots(figsize=(7, 4.2))
# direction field (unit vectors) on a coarse grid
xs, ys = np.meshgrid(np.linspace(-1.5, 1.5, 21), np.linspace(-2.8, 2.8, 16))
S = (1 + 3 * xs**2) / (2 * ys)
L = np.sqrt(1 + S**2)
ax.quiver(xs, ys, 1 / L, S / L, angles="xy", color="0.7", width=0.003)
ax.contour(X, Y, G, levels=np.arange(-2, 4.5, 0.5), colors="C0", linewidths=1)
# the solution through (0,1): C = 1, upper branch only
xx = np.linspace(-0.6823, 1.6, 400)
ax.plot(xx, np.sqrt(np.maximum(1 + xx + xx**3, 0)), "C3", lw=2.5,
        label="solution with y(0)=1")
ax.axhline(0, color="k", lw=0.8)
ax.plot([-0.6823], [0], "ko")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_ylim(-3, 3)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("ch02_separable_family.pdf", bbox_inches="tight")
plt.close()
