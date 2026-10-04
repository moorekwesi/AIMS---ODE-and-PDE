# Laplace's equation on a square: u(x,0) = sin^3 x
# Applied ODE & PDE with Python, Ch. 8 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

pi = np.pi


def u(x, y):
    """Harmonic on (0,pi)^2, u = sin^3 x on y = 0 and u = 0 on the other sides."""
    # sinh(n(pi-y))/sinh(n pi) written with exponentials to avoid overflow
    r = lambda n: (np.exp(-n * y) - np.exp(-n * (2 * pi - y))) / (1 - np.exp(-2 * n * pi))
    return 0.75 * np.sin(x) * r(1) - 0.25 * np.sin(3 * x) * r(3)


X, Y = np.meshgrid(np.linspace(0, pi, 201), np.linspace(0, pi, 201))
U = u(X, Y)
h = X[0, 1] - X[0, 0]
lapU = (U[1:-1, 2:] + U[1:-1, :-2] + U[2:, 1:-1] + U[:-2, 1:-1]
        - 4 * U[1:-1, 1:-1]) / h**2
print(f"u(pi/2, pi/2)        = {u(pi / 2, pi / 2):.8f}")
print(f"bottom edge error    = {np.max(np.abs(U[0] - np.sin(X[0])**3)):.2e}")
print(f"max |5-point Lap u|  = {np.max(np.abs(lapU)):.2e}   (h = {h:.4f})")
print(f"max u on the square  = {U.max():.6f}   attained at y = {Y.flat[U.argmax()]:.3f}")
# mean value property on the circle of radius 0.5 about the centre
th = np.linspace(0, 2 * pi, 64, endpoint=False)
mean = np.mean(u(pi / 2 + 0.5 * np.cos(th), pi / 2 + 0.5 * np.sin(th)))
print(f"mean over circle r=0.5 = {mean:.8f}")

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(1, 2, 1)
cs = ax1.contourf(X, Y, U, levels=20, cmap="viridis")
for coll in cs.collections:
    coll.set_rasterized(True)               # avoids white seams in PDF viewers
fig.colorbar(cs, ax=ax1); ax1.set_xlabel("$x$"); ax1.set_ylabel("$y$")
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(X, Y, U, cmap="viridis", rstride=4, cstride=4, linewidth=0,
                 rasterized=True)
ax2.set_xlabel("$x$"); ax2.set_ylabel("$y$"); ax2.view_init(25, -60)
plt.tight_layout()
plt.savefig("ch08_laplace_rect.pdf", bbox_inches="tight", dpi=200)
