# Poisson's integral formula for the unit disc
# Applied ODE & PDE with Python, Ch. 8 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt


def poisson(g, r, theta, M=512):
    """u(r,theta) = (1/2pi) int P_r(theta - phi) g(phi) dphi, trapezoid rule, M nodes."""
    phi = 2 * np.pi * np.arange(M) / M
    P = (1 - r**2) / (1 - 2 * r * np.cos(theta - phi) + r**2)
    return np.mean(P * g(phi))


g = lambda p: (p % (2 * np.pi) < np.pi).astype(float)    # 1 on top half, 0 below
exact = lambda r, th: 0.5 + np.arctan(2 * r * np.sin(th) / (1 - r**2)) / np.pi

print("  r     theta    M=64 error    M=512 error   M=4096 error   exact u")
for r, th in [(0.0, 0.0), (0.5, 1.0), (0.9, 2.0), (0.99, -0.5)]:
    errs = [abs(poisson(g, r, th, M) - exact(r, th)) for M in (64, 512, 4096)]
    print(f"{r:4.2f}  {th:5.2f}    " + "    ".join(f"{e:.2e}" for e in errs)
          + f"     {exact(r, th):.6f}")

g2 = lambda p: np.cos(p) ** 2            # smooth data: u = 1/2 + r^2 cos(2 theta)/2
u2 = 0.5 + 0.5 * 0.9**2 * np.cos(2 * 0.3)
print("\nsmooth data g = cos^2, r = 0.9, theta = 0.3:")
for M in [16, 64, 128, 256]:
    print(f"   M = {M:3d}: error = {abs(poisson(g2, 0.9, 0.3, M) - u2):.1e}"
          f"   (r^M = {0.9**M:.1e})")

r, th = np.meshgrid(np.linspace(0, 0.999, 120), np.linspace(0, 2 * np.pi, 241))
X, Y = r * np.cos(th), r * np.sin(th)
fig, ax = plt.subplots(figsize=(5.2, 4.2))
cs = ax.contourf(X, Y, exact(r, th), levels=np.linspace(0, 1, 21), cmap="coolwarm")
for coll in cs.collections:
    coll.set_rasterized(True)               # avoids white seams in PDF viewers
ax.contour(X, Y, exact(r, th), levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors="k", linewidths=0.6)
fig.colorbar(cs, ax=ax, ticks=np.linspace(0, 1, 6))
ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
plt.tight_layout()
plt.savefig("ch08_poisson_disc.pdf", bbox_inches="tight", dpi=200)
