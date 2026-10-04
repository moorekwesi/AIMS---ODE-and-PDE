# Euler's method for a scalar initial value problem
# Applied ODE & PDE with Python, Ch. 4 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt


def euler(f, t0, y0, T, N):
    """Euler's method y_{n+1} = y_n + h f(t_n, y_n) with N steps on [t0, T]."""
    h = (T - t0) / N
    t = t0 + h * np.arange(N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    for n in range(N):
        y[n + 1] = y[n] + h * f(t[n], y[n])
    return t, y


f = lambda t, y: y - t**2 + 1                       # right-hand side
exact = lambda t: (t + 1)**2 - 0.5 * np.exp(t)      # exact solution, y(0) = 0.5

t, y = euler(f, 0.0, 0.5, 2.0, 10)                  # h = 0.2
print(" n    t_n      y_n        y(t_n)     error")
for n in range(0, 11, 2):
    print(f"{n:2d}  {t[n]:4.1f}  {y[n]:9.6f}  {exact(t[n]):9.6f}  "
          f"{abs(y[n] - exact(t[n])):.2e}")

# figure: slope field, exact solution and Euler polygons
fig, ax = plt.subplots(figsize=(7, 4))
TT, YY = np.meshgrid(np.linspace(0, 2, 17), np.linspace(0, 6, 13))
S = f(TT, YY)
L = np.sqrt(1 + S**2)
ax.quiver(TT, YY, 1 / L, S / L, color="0.7", angles="xy", width=0.002)
ts = np.linspace(0, 2, 200)
ax.plot(ts, exact(ts), "k", lw=2, label="exact solution")
for N, mk in [(4, "o-"), (10, "s-")]:
    tN, yN = euler(f, 0.0, 0.5, 2.0, N)
    ax.plot(tN, yN, mk, ms=4, label=f"Euler, h = {2 / N:.1f}")
ax.set_xlabel("t")
ax.set_ylabel("y")
ax.set_ylim(0, 6)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("ch04_euler_geometry.pdf", bbox_inches="tight")
plt.close()
