# Energy identity and maximum principle for a heat series solution
# Applied ODE & PDE with Python, Ch. 8 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
from scipy.integrate import trapezoid

# u_t = u_xx on (0,1), u = 0 at x = 0, 1, u(x,0) = sin(pi x) - 0.8 sin(3 pi x)
b = {1: 1.0, 3: -0.8}
x = np.linspace(0, 1, 2001)


def u(t):
    return sum(c * np.exp(-(n * np.pi) ** 2 * t) * np.sin(n * np.pi * x)
               for n, c in b.items())


def ux(t):
    return sum(c * n * np.pi * np.exp(-(n * np.pi) ** 2 * t) * np.cos(n * np.pi * x)
               for n, c in b.items())


E = lambda t: 0.5 * trapezoid(u(t) ** 2, x)          # E(t) = 1/2 int u^2
print("    t       E(t)        dE/dt (FD)    -int u_x^2")
for t in [0.0, 0.01, 0.05, 0.1]:
    d = 1e-6
    dE = (E(t + d) - E(max(t - d, 0))) / (t + d - max(t - d, 0))
    print(f"{t:5.2f}  {E(t):.8f}  {dE: .6e}  {-trapezoid(ux(t) ** 2, x): .6e}")

ts = np.linspace(0, 0.5, 501)
U = np.array([u(t) for t in ts])
i, j = np.unravel_index(np.argmax(U), U.shape)
k, l = np.unravel_index(np.argmin(U), U.shape)
print(f"\nmax u = {U[i, j]:.6f} at (x, t) = ({x[j]:.4f}, {ts[i]:.3f})")
print(f"min u = {U[k, l]:.6f} at (x, t) = ({x[l]:.4f}, {ts[k]:.3f})")
