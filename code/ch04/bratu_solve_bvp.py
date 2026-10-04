# The Bratu problem with scipy.integrate.solve_bvp
# Applied ODE & PDE with Python, Ch. 4 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.optimize import brentq

lam = 1.0                                   # y'' + lam exp(y) = 0, y(0) = y(1) = 0


def rhs(x, Y):                              # Y = (y, y'); x is a vector of mesh points
    return np.vstack([Y[1], -lam * np.exp(Y[0])])


def bc(Ya, Yb):                             # residuals of the boundary conditions
    return np.array([Ya[0], Yb[0]])


def exact(x, th):
    return -2 * np.log(np.cosh((x - 0.5) * th / 2) / np.cosh(th / 4))


# the exact solutions: theta = sqrt(2 lam) cosh(theta/4) has two roots
g = lambda th: th - np.sqrt(2 * lam) * np.cosh(th / 4)
thetas = [brentq(g, 0.1, 4.0), brentq(g, 4.0, 20.0)]

x = np.linspace(0, 1, 11)
plt.figure(figsize=(6.5, 3.6))
for amp, th in zip([0.1, 3.0], thetas):     # two initial guesses -> two solutions
    guess = np.vstack([amp * 4 * x * (1 - x), amp * 4 * (1 - 2 * x)])
    sol = solve_bvp(rhs, bc, x, guess, tol=1e-8, max_nodes=10000)
    xx = np.linspace(0, 1, 201)
    err = np.max(abs(sol.sol(xx)[0] - exact(xx, th)))
    print(f"guess amplitude {amp}: status {sol.status}, mesh points {sol.x.size}, "
          f"max y = {sol.sol(0.5)[0]:.6f}, error {err:.1e}")
    plt.plot(xx, sol.sol(xx)[0], label=f"solution with y(1/2) = {sol.sol(0.5)[0]:.3f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("ch04_bratu.pdf", bbox_inches="tight")
plt.close()
