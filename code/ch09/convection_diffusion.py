# Convection-diffusion -eps u'' + u' = 1: central versus upwind differences
# Applied ODE & PDE with Python, Ch. 9 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

eps = 0.01


def exact(x):
    """Exact solution with a boundary layer of width ~eps at x = 1."""
    return x - (np.exp((x - 1) / eps) - np.exp(-1 / eps)) / (1 - np.exp(-1 / eps))


def solve_cd(N, scheme):
    """u(0) = u(1) = 0. Central: u' ~ (U_{j+1}-U_{j-1})/(2h);
    upwind: u' ~ (U_j - U_{j-1})/h  (the flow comes from the left)."""
    h = 1.0 / N
    d = eps / h**2
    if scheme == "central":
        lo, di, up = -d - 1 / (2 * h), 2 * d, -d + 1 / (2 * h)
    else:
        lo, di, up = -d - 1 / h, 2 * d + 1 / h, -d
    A = sps.diags([lo, di, up], [-1, 0, 1], shape=(N - 1, N - 1), format="csr")
    U = spsolve(A, np.ones(N - 1))
    return np.linspace(0, 1, N + 1), np.concatenate(([0], U, [0]))


print(f"eps = {eps}")
print(f"{'N':>5} {'Pe = h/(2 eps)':>15} {'central err':>12} {'upwind err':>11}")
for N in [10, 20, 40, 80, 160, 320, 640]:
    err = {}
    for s in ["central", "upwind"]:
        x, U = solve_cd(N, s)
        err[s] = np.max(np.abs(U - exact(x)))
    print(f"{N:5d} {1 / (2 * N * eps):15.3f} {err['central']:12.3e} "
          f"{err['upwind']:11.3e}")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
xf = np.linspace(0, 1, 1000)
for a, N in zip(ax, [20, 100]):
    a.plot(xf, exact(xf), "k-", lw=1, label="exact")
    for s, m in [("central", "o--"), ("upwind", "s:")]:
        x, U = solve_cd(N, s)
        a.plot(x, U, m, ms=3, label=s)
    a.set_title(f"N = {N},  Pe = {1 / (2 * N * eps):.2f}")
    a.set_xlabel("x")
ax[0].legend(loc="upper left")
plt.tight_layout()
plt.savefig("ch09_convection_diffusion.pdf", bbox_inches="tight")
plt.close()
