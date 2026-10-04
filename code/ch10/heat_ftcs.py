# The explicit FTCS scheme for the heat equation
# Applied ODE & PDE with Python, Ch. 10 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

kappa = 1.0


def exact(x, t, nterms=200):
    """Fourier series solution for u(x,0) = x(1-x): only odd sine modes."""
    u = np.zeros_like(x)
    for n in range(1, 2 * nterms, 2):
        bn = 8.0 / (n * np.pi)**3
        u += bn * np.exp(-kappa * (n * np.pi)**2 * t) * np.sin(n * np.pi * x)
    return u


def ftcs(u0, h, dt, nsteps):
    """Advance U^{n+1}_j = U^n_j + r (U^n_{j+1} - 2U^n_j + U^n_{j-1}), Dirichlet 0."""
    r = kappa * dt / h**2
    U = u0.copy()
    for n in range(nsteps):
        U[1:-1] = U[1:-1] + r * (U[2:] - 2 * U[1:-1] + U[:-2])
        U[0] = U[-1] = 0.0
    return U


T, r = 0.1, 0.4
print("  N      h        dt     steps   max error   ratio")
prev = None
for N in (10, 20, 40, 80):
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N + 1)
    nsteps = int(round(T / (r * h**2 / kappa)))
    dt = T / nsteps
    U = ftcs(x * (1 - x), h, dt, nsteps)
    err = np.max(np.abs(U - exact(x, T)))
    ratio = "" if prev is None else f"{prev / err:6.2f}"
    print(f"{N:3d}  {h:.4f}  {dt:.2e}  {nsteps:5d}   {err:.3e}   {ratio}")
    prev = err

# snapshots for N = 20
N = 20; h = 1.0 / N; dt = r * h**2 / kappa
x = np.linspace(0.0, 1.0, N + 1); xf = np.linspace(0, 1, 401)
plt.figure(figsize=(7, 4))
for t in (0.0, 0.02, 0.05, 0.1, 0.2):
    n = int(round(t / dt))
    plt.plot(xf, exact(xf, n * dt), "k-", lw=0.8)
    plt.plot(x, ftcs(x * (1 - x), h, dt, n), "o", ms=4, label=f"t = {t:.2f}")
plt.xlabel("x"); plt.ylabel("u(x,t)"); plt.legend()
plt.title("FTCS (dots, N = 20, r = 0.4) and exact solution (lines)")
plt.tight_layout(); plt.savefig("ch10_ftcs_snapshots.pdf", bbox_inches="tight")
plt.close()
