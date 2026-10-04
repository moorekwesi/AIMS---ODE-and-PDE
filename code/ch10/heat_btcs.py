# BTCS: the Thomas algorithm, solve_banded and a sparse LU factorised once
# Applied ODE & PDE with Python, Ch. 10 | (c) 2026 Stephen E. Moore | MIT Licence
import time
import numpy as np
import scipy.sparse as sps
from scipy.linalg import solve_banded
from scipy.sparse.linalg import splu


def thomas(a, b, c, d):
    """Solve a tridiagonal system. a: sub-diagonal (a[0] unused), b: diagonal,
    c: super-diagonal (c[-1] unused), d: right-hand side. O(n) operations."""
    n = len(d)
    cp, dp = np.empty(n), np.empty(n)
    cp[0], dp[0] = c[0] / b[0], d[0] / b[0]
    for i in range(1, n):                      # forward elimination
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):             # back substitution
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


N, T, nsteps = 1000, 0.1, 200
h, dt = 1.0 / N, T / nsteps
r = dt / h**2                                  # kappa = 1
x = np.linspace(0.0, 1.0, N + 1)[1:-1]
u0 = x * (1 - x)
n = N - 1
a = np.full(n, -r); b = np.full(n, 1 + 2 * r); c = np.full(n, -r)
ab = np.vstack([np.r_[0, c[:-1]], b, np.r_[a[1:], 0]])   # banded storage
A = sps.diags([a[1:], b, c[:-1]], [-1, 0, 1], format="csc")


def run(solve, steps=nsteps):
    """Take `steps` BTCS steps; return U and the time per step in milliseconds."""
    U = u0.copy(); t0 = time.perf_counter()
    for k in range(steps):
        U = solve(U)
    return U, 1000 * (time.perf_counter() - t0) / steps


lu = splu(A)                                   # factorise ONCE
Ad = A.toarray()
methods = {"Thomas (Python loops)": lambda U: thomas(a, b, c, U),
           "solve_banded": lambda U: solve_banded((1, 1), ab, U),
           "sparse LU, factorised once": lambda U: lu.solve(U),
           "dense np.linalg.solve": lambda U: np.linalg.solve(Ad, U)}
print(f"BTCS with N = {N}, {nsteps} steps, r = {r:.0f}")
U10, _ = run(lu.solve, 10)
print("  method                       ms per step   diff after 10 steps")
for name, solve in methods.items():
    U, ms = run(solve, 10)
    print(f"  {name:28s} {ms:9.3f}     {np.max(abs(U - U10)):.1e}")
Uref, _ = run(lu.solve)                        # all 200 steps
uex = sum(8 / (k * np.pi)**3 * np.exp(-(k * np.pi)**2 * T) * np.sin(k * np.pi * x)
          for k in range(1, 400, 2))
print(f"max error at T = {T}: {np.max(abs(Uref - uex)):.3e}  (stable although r >> 1/2)")
