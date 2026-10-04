# Peaceman-Rachford ADI for the 2D heat equation with a source term
# Applied ODE & PDE with Python, Ch. 10 | (c) 2026 Stephen E. Moore | MIT Licence
import time
import numpy as np
from scipy.linalg import solve_banded

T = 0.5
S = lambda X, Y: np.sin(np.pi * X) * np.sin(np.pi * Y)
exact = lambda X, Y, t: (1 - np.exp(-t)) * S(X, Y)
source = lambda X, Y, t: (np.exp(-t) + 2 * np.pi**2 * (1 - np.exp(-t))) * S(X, Y)


def adi(N, dt):
    """u_t = u_xx + u_yy + f on (0,1)^2, u = 0 on the boundary and at t = 0."""
    h = 1.0 / N; m = N - 1; s = 0.5 * dt / h**2
    x = np.linspace(0, 1, N + 1)[1:-1]
    X, Y = np.meshgrid(x, x, indexing="ij")
    ab = np.zeros((3, m)); ab[0, 1:] = -s; ab[1, :] = 1 + 2 * s; ab[2, :-1] = -s

    def d2(V, axis):                       # (h^2 * second difference) with zero BCs
        P = np.pad(V, 1)
        if axis == 0:
            return P[2:, 1:-1] - 2 * V + P[:-2, 1:-1]
        return P[1:-1, 2:] - 2 * V + P[1:-1, :-2]

    U = np.zeros((m, m)); nsteps = int(round(T / dt)); dt = T / nsteps
    for n in range(nsteps):
        F = 0.5 * dt * source(X, Y, (n + 0.5) * dt)
        Ustar = solve_banded((1, 1), ab, U + s * d2(U, 1) + F)          # implicit in x
        U = solve_banded((1, 1), ab, (Ustar + s * d2(Ustar, 0) + F).T).T  # implicit in y
    return np.max(np.abs(U - exact(X, Y, T))), nsteps


print("    N   steps   max error    rate   seconds")
prev = None
for N in (16, 32, 64, 128, 256):
    t0 = time.perf_counter()
    e, ns = adi(N, dt=1.0 / N)
    rate = "  -- " if prev is None else f"{np.log2(prev / e):5.2f}"
    print(f"{N:5d}  {ns:5d}   {e:.3e}   {rate}   {time.perf_counter() - t0:6.2f}")
    prev = e
