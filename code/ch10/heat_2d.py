# The heat equation on the unit square: FTCS and Crank-Nicolson with Kronecker products
# Applied ODE & PDE with Python, Ch. 10 | (c) 2026 Stephen E. Moore | MIT Licence
import time
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt

T = 1 / 16
exact = lambda X, Y, t: np.exp(-2 * np.pi**2 * t) * np.sin(np.pi * X) * np.sin(np.pi * Y)


def grid(N):
    x = np.linspace(0, 1, N + 1)
    return np.meshgrid(x, x, indexing="ij")          # U[i, j] ~ u(x_i, y_j)


def ftcs_2d(N, r=0.2):
    h = 1.0 / N; X, Y = grid(N)
    nsteps = int(np.ceil(T / (r * h**2))); r = T / nsteps / h**2
    U = exact(X, Y, 0.0)
    for n in range(nsteps):                          # boundary values stay 0
        U[1:-1, 1:-1] += r * (U[2:, 1:-1] + U[:-2, 1:-1] + U[1:-1, 2:]
                              + U[1:-1, :-2] - 4 * U[1:-1, 1:-1])
    return np.max(np.abs(U - exact(X, Y, T))), nsteps


def laplacian_2d(N):
    """5-point Laplacian on the (N-1)^2 interior nodes via Kronecker products."""
    h = 1.0 / N
    D = sps.diags([1, -2, 1], [-1, 0, 1], shape=(N - 1, N - 1)) / h**2
    I = sps.identity(N - 1)
    return (sps.kron(D, I) + sps.kron(I, D)).tocsc()


def cn_2d(N):
    h = 1.0 / N; dt = h / 8; nsteps = int(np.ceil(T / dt)); dt = T / nsteps
    X, Y = grid(N); L = laplacian_2d(N); I = sps.identity(L.shape[0])
    lu = splu((I - 0.5 * dt * L).tocsc()); B = (I + 0.5 * dt * L).tocsr()
    U = exact(X, Y, 0.0)[1:-1, 1:-1].ravel()         # row-major = kron(D,I)+kron(I,D)
    for n in range(nsteps):
        U = lu.solve(B @ U)
    return np.max(np.abs(U - exact(X, Y, T)[1:-1, 1:-1].ravel())), nsteps


print("   N    FTCS steps  error     rate  sec  |  CN steps  error     rate  sec")
hs, errs, prev = [], {"FTCS": [], "CN": []}, None
for N in (8, 16, 32, 64, 128):
    out = []
    for name, solver in (("FTCS", ftcs_2d), ("CN", cn_2d)):
        t0 = time.perf_counter(); e, ns = solver(N); sec = time.perf_counter() - t0
        errs[name].append(e); out.append((ns, e, sec))
    hs.append(1.0 / N)
    rates = ["  -- " if prev is None else f"{np.log2(p / o[1]):5.2f}"
             for p, o in zip(prev or [0, 0], out)]
    print(f"{N:5d}  " + "  |  ".join(f"{o[0]:6d}   {o[1]:.2e} {rt} {o[2]:4.1f}"
                                     for o, rt in zip(out, rates)))
    prev = [o[1] for o in out]

hs = np.array(hs)
plt.figure(figsize=(6, 4))
plt.loglog(hs, errs["FTCS"], "o-", label="FTCS, r = 0.2")
plt.loglog(hs, errs["CN"], "s-", label="Crank-Nicolson, dt = h/8")
plt.loglog(hs, 0.3 * hs**2, "k--", label="slope 2")
plt.xlabel("h"); plt.ylabel("max error at T = 1 / 16"); plt.legend()
plt.tight_layout(); plt.savefig("ch10_heat2d_error.pdf", bbox_inches="tight")
plt.close()
