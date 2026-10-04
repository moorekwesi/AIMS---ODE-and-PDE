# Heat equation with a source term (AIMS Senegal 2024 project)
# Applied ODE & PDE with Python, Ch. 10 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt

kappa, T = 1.0 / 6.0, 0.25
f = lambda x: x * (1 - x) * (10 - 22 * x)
g = lambda x: np.where(x <= 1/3, 0.5 - 3 * np.abs(x - 1/6),
                       np.where(x < 2/3, 0.0, 0.5 - 3 * np.abs(x - 5/6)))
us = lambda x: 0.6 * x - 10 * x**3 + 16 * x**4 - 6.6 * x**5   # kappa u'' = -f

# analytic solution u = us + sum b_n exp(-kappa n^2 pi^2 t) sin(n pi x)
xq = np.linspace(0, 1, 60001); nn = np.arange(1, 401)[:, None]
bn = 2 * np.trapz((g(xq) - us(xq)) * np.sin(nn * np.pi * xq), xq, axis=1)


def exact(x, t):
    decay = np.exp(-kappa * (nn[:, 0] * np.pi)**2 * t)
    return us(x) + (bn * decay) @ np.sin(nn * np.pi * x)


def solve(N, dt, theta, T=T):
    """theta = 0: explicit FTCS;  theta = 1/2: Crank-Nicolson."""
    h = 1.0 / N; x = np.linspace(0, 1, N + 1)
    nsteps = max(1, int(np.ceil(T / dt - 1e-9))); dt = T / nsteps
    D2 = kappa * sps.diags([1, -2, 1], [-1, 0, 1], shape=(N - 1, N - 1), format="csr") / h**2
    I = sps.identity(N - 1, format="csr"); B = (I + (1 - theta) * dt * D2).tocsr()
    lu = splu((I - theta * dt * D2).tocsc()) if theta > 0 else None
    U, F = g(x[1:-1]), dt * f(x[1:-1])
    for n in range(nsteps):
        rhs = B @ U + F
        U = rhs if lu is None else lu.solve(rhs)
    return x, np.r_[0, U, 0], nsteps


print(" n     h      FTCS steps  max error  |  CN steps  max error")
for n in range(1, 8):
    N = 2**n; h = 1.0 / N
    x, U1, s1 = solve(N, 0.4 * h**2 / kappa, 0.0)
    x, U2, s2 = solve(N, h / 10, 0.5)
    e1, e2 = (np.max(np.abs(U - exact(x, T))) for U in (U1, U2))
    print(f"{n:2d}  {h:.5f}  {s1:8d}   {e1:.3e}  | {s2:6d}    {e2:.3e}")

xf = np.linspace(0, 1, 400)
plt.figure(figsize=(7, 4))
for t in (0.0, 0.02, 0.1, 0.3, 1.0):
    l, = plt.plot(xf, exact(xf, t), label=f"t = {t}")
    if t > 0:
        x, U, _ = solve(32, 1 / 320, 0.5, T=t)
        plt.plot(x, U, "o", ms=3, color=l.get_color())
plt.plot(xf, us(xf), "k--", label="steady state")
plt.xlabel("x"); plt.ylabel("u(x,t)"); plt.legend(fontsize=9)
plt.tight_layout(); plt.savefig("ch10_source_aims.pdf", bbox_inches="tight")
plt.close()
