# Sample project: explicit scheme for u_t = eps u_xx + u_x and its convergence
# Applied ODE & PDE with Python, Ch. 13 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

EPS, T = 0.1, 0.5


def reference(x, t, eps=EPS, K=60):
    """Series solution via u = exp(-x/(2eps) - t/(4eps)) v, where v_t = eps v_xx."""
    xg, wg = np.polynomial.legendre.leggauss(200)        # Gauss-Legendre on [0, 1]
    xg, wg = 0.5 * (xg + 1), 0.5 * wg
    v = np.zeros_like(x)
    for k in range(1, K + 1):
        bk = 2 * np.sum(wg * np.exp(xg / (2 * eps)) * xg * (1 - xg) * np.sin(k * np.pi * xg))
        v += bk * np.exp(-eps * (k * np.pi)**2 * t) * np.sin(k * np.pi * x)
    return np.exp(-x / (2 * eps) - t / (4 * eps)) * v


def ftcs(N, r=0.4, eps=EPS, T=T):
    """Forward Euler in time, central differences in space; dt <= r h^2 / eps."""
    h = 1.0 / N
    x = np.linspace(0, 1, N + 1)
    M = int(np.ceil(T / (r * h**2 / eps)))       # number of time steps
    dt = T / M
    lam, mu = eps * dt / h**2, dt / (2 * h)
    U = x * (1 - x)
    for n in range(M):
        U[1:-1] = U[1:-1] + lam * (U[2:] - 2 * U[1:-1] + U[:-2]) + mu * (U[2:] - U[:-2])
    return x, U, dt


print(f"eps = {EPS}, T = {T}, dt = 0.4 h^2/eps (rounded so that T/dt is an integer)")
print(" n     h        dt        max error   ratio   order")
prev = None
for n in range(1, 8):
    x, U, dt = ftcs(2**n)
    err = np.max(np.abs(U - reference(x, T)))
    if prev is None:
        print(f"{n:2d}  {1 / 2**n:.5f}  {dt:.3e}  {err:.3e}")
    else:
        print(f"{n:2d}  {1 / 2**n:.5f}  {dt:.3e}  {err:.3e}  {prev / err:6.2f}  "
              f"{np.log2(prev / err):5.2f}")
    prev = err
xf = np.linspace(0, 1, 401)
print(f"max of reference solution at T: {np.max(reference(xf, T)):.6f}")

plt.figure(figsize=(6.5, 3.6))
for t in [0.0, 0.1, 0.25, 0.5]:
    line, = plt.plot(xf, reference(xf, t), label=f"t = {t}")
    if t > 0:
        x, U, dt = ftcs(16, T=t)
        plt.plot(x, U, "o", ms=3, color=line.get_color())
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.tight_layout()
plt.savefig("ch13_cd_profiles.pdf", bbox_inches="tight")
plt.close()
