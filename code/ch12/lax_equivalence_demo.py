# Consistent but unstable: FTCS for u_t = u_xx under grid refinement
# Applied ODE & PDE with Python, Ch. 12 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np


def ftcs(N, r, T=0.1):
    """FTCS with dt = r h^2 for u_t = u_xx, u(0,t) = u(1,t) = 0, u(x,0) = sin(pi x).
    Returns the max error at time T."""
    h = 1.0 / N
    dt = r * h**2
    nsteps = int(round(T / dt))
    x = np.linspace(0, 1, N + 1)
    U = np.sin(np.pi * x)
    for n in range(nsteps):
        U[1:-1] = U[1:-1] + r * (U[2:] - 2 * U[1:-1] + U[:-2])
    exact = np.exp(-np.pi**2 * nsteps * dt) * np.sin(np.pi * x)
    return np.max(np.abs(U - exact)), nsteps


print(f"{'N':>4} {'steps':>6} {'error, r = 0.4':>15} {'steps':>6} {'error, r = 0.6':>15}")
for N in [10, 20, 40, 80]:
    e1, n1 = ftcs(N, 0.4)
    e2, n2 = ftcs(N, 0.6)
    print(f"{N:4d} {n1:6d} {e1:15.3e} {n2:6d} {e2:15.3e}")
