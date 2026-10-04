# Convergence of FTCS, BTCS and Crank-Nicolson for the heat equation
# Applied ODE & PDE with Python, Ch. 10 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu

T = 0.1
exact = lambda x, t: np.exp(-np.pi**2 * t) * np.sin(np.pi * x)   # kappa = 1


def theta_method(theta, N, dt):
    """Solve u_t = u_xx, u(0)=u(1)=0, u(x,0)=sin(pi x) to time T; return max error."""
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N + 1)[1:-1]
    D2 = sps.diags([1, -2, 1], [-1, 0, 1], shape=(N - 1, N - 1)) / h**2
    I = sps.identity(N - 1)
    nsteps = int(round(T / dt)); dt = T / nsteps
    B = (I + (1 - theta) * dt * D2).tocsr()
    if theta == 0:
        step = lambda U: B @ U                   # explicit: no linear solve
    else:
        lu = splu((I - theta * dt * D2).tocsc())
        step = lambda U: lu.solve(B @ U)
    U = exact(x, 0.0)
    for n in range(nsteps):
        U = step(U)
    return np.max(np.abs(U - exact(x, T))), nsteps


print("   N   | FTCS (r=0.4)       | BTCS (dt=h)        | CN (dt=h)")
print("       | steps  error  rate | steps  error  rate | steps  error  rate")
prev = None
for N in (10, 20, 40, 80, 160, 320):
    h = 1.0 / N
    res = [theta_method(0.0, N, 0.4 * h**2), theta_method(1.0, N, h),
           theta_method(0.5, N, h)]
    line = f"{N:5d}  "
    for k, (e, ns) in enumerate(res):
        rate = "  -- " if prev is None else f"{np.log2(prev[k] / e):5.2f}"
        line += f"|{ns:6d} {e:.1e} {rate} "
    print(line)
    prev = [e for e, _ in res]
