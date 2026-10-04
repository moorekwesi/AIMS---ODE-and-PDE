# Order of accuracy of four advection schemes for smooth periodic data
# Applied ODE & PDE with Python, Ch. 11 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np

R = lambda U, k: np.roll(U, k)                  # R(U, 1)[j] = U[j-1]


def step(U, nu, scheme):
    """One time step of u_t + c u_x = 0 (c > 0) on a periodic grid."""
    if scheme == "upwind":
        return U - nu * (U - R(U, 1))
    if scheme == "LF":
        return 0.5 * (R(U, -1) + R(U, 1)) - 0.5 * nu * (R(U, -1) - R(U, 1))
    if scheme == "LW":
        return (U - 0.5 * nu * (R(U, -1) - R(U, 1))
                + 0.5 * nu**2 * (R(U, -1) - 2 * U + R(U, 1)))
    if scheme == "BW":
        return (U - 0.5 * nu * (3 * U - 4 * R(U, 1) + R(U, 2))
                + 0.5 * nu**2 * (U - 2 * R(U, 1) + R(U, 2)))


def error(N, scheme, nu=0.8, T=1.0):
    h = 1.0 / N; x = np.arange(N) * h
    nsteps = int(np.ceil(T / (nu * h))); nu = T / nsteps / h   # c = 1
    U = np.sin(2 * np.pi * x)
    for n in range(nsteps):
        U = step(U, nu, scheme)
    return np.max(np.abs(U - np.sin(2 * np.pi * (x - T))))


names = ("upwind", "LF", "LW", "BW")
print("    N  " + "".join(f"{s:>9s}   rate " for s in names))
prev = None
for N in (25, 50, 100, 200, 400, 800):
    e = [error(N, s) for s in names]
    line = f"{N:5d}  "
    for k in range(4):
        rate = "  -- " if prev is None else f"{np.log2(prev[k] / e[k]):5.2f}"
        line += f"{e[k]:9.2e}  {rate}"
    print(line)
    prev = e
