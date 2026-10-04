# An insulated end: ghost point versus one-sided Neumann condition
# Applied ODE & PDE with Python, Ch. 10 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np

T, r = 0.5, 0.4
exact = lambda x, t: np.exp(-np.pi**2 * t / 4) * np.sin(np.pi * x / 2)


def ftcs_neumann(N, ghost=True):
    """u_t = u_xx on [0,1], u(0,t) = 0, u_x(1,t) = 0, u(x,0) = sin(pi x / 2)."""
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N + 1)
    nsteps = int(round(T / (r * h**2))); rr = T / nsteps / h**2
    U = exact(x, 0.0)
    for n in range(nsteps):
        Unew = U.copy()
        Unew[1:-1] = U[1:-1] + rr * (U[2:] - 2 * U[1:-1] + U[:-2])
        if ghost:   # ghost value U_{N+1} = U_{N-1}  (central difference = 0)
            Unew[-1] = U[-1] + 2 * rr * (U[-2] - U[-1])
        else:       # one-sided (U_N - U_{N-1})/h = 0, first order
            Unew[-1] = Unew[-2]
        Unew[0] = 0.0
        U = Unew
    return np.max(np.abs(U - exact(x, T)))


print("   N    ghost point   rate    one-sided    rate")
prev = None
for N in (10, 20, 40, 80, 160):
    e = (ftcs_neumann(N, True), ftcs_neumann(N, False))
    rates = ("  -- ", "  -- ") if prev is None else \
        tuple(f"{np.log2(p / q):5.2f}" for p, q in zip(prev, e))
    print(f"{N:4d}   {e[0]:.3e}   {rates[0]}   {e[1]:.3e}   {rates[1]}")
    prev = e
