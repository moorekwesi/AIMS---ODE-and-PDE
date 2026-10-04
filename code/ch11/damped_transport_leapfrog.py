# Leapfrog (CTCS) scheme for the damped transport equation u_t + (3/4) u_x + u = 0
# Applied ODE & PDE with Python, Ch. 11 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np

a = 0.75
exact = lambda x, t: (x - a * t)**2 * np.exp(-t)    # u(x, 0) = x^2


def leapfrog(N, T, nu=0.5, averaged=False):
    """CTCS on [0, 1]; exact inflow data at x = 0, upwind outflow at x = 1.
    averaged=True replaces u^n_j in the damping term by (u^{n+1}_j + u^{n-1}_j)/2."""
    h = 1.0 / N; x = np.linspace(0, 1, N + 1)
    nsteps = int(np.ceil(T / (nu * h / a))); dt = T / nsteps; nu = a * dt / h
    Uold = exact(x, 0.0)
    # starting step: second-order Taylor, u_t = -(a u_x + u), u_tt = (a d_x + 1)^2 u
    D0 = lambda V: (V[2:] - V[:-2]) / (2 * h)
    D2 = lambda V: (V[2:] - 2 * V[1:-1] + V[:-2]) / h**2
    V = Uold[1:-1]
    U = Uold.copy()
    U[1:-1] = V - dt * (a * D0(Uold) + V) + 0.5 * dt**2 * (a**2 * D2(Uold)
                                                            + 2 * a * D0(Uold) + V)
    U[0] = exact(0.0, dt); U[-1] = Uold[-1] - nu * (Uold[-1] - Uold[-2]) - dt * Uold[-1]
    for n in range(1, nsteps):
        Unew = np.empty_like(U)
        flux = nu * (U[2:] - U[:-2])
        if averaged:
            Unew[1:-1] = ((1 - dt) * Uold[1:-1] - flux) / (1 + dt)
        else:
            Unew[1:-1] = Uold[1:-1] - flux - 2 * dt * U[1:-1]
        Unew[0] = exact(0.0, (n + 1) * dt)                        # inflow
        Unew[-1] = U[-1] - nu * (U[-1] - U[-2]) - dt * U[-1]      # outflow (upwind)
        Uold, U = U, Unew
    return np.max(np.abs(U - exact(x, T))), np.max(np.abs(exact(x, T)))


print("Leapfrog, nu = 0.5, T = 1")
print("    N    max error    rate")
prev = None
for N in (10, 20, 40, 80, 160, 320):
    e, _ = leapfrog(N, 1.0)
    print(f"{N:5d}   {e:.3e}   " + ("  --" if prev is None else f"{np.log2(prev / e):5.2f}"))
    prev = e
print("Long-time behaviour, N = 40")
print("    T    max|u|      error (plain)   error (averaged damping)")
for T in (5.0, 10.0, 20.0, 30.0):
    e1, umax = leapfrog(40, T); e2, _ = leapfrog(40, T, averaged=True)
    print(f"{T:5.0f}   {umax:.2e}    {e1:.3e}       {e2:.3e}")
