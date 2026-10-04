# Method of lines for the heat equation with solve_ivp
# Applied ODE & PDE with Python, Ch. 10 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import scipy.sparse as sps
from scipy.integrate import solve_ivp

kappa, T = 1.0, 0.1


def exact(x, t):
    """Two-mode Fourier solution of u_t = kappa u_xx, u(0,t) = u(1,t) = 0."""
    return (np.exp(-np.pi**2 * kappa * t) * np.sin(np.pi * x)
            + 0.5 * np.exp(-9 * np.pi**2 * kappa * t) * np.sin(3 * np.pi * x))


def second_difference(N, h):
    """Sparse (N-1)x(N-1) matrix of (U[j-1] - 2U[j] + U[j+1]) / h^2."""
    e = np.ones(N - 1)
    return sps.diags([e[:-1], -2 * e, e[:-1]], [-1, 0, 1], format="csr") / h**2


print("   N  lambda_max   method   nfev   max error")
for N in (20, 40, 80):
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N + 1)
    A = kappa * second_difference(N, h)
    U0 = exact(x[1:-1], 0.0)
    lam_max = 4 * kappa / h**2 * np.sin((N - 1) * np.pi / (2 * N))**2
    for method in ("RK45", "BDF"):
        extra = {"jac": A} if method == "BDF" else {}
        sol = solve_ivp(lambda t, U: A @ U, (0.0, T), U0, method=method,
                        rtol=1e-6, atol=1e-9, **extra)
        err = np.max(np.abs(sol.y[:, -1] - exact(x[1:-1], T)))
        print(f"{N:4d}  {lam_max:9.1f}   {method:5s} {sol.nfev:6d}   {err:.2e}")
