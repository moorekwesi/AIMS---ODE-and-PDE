# Three ways to impose a Neumann condition u'(0) = sigma
# Applied ODE & PDE with Python, Ch. 9 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

exact = lambda x: np.exp(x) * np.sin(np.pi * x) + x
f = lambda x: np.exp(x) * ((np.pi**2 - 1) * np.sin(np.pi * x)
                           - 2 * np.pi * np.cos(np.pi * x))
sigma, beta = np.pi + 1.0, 1.0           # u'(0) = pi + 1,  u(1) = 1


def solve_neumann(N, method):
    """-u'' = f on (0,1), u'(0) = sigma, u(1) = beta. Unknowns U_0..U_{N-1}."""
    h = 1.0 / N
    x = np.linspace(0, 1, N + 1)
    A = sps.lil_matrix((N, N))
    b = f(x[:-1]).copy()
    for j in range(1, N):                 # interior rows: (-U_{j-1}+2U_j-U_{j+1})/h^2
        A[j, j - 1], A[j, j] = -1 / h**2, 2 / h**2
        if j + 1 < N:
            A[j, j + 1] = -1 / h**2
    b[-1] += beta / h**2                  # U_N = beta is known
    if method == "one-sided O(h)":        # (U_1 - U_0)/h = sigma
        A[0, 0], A[0, 1], b[0] = -1 / h, 1 / h, sigma
    elif method == "one-sided O(h^2)":    # (-3U_0 + 4U_1 - U_2)/(2h) = sigma
        A[0, 0], A[0, 1], A[0, 2] = -3 / (2*h), 4 / (2*h), -1 / (2*h)
        b[0] = sigma
    else:                                 # ghost point U_{-1} = U_1 - 2 h sigma
        A[0, 0], A[0, 1] = 1 / h**2, -1 / h**2
        b[0] = f(0.0) / 2 - sigma / h
    U = spsolve(A.tocsr(), b)
    return x, np.append(U, beta)


methods = ["one-sided O(h)", "one-sided O(h^2)", "ghost point"]
print(f"{'N':>5} " + "".join(f"{m:>22}" for m in methods))
prev = {}
for N in [10, 20, 40, 80, 160, 320]:
    row = f"{N:5d} "
    for m in methods:
        x, U = solve_neumann(N, m)
        e = np.max(np.abs(U - exact(x)))
        p = f"({np.log2(prev[m] / e):4.2f})" if m in prev else " " * 6
        row += f"{e:15.3e} {p}"
        prev[m] = e
    print(row)
