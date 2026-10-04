# The 1D Poisson problem -u'' = f, u(0) = alpha, u(1) = beta
# Applied ODE & PDE with Python, Ch. 9 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt


def exact(x):
    return np.exp(x) * np.sin(np.pi * x) + x


def f(x):                                   # f = -u'' for the exact solution
    return np.exp(x) * ((np.pi**2 - 1) * np.sin(np.pi * x)
                        - 2 * np.pi * np.cos(np.pi * x))


def poisson1d(f, alpha, beta, N, banded=False):
    """Second-order FD solution on N subintervals; returns grid and U_0..U_N."""
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N + 1)
    b = f(x[1:-1])                          # right-hand side at interior points
    b[0] += alpha / h**2                    # known boundary values move to the RHS
    b[-1] += beta / h**2
    m = N - 1                               # number of unknowns
    if banded:                              # 3 x m array of diagonals for LAPACK
        ab = np.zeros((3, m))
        ab[0, 1:], ab[1, :], ab[2, :-1] = -1.0, 2.0, -1.0
        Ui = solve_banded((1, 1), ab / h**2, b)
    else:
        A = sps.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m), format="csr")
        Ui = spsolve(A / h**2, b)
    return x, np.concatenate(([alpha], Ui, [beta]))


alpha, beta = exact(0.0), exact(1.0)
print(f"{'n':>2} {'N':>5} {'h':>10} {'max error':>11} {'ratio':>7} {'order':>6}")
prev = None
for n in range(1, 11):
    N = 2**n
    x, U = poisson1d(f, alpha, beta, N)
    err = np.max(np.abs(U - exact(x)))
    if prev is None:
        print(f"{n:2d} {N:5d} {1/N:10.3e} {err:11.3e}")
    else:
        print(f"{n:2d} {N:5d} {1/N:10.3e} {err:11.3e} {prev/err:7.3f} "
              f"{np.log2(prev/err):6.3f}")
    prev = err
x, U1 = poisson1d(f, alpha, beta, 1000)
x, U2 = poisson1d(f, alpha, beta, 1000, banded=True)
print(f"N = 1000: max |spsolve - solve_banded| = {np.max(np.abs(U1 - U2)):.2e}")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
xf = np.linspace(0, 1, 400)
ax[0].plot(xf, exact(xf), "k-", lw=1, label="exact u")
for N, m in [(4, "s"), (8, "o")]:
    x, U = poisson1d(f, alpha, beta, N)
    ax[0].plot(x, U, m + "--", label=f"U, N = {N}")
ax[0].set_xlabel("x"); ax[0].legend()
x, U = poisson1d(f, alpha, beta, 32)
ax[1].plot(x, U - exact(x), "o-", ms=3)
ax[1].set_xlabel("x"); ax[1].set_ylabel("U_j - u(x_j)")
ax[1].set_title("error, N = 32")
plt.tight_layout()
plt.savefig("ch09_poisson1d.pdf", bbox_inches="tight")
plt.close()
