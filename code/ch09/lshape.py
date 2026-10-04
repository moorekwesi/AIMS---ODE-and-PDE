# Poisson's equation -Laplace(u) = 1 on an L-shaped domain
# Applied ODE & PDE with Python, Ch. 9 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def laplacian_2d(N):
    """-Laplace_h on the (N-1)^2 interior points of the unit square."""
    h, m = 1.0 / N, N - 1
    T = sps.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m), format="csr") / h**2
    I = sps.identity(m, format="csr")
    return sps.kron(I, T, format="csr") + sps.kron(T, I, format="csr")


def solve_lshape(N):
    """L = (0,1)^2 minus [1/2,1]x[1/2,1], u = 0 on the boundary, N even."""
    x = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, x)
    Xi, Yi = X[1:-1, 1:-1].ravel(), Y[1:-1, 1:-1].ravel()
    inside = ~((Xi >= 0.5) & (Yi >= 0.5))      # interior nodes of the L
    A = laplacian_2d(N)[inside][:, inside]     # zero Dirichlet data: just drop
    Ui = np.zeros((N - 1) ** 2)
    Ui[inside] = spsolve(A.tocsc(), np.ones(inside.sum()))
    U = np.zeros((N + 1, N + 1))               # boundary values are zero
    U[1:-1, 1:-1] = Ui.reshape(N - 1, N - 1)
    U[(X > 0.5) & (Y > 0.5)] = np.nan          # outside the domain
    return X, Y, U, inside.sum()


print(f"{'N':>4} {'unknowns':>9} {'U(1/4,1/4)':>12} {'difference':>11} {'ratio':>6}")
vals = []
for N in [8, 16, 32, 64, 128, 256]:
    X, Y, U, n = solve_lshape(N)
    vals.append(U[N // 4, N // 4])
    line = f"{N:4d} {n:9d} {vals[-1]:12.8f}"
    if len(vals) > 1:
        line += f" {vals[-1] - vals[-2]:11.3e}"
    if len(vals) > 2:
        line += f" {(vals[-2] - vals[-3]) / (vals[-1] - vals[-2]):6.2f}"
    print(line)

X, Y, U, _ = solve_lshape(64)
plt.figure(figsize=(5, 4.2))
cs = plt.contourf(X, Y, np.ma.masked_invalid(U), 20, cmap="viridis")
plt.colorbar(cs)
plt.plot([0, 1, 1, 0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5, 1, 1, 0], "k-", lw=1.5)
plt.gca().set_aspect("equal")
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout()
plt.savefig("ch09_lshape.pdf", bbox_inches="tight")
plt.close()
