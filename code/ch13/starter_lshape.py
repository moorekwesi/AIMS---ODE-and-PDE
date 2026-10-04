# Starter code: Laplace equation on an L-shaped domain (corner singularity)
# Applied ODE & PDE with Python, Ch. 13 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def exact(x, y):
    """u = r^(2/3) sin(2 phi/3), phi in [0, 3pi/2] measured from the positive y-axis."""
    r = np.hypot(x, y)
    phi = np.mod(np.arctan2(y, x) - np.pi / 2, 2 * np.pi)
    return r**(2 / 3) * np.sin(2 * phi / 3)


def solve_lshape(n):
    h = 1.0 / 2**n
    M = 2 * 2**n                                  # grid points -1 + i h, i = 0..M
    s = np.linspace(-1, 1, M + 1)
    X, Y = np.meshgrid(s, s, indexing="ij")
    inside = (np.abs(X) < 1) & (np.abs(Y) < 1) & ((X < 0) | (Y < 0))
    idx = -np.ones(X.shape, int)
    idx[inside] = np.arange(inside.sum())         # number the unknowns
    G = exact(X, Y)                               # boundary values (and exact solution)
    A = sps.lil_matrix((inside.sum(), inside.sum()))
    b = np.zeros(inside.sum())
    for i, j in zip(*np.nonzero(inside)):
        k = idx[i, j]
        A[k, k] = 4.0
        for ii, jj in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
            if inside[ii, jj]:
                A[k, idx[ii, jj]] = -1.0
            else:
                b[k] += G[ii, jj]                 # known boundary value moves to the rhs
    U = G.copy()
    U[inside] = spla.spsolve(A.tocsr(), b)
    return X, Y, U, np.max(np.abs(U - G)[inside])


print(" n     h        unknowns   max error   order")
prev = None
for n in range(1, 8):
    X, Y, U, err = solve_lshape(n)
    unk = int(((np.abs(X) < 1) & (np.abs(Y) < 1) & ((X < 0) | (Y < 0))).sum())
    order = "" if prev is None else f"{np.log2(prev / err):6.3f}"
    print(f"{n:2d}  {1 / 2**n:.5f}  {unk:8d}   {err:.3e}   {order}")
    prev = err

X, Y, U, err = solve_lshape(5)
U[(X >= 0) & (Y >= 0) & ~((X == 0) | (Y == 0))] = np.nan     # hide the cut-out square
plt.figure(figsize=(5, 4.2))
plt.contourf(X, Y, U, levels=20, cmap="viridis")
plt.colorbar(label="u")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.savefig("ch13_lshape_starter.pdf", bbox_inches="tight")
plt.close()
