# The five-point Laplacian on the unit square via Kronecker products
# Applied ODE & PDE with Python, Ch. 9 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

p = lambda s: s * (1 - s) * np.exp(s)          # u(x,y) = p(x) p(y), zero on boundary
p2 = lambda s: -s * (s + 3) * np.exp(s)        # p''(s)
exact = lambda X, Y: p(X) * p(Y)
f = lambda X, Y: -(p2(X) * p(Y) + p(X) * p2(Y))   # f = -Laplace(u)


def laplacian_2d(N):
    """Matrix of -Laplace_h on the (N-1)^2 interior points, lexicographic
    ordering k = i + j (N-1) (x index fastest)."""
    h = 1.0 / N
    m = N - 1
    T = sps.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m), format="csr") / h**2
    I = sps.identity(m, format="csr")
    A = sps.kron(I, T, format="csr") + sps.kron(T, I, format="csr")
    return A.tocsc()


def solve_poisson_2d(N):
    x = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, x)                   # X[j, i] = x_i, Y[j, i] = y_j
    U = np.zeros_like(X)                       # boundary values are zero
    b = f(X[1:-1, 1:-1], Y[1:-1, 1:-1]).ravel()   # row j, column i -> k
    U[1:-1, 1:-1] = spsolve(laplacian_2d(N), b).reshape(N - 1, N - 1)
    return X, Y, U


A = laplacian_2d(4)
print("N = 4: A is", A.shape, "with", A.nnz, "non-zeros; h^2 A =")
print((A.toarray() / 16).astype(int))
print(f"{'N':>5} {'unknowns':>9} {'max error':>11} {'order':>6}")
prev = None
for N in [8, 16, 32, 64, 128, 256]:
    X, Y, U = solve_poisson_2d(N)
    err = np.max(np.abs(U - exact(X, Y)))
    order = f"{np.log2(prev / err):6.3f}" if prev else ""
    print(f"{N:5d} {(N - 1)**2:9d} {err:11.3e} {order}")
    prev = err

X, Y, U = solve_poisson_2d(32)
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot_surface(X, Y, U, cmap="viridis", linewidth=0)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("U, N = 32")
ax2 = fig.add_subplot(1, 2, 2)
cs = ax2.contourf(X, Y, U - exact(X, Y), 20, cmap="RdBu_r")
fig.colorbar(cs, ax=ax2)
ax2.set_aspect("equal"); ax2.set_title("error U - u")
ax2.set_xlabel("x"); ax2.set_ylabel("y")
plt.tight_layout()
plt.savefig("ch09_poisson2d.pdf", bbox_inches="tight")
plt.close()
