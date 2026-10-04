# Dense, banded and sparse linear algebra for differential equations
# Applied ODE & PDE with Python, App. A | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spla

# Model problem: -u'' = pi^2 sin(pi x), u(0) = u(1) = 0, exact u = sin(pi x)
N = 50
h = 1.0 / N
x = np.linspace(0, 1, N + 1)[1:-1]            # N - 1 interior points
b = np.pi**2 * np.sin(np.pi * x)
m = N - 1

# 1. Dense matrix and la.solve
A = (np.diag(2*np.ones(m)) - np.diag(np.ones(m-1), 1) - np.diag(np.ones(m-1), -1)) / h**2
u1 = la.solve(A, b)

# 2. Banded storage: rows = super-diagonal, diagonal, sub-diagonal
ab = np.zeros((3, m))
ab[0, 1:] = -1 / h**2; ab[1, :] = 2 / h**2; ab[2, :-1] = -1 / h**2
u2 = la.solve_banded((1, 1), ab, b)

# 3. Sparse matrix built with diags; spsolve and a reusable LU factorisation (splu)
As = sps.diags([-1, 2, -1], [-1, 0, 1], shape=(m, m), format="csc") / h**2
u3 = spla.spsolve(As, b)
lu = spla.splu(As)
u4 = lu.solve(b)
for name, u in (("solve", u1), ("solve_banded", u2), ("spsolve", u3), ("splu", u4)):
    print(f"{name:13s} max error = {np.max(np.abs(u - np.sin(np.pi*x))):.3e}")
print(f"dense storage {A.size} numbers, sparse storage {As.nnz} nonzeros")

# 4. Two dimensions with kron: the 5-point Laplacian on an m x m interior grid
I = sps.identity(m, format="csc")
A2 = sps.kron(I, As) + sps.kron(As, I)
print(f"2D Laplacian: shape {A2.shape}, nonzeros {A2.nnz}")

# 5. Matrix exponential: the system y' = M y has solution y(t) = expm(t M) y(0)
M = np.array([[0.0, 1.0], [-1.0, 0.0]])        # y1'' = -y1 written as a system
print("expm(pi/2 M) @ [1, 0] =", np.round(la.expm(np.pi/2 * M) @ np.array([1.0, 0.0]), 12))
