# NumPy essentials: grids, slicing and finite differences
# Applied ODE & PDE with Python, App. A | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np

# Arrays and grids
x = np.linspace(0.0, 1.0, 6)              # 6 points = 5 subintervals, endpoints included
print("x          =", x)
print("x[0], x[-1] =", x[0], x[-1], "  x[1:-1] =", x[1:-1])
A = np.arange(12).reshape(3, 4)            # integers 0..11 in 3 rows and 4 columns
print("A.shape =", A.shape, "  A[1, :] =", A[1, :], "  A[:, 2] =", A[:, 2])

# Finite differences by slicing, on u(x) = sin(pi x)
for N in (10, 20, 40, 80):
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N + 1)       # x_j = j h, j = 0..N
    u = np.sin(np.pi * x)
    du = (u[2:] - u[:-2]) / (2 * h)                # centred first derivative
    d2u = (u[2:] - 2 * u[1:-1] + u[:-2]) / h**2    # second derivative, interior points
    e1 = np.max(np.abs(du - np.pi * np.cos(np.pi * x[1:-1])))
    e2 = np.max(np.abs(d2u + np.pi**2 * u[1:-1]))
    print(f"N = {N:3d}  h = {h:.4f}  error u' = {e1:.3e}  error u'' = {e2:.3e}")

# Broadcasting and meshgrid: a function of two variables on a grid
xg, yg = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 2, 4))   # xy indexing
print("meshgrid shapes:", xg.shape, yg.shape)
p, q = np.arange(3.0), np.arange(4.0)
print("broadcasting p[:, None] + q[None, :] has shape", (p[:, None] + q[None, :]).shape)
