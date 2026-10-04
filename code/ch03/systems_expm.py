# Linear systems x' = A x: eigenvalue method versus the matrix exponential
# Applied ODE & PDE with Python, Ch. 3 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

np.set_printoptions(precision=6, suppress=True)

# companion matrix of y'' + 3y' + 2y = 0 (x1 = y, x2 = y')
A = np.array([[0.0, 1.0], [-2.0, -3.0]])
x0 = np.array([1.0, 0.0])
lam, V = np.linalg.eig(A)
print("eigenvalues :", lam)
print("eigenvectors (columns):\n", V)

t = 1.5
c = np.linalg.solve(V, x0)                      # x0 = c1 v1 + c2 v2
x_eig = V @ (c * np.exp(lam * t))               # sum c_i e^{lam_i t} v_i
x_exp = expm(A * t) @ x0
sol = solve_ivp(lambda s, x: A @ x, (0, t), x0, rtol=1e-12, atol=1e-14)
y_exact = 2 * np.exp(-t) - np.exp(-2 * t)       # from the scalar equation
print(f"x(1.5) eigen : {x_eig}")
print(f"x(1.5) expm  : {x_exp}")
print(f"x(1.5) ivp   : {sol.y[:, -1]}")
print(f"y(1.5) exact : {y_exact:.6f}")

# a defective matrix: one eigenvector only, exp(tB) contains t e^{-2t}
B = np.array([[-2.0, 1.0], [0.0, -2.0]])
print("eig(B):", np.linalg.eig(B)[0])
print("expm(B) :\n", expm(B))
print("formula e^{-2}[[1, 1], [0, 1]]:\n", np.exp(-2) * np.array([[1, 1], [0, 1]]))
