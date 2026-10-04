# Method of manufactured solutions for a 2D Crank-Nicolson heat code
# Applied ODE & PDE with Python, Ch. 12 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import sympy as sp
import scipy.sparse as sps
from scipy.sparse.linalg import splu

# 1. Manufacture a solution and let SymPy compute the source f = u_t - Laplace(u)
x, y, t = sp.symbols("x y t")
u_sym = sp.exp(-t) * sp.sin(sp.pi * x) * sp.sin(sp.pi * y) * sp.exp(x + y)
f_sym = sp.simplify(sp.diff(u_sym, t) - sp.diff(u_sym, x, 2) - sp.diff(u_sym, y, 2))
print("f =", sp.sstr(f_sym))
u_ex = sp.lambdify((x, y, t), u_sym, "numpy")
f_ex = sp.lambdify((x, y, t), f_sym, "numpy")


# 2. The code under test: Crank-Nicolson + five-point Laplacian, u = 0 on boundary
def heat2d_cn(N, T=0.5, buggy=False):
    h = 1.0 / N
    dt = h                                      # balanced: error O(dt^2 + h^2)
    m = N - 1
    D = sps.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(m, m), format="csr") / h**2
    I = sps.identity(m, format="csr")
    L = sps.kron(I, D, format="csr") + sps.kron(D, I, format="csr")   # Laplacian
    Id = sps.identity(m * m, format="csc")
    lu = splu((Id - dt / 2 * L).tocsc())        # factorise once
    Bexp = Id + dt / 2 * L
    xs = np.linspace(0, 1, N + 1)[1:-1]
    X, Y = np.meshgrid(xs, xs)
    U = u_ex(X, Y, 0.0).ravel()
    nsteps = int(round(T / dt))
    for n in range(nsteps):
        tn = n * dt
        if buggy:                               # source at the old time level only
            F = f_ex(X, Y, tn).ravel()
        else:                                   # source at t^{n+1/2} (average)
            F = 0.5 * (f_ex(X, Y, tn) + f_ex(X, Y, tn + dt)).ravel()
        U = lu.solve(Bexp @ U + dt * F)
    return np.max(np.abs(U - u_ex(X, Y, nsteps * dt).ravel()))


# 3. Grid refinement: the observed order must approach the theoretical order 2
print(f"{'N':>5} {'error (correct)':>16} {'p':>6} {'error (bug)':>12} {'p':>6}")
prev = None
for N in [8, 16, 32, 64, 128]:
    e, eb = heat2d_cn(N), heat2d_cn(N, buggy=True)
    if prev:
        print(f"{N:5d} {e:16.3e} {np.log2(prev[0] / e):6.3f} {eb:12.3e} "
              f"{np.log2(prev[1] / eb):6.3f}")
    else:
        print(f"{N:5d} {e:16.3e} {'':6} {eb:12.3e}")
    prev = (e, eb)
